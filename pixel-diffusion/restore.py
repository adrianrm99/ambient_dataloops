# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
from tqdm import tqdm
import pickle
import numpy as np
from ambient_utils import save_image
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import joblib
from huggingface_hub import hf_hub_download
import json
from generate import StackedRandomGenerator, load_hf_checkpoint, parse_int_list
from ambient_utils.dataset import SyntheticallyCorruptedImageFolderDataset
import importlib
from torch_utils.ambient import load_annotations
from collections import defaultdict
#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_restorer(
    x, sigma_start, sigma_end,
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    add_noise=True,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Get indexes where do not touch
    is_clean_sample = sigma_start == 0

    # Adjust bounds
    sigma_start_clamp = torch.max(sigma_start, torch.full_like(sigma_start, sigma_min)).to(device=x.device, dtype=x.dtype)
    sigma_end_clamp = torch.max(sigma_end, torch.full_like(sigma_end, sigma_min)).to(device=x.device, dtype=x.dtype)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_start_clamp[None, :] ** (1 / rho) + step_indices[:, None] / (num_steps - 1) * (sigma_end_clamp[None, :] ** (1 / rho) - sigma_start_clamp[None, :] ** (1 / rho))) ** rho
    final_step = sigma_end.to(device=x.device, dtype=x.dtype)
    t_steps = torch.cat([net.round_sigma(t_steps), final_step[None, :]], 0) # t_N = 0
    num_steps = len(t_steps) - 1

    # Main sampling loop.
    # print(x.shape, latents.shape, t_steps.shape)
    x_next = x + latents.to(torch.float64) * t_steps[0, :, None, None, None] * float(add_noise == True)
    expected = None
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = torch.where((S_min <= t_cur) & (t_cur <= S_max), min(S_churn / num_steps, np.sqrt(2) - 1), 0.0)
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt()[:, None, None, None] * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        if expected is None:
            expected = denoised

        d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
        x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next[:, None, None, None]
            x_next = x_hat + (t_next - t_hat)[:, None, None, None] * (0.5 * d_cur + 0.5 * d_prime)

    x_next[is_clean_sample] = x[is_clean_sample].to(dtype=x_next.dtype)
    return x_next

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--dataset_path',                  help='Where the images come from', metavar='DIR',                   type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--rho_ambient', help="How much to decrease the variance", type=float, default=None)

def main(network_pkl, dataset_path, outdir, subdirs, seeds, class_idx, max_batch_size, rho_ambient, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')

    if "pkl" in network_pkl:
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
    else:
        net = load_hf_checkpoint(network_pkl).to(device).eval()

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Load options and dataset
    base_folder = os.path.dirname(network_pkl)
    with open(os.path.join(base_folder, "training_options.json"), "r", encoding="utf-8") as f:
        options = json.load(f)
    corruptions_dict = importlib.import_module(f"noise_configs.inference.identity").corruptions_dict
    options['dataset_kwargs']['corruptions_dict'] = corruptions_dict
    del options['dataset_kwargs']['noise_config']
    del options['dataset_kwargs']['dataset_keep_percentage']  # TODO: ?
    options['dataset_kwargs']['path'] = dataset_path
    options['dataset_kwargs']['corruption_probability'] = 0.0
    options['dataset_kwargs']['normalize'] = not dataset_path.endswith('/npy')

    dataset_obj = SyntheticallyCorruptedImageFolderDataset(**options['dataset_kwargs'])
    dataset_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=max_batch_size,
            shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_obj, shuffle=False)
    )

    # Get old annotations
    print('Checking old annotations')
    prev_annotations = load_annotations(options['dataset_kwargs']['path'])

    # Loop over dataset
    torch.distributed.barrier()
    process_id = torch.distributed.get_rank()

    if process_id == 0:
        # Make dirs
        os.makedirs(outdir, exist_ok=True)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

    seeds_idx = 0
    for dataset_item in tqdm(dataset_loader):

        # Image information
        images = dataset_item["image"].to("cuda")
        class_labels = dataset_item["label"].to("cuda")  # attention: this is NOT the label for good or bad image. This is more like a class label (dog, cat, etc.)
        image_names = [dataset_item['filename'][i].split("/")[-1] for i in range(images.size(0))]
        idxs = [int(image_name.replace('.png', '').replace('img', '').replace('.npy', '')) for image_name in image_names]

        # Pick latents
        batch_seeds = rank_batches[seeds_idx:seeds_idx + images.size(0)]
        seeds_idx += images.size(0)
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([images.size(0), net.img_channels, net.img_resolution, net.img_resolution], device=device)

        # Get sigma_tn
        sigma_tn = torch.Tensor([prev_annotations[image_name][0] for image_name in image_names])
        # images = images + sigma_tn * latents

        # Restore
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        sampler_kwargs['add_noise'] = not dataset_path.endswith('/npy')
        sampler_fn = edm_restorer
        sigma_start = sigma_tn
        sigma_end = sigma_start / rho_ambient if rho_ambient is not None else torch.zeros_like(sigma_start)
        with torch.no_grad():
            restored_images = sampler_fn(images, sigma_start, torch.zeros_like(sigma_end), net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
            restored_images = restored_images.to(dtype=images.dtype)

        # Save restored images
        restored_images_paths = [os.path.join(outdir, image_name.replace('.npy', '.png')) for image_name in image_names]
        for x, path in zip(restored_images, restored_images_paths):
            save_image(x, path)

        # Write to process-specific annotation file
        process_id = torch.distributed.get_rank()
        # Iterate PNG
        process_annotations_path = os.path.join(outdir, f"annotations_{process_id}.jsonl")
        with open(process_annotations_path, "a", encoding="utf-8") as f:
            for i, image_name in enumerate(image_names):
                annotation = {
                    "filename": image_name,
                    "sigma_min": sigma_end[i].item(),
                    "sigma_max": 0.0,
                }
                f.write(json.dumps(annotation) + "\n")


    # After all processes finish, merge files
    torch.distributed.barrier()
    # torch.distributed.monitored_barrier(timeout=datetime.timedelta(days=1))
    if process_id == 0:
        # Merge iterate PNG
        final_annotations_path = os.path.join(outdir, "annotations.jsonl")
        with open(final_annotations_path, "a+", encoding="utf-8") as outfile:
            world_size = torch.distributed.get_world_size()
            for pid in range(world_size):
                proc_file = os.path.join(outdir, f"annotations_{pid}.jsonl")
                if os.path.exists(proc_file):
                    with open(proc_file, encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    os.remove(proc_file)  # Clean up process file

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

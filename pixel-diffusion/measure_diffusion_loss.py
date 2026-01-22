import argparse
from ambient_utils import save_image
from ambient_utils import dist_utils
from ambient_utils.dataset import SyntheticallyCorruptedImageFolderDataset
from ambient_utils.classifier import get_classifier_trajectory
import torch
from torch_utils.misc import copy_params_and_buffers
import pickle
import dnnlib
import os
import json
import importlib
from slurm_jobs.utils import find_training_folders_based_on_params, find_nearest_checkpoint, find_latest_checkpoint
import matplotlib.pyplot as plt
from filelock import FileLock
from tqdm import tqdm
import datetime
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the clean dataset.")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the losses.")
parser.add_argument("--dataset", type=str, default="cifar", help="Dataset name.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--num_sigmas", type=int, default=256, help="How many times to query the classifier for each t.")
parser.add_argument("--num_trials_per_t", type=int, default=4, help="How many times to query the classifier for each t.")
# TODO (@giannisdaras): update this path once we get a better one
parser.add_argument("--checkpoint_path", type=str, 
                    default="/scratch/07362/gdaras/cls_runs/00005-cifar10-32x32-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-cQrt8/network-snapshot-005018.pkl", 
                    help="Checkpoint path.")
parser.add_argument("--checkpoint_index", type=int, default=200_000, help="Checkpoint index.")

def load_net_from_pkl(ckpt_file):
    base_folder = os.path.dirname(ckpt_file)
    with open(os.path.join(base_folder, "training_options.json"), "r", encoding="utf-8") as f:
        options = json.load(f)

    interface_kwargs = dict(img_resolution=options['dataset_kwargs']['resolution'], img_channels=3, label_dim=0)
    
    net = dnnlib.util.construct_class_by_name(**options['network_kwargs'], **interface_kwargs)
    with dnnlib.util.open_url(ckpt_file) as f:
        data = pickle.load(f)
    copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    return net, options


import torch
from typing import Callable

def get_loss_trajectory(
    loss_fn,
    input: torch.Tensor,
    model: torch.nn.Module,
    diffusion_times: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 1,
    **model_kwargs,
) -> torch.Tensor:
    """
    Get the trajectory of the classifier for a given input and diffusion times.
    
    Args:
        input: Input tensor to classify
        model: The classifier model
        scheduler: Function that takes two tensor arguments and returns a tensor
        diffusion_times: Tensor of diffusion timesteps
        device: Device to run model on
        batch_size: Number of inputs to process in parallel at once.
        model_output_type: The type of output that the model returns.
    Returns:
        torch.Tensor: Model output predictions
    """
    # model.eval()
    predictions = []

    def process_t(t):
        t = t.unsqueeze(0).repeat(input.shape[0])
        output = model(input, t, **model_kwargs).squeeze()
        loss, _, _, _ = loss_fn(net=model, x_tn=input, sigma_tn=torch.zeros_like(t), sigma_t=t)
        loss = loss.mean()
        return loss

    vmapped_fn = torch.func.vmap(process_t, randomness="different", chunk_size=batch_size)
    with torch.no_grad():
        predictions = vmapped_fn(diffusion_times)
    return predictions

def main(args):
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    
    dist_utils.print0("Total nodes: ", dist_utils.get_world_size())

    if args.checkpoint_path is None:
        training_folders = find_training_folders_based_on_params(corruption_probability=0.5, 
                                                                noise_config=args.training_noise_config, 
                                                                dataset_keep_percentage=1.0, 
                                                                dataset=args.dataset)        
        checkpoint_path = find_nearest_checkpoint(training_folders, checkpoint_index=args.checkpoint_index, pkl=True)    
    else:
        checkpoint_path = args.checkpoint_path

    net, options = load_net_from_pkl(checkpoint_path)
    net.eval().to("cuda")
    
    # prepare params for synthetic dataset corruption
    corruptions_dict = importlib.import_module(f"noise_configs.inference.identity").corruptions_dict
    options['dataset_kwargs']['corruptions_dict'] = corruptions_dict
    del options['dataset_kwargs']['noise_config']
    del options['dataset_kwargs']['dataset_keep_percentage']  # TODO: ?
    # overwrite the corruption probability so that we can create different types of datasets, as needed.

    options['dataset_kwargs']['path'] = args.dataset_path
    options['dataset_kwargs']['corruption_probability'] = 0.0


    dataset_obj = SyntheticallyCorruptedImageFolderDataset(**options['dataset_kwargs'])
    dataset_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=1,
            shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_obj, shuffle=False)
            )

    # Get loss fn
    loss_fn = dnnlib.util.construct_class_by_name(**options['loss_kwargs'])
    print('loss_fn', loss_fn)

    # rnd_normal = torch.randn([4096, 1, 1, 1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))  # it is very important to set this seed to have consistency with the seed used during training.
    # rnd_normal = torch.randn([2048, 1, 1, 1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))  # it is very important to set this seed to have consistency with the seed used during training.
    # rnd_normal = torch.randn([256, 1, 1, 1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))  # it is very important to set this seed to have consistency with the seed used during training.
    rnd_normal = torch.randn([args.num_sigmas, 1, 1, 1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))  # it is very important to set this seed to have consistency with the seed used during training.
    sigmas, _ = (rnd_normal * 1.2 - 1.2).exp().sort(dim=0)
    sigmas = sigmas.squeeze()

    if dist_utils.get_rank() == 0:
        os.makedirs(args.output_path, exist_ok=True)
        # dump the sigmas into a file
        with open(os.path.join(args.output_path, "sigmas_measure_diffusion_loss.txt"), "w") as f:
            for sigma in sigmas:
                f.write(f"{sigma.item()}\n")

    # Save config
    with open(os.path.join(args.output_path, 'config_dump.json'), "a", encoding="utf-8") as f:
        del options['dataset_kwargs']['corruptions_dict']
        f.write(json.dumps(options) + "\n")

    # Resume
    process_id = torch.distributed.get_rank()
    annotations_file = os.path.join(args.output_path, f"measure_diffusion_loss_{process_id}.jsonl")
    annotations = {}
    if os.path.exists(annotations_file):
        with open(annotations_file, "r") as f:
            for line in f:
                line_json = json.loads(line)
                filename = line_json["filename"]
                annotations[filename] = True

    # all models wait for the path to be created
    torch.distributed.barrier()
    process_id = torch.distributed.get_rank()
    for dataset_item in tqdm(dataset_loader):
        images = dataset_item["image"].to("cuda").repeat(args.num_trials_per_t, 1, 1, 1)
        labels = dataset_item["label"].to("cuda").repeat(args.num_trials_per_t, 1)  # attention: this is NOT the label for good or bad image. This is more like a class label (dog, cat, etc.)

        image_name = dataset_item['filename'][0].split("/")[-1]
        if image_name not in annotations:

            # Write to process-specific annotation file
            process_id = torch.distributed.get_rank()
            process_annotations_path = os.path.join(args.output_path, f"measure_diffusion_loss_{process_id}.jsonl")

            image_losses = get_loss_trajectory(
                loss_fn=loss_fn,
                input=images,
                model=net,
                diffusion_times=sigmas,
                batch_size=args.batch_size,
            ).detach().cpu().numpy().tolist()
            
            with open(process_annotations_path, "a", encoding="utf-8") as f:
                annotation = {
                    "filename": image_name,
                    "image_losses": image_losses
                }
                f.write(json.dumps(annotation) + "\n")
    
    # After all processes finish, merge files
    torch.distributed.barrier()
    # torch.distributed.monitored_barrier(timeout=datetime.timedelta(days=1))
    if process_id == 0:
        # Merge all process files into single annotations file
        final_annotations_path = os.path.join(args.output_path, "measure_diffusion_loss.jsonl")
        with open(final_annotations_path, "a+", encoding="utf-8") as outfile:
            world_size = torch.distributed.get_world_size()
            for pid in range(world_size):
                proc_file = os.path.join(args.output_path, f"measure_diffusion_loss_{pid}.jsonl")
                if os.path.exists(proc_file):
                    with open(proc_file, encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    os.remove(proc_file)  # Clean up process file

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

import os
import io
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from composer.utils import dist, reproducibility
from micro_diffusion.models.utils import text_encoder_embedding_format, DATA_TYPES
import torch.distributed as dist
from ambient_utils import dist_utils
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from typing import Optional
from functools import partial
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import numpy as np
from torchvision.utils import make_grid, save_image

torch.backends.cudnn.benchmark = True  # 3-5% speedup

@torch.no_grad()
def latents_to_image(model, latents):
    latents = 1 / model.latent_scale * latents
    torch_dtype = DATA_TYPES[model.dtype]
    image = model.vae.decode(latents.to(torch_dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float().detach()
    return image

@torch.no_grad()
def edm_restore_loop(
    model, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    sigma_tn: torch.Tensor,
    steps: Optional[int] = None, 
    cfg: float = 1.0, 
    **kwargs
) -> torch.Tensor:
    mask_ratio = 0  # no masking during image generation
    model_forward_fxn = (
        partial(model.dit.forward, cfg=cfg) if cfg > 1.0
        else model.dit.forward
    )

    # Time step discretization.
    num_steps = model.edm_config.num_steps if steps is None else steps
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=x.device)
    t_steps = (
        model.edm_config.sigma_max ** (1 / model.edm_config.rho) +
        step_indices / (num_steps - 1) *
        (model.edm_config.sigma_min ** (1 / model.edm_config.rho) -
          model.edm_config.sigma_max ** (1 / model.edm_config.rho))
    ) ** model.edm_config.rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

    # Start at sigma_tn
    i = torch.where(t_steps >= sigma_tn)[0][-1].item()
    t_steps[i] = sigma_tn
    t_steps = t_steps[i:]
    num_steps = len(t_steps) - 1

    # Main sampling loop.
    x_next = x.to(torch.float64) + torch.randn_like(x, dtype=torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = (
            min(model.edm_config.S_churn / num_steps, np.sqrt(2) - 1)
            if model.edm_config.S_min <= t_cur <= model.edm_config.S_max else 0
        )
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = (
            x_cur +
            (t_hat ** 2 - t_cur ** 2).sqrt() *
            model.edm_config.S_noise *
            model.randn_like(x_cur)
        )

        # Euler step.
        denoised = model.model_forward_wrapper(
            x_hat.to(torch.float32),
            t_hat.to(torch.float32),
            y,
            model_forward_fxn,
            mask_ratio=mask_ratio,
            **kwargs
        )['sample'].to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model.model_forward_wrapper(
                x_next.to(torch.float32),
                t_next.to(torch.float32),
                y,
                model_forward_fxn,
                mask_ratio=mask_ratio,
                **kwargs
            )['sample'].to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.to(torch.float32)

@torch.no_grad()
def edm_restore(
    model,
    latents,
    text_embeddings,
    sigma_tn,
    guidance_scale,
    num_inference_steps,
    seed: Optional[int] = None,
    **kwargs
) -> torch.Tensor:
    # _check_prompt_given(prompt, tokenized_prompts, prompt_embeds=None)
    device = model.vae.device  # hack to identify model device during training
    rng_generator = torch.Generator(device=device)
    if seed:
        rng_generator = rng_generator.manual_seed(seed)

    # iteratively denoise latents
    latents = edm_restore_loop(
        model,
        latents,
        text_embeddings,
        sigma_tn,
        num_inference_steps,
        cfg=guidance_scale
    )

    # Decode latents with VAE
    image = latents_to_image(model, latents)
    return latents, image

@hydra.main(version_base=None)
def generate(cfg: DictConfig) -> None:
    """Generate images using a trained micro-diffusion model."""
    # Set NODE_RANK for distributed training
    NODE_RANK = int(os.environ.get('RANK', 0))
    os.environ['NODE_RANK'] = str(NODE_RANK)

    if not cfg:
        raise ValueError('Config not specified. Please provide --config-path and --config-name, respectively.')
    
    reproducibility.seed_all(cfg.get('seed', 42))

    # Check if checkpoint path is provided
    if 'checkpoint_path' not in cfg:
        raise ValueError('checkpoint_path must be specified in config for inference')
    
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path.startswith('giannisdaras/'):
        checkpoint_path = hf_hub_download(repo_id="giannisdaras/ambient-o", filename="model.safetensors")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("Instantiating model")
    model = hydra.utils.instantiate(cfg.model)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    if checkpoint_path.endswith('.safetensors'):
        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'dit.' prefix if present in checkpoint keys
        if any(key.startswith('dit.') for key in state_dict.keys()):
            state_dict = {k.replace('dit.', ''): v for k, v in state_dict.items() if k.startswith('dit.')}
        model.dit.load_state_dict(state_dict)
    elif 'state' in checkpoint:
        state_dict = checkpoint['state']['model']
        # Remove 'dit.' prefix if present in checkpoint keys
        if any(key.startswith('dit.') for key in state_dict.keys()):
            state_dict = {k.replace('dit.', ''): v for k, v in state_dict.items() if k.startswith('dit.')}
        model.dit.load_state_dict(state_dict)
    elif 'model' in checkpoint:
        model.dit.load_state_dict(checkpoint['model'])
    else:
        model.dit.load_state_dict(checkpoint)
    
    print("Checkpoint loaded successfully")

    # Set up data loader
    cap_seq_size, cap_emb_dim = text_encoder_embedding_format(cfg.model.text_encoder_name)
    
    data_loader = hydra.utils.instantiate(
        cfg.dataset.eval,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.dataset.get('gen_batch_size', cfg.dataset.eval_batch_size) // dist.get_world_size(),
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim,
        shuffle=False,
        drop_last=False
    )

    print(f"Found {len(data_loader.dataset)*dist.get_world_size()} samples in the dataset")

    # Set models to eval mode and move to device
    device = next(model.dit.parameters()).device
    model.vae.to(device)
    model.text_encoder.to(device)
    model.dit.eval()
    model.vae.eval()
    model.text_encoder.eval()

    # Set up output directories
    output_dir = cfg.get('output_dir', './outputs/restored_images')
    output_dir = os.path.join(output_dir, f'sigma-tn-{cfg.sigma_tn}', f'guidance-scale-{cfg.guidance_scale}_sampling-steps-{cfg.sampling_steps}')
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(output_dir, 'cfg.yaml'))
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank_output_dir = os.path.join(output_dir, str(rank))
    
    # Define MDS columns
    columns = {
        'caption': 'str',
        'caption_latents': 'bytes',
        'latents_256': 'bytes',
        'latents_512': 'bytes'
    }
    
    # Initialize MDS writer
    writer = MDSWriter(
        out=rank_output_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )
    
    # Handle num_batches configuration
    num_batches = cfg.get('num_batches', len(data_loader))
    if num_batches == -1:
        num_batches = len(data_loader)

    # Get sigma_tn
    sigma_tn = cfg.get('sigma_tn')

    # Send model to device
    device = 'cuda'
    model.dit.to(device)
    model.vae.to(device)
    model.text_encoder.to(device)

    # Restore images
    print(f"Starting image restoration with sigma_tn {sigma_tn}")


    torch_dtype = DATA_TYPES[model.dtype]
    print(f'model_device={device}, model_dtype={torch_dtype}')
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating images", total=num_batches)):
            if batch_idx >= num_batches:
                break

            # Get latents
            latents_512 = batch['latents_512'].to(device=device, dtype=torch_dtype)
            caption_latents = batch['caption_latents'].to(device=device, dtype=torch_dtype)

            # Save example
            if batch_idx == 0:
                original_images_512 = latents_to_image(model, latents_512)
                # save_image(make_grid(original_images_512, nrow=int(original_images_512.size(0)**0.5)), os.path.join(output_dir, f'original_images_512_{str(rank)}.png'))
            
            # restore images
            latents_512_list = []
            images_512_list = []
            for restoration_idx in range(cfg.num_restorations):
                latents_512, images_512 = edm_restore(
                    model,
                    latents=latents_512,
                    text_embeddings=caption_latents,
                    sigma_tn=sigma_tn,
                    guidance_scale=cfg.guidance_scale,
                    num_inference_steps=cfg.sampling_steps,
                    seed=cfg.seed + len(data_loader.dataset) * restoration_idx,
                )
                latents_512_list.append(latents_512)
                images_512_list.append(images_512)
            latents_512 = torch.stack(latents_512_list)
            images_512 = torch.stack(images_512_list)

            # print(latents_512.shape, images_512.shape)

            # Save example
            if batch_idx == 0:
                restored_images_512 = images_512
                restored_images_512 = restored_images_512.permute(1, 2, 0, 3, 4).flatten(2, 3)
                images_to_save = torch.cat((original_images_512, restored_images_512), dim=-2)
                save_image(make_grid(images_to_save, nrow=int(restored_images_512.size(0))), os.path.join(output_dir, f'original_vs_restored_images_512_{str(rank)}.png'))
            images_512 = images_512[0]
            latents_512 = latents_512[0]

            # Resize images from 512x512 to 256x256
            images_256 = F.interpolate(
                images_512,
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            )
            
            # Re-encode 256x256 images to get latents_256
            latents_256 = model.vae.encode(images_256.to(torch_dtype))['latent_dist'].sample().data * model.vae.config.scaling_factor

            # Convert to save dtype
            caption_latents = caption_latents.to(DATA_TYPES[cfg.save_dtype])
            latents_256 = latents_256.to(DATA_TYPES[cfg.save_dtype])
            latents_512 = latents_512.to(DATA_TYPES[cfg.save_dtype])
            
            # Save each sample to MDS
            for i in range(len(latents_512)):
                mds_sample = {
                    'caption': batch['caption'][i],
                    'caption_latents': caption_latents[i].cpu().numpy().tobytes(),
                    'latents_256': latents_256[i].cpu().numpy().tobytes(),
                    'latents_512': latents_512[i].cpu().numpy().tobytes()
                }
                writer.write(mds_sample)
    
    # Finish writing
    writer.finish()
    
    # Wait for all processes to finish
    if dist.is_initialized():
        dist.barrier()
    
    print(f"Process {rank} finished")
    
    # Merge shards on main process
    if rank == 0:
        print("Merging MDS shards...")
        shards_metadata = [
            os.path.join(output_dir, str(i), 'index.json')
            for i in range(world_size)
        ]
        merge_index(shards_metadata, out=output_dir, keep_local=True)
        print(f"Generation complete! MDS dataset saved to {output_dir}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
    dist_utils.print0("Total nodes: ", dist_utils.get_world_size())
    generate()
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
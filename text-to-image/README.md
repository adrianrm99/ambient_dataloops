
# Ambient Dataloops + Text-to-Image Diffusion w/ Micro-Diffusion
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/adrianrm/ambient-dataloops-text2image)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/adrianrm/ambient-dataloops)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

This part of the repo focuses on large-scale text-to-image diffusion experiments with [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion) using Conceptual Captions, Segment Anything-1B, TextCaps, JourneyDB, and DiffusionDB. Here, we show how just by using the data better, you can improve the performance and quality of your generative models. The key idea is to refine our lower quality model-generated images. High-quality data (real or strong model synthetic) is kept as-is, while low-quality data is refined by using posterior sampling starting from the diffusion time it was used during the Ambient Omni stage (loop 0).

## Results
This allows us to obtain much better generation quality than without data refinement, as we can see the following figure:
![Generated Images](../figs/loops_tti.jpg)

## Using our models
You can generate your own images with the following snippet, which uses our ambient model on [huggingface](https://huggingface.co/adrianrm/ambient-dataloops)
```python
import torch
from micro_diffusion.models.model import create_latent_diffusion
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Init model
params = {
    'latent_res': 64,
    'in_channels': 4,
    'pos_interp_scale': 2.0,
}
model = create_latent_diffusion(**params).to('cuda')

# Download weights from HF
model_dict_path = hf_hub_download(repo_id="adrianrm/ambient-dataloops", filename="model.safetensors")
model_dict = {}
with safe_open(model_dict_path, framework="pt", device="cpu") as f:
   for key in f.keys():
       model_dict[key] = f.get_tensor(key)

# Convert parameters to float32 + load
float_model_params = {
    k: v.to(torch.float32) for k, v in model_dict.items()
}
model.dit.load_state_dict(float_model_params)

# Eval mode
model = model.eval()

# Generate images
prompts = [
    "A giraffe standing in an open field next to some rocks.",
    "A bike parked next to a red door on the front of a house.",
    "An apple tree filled with lots of apples.",
    "An empty train station has very nice clocks.",
    "A parking lot filled with buses parked next to each other."
    "Panda mad scientist mixing sparkling chemicals, artstation",
    "the sailor galaxia. beautiful, realistic painting by mucha and kuvshinov and bilibin. watercolor, thick lining, manga, soviet realism",
]
images = model.generate(prompt=prompts, num_inference_steps=30, guidance_scale=5.0, seed=42)
```

## Training your own

If you want to train your own, you must follow four steps:
1. Prepare your environment and data following the instructions in the Micro-Diffusion repo
2. Download ours, or train your own loop0 generative model (using the ambient-o algorithm).
3. Refine the data from DiffusionDB using the ambient-o model.
4. Train your loop1 generative model.

### 1. Prepare your environment and data in the Micro-Diffusion format

Follow the instructions in the original [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion) repository (but using our edited version of their code).

### 2. Download or train your loop0 (ambient-o) diffusion model

We provide scripts for training the loop0 model (`scripts/train_e2e_ambient.sh`), or you can download it from our [huggingface repository](https://huggingface.co/giannisdaras/ambient-o).

### 3. Refine the data from Diffusion-DB using the ambient-o model

Use the script `scripts/restore_diffdb_ambient.sh` to refine the DiffusionDB dataset. It uses the public huggingface ambient checkpoint by default, but you can change this by setting the `checkpoint_path` field `configs_restore/restore_ambient-sigma-2_diffdb_sigma-2.yaml`.

### 4. Train your loop1 diffusion model on the refined data

Use the script `scripts/train_e2e_loops.sh` to train the loop1 model. It fine-tunes an intermediate checkpoint from the loop0 run. You can obtain it by training loop0 yourself or by downloading from our [huggingface](https://huggingface.co/adrianrm/ambient-dataloops). Either way, you should have the file `./trained_models/OmniDiTXL_mask_75_res_512_pretrain/latest-rank0.pt`.

## COCO Evaluation

The script for generating the images is `scripts/generate_coco_loops.sh`. The generation scripts use our [huggingface loops checkpoint](https://huggingface.co/adrianrm/ambient-dataloops) by default, but you can change the path to your own models. The script for evaluating FID is `scripts/eval_fid.sh`.

# 🔗 Related Codebases

* [Ambient Omni](https://github.com/giannisdaras/ambient-omni): starting point for this repository, which itself started from [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion).
* [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion): starting point for Ambient Omni, and the code for obtaining the datasets.
* [Ambient utils](https://github.com/giannisdaras/ambient-utils): helper functions for training diffusion models (or flow matching models) in settings with limited access to high-quality data.
* [Ambient Laws](https://github.com/giannisdaras/ambient-laws): trains models with a mix of clean and noisy data.
* [Ambient Diffusion](https://github.com/giannisdaras/ambient-diffusion): trains models for linear corruptions.
* [Consistent Diffusion Meets Tweedie](https://github.com/giannisdaras/ambient-tweedie): trains models with only noisy data, with support for Stable Diffusion finetuning.
* [Consistent Diffusion Models](https://github.com/giannisdaras/cdm): original implementation of the consistency loss.


# 📧 Contact

If you are interested in colaborating, please reach out to adrianrm[at]mit[dot]edu and gdaras[at]mit[dot]edu.

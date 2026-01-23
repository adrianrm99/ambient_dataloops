
# Ambient Dataloops + Pixel Diffusion w/ EDM

This part of the repo focuses on small-scale experiments with pixel diffusion using Cifar-10. Here we investigate the effects of dataloops in a controlled corruption setting. We provide scripts for:
1. Artificially corrupt 90\% of Cifar-10
2. Train an Ambient Omni model using the corrupted data, and evaluate it on the uncorrupted Cifar-10
3. Restore the corrupted data using the model + posterior sampling
4. Train a new Ambient Dataloops model using the restored data, and evaluate it on the uncorrupted Cifar-10

## 1. Artificially corrupt 90% of Cifar-10

Use the scripts in `scripts/annotate_fixed_sigma`. This will require access to Cifar-10, which we assume is stored in `./data/cifar10/train`. We suggest either making a symlink, or copying the data.

For example `scripts/annotate_fixed_sigma/blur0-6_fixed_1-2.sh` will artifically corrupt a random 90% of Cifar-10 with a blur corruption with $\sigma_B=0.6$, and annotate it with $\sigma_{\text{min}}=1.2$

## 2. Train an Ambient Omni model using the corrupted data, and evaluate it on the uncorrupted Cifar-10

### 2.a Train an Ambient Omni model using the corrupted data

Use the scripts in `scripts/train_omni`. The code assumes that a folder such as `dataset_path=./outputs/annotated_cifar10/fixed_sigma/blur0-6/sigma_1.2` exists and contains the corrupted Cifar-10 data from step 1.

For example `scripts/train_omni/blur0-6_fixed_1-2.sh` will train an Ambient Omni model from the corrupted Cifar-10 with blur corruption with $\sigma_B=0.6$ and $\sigma_{\text{min}}=1.2$.

### 2.b Evaluate it on the uncorrupted Cifar-10

Use the scripts in `scripts/eval_omni`. The script will require you to set path to the trained checkpoint from 2.a, i.e. `ckpt_dir=fixed_sigma/blur0-6/sigma_1.2/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-8DVvm`. The code expects the path to be relative to `./outputs/ambient-syn-runs/cifar/train_omni/`.

## 3. Restore the corrupted data using the model + posterior sampling

Use the scripts in `scripts/restore_omni`. Same as with 2.b, the script will require you to set path to the trained checkpoint from 2.a, i.e. `ckpt_dir=fixed_sigma/blur0-6/sigma_1.2/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-8DVvm`. Additionally, it will need the iteration number of the best checkpoint (or whichever you want to use to restore) in list form, i.e. `iters_list=("017060")`. The `rho_ambient=8` parameter controls how much we reduce the minimum noise level $\sigma_{\text{min}}$ (in this case by a factor of 8).

## 4. Train a new Ambient Dataloops model using the restored data, and evaluate it on the uncorrupted Cifar-10

### 4.a Train a new Ambient Dataloops model using the restored data

Use the scripts in `scripts/train_loop1`. Same as with step 2.a, the code assumes that a folder such as `dataset_path=./outputs/ambient-syn-runs/cifar/restore_omni/fixed_sigma/blur0-6/sigma_1.2/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-8DVvm/rho_ambient_8/seeds0-49999` exists and contains the restored Cifar-10 data from step 3.

### 4.b Evaluate it on the uncorrupted Cifar-10

Use the scripts in `scripts/eval_loop1`. The script will require you to set path to the trained checkpoint from 4.a, i.e. `ckpt_dir=fixed_sigma/blur0-6/sigma_1.2/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-8DVvm/rho_ambient_8/seeds0-49999/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-6TDmr`. The code expects the path to be relative to `./outputs/ambient-syn-runs/cifar/train_loop1/`.

# 🔗 Related Codebases

* [Ambient Omni](https://github.com/giannisdaras/ambient-omni): starting point for this repository, which itself started from [EDM](https://github.com/NVlabs/edm).
* [EDM](https://github.com/NVlabs/edm): starting point for Ambient Omni.
* [Ambient Laws](https://github.com/giannisdaras/ambient-laws): trains models with a mix of clean and noisy data.
* [Ambient Diffusion](https://github.com/giannisdaras/ambient-diffusion): trains models for linear corruptions.
* [Consistent Diffusion Meets Tweedie](https://github.com/giannisdaras/ambient-tweedie): trains models with only noisy data, with support for Stable Diffusion finetuning.
* [Consistent Diffusion Models](https://github.com/giannisdaras/cdm): original implementation of the consistency loss.


# 📧 Contact

If you are interested in colaborating, please reach out to adrianrm[at]mit[dot]edu and gdaras[at]mit[dot]edu.



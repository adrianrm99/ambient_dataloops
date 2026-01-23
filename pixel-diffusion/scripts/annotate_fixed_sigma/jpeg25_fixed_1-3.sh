#!/bin/bash
PYTHONPATH=.
annotated_dataset_path=./outputs/annotated_cifar10/fixed_sigma
dataset_path=./data/cifar10/train # Replace with your own
noise_config=jpeg25
corruption_probability=0.9
sigma_min=1.3

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 ) + 10000 ))
export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=72000

torchrun --nproc_per_node=1 --standalone annotate_fixed_sigma.py \
--annotated_dataset_path=${annotated_dataset_path}/${noise_config}/sigma_${sigma_min} \
--dataset_path=${dataset_path} \
--inference_noise_config=${noise_config} \
--corruption_probability=${corruption_probability} \
--min_fixed_sigma=${sigma_min} \
--max_fixed_sigma=0.0
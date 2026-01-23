#!/bin/bash
PYTHONPATH=.
outdir=./outputs # Replace with your own
# Replace with your own; path to annotated dataset e.g. ${outdir}/ambient-syn-runs/cifar/restore_omni/fixed_sigma/jpeg25/sigma_1.3/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-8DVvm/rho_ambient_8/seeds0-49999
dataset_path=
s_max=4
dp=1.0
weight_decay=0.0
cls_epsilon=0.05

# Set experiment dir
save_path=$(echo "$dataset_path" | sed 's|.*/restore_omni/||')
outdir=${outdir}/ambient-syn-runs/cifar/train_loop1/${save_path}/s-max-${s_max}

# Extract jpeg and sigma
jpeg_config=$(echo "$dataset_path" | sed -n 's|.*jpeg\([0-9.]*\).*|\1|p')
sigma_min=$(echo "$dataset_path" | sed -n 's|.*sigma_\([0-9.]*\).*|\1|p')

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

mkdir -p $outdir
torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=20 --duration=200 \
            --precond=edm \
            --corruption_probability=0.0 \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --workers 2 \
            --cls_epsilon=${cls_epsilon} \
            --expr_id=train_low_quality_data_diffusion_cifar10_jpeg${jpeg_config}_sigma-min-${sigma_min}_s-max-${s_max}_dp${dp}_wd${weight_decay} \
            --s_max=${s_max}
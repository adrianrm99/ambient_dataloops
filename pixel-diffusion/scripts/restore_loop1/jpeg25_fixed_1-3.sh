#!/bin/bash
PYTHONPATH=.
ref_dataset_path=./data/cifar10/train # Replace with your own; path to clean cifar10
annotated_dataset_path=./outputs/annotated_cifar10/fixed_sigma/jpeg25/sigma_1.3
# Replace with your own e.g. fixed_sigma/jpeg25/sigma_1.3/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-xN0co
ckpt_dir=
dp=1.0
weight_decay=0.0
sigma_tn=$(echo "$dataset_path" | sed -n 's|.*sigma_\([0-9.]*\).*|\1|p')
rho_ambient=8
iters_list=("018063")
seeds=0-49999
#seeds=50000-99999
#seeds=100000-149999

# MASTER PORT
MASTER_PORT=$(( ( RANDOM % 1000 ) + 10000 ))
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=72000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_BLOCKING_WAIT=1

# Double for loop to iterate over both lists
for iter_num in "${iters_list[@]}"; do
    # For FID purposes
    ckpt_name=${ckpt_dir}/network-snapshot-$iter_num
    ckpt_path=./outputs/ambient-syn-runs/cifar/train_loop1/${ckpt_name}.pkl
    restore_path=./outputs/ambient-syn-runs/cifar/restore_loop1/${ckpt_name}/rho_ambient_{rho_ambient}/seeds${seeds}
    mkdir -p $restore_path
    
    # Randomize torchrun master_port
    MASTER_PORT=$(( ( RANDOM % 1000 ) + 10000 ))
    echo "Using pretrained checkpoint: $ckpt_path"
    echo "Checkpoint found: $ckpt_path"
    echo "Using sigma_tn=$sigma_tn, rho_ambient=$rho_ambient"
    
    # Generate
    torchrun --master_port $MASTER_PORT --nproc_per_node=8 restore.py --seeds=$seeds --network=$ckpt_path \
    --outdir=$restore_path  --steps=18 --dataset_path=$annotated_dataset_path --rho_ambient=$rho_ambient

    # FID
    output=$(torchrun --standalone eval_fid.py --gen_path=$restore_path --ref_path=$ref_dataset_path)
    FID=$(echo "$output" | grep "FID score:" | awk '{print $3}')
    INCEPTION=$(echo "$output" | grep "Inception score:" | awk '{print $3}')
    echo "Dataset=cifar, Checkpoint=$ckpt_path, sigma_tn=$sigma_tn, rho_ambient=$rho_ambient, FID=$FID, INCEPTION=$INCEPTION"
    echo "Dataset=cifar, Checkpoint=$ckpt_path, sigma_tn=$sigma_tn, rho_ambient=$rho_ambient, FID=$FID, INCEPTION=$INCEPTION" >> $restore_path/eval.txt
done
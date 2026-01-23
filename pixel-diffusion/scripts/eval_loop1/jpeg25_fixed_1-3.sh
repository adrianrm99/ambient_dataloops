#!/bin/bash
PYTHONPATH=.
dataset_path=./data/cifar10/train # Replace with your own; path to clean cifar10
# Replace with your own e.g. fixed_sigma/blur0-6/sigma_1.2/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-8DVvm/rho_ambient_8/seeds0-49999/s-max-4/00000-npy-uncond-ddpmpp-edm-gpus8-batch512-fp32-6TDmr
ckpt_dir=
dp=1.0
weight_decay=0.0
seeds=0-49999
#seeds=50000-99999
#seeds=100000-149999

iters_dir=./outputs/ambient-syn-runs/cifar/train_loop1/$ckpt_dir
iters_list=($(ls "$iters_dir" | grep 'network-snapshot-' | sed 's/network-snapshot-\([0-9]*\)\.pkl/\1/' | sort -n))

# Filter to keep only checkpoints between 10000 and 31000 (inclusive)
filtered_iters=()
for iter in "${iters_list[@]}"; do
    if [ "$iter" -ge 10000 ] && [ "$iter" -le 31000 ]; then
        filtered_iters+=("$iter")
    fi
done
iters_list=("${filtered_iters[@]}")

# Double for loop to iterate over both lists
for iter_num in "${iters_list[@]}"; do
  ckpt_name=${ckpt_dir}/network-snapshot-$iter_num
  ckpt_path=./outputs/ambient-syn-runs/cifar/train_loop1/${ckpt_name}.pkl
  eval_path=./outputs/ambient-syn-evals/cifar/train_loop1/${ckpt_name}/seeds${seeds}

  mkdir -p $eval_path

  # Randomize torchrun master_port
  MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

  echo "Using pretrained checkpoint: $ckpt_path"
  echo "Checkpoint found: $ckpt_path"
  echo "Using seeds: $seeds"

  # Generate
  torchrun --master_port $MASTER_PORT --nproc_per_node=8 generate.py --seeds=$seeds --network=$ckpt_path \
  --outdir=$eval_path  --steps=18

  # FID
  output=$(torchrun --standalone eval_fid.py --gen_path=$eval_path --ref_path=$dataset_path)
  FID=$(echo "$output" | grep "FID score:" | awk '{print $3}')
  INCEPTION=$(echo "$output" | grep "Inception score:" | awk '{print $3}')
  echo "Dataset=cifar, Checkpoint=$ckpt_path, FID=$FID, INCEPTION=$INCEPTION"
  echo "Dataset=cifar, Checkpoint=$ckpt_path, FID=$FID, INCEPTION=$INCEPTION" >> $eval_path/eval.txt
done
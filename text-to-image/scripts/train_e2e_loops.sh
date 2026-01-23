#!/bin/bash

train() {
   # 1: yaml_path, 2: yaml_name, $3: update_args
   pkill -9 python; python -c 'import streaming; streaming.base.util.clean_stale_shared_memory()' # alternative hack: rm -rf /dev/shm/0000*
   rm -rf /tmp/streaming/*
   wait;
   sleep 3

   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HYDRA_FULL_ERROR=1 composer train.py --config-path $1 --config-name $2  $3
}

# Step-4: Finetuning at 512x512 image resolution with no patch masking
train ./configs_loops res_512_finetune.yaml "exp_name=LoopsDiTXL_mask_0_res_512_finetune model.train_mask_ratio=0.0 trainer.load_path=./trained_models/OmniDiTXL_mask_75_res_512_pretrain/latest-rank0.pt"
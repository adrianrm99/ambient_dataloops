#!/bin/bash

restore() {
   # 1: yaml_path, 2: yaml_name, $3: update_args
   # pkill -9 python
   python -c 'import streaming; streaming.base.util.clean_stale_shared_memory()' # alternative hack: rm -rf /dev/shm/0000*
   rm -rf /tmp/streaming/*
   wait;
   sleep 3

   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 HYDRA_FULL_ERROR=1 composer restore.py --config-path $1 --config-name $2  $3
}

# Restore diffdb with sigma=2
diffdb_dir=./datadir/diffdb/mds_latents_sdxl1_dfnclipH14/
restore ./configs_restore restore_ambient-sigma-2_diffdb_sigma-2.yaml "dataset.eval.datadir=${diffdb_dir} guidance_scale=3.5"
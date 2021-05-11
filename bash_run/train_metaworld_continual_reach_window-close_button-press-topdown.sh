#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$SCRIPT_DIR/..

export PYTHONPATH=$PROJECT_DIR
export PWD=$PROJECT_DIR
export CUDA_VISIBLE_DEVICES=7

python src/train_locomotion.py \
  --env_names reach-v2 window-close-v2 button-press-topdown-v2 \
  --env_type metaworld \
  --algo sac_mlp_ss_ensem \
  --train_steps_per_task 1000000 \
  --eval_freq_per_task 50000 \
  --init_steps 1500 \
  --work_dir vec_logs/reach_window-close_button-press-topdown/no_ensem/0 \
  --save_model \
  --save_video \
  --save_tb

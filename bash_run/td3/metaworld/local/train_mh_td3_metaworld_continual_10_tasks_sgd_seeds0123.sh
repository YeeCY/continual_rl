#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES="$seed"
  nohup \
  python $PROJECT_DIR/src/train_td3.py \
    --env_names \
      window-close-v2 \
      button-press-topdown-v2 \
      door-open-v2 \
      shelf-place-v2 \
      door-lock-v2 \
      box-close-v2 \
      sweep-into-v2 \
      faucet-close-v2 \
      coffee-push-v2 \
      assembly-v2 \
    --env_type metaworld \
    --algo mh_td3_mlp \
    --train_steps_per_task 500000 \
    --eval_freq 10 \
    --discount 0.99 \
    --td3_init_steps 1000 \
    --td3_num_expl_steps_per_process 1000 \
    --td3_num_processes 1 \
    --td3_num_train_iters 1000 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/mh_td3_mlp_metaworld_10_tasks/sgd/$seed \
    > $PROJECT_DIR/terminal_logs/mh_td3_mlp_metaworld_10_tasks-sgd-seed"$seed".log 2>&1 &
done

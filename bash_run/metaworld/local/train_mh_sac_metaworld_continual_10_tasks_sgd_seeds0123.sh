#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES="$seed"
  nohup \
  python $PROJECT_DIR/src/train_sac.py \
    --env_names \
      window-close-v1 \
      button-press-topdown-v1 \
      door-open-v1 \
      peg-insert-side-v1 \
      door-lock-v1 \
      push-v1 \
      sweep-into-v1 \
      faucet-close-v1 \
      coffee-push-v1 \
      assembly-v1 \
    --env_type metaworld \
    --algo mh_sac_mlp \
    --train_steps_per_task 500000 \
    --eval_freq 10 \
    --discount 0.99 \
    --sac_init_steps 1000 \
    --sac_num_expl_steps_per_process 1000 \
    --sac_num_processes 1 \
    --sac_num_train_iters 1000 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/mh_sac_mlp_metaworld_5_tasks/sgd/$seed \
    > $PROJECT_DIR/terminal_logs/mh_sac_mlp_metaworld_5_tasks-sgd-seed"$seed".log 2>&1 &
done

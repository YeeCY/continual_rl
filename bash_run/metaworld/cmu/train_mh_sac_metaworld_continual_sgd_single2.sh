#!/bin/bash

source $HOME/.bashrc
source $HOME/cyzheng/env_vars

conda activate pad

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a all_env_names=(
  handle-pull-side-v2
  handle-pull-v2
  plate-slide-back-v2
  plate-slide-side-v2
  plate-slide-v2
)

declare -a seeds=(4 5 6 7)

for env_names in "${all_env_names[@]}"; do
  for seed in "${seeds[@]}"; do
    export CUDA_VISIBLE_DEVICES="$(("$seed" - 4))"
    nohup \
    python $PROJECT_DIR/src/train_sac.py \
      --env_names $env_names \
      --env_type metaworld \
      --algo mh_sac_mlp \
      --train_steps_per_task 500000 \
      --eval_freq 10 \
      --num_eval_episodes 10 \
      --discount 0.99 \
      --sac_init_steps 1000 \
      --sac_num_expl_steps_per_process 1000 \
      --sac_num_processes 1 \
      --sac_num_train_iters 1000 \
      --seed $seed \
      --save_video \
      --work_dir $PROJECT_DIR/vec_logs/mh_sac_mlp_metaworld_single/sgd/$env_names/$seed \
      > $PROJECT_DIR/terminal_logs/mh_sac_mlp_metaworld_single-sgd-"$env_names"-seed"$seed".log 2>&1 &
  done
done

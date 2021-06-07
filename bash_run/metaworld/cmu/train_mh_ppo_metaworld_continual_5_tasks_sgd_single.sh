#!/bin/bash

source $HOME/.bashrc
source $HOME/cyzheng/env_vars

conda activate continual_rl

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a all_env_names=(window-close-v2 button-press-topdown-v2 door-open-v2 peg-insert-side-v2 door-lock-v2)
declare -a seeds=(0)

for env_names in "${all_env_names[@]}"; do
  for seed in "${seeds[@]}"; do
    export CUDA_VISIBLE_DEVICES=$seed
    nohup \
    python $PROJECT_DIR/src/train_ppo.py \
      --env_names $env_names \
      --env_type metaworld \
      --algo mh_ppo_mlp \
      --train_steps_per_task 1000000 \
      --eval_freq 10 \
      --discount 0.99 \
      --ppo_num_batch 256 \
      --ppo_num_rollout_steps_per_process 1000 \
      --ppo_num_processes 1 \
      --ppo_use_clipped_critic_loss \
      --ppo_use_proper_time_limits \
      --seed $seed \
      --work_dir $PROJECT_DIR/vec_logs/mh_ppo_mlp_metaworld_5_tasks_single/sgd/$env_names/$seed \
        > $PROJECT_DIR/terminal_logs/mh_ppo_mlp_metaworld_5_tasks_single-sgd-"$env_names"-seed"$seed".log 2>&1 &
  done
done

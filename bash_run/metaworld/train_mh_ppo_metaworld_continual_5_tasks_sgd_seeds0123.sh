#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$SCRIPT_DIR/..

export PYTHONPATH=$PROJECT_DIR
export CUDA_VISIBLE_DEVICES=0

declare -a seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
  python $PROJECT_DIR/src/train_ppo.py \
    --env_names window-close-v2 button-press-topdown-v2 peg-insert-side-v2 door-open-v2 push-v2 \
    --env_type metaworld \
    --algo mh_ppo_mlp \
    --train_steps_per_task 1000000 \
    --eval_freq 10 \
    --discount 0.99 \
    --ppo_num_batch 256 \
    --ppo_num_rollout_steps_per_process 2048 \
    --ppo_num_processes 1 \
    --ppo_use_clipped_critic_loss \
    --ppo_use_proper_time_limits \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/mh_metaworld_5_tasks/sgd/$seed \
    --save_model
done

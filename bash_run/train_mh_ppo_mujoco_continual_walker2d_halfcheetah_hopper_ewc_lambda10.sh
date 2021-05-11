#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$SCRIPT_DIR/..

export PYTHONPATH=$PROJECT_DIR
export CUDA_VISIBLE_DEVICES=2

declare -a seeds=(0 1 2)

for seed in "${seeds[@]}"; do
  xvfb-run -a -s "-screen 0 1400x900x24" python $PROJECT_DIR/src/train_ppo.py \
    --env_names Walker2d-v3 HalfCheetah-v3 Hopper-v3 \
    --env_type mujoco \
    --algo ewc_mh_ppo_mlp \
    --train_steps_per_task 1000000 \
    --save_freq 1 \
    --eval_freq 1 \
    --discount 0.99 \
    --ppo_num_batch 64 \
    --ppo_num_rollout_steps_per_process 2048 \
    --ppo_num_processes 8 \
    --ppo_use_clipped_critic_loss \
    --ppo_ewc_lambda 10 \
    --ppo_ewc_estimate_fisher_epochs 100 \
    --ppo_ewc_rollout_steps_per_process 20 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/mh_walker2d_halfcheetah_hopper/ewc_lambda10/$seed \
    --save_model
done

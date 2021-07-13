#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES=$seed
  nohup \
  python $PROJECT_DIR/src/train_ppo.py \
    --env_names \
      Walker2d-v3 \
      HalfCheetah-v3 \
      Hopper-v3 \
      Pusher-v2 \
      Ant-v3 \
      Striker-v2 \
      InvertedPendulum-v2 \
      Reacher-v2 \
      Swimmer-v3 \
      Humanoid-v3 \
    --env_type mujoco \
    --algo ewc_mh_ppo_mlp_v2 \
    --train_steps_per_task 1000000 \
    --eval_freq 10 \
    --discount 0.99 \
    --ppo_num_rollout_steps_per_process 2048 \
    --ppo_num_processes 1 \
    --ppo_use_clipped_critic_loss \
    --ppo_use_proper_time_limits \
    --ppo_ewc_lambda 5000 \
    --ppo_ewc_estimate_fisher_epochs 10 \
    --ppo_ewc_rollout_steps_per_process 2048 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/mh_ppo_mlp_v2_mujoco_10_tasks/ewc_lambda5000/$seed \
    > $PROJECT_DIR/terminal_logs/mh_ppo_mlp_v2_mujoco_10_tasks-ewc_lambda5000-seed"$seed".log 2>&1 &
done

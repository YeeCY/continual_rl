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
      window-close-v2 \
      button-press-topdown-v2 \
      door-open-v2 \
      coffee-button-v2 \
      plate-slide-side-v2 \
      push-v2 \
      faucet-close-v2 \
      handle-press-v2 \
      reach-wall-v2 \
      window-open-v2 \
    --env_type metaworld \
    --add_onehot True \
    --algo agem_ppo_mlp_v2 \
    --train_steps_per_task 500000 \
    --eval_freq 10 \
    --discount 0.99 \
    --ppo_hidden_dim 512 \
    --ppo_num_rollout_steps_per_process 1000 \
    --ppo_num_processes 1 \
    --ppo_use_clipped_critic_loss \
    --ppo_use_proper_time_limits \
    --ppo_agem_memory_budget 5000 \
    --ppo_agem_ref_grad_batch_size 2500 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/ppo_mlp_v2_overparam512_onehot_metaworld_10_tasks/agem_ref_grad_batch_size2500/$seed \
  > $PROJECT_DIR/terminal_logs/ppo_mlp_v2_overparam512_onehot_metaworld_10_tasks-agem_ref_grad_batch_size2500-seed"$seed".log 2>&1 &
done

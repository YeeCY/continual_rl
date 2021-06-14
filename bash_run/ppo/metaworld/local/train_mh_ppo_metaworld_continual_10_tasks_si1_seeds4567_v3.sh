#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(4 5 6 7)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES="$(($seed - 4))"
  nohup \
  python $PROJECT_DIR/src/train_ppo.py \
    --env_names \
      window-close-v2 \
      button-press-topdown-v2 \
      door-open-v2 \
      coffee-button-v2 \
      plate-slide-side-v2 \
      sweep-into-v2 \
      faucet-close-v2 \
      window-open-v2 \
      handle-press-v2 \
      drawer-close-v2 \
    --env_type metaworld \
    --algo si_mh_ppo_mlp \
    --train_steps_per_task 500000 \
    --eval_freq 10 \
    --discount 0.99 \
    --ppo_num_rollout_steps_per_process 1000 \
    --ppo_num_processes 1 \
    --ppo_use_clipped_critic_loss \
    --ppo_use_proper_time_limits \
    --ppo_si_c 1 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/mh_ppo_mlp_metaworld_10_tasks_v3/si_c1/$seed \
  > $PROJECT_DIR/terminal_logs/mh_ppo_mlp_metaworld_10_tasks_v3-si_c1-seed"$seed".log 2>&1 &
done

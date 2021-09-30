#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES="$seed"
  python $PROJECT_DIR/src/train_sac.py \
    --env_names \
      window-close-v2 \
      button-press-topdown-v2 \
      door-open-v2 \
      coffee-button-v2 \
      plate-slide-side-v2 \
      sweep-into-v2 \
      faucet-close-v2 \
      door-lock-v2 \
      handle-pull-side-v2 \
      window-open-v2 \
    --env_type metaworld \
    --algo sac_mlp \
    --train_steps_per_task 500000 \
    --eval_freq 10 \
    --log_freq 5 \
    --discount 0.99 \
    --replay_buffer_capacity 500000 \
    --sac_actor_hidden_dim 256 \
    --sac_critic_hidden_dim 256 \
    --sac_init_steps 1000 \
    --sac_num_expl_steps_per_process 125 \
    --sac_num_processes 8 \
    --sac_num_train_iters 1000 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/vectorized_sac_mlp_metaworld/sgd/$seed \
    > $PROJECT_DIR/terminal_logs/vectorized_sac_mlp_metaworld-sgd-seed"$seed".log 2>&1 &
done

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
    --env_type metaworld \
    --algo sparse_gp_hypernet_actor_sac_mlp \
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
    --sac_num_train_iters 300 \
    --sac_gp_num_inducing_points 2000 \
    --seed $seed \
    --save_task_model True \
    --work_dir $PROJECT_DIR/vec_logs/gp_hypernet_sac_mlp_metaworld/inducing_points2000/$seed \
    > $PROJECT_DIR/terminal_logs/gp_hypernet_sac_mlp_metaworld-inducing_points2000-seed"$seed".log 2>&1 &
done

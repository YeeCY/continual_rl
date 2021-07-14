#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES="$seed"
  nohup \
  python $PROJECT_DIR/src/train_sac.py \
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
    --algo mh_sac_mlp_v2 \
    --train_steps_per_task 1000000 \
    --eval_freq 10 \
    --discount 0.99 \
    --sac_actor_hidden_dim 256 \
    --sac_init_steps 2048 \
    --sac_num_expl_steps_per_process 2048 \
    --sac_num_processes 1 \
    --sac_num_train_iters 2048 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/mh_sac_mlp_v2_mujoco_10_tasks_improvement/sgd/$seed \
    > $PROJECT_DIR/terminal_logs/mh_sac_mlp_v2_mujoco_10_tasks_improvement-sgd-seed"$seed".log 2>&1 &
done

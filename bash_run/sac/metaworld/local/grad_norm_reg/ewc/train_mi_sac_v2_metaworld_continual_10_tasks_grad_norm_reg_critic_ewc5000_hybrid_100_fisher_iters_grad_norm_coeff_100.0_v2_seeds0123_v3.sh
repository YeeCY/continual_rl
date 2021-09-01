#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../../../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2 3)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES="$seed"
  nohup \
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
    --algo ewc_v2_grad_norm_reg_critic_mi_sac_mlp_v2 \
    --train_steps_per_task 500000 \
    --eval_freq 10 \
    --discount 0.99 \
    --sac_init_steps 1000 \
    --sac_num_expl_steps_per_process 1000 \
    --sac_num_processes 1 \
    --sac_num_train_iters 1000 \
    --sac_ewc_lambda 5000 \
    --sac_ewc_estimate_fisher_iters 100 \
    --sac_ewc_estimate_fisher_sample_num 1000 \
    --sac_ewc_estimate_fisher_sample_src hybrid \
    --sac_ewc_critic_grad_norm_reg_coeff 100.0 \
    --seed $seed \
    --work_dir $PROJECT_DIR/vec_logs/grad_norm_reg_critic_mi_sac_mlp_metaworld_10_tasks/ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_100.0/$seed \
    > $PROJECT_DIR/terminal_logs/grad_norm_reg_critic_mi_sac_mlp_metaworld_10_tasks-ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_100.0-seed"$seed".log 2>&1 &
done

export PYTHONPATH=$HOME/policy-adaptation-during-deployment/
export PWD=$HOME/policy-adaptation-during-deployment
export CUDA_VISIBLE_DEVICES=0

python src/ppo.py \
  --env_names Reacher-v2 HalfCheetah-v3 Hopper-v3 \
  --env_type mujoco \
  --algo ppo_mlp \
  --train_steps_per_task 1000000 \
  --save_freq 10 \
  --eval_freq 10 \
  --discount 0.99 \
  --batch_size 32 \
  --ppo_num_rollout_steps_per_process 2048 \
  --ppo_num_processes 1 \
  --ppo_use_clipped_critic_loss \
  --ppo_use_proper_time_limits \
  --seed 0 \
  --work_dir vec_logs/reacher_halfcheetah_hopper/0 \
  --save_model \
  --save_video

import argparse
import numpy as np

from agent import ALGOS

ENV_TYPES = [
	'atari',
	'dmc_locomotion',
	'metaworld',
	'mujoco'
]


def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	# parser.add_argument('--domain_name', default='walker')
	# parser.add_argument('--task_name', default='walk')
	parser.add_argument('--env_name', default='walker_run')  # (chongyi zheng)
	parser.add_argument('--env_names', nargs='+', type=str,
						default=['reach-v2', 'window-close-v2', 'button-press-topdown-v2'])
	parser.add_argument('--env_type', default='dmc_locomotion', type=str, choices=ENV_TYPES)
	parser.add_argument('--algo', default='sac_mlp_ss_ensem', type=str, choices=list(ALGOS))
	parser.add_argument('--video_camera_id', default=0, type=int)  # (chongyi zheng)
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=1, type=int)  # 1
	parser.add_argument('--mode', default='train', type=str)

	# locomotion tasks
	parser.add_argument('--pixel_obs', default=False, action='store_true')  # (chongyi zheng)
	parser.add_argument('--obs_height', default=84, type=int)
	parser.add_argument('--obs_width', default=84, type=int)
	parser.add_argument('--obs_pad', default=4, type=int)
	parser.add_argument('--env_camera_id', default=0, type=int)  # (chongyi zheng)
	parser.add_argument('--episode_length', default=1000, type=int)

	# agent
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--num_train_iters', default=1, type=int)
	parser.add_argument('--train_freq', default=1, type=int)  # 4 for dqn?
	parser.add_argument('--train_steps', default=1000000, type=int)
	parser.add_argument('--train_steps_per_task', default=1000000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)  # 32 for dqn?
	parser.add_argument('--device', default='cuda', type=str)

	# eval
	parser.add_argument('--save_freq', default=100000, type=int)
	parser.add_argument('--eval_freq', default=100000, type=int)
	parser.add_argument('--num_eval_episodes', default=5, type=int)  # default = 3
	parser.add_argument('--eval_results', default=False, action='store_true')  # (chongyi zheng): save evalution results or not

	# algorithm common
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--encoder_feature_dim', default=50, type=int)
	# parser.add_argument('--encoder_lr', default=1e-3, type=float)  # TODO (chongyi zheng): delete this line
	# parser.add_argument('--encoder_tau', default=0.05, type=float)  # TODO (chongyi zheng): delete this line

	# self-supervision
	parser.add_argument('--use_rot', default=False, action='store_true')  # rotation prediction
	parser.add_argument('--use_fwd', default=False, action='store_true')  # forward dynamics model
	parser.add_argument('--use_inv', default=False, action='store_true')  # inverse dynamics model
	parser.add_argument('--use_curl', default=False, action='store_true')  # CURL
	parser.add_argument('--ss_lr', default=3e-4, type=float)   # self-supervised learning rate, 1e-3
	parser.add_argument('--ss_update_freq', default=2, type=int)  # self-supervised update frequency
	# (chongyi zheng) stop gradients flow into the shared encoder from self-supervised predictors other than the first
	# one in the ensemble
	# parser.add_argument('--ss_stop_shared_layers_grad', default=False, action='store_true')
	# parser.add_argument('--num_layers', default=4, type=int)  # number of conv layers
	# parser.add_argument('--num_shared_layers', default=-1, type=int)  # number of shared conv layers
	# parser.add_argument('--num_filters', default=32, type=int)  # number of filters in conv
	parser.add_argument('--curl_latent_dim', default=128, type=int)  # latent dimension for curl
	parser.add_argument('--use_ensemble', default=False, action='store_true')  # ensemble
	parser.add_argument('--num_ensem_comps', default=4, type=int)  # number of components in ensemble
	
	# sac
	parser.add_argument('--hidden_dim', default=400, type=int)  # 1024
	parser.add_argument('--init_temperature', default=1.0, type=float)  # 0.1
	parser.add_argument('--alpha_lr', default=3e-4, type=float)  # (chongyi zheng): 1e-4, try 3e-4?
	parser.add_argument('--grad_clip_norm', default=10.0, type=float)  # tuning this

	# sac actor
	parser.add_argument('--actor_lr', default=3e-4, type=float)  # 1e-3
	parser.add_argument('--actor_log_std_min', default=-20, type=float)  # -10
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=1, type=int)  # (chongyi zheng): default = 2

	# sac critic
	parser.add_argument('--critic_lr', default=3e-4, type=float)  # 1e-3
	parser.add_argument('--critic_tau', default=0.005, type=float)  # 0.01
	parser.add_argument('--critic_target_update_freq', default=1, type=int)  # 1

	# dqn
	parser.add_argument('--double_q', default=False, action='store_true')
	parser.add_argument('--dueling', default=False, action='store_true')
	parser.add_argument('--exploration_fraction', default=0.1, type=float)
	parser.add_argument('--exploration_initial_eps', default=1.0, type=float)
	parser.add_argument('--exploration_final_eps', default=0.01, type=float)
	parser.add_argument('--target_update_interval', default=1000, type=int)
	parser.add_argument('--max_grad_norm', default=10, type=float)
	parser.add_argument('--q_net_lr', default=1e-4, type=float)  # try 3e-4?
	parser.add_argument('--q_net_tau', default=1.0, type=float)

	# ppo
	parser.add_argument('--ppo_num_rollout_steps_per_process', default=2048, type=int)
	parser.add_argument('--ppo_num_processes', default=1, type=int)
	parser.add_argument('--ppo_hidden_dim', default=64, type=int)
	parser.add_argument('--ppo_clip_param', default=0.2, type=float)
	parser.add_argument('--ppo_epoch', default=10, type=int)
	parser.add_argument('--ppo_critic_loss_coef', default=0.5, type=float)
	parser.add_argument('--ppo_entropy_coef', default=0.0, type=float)
	parser.add_argument('--ppo_lr', default=3e-4, type=float)
	parser.add_argument('--ppo_eps', default=1e-5, type=float)
	parser.add_argument('--ppo_grad_clip_norm', default=0.5, type=float)
	parser.add_argument('--ppo_use_clipped_critic_loss', default=False, action='store_true')
	parser.add_argument('--ppo_gae_lambda', default=0.95, type=float)
	parser.add_argument('--ppo_use_proper_time_limits', default=False, action='store_true')

	# ppo ewc
	parser.add_argument('--ppo_ewc_lambda', default=5000, type=float)
	parser.add_argument('--ppo_ewc_estimate_fisher_epochs', default=100, type=int)
	parser.add_argument('--ppo_ewc_rollout_steps_per_process', default=1024, type=int)
	parser.add_argument('--ppo_online_ewc', default=False, action='store_true')
	parser.add_argument('--ppo_online_ewc_gamma', default=1.0, type=float)

	# ewc
	parser.add_argument('--ewc_lambda', default=5000, type=float)
	parser.add_argument('--ewc_fisher_sample_size', default=1024, type=int)
	parser.add_argument('--online_ewc', default=False, action='store_true')
	parser.add_argument('--online_ewc_gamma', default=1.0, type=float)

	# si
	parser.add_argument('--si_c', default=1.0, type=float)
	parser.add_argument('--si_epsilon', default=0.1, type=float)

	# agem
	parser.add_argument('--agem_memory_budget', default=3000, type=int)
	parser.add_argument('--agem_ref_grad_batch_size', default=1024)

	# misc
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument('--work_dir', default=None, type=str)
	parser.add_argument('--load_checkpoint', default=None, type=str)
	parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)  # (chongyi zheng), 100000
	parser.add_argument('--save_model', default=False, action='store_true')
	parser.add_argument('--save_video', default=False, action='store_true')
	parser.add_argument('--log_freq', default=20000, type=int)
	parser.add_argument('--save_tb', default=False, action='store_true')  # (chongyi zheng)

	# pad
	# parser.add_argument('--pad_checkpoint', default=None, type=str)
	# parser.add_argument('--pad_batch_size', default=32, type=int)
	# parser.add_argument('--pad_num_episodes', default=100, type=int)

	args = parser.parse_args()

	assert args.mode in {'train', 'eval', 'eval_color_easy', 'eval_color_hard'} or 'eval_video' in args.mode, \
		f'unrecognized mode "{args.mode}"'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.work_dir is not None, 'must provide a working directory for experiment'

	assert np.sum([args.use_inv, args.use_rot, args.use_curl]) <= 1, \
		'can use at most one self-supervised task'

	if args.load_checkpoint is not None:
		try:
			args.load_checkpoint = args.load_checkpoint.replace('k', '000')
			args.load_checkpoint = int(args.load_checkpoint)
		except:
			return ValueError('load_checkpoint must be int, received', args.load_checkpoint)

	if args.seed is None:
		args.seed = np.random.randint(int(1e9))

	return args

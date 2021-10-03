import os
# import argparse

import numpy as np
import torch
# from copy import deepcopy
# from tqdm import tqdm


import utils
from video import VideoRecorder

from arguments import parse_args
from environment import make_continual_vec_envs
from agent import make_agent
# from utils import get_curl_pos_neg

from src.arguments import ENV_TYPES, ALGOS


# def evaluate(env, agent, args, video, adapt=False):
# 	"""Evaluate an agent, optionally adapt using PAD"""
# 	episode_rewards = []
#
# 	for i in tqdm(range(args.pad_num_episodes)):
# 		ep_agent = deepcopy(agent)  # make a new copy
#
# 		if args.use_curl:  # initialize replay buffer for CURL
# 			replay_buffer = utils.ReplayBuffer(
# 				obs_shape=env.observation_space.shape,
# 				action_shape=env.action_space.shape,
# 				capacity=args.train_steps,
# 				batch_size=args.pad_batch_size
# 			)
# 		video.init(enabled=True)
#
# 		obs = env.reset()
# 		done = False
# 		episode_reward = 0
# 		losses = []
# 		step = 0
# 		ep_agent.train()
#
# 		while not done:
# 			# Take step
# 			with utils.eval_mode(ep_agent):
# 				action = ep_agent.select_action(obs)
# 			next_obs, reward, done, _ = env.step(action)
# 			episode_reward += reward
#
# 			# Make self-supervised update if flag is true
# 			if adapt:
# 				if args.use_rot:  # rotation prediction
#
# 					# Prepare batch of cropped observations
# 					batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
# 					batch_next_obs = utils.random_crop(batch_next_obs)
#
# 					# Adapt using rotation prediction
# 					losses.append(ep_agent.update_rot(batch_next_obs))
#
# 				if args.use_inv: # inverse dynamics model
#
# 					# Prepare batch of observations
# 					batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
# 					batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
# 					batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)
#
# 					# Adapt using inverse dynamics prediction
# 					losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs), batch_action))
#
# 				if args.use_curl:  # CURL
#
# 					# Add observation to replay buffer for use as negative samples
# 					# (only first argument obs is used, but we store all for convenience)
# 					replay_buffer.add(obs, action, reward, next_obs, True)
#
# 					# Prepare positive and negative samples
# 					obs_anchor, obs_pos = get_curl_pos_neg(next_obs, replay_buffer)
#
# 					# Adapt using CURL
# 					losses.append(ep_agent.update_curl(obs_anchor, obs_pos, ema=True))
#
# 			video.record(env, losses)
# 			obs = next_obs
# 			step += 1
#
# 		video.save(f'{args.mode}_pad_{i}.mp4' if adapt else f'{args.mode}_{i}.mp4')
# 		episode_rewards.append(episode_reward)
#
# 	return np.mean(episode_rewards)


def evaluate_task(env, task_name, agent, video, num_episodes, **act_kwargs):
	"""Evaluate agent"""

	# for task_id, task_name in enumerate(env.get_attr('env_names')[0]):
	episode_rewards = []
	episode_successes = []
	video.init(enabled=True)
	# env.env_method('sample_task')
	# FIXME (cyzheng): Fix the following method
	env.env_method('set_task', task_name)
	task_id = env.get_attr('active_task_index')[0]
	obs = env.reset()
	video.record(env)

	if 'task_embedding_hypernet' in args.algo:
		agent.infer_weights(task_id)

	while len(episode_rewards) < num_episodes:
		with utils.eval_mode(agent):
			with utils.eval_mode(agent):
				if any(x in args.algo for x in ['mh', 'mi', 'individual', 'hypernet', 'distilled']):
					action = agent.act(obs, sample=False, head_idx=task_id, **act_kwargs)
				else:
					action = agent.act(obs, sample=False, **act_kwargs)

			obs, _, done, infos = env.step(action)
			video.record(env)

			for done_ in done:
				if done_ and len(episode_rewards) == 0:
					video.save('%s.mp4' % task_name)
					video.init(enabled=False)

			for info in infos:
				if 'episode' in info.keys():
					episode_successes.append(info.get('success', 0.0))
					episode_rewards.append(info['episode']['r'])

	episode_rewards = episode_rewards[:num_episodes]
	episode_successes = episode_successes[:num_episodes]

	if 'task_embedding_hypernet' in args.algo:
		agent.clear_weights()

	# if len(episode_successes) > 0:
	# 	logger.log('eval/success_rate', np.mean(episode_successes), step)
	# logger.log('eval/episode_reward', np.mean(episode_rewards), step)
	# log_info = {
	# 	'eval/task_name': task_name
	# }
	# logger.dump(step, ty='eval', info=log_info)

	if len(episode_successes) > 0:
		return np.mean(episode_successes), np.mean(episode_rewards)
	else:
		return None, np.mean(episode_rewards)


def main(args):
	if args.env_type == 'mujoco':
		# train_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'train_env'))
		eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'dummy'))
		# env = make_continual_vec_envs(
		# 	args.env_names, args.seed, args.sac_num_processes,
		# 	args.discount, train_env_log_dir,
		# 	allow_early_resets=True,
		# 	normalize=False,
		# 	add_onehot=args.add_onehot,
		# )
		env = make_continual_vec_envs(
			args.env_names, args.seed, args.sac_num_processes,
			None, eval_env_log_dir,
			allow_early_resets=True,
			normalize=False,
			add_onehot=args.add_onehot,
		)
	elif args.env_type == 'metaworld':
		# train_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'train_env'))
		eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'dummy'))

		# env = make_continual_vec_envs(
		# 	args.env_names, args.seed, args.sac_num_processes,
		# 	args.discount, train_env_log_dir,
		# 	allow_early_resets=True,
		# 	normalize=False,
		# 	add_onehot=args.add_onehot,
		# )

		env = make_continual_vec_envs(
			args.env_names, args.seed, args.sac_num_processes,
			None, eval_env_log_dir,
			allow_early_resets=True,
			normalize=False,
			add_onehot=args.add_onehot,
		)

	# Initialize environment
	# env = init_env(args)
	# model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	# video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	# video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.set_seed_everywhere(args.seed)
	utils.make_dir(args.work_dir)
	# model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, args.env_type,
						  height=448, width=448, camera_id=args.video_camera_id)

	# Prepare agent
	# assert torch.cuda.is_available(), 'must have cuda enabled'
	# cropped_obs_shape = (3*args.frame_stack, 84, 84)
	# agent = make_agent(
	# 	obs_shape=cropped_obs_shape,
	# 	action_shape=env.action_space.shape,
	# 	args=args
	# )
	device = torch.device(args.device)
	agent = make_agent(
		obs_space=env.observation_space,
		action_space=[env.action_space for _ in range(env.num_tasks)]
		if any(x in args.algo for x in ['mh', 'mi', 'individual', 'hypernet', 'distilled'])
		else env.action_space,
		device=device,
		args=args
	)

	for task_name in args.env_names:
		task_load_dir = os.path.join(args.load_dir, task_name)

		assert os.path.exists(task_load_dir)

		agent.load(task_load_dir)

		# Evaluate agent without PAD
		print(f'Evaluating {task_name} for {args.num_eval_episodes} episodes...')
		task_successes, task_rewards = evaluate_task(env, task_name, agent, video, args.num_eval_episodes)
		print(f'Success Rate: {task_successes}, Return: {task_rewards}')


if __name__ == '__main__':
	# def str2bool(v):
	# 	if isinstance(v, bool):
	# 		return v
	# 	if v.lower() in ('yes', 'true', 't', 'y', '1'):
	# 		return True
	# 	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
	# 		return False
	# 	else:
	# 		raise argparse.ArgumentTypeError('Boolean value expected.')
	#
	# parser = argparse.ArgumentParser()
	#
	# parser.add_argument('--env_names', nargs='+', type=str,
	# 					default=['reach-v2', 'window-close-v2', 'button-press-topdown-v2'])
	# parser.add_argument('--env_type', default='metaworld', type=str, choices=ENV_TYPES)
	# parser.add_argument('--algo', default='sac_mlp', type=str, choices=list(ALGOS))
	# parser.add_argument('--add_onehot', default=False, type=str2bool)
	# parser.add_argument('--sac_num_processes', default=1, type=int)
	# parser.add_argument('--seed', default=0, type=int)
	# parser.add_argument('--save_video', default=False, type=str2bool)
	# parser.add_argument('--video_camera_id', default=0, type=int)
	# parser.add_argument('--work_dir', default=None, type=str)
	# parser.add_argument('--discount', default=0.99, type=float)  # (cyzheng): for the sake of constructing agent
	# parser.add_argument('--device', default='cuda', type=str)

	args = parse_args()

	main(args)

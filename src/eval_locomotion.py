import numpy as np
import torch
import os
from copy import deepcopy
from tqdm import tqdm
import utils
from video import VideoRecorder

from arguments import parse_args
from env.wrappers import make_locomotion_env
from agent.agent import make_agent


def evaluate(env, agent, args, video):
	"""Evaluate an agent, optionally adapt using PAD"""
	episode_rewards = []
	episode_invs_pred_vars = []
	obs_buf = []
	next_obs_buf = []
	action_buf = []

	for i in tqdm(range(args.eval_episodes)):
		ep_agent = deepcopy(agent)  # make a new copy
		video.init(enabled=True)

		obs = env.reset()
		done = False
		episode_reward = 0
		losses = []
		step = 0
		ep_agent.train()

		while not done:
			# Take step
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			episode_reward += reward

			obs_buf.append(obs)
			next_obs_buf.append(next_obs)
			action_buf.append(action)
			video.record(env, losses)
			obs = next_obs
			step += 1

		video.save('{}_{}.mp4'.format(args.mode, i))
		episode_rewards.append(episode_reward)
		# Compute self-supervised ensemble variance
		if args.use_inv:
			episode_invs_pred_vars.append(np.mean(
				agent.invs_pred_var(
					np.asarray(obs_buf, dtype=obs.dtype),
					np.asarray(next_obs_buf, dtype=obs.dtype),
					np.asarray(action_buf, dtype=action.dtype))
			))

	return np.mean(episode_rewards), np.mean(episode_invs_pred_vars)


def init_env(args):
	utils.set_seed_everywhere(args.seed)
	return make_locomotion_env(
		env_name=args.env_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		obs_height=args.obs_height,
		obs_width=args.obs_width,
		camera_id=args.env_camera_id,
		mode=args.mode
	)


def main(args):
	# Initialize environment
	env = init_env(args)
	model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448, camera_id=args.video_camera_id)

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, 84, 84)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	agent.load(model_dir, args.load_checkpoint)

	# Evaluate agent without PAD
	print(f'Evaluating {args.work_dir} for {args.eval_episodes} episodes (mode: {args.mode})')
	eval_reward, eval_invs_pred_var = evaluate(env, agent, args, video)
	print('eval reward:', int(eval_reward))
	print('eval_invs_pred_var: ', eval_invs_pred_var)

	# # Evaluate agent with PAD (if applicable)
	# pad_reward = None
	# if args.use_inv or args.use_curl or args.use_rot:
	# 	env = init_env(args)
	# 	print(f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes '
	# 		  f'(mode: {args.mode})')
	# 	pad_reward = evaluate(env, agent, args, video, adapt=True)
	# 	print('pad reward:', int(pad_reward))

	# Save results
	results_fp = os.path.join(args.work_dir, '{}.pt'.format(args.mode))
	torch.save({
		'args': args,
		'eval_reward': eval_reward,
		'eval_invs_pred_var': eval_invs_pred_var
	}, results_fp)
	print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)

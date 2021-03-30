import torch
import numpy as np
import os
import copy

from arguments import parse_args
from env import make_locomotion_env
from agent.agent import make_agent
import utils
import buffers
import time
from logger import Logger
from video import VideoRecorder


def evaluate(env, agent, video, num_episodes, logger, step):
    """Evaluate agent"""
    episode_rewards = []
    episode_fwd_pred_vars = []
    episode_inv_pred_vars = []
    for episode in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        obs_buf = []
        next_obs_buf = []
        action_buf = []
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            next_obs, reward, done, _ = env.step(action)

            obs_buf.append(obs)
            next_obs_buf.append(next_obs)
            action_buf.append(action)
            episode_reward += reward

            video.record(env)
            obs = next_obs
        episode_rewards.append(episode_reward)
        if agent.use_fwd:
            episode_fwd_pred_vars.append(np.mean(
                agent.ss_preds_var(
                    np.asarray(obs_buf, dtype=obs.dtype),
                    np.asarray(next_obs_buf, dtype=obs.dtype),
                    np.asarray(action_buf, dtype=action.dtype))
            ))
        if agent.use_inv:
            episode_inv_pred_vars.append(np.mean(
                agent.ss_preds_var(
                    np.asarray(obs_buf, dtype=obs.dtype),
                    np.asarray(next_obs_buf, dtype=obs.dtype),
                    np.asarray(action_buf, dtype=action.dtype))
            ))
        video.save('%d.mp4' % step)
    logger.log('eval/episode_reward', np.mean(episode_rewards), step)
    if agent.use_fwd:
        logger.log('eval/episode_fwd_pred_var', np.mean(episode_fwd_pred_vars), step)
    if agent.use_inv:
        logger.log('eval/episode_inv_pred_var', np.mean(episode_inv_pred_vars), step)
    logger.dump(step)


def main(args):
    # Initialize environment
    utils.set_seed_everywhere(args.seed)
    env = make_locomotion_env(
        env_name=args.env_name,
        seed=args.seed,
        episode_length=args.episode_length,
        from_pixels=args.pixel_obs,
        action_repeat=args.action_repeat,
        obs_height=args.obs_height,
        obs_width=args.obs_width,
        camera_id=args.env_camera_id,
        mode=args.mode
    )
    eval_env = make_locomotion_env(
        env_name=args.env_name,
        seed=args.seed,
        episode_length=args.episode_length,
        from_pixels=args.pixel_obs,
        action_repeat=args.action_repeat,
        obs_height=args.obs_height,
        obs_width=args.obs_width,
        camera_id=args.env_camera_id,
        mode=args.mode
    )

    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None,
                          height=448, width=448, camera_id=args.video_camera_id)

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    device = torch.device(args.device)

    replay_buffer = buffers.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        device=device,
        optimize_memory_usage=True,
    )
    # replay_buffer = buffers.AugmentFrameStackReplayBuffer(
    #     obs_shape=env.unwrapped.observation_space.shape,
    #     action_shape=env.action_space.shape,
    #     capacity=args.replay_buffer_capacity,
    #     frame_stack=args.frame_stack,
    #     image_pad=args.obs_pad,
    #     device=device,
    #     optimize_memory_usage=True,
    # )
    # cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_range=[
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ],
        device=device,
        args=args
    )

    logger = Logger(args.work_dir,
                    log_frequency=args.log_freq,
                    action_repeat=args.action_repeat,
                    save_tb=args.save_tb)
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    start_time = time.time()
    for step in range(args.train_steps + 1):
        # (chongyi zheng): we can also evaluate and save model when current episode is not finished
        # Evaluate agent periodically
        if step % args.eval_freq == 0:
            print('Evaluating:', args.work_dir)
            logger.log('eval/episode', episode, step)
            evaluate(eval_env, agent, video, args.num_eval_episodes, logger, step)

        # Save agent periodically
        if step % args.save_freq == 0 and step > 0:
            if args.save_model:
                agent.save(model_dir, step)

        if done:
            if step > 0:
                logger.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                logger.dump(step, save=(step > args.init_steps))

            logger.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            episode_reward = 0
            episode_step = 0
            episode += 1

            logger.log('train/episode', episode, step)

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        # Run training update
        if step >= args.init_steps:
            # TODO (chongyi zheng): Do we need multiple updates after initial data collection?
            # num_updates = args.init_steps if step == args.init_steps else 1
            # for _ in range(num_updates):
            # 	agent.update(replay_buffer, logger, step)
            for _ in range(args.num_train_iters):
                agent.update(replay_buffer, logger, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        # (chongyi zheng): example to add transition into FrameStackReplayBuffer
        # 	frame_stack = 4, repeated_frame_dists + new_frame_dists
        # 			episode_step = 0, [0, 0, 0] + [0]
        # 			episode_step = 1, [-1, -1] + [-1, 0]
        # 			episode_step = 2, [-2] + [-2 ,-1, 0]
        # 			episode_step = 3, [] + [-3, -2, -1, 0]
        #			...
        # new_frame_dists = [-idx for idx in reversed(range(0, min(episode_step + 1, args.frame_stack)))]
        # stack_frame_dists = np.array(
        #     [-episode_step] * (args.frame_stack - len(new_frame_dists)) + new_frame_dists
        # )
        # replay_buffer.add(obs, action, reward, next_obs, done_bool, stack_frame_dists=stack_frame_dists)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)

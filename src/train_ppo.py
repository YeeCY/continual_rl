import torch
import numpy as np
import os
from collections import deque
import copy
import time


from arguments import parse_args
from environment import make_atari_env, make_locomotion_env, make_single_metaworld_env, make_continual_metaworld_env, \
    make_vec_envs, make_continual_vec_envs
from environment.metaworld import MultiEnvWrapper
from agent import make_agent
import utils
from environment.utils import get_vec_normalize, get_render_func
import storages
from logger import Logger
from video import VideoRecorder


def evaluate(env, agent, video, obs_rms, num_episodes, logger, step):
    """Evaluate agent"""
    if isinstance(env, MultiEnvWrapper):
        assert env.env_names is not None, "Environment name must exist!"

        for task_name in env.env_names:
            vec_norm = get_vec_normalize(env.env)  # use actually vector env
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.obs_rms = obs_rms

            episode_rewards = []
            episode_successes = []
            video.init(enabled=True)
            obs = env.reset()
            video.record(env.env)  # use actually vector env

            while len(episode_rewards) < num_episodes:
                with utils.eval_mode(agent):
                    action, _ = agent.act(obs, sample=False, compute_log_pi=False)

                obs, _, done, infos = env.step(action)
                video.record(env)

                for done_ in done:
                    if done_ and len(episode_rewards) == 0:
                        video.save('%s_%d.mp4' % (args.env_name, step))
                        video.init(enabled=False)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_successes.append(info.get('success', False))
                        episode_rewards.append(info['episode']['r'])
            if len(episode_successes) > 0:
                logger.log('eval/success_rate', np.mean(episode_successes), step)
            logger.log('eval/episode_reward', np.mean(episode_rewards), step, sw_prefix=task_name + '_')
            log_info = {
                'train/task_name': task_name
            }
            logger.dump(step, ty='eval', info=log_info)


def main(args):
    utils.set_seed_everywhere(args.seed)
    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, args.env_type,
                          height=448, width=448, camera_id=args.video_camera_id)

    # Prepare agent
    # assert torch.cuda.is_available(), 'must have cuda enabled'
    device = torch.device(args.device)

    # Create environments
    if args.env_type == 'atari':
        # environment = make_atari_env(
        #     env_name=args.env_name,
        #     seed=args.seed,
        #     action_repeat=args.action_repeat,
        #     frame_stack=args.frame_stack
        # )
        # eval_env = make_atari_env(
        #     env_name=args.env_name,
        #     seed=args.seed,
        #     action_repeat=args.action_repeat,
        #     frame_stack=args.frame_stack
        # )
        # environment = make_continual_atari_env(
        #     env_names=args.env_names,
        #     seed=args.seed,
        #     action_repeat=args.action_repeat,
        #     frame_stack=args.frame_stack
        # )
        # eval_env = make_continual_atari_env(
        #     env_names=args.env_names,
        #     seed=args.seed,
        #     action_repeat=args.action_repeat,
        #     frame_stack=args.frame_stack
        # )
        pass
    elif args.env_type == 'dmc_locomotion':
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
    elif args.env_type == 'mujoco':
        # env = make_vec_envs(args.env_name, args.seed, args.ppo_num_processes,
        #                     args.discount, args.work_dir)
        # eval_env = make_vec_envs(args.env_name, args.seed + args.ppo_num_processes,
        #                          args.ppo_num_processes, None, args.work_dir, True)
        train_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'train_env'))
        eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'eval_env'))
        env = make_continual_vec_envs(args.env_names, args.seed, args.ppo_num_processes,
                                      args.discount, train_env_log_dir)
        eval_env = make_vec_envs(args.env_names, args.seed + args.ppo_num_processes,
                                 args.ppo_num_processes, None, eval_env_log_dir, True)
    elif args.env_type == 'metaworld':
        # environment = make_single_metaworld_env(
        #     env_name=args.env_name,
        #     seed=args.seed
        # )
        # eval_env = make_single_metaworld_env(
        #     env_name=args.env_name,
        #     seed=args.seed
        # )
        env = make_continual_metaworld_env(
            env_names=args.env_names,
            seed=args.seed
        )
        eval_env = copy.deepcopy(env)
        # eval_env = make_continual_metaworld_env(
        #     env_names=args.env_names,
        #     seed=args.seed
        # )

    # if args.env_type == 'atari':
    #     # replay_buffer = buffers.FrameStackReplayBuffer(
    #     #     obs_space=environment.observation_space,
    #     #     action_space=environment.action_space,
    #     #     capacity=args.replay_buffer_capacity,
    #     #     frame_stack=args.frame_stack,
    #     #     device=device,
    #     #     optimize_memory_usage=True,
    #     # )
    #     # from stable_baselines3.common.buffers import ReplayBuffer
    #     # replay_buffer = ReplayBuffer(
    #     #     args.replay_buffer_capacity,
    #     #     environment.observation_space,
    #     #     environment.action_space,
    #     #     device,
    #     #     optimize_memory_usage=True,
    #     # )
    #     replay_buffer = buffers.ReplayBuffer(
    #         obs_space=environment.observation_space,
    #         action_space=environment.action_space,
    #         capacity=args.replay_buffer_capacity,
    #         device=device,
    #         optimize_memory_usage=True,
    #     )
    # elif args.env_type == 'dmc_locomotion' or 'metaworld':
    #     replay_buffer = buffers.ReplayBuffer(
    #         obs_space=environment.observation_space,
    #         action_space=environment.action_space,
    #         capacity=args.replay_buffer_capacity,
    #         device=device,
    #         optimize_memory_usage=True,
    #     )
    rollouts = storages.RolloutStorage(args.ppo_num_rollout_steps_per_process,
                                       args.ppo_num_processes,
                                       env.observation_space.shape,
                                       env.action_space,
                                       device)
    
    if 'ewc' in args.algo:
        est_fisher_rollouts = storages.RolloutStorage(args.ewc_rollout_steps_per_process,
                                                      args.ppo_num_processes,
                                                      env.observation_space.shape,
                                                      env.action_space,
                                                      device)

    agent = make_agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        args=args
    )

    logger = Logger(args.work_dir,
                    log_frequency=args.log_freq,
                    action_repeat=args.action_repeat,
                    save_tb=args.save_tb)

    # log arguments
    args_dict = vars(args)
    logger.log_and_dump_arguments(args_dict)

    episode = 0
    total_steps = 0
    recent_success = deque(maxlen=100)
    recent_episode_reward = deque(maxlen=100)
    # for step in range(args.train_steps + 1):
    #     # (chongyi zheng): we can also evaluate and save model when current episode is not finished
    #     # Evaluate agent periodically
    #     if step % args.eval_freq == 0:
    #         print('Evaluating:', args.work_dir)
    #         logger.log('eval/episode', episode, step)
    #         evaluate(eval_env, agent, video, args.num_eval_episodes, logger, step)
    #
    #     # Save agent periodically
    #     if step % args.save_freq == 0 and step > 0:
    #         if args.save_model:
    #             agent.save(model_dir, step)
    #
    #     if done:
    #         if step > 0:
    #             logger.log('train/duration', time.time() - start_time, step)
    #             start_time = time.time()
    #             logger.dump(step, ty='train', save=(step > args.init_steps))
    #
    #         recent_episode_reward.append(episode_reward)
    #         logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), step)
    #         logger.log('train/episode_reward', episode_reward, step)
    #
    #         obs = environment.reset()
    #         episode_reward = 0
    #         episode_step = 0
    #         episode += 1
    #
    #         logger.log('train/episode', episode, step)
    #
    #     # Sample action for data collection
    #     if step < args.init_steps:
    #         action = environment.action_space.sample()
    #     else:
    #         # with utils.eval_mode(agent):
    #         action = agent.act(obs, False)
    #
    #     if 'dqn' in args.algo:
    #         agent.on_step(step, args.train_steps, logger)
    #
    #     # Run training update
    #     if step >= args.init_steps and step % args.train_freq == 0:
    #         # TODO (chongyi zheng): Do we need multiple updates after initial data collection?
    #         # num_updates = args.init_steps if step == args.init_steps else 1
    #         # for _ in range(num_updates):
    #         # 	agent.update(replay_buffer, logger, step)
    #         for _ in range(args.num_train_iters):
    #             agent.update(replay_buffer, logger, step)
    #
    #     # Take step
    #     next_obs, reward, done, _ = environment.step(action)
    #     # replay_buffer.add(obs, action, reward, next_obs, done)
    #     replay_buffer.add(np.expand_dims(obs, axis=0),
    #                       np.expand_dims(next_obs, axis=0),
    #                       np.expand_dims(action, axis=0),
    #                       np.expand_dims(reward, axis=0),
    #                       np.expand_dims(done, axis=0))
    #     episode_reward += reward
    #     obs = next_obs
    #     episode_step += 1
    # train_steps_per_task = args.train_steps_per_task
    # if isinstance(environment, MultiEnvWrapper):
    #     for step in range(train_steps_per_task * environment.num_tasks):
    #         # (chongyi zheng): we can also evaluate and save model when current episode is not finished
    #         # Evaluate agent periodically
    #         if step % args.eval_freq_per_task == 0:
    #             print('Evaluating:', args.work_dir)
    #             logger.log('eval/episode', episode, step)
    #             evaluate(eval_env, agent, video, args.num_eval_episodes, logger, step)
    #
    #         # Save agent periodically
    #         if step % args.save_freq == 0 and step > 0:
    #             if args.save_model:
    #                 train_steps_per_task = args.train_steps_per_task
    # if isinstance(environment, MultiEnvWrapper):
    #     for total_step in range(train_steps_per_task * environment.num_tasks):
    #         # (chongyi zheng): we can also evaluate and save model when current episode is not finished
    #         # Evaluate agent periodically
    #         if total_step % args.eval_freq_per_task == 0:
    #             print('Evaluating:', args.work_dir)
    #             logger.log('eval/episode', episode, total_step)
    #             evaluate(eval_env, agent, video, args.num_eval_episodes, logger, total_step)
    #
    #         # Save agent periodically
    #         if total_step % args.save_freq == 0 and total_step > 0:
    #             if args.save_model:
    #                 agent.save(model_dir, total_step)
    #
    #         # (chongyi zheng): force reset outside done = True when step reach train_steps_per_task
    #         if task_step >= train_steps_per_task:
    #             obs = environment.reset(sample_task=True)
    #
    #             if 'ewc' in args.algo:
    #                 agent.estimate_fisher(replay_buffer)
    #             elif 'si' in args.algo:
    #                 agent.update_omegas()
    #
    #             agent.reset_target_critic()
    #             replay_buffer.reset()
    #             task_step = 0
    #
    #         # if done[0]:
    #         if done:
    #             # if step > 0:
    #             #     recent_episode_reward.append(episode_reward)
    #             #     logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), step)
    #             #     logger.log('train/episode_reward', episode_reward, step)
    #             #     logger.dump(step, ty='train', save=(step > args.init_steps))
    #             success = np.any(episode_successes).astype(np.float)
    #             recent_success.append(success)
    #             recent_episode_reward.append(episode_reward)
    #
    #             logger.log(f'train/episode_success', success, total_step)
    #             logger.log(f'train/recent_success_rate', np.mean(recent_success), total_step)
    #             logger.log('train/episode_reward', episode_reward, total_step)
    #             logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), total_step)
    #             logger.log('train/episode', episode, total_step)
    #
    #             if total_step > 0:
    #                 # save non-scalar info
    #                 log_info = {
    #                     'train/task_name': info['task_name']
    #                 }
    #                 logger.log('train/duration', time.time() - start_time, total_step)
    #                 start_time = time.time()
    #                 # logger.dump(step, ty='train', save=(step > args.init_steps), info=log_info)
    #                 logger.dump(total_step, ty='train', save=(task_step > args.init_steps), info=log_info)
    #
    #             obs = environment.reset()
    #             episode_reward = 0
    #             episode_step = 0
    #             episode_successes.clear()
    #             episode += 1
    #
    #         # Sample action for data collection
    #         if task_step < args.init_steps:
    #             action = np.array(environment.action_space.sample())
    #         else:
    #             with utils.eval_mode(agent):
    #                 action = agent.act(obs, True)
    #
    #         if 'dqn' in args.algo:
    #             agent.on_step(task_step, train_steps_per_task, logger)
    #
    #         # Run training update
    #         if task_step >= args.init_steps and total_step % args.train_freq == 0:
    #             # TODO (chongyi zheng): Do we need multiple updates after initial data collection?
    #             # num_updates = args.init_steps if step == args.init_steps else 1
    #             for _ in range(args.num_train_iters):
    #                 agent.update(replay_buffer, logger, total_step)
    #
    #         # Take step
    #         next_obs, reward, done, info = environment.step(action)
    #
    #         if info.get('success') is not None:
    #             episode_successes.append(info.get('success'))
    #
    #         replay_buffer.add(obs, action, reward, next_obs, done)
    #         # replay_buffer.add(obs, next_obs, action, reward, done)
    #         # self.replay_buffer.add(np.expand_dims(obs, axis=0),
    #         #                        np.expand_dims(next_obs, axis=0),
    #         #                        np.expand_dims(action, axis=0),
    #         #                        np.expand_dims(reward, axis=0),
    #         #                        np.expand_dims(done, axis=0))
    #         episode_reward += reward
    #         obs = next_obs
    #         episode_step += 1
    #         task_step += 1
    if isinstance(env, MultiEnvWrapper):
        total_epochs_per_task = int(args.train_steps_per_task) // args.ppo_num_rollout_steps_per_process \
                                // args.ppo_num_processes

        for _ in range(env.num_tasks):
            task_steps = 0
            start_time = time.time()
            obs = env.reset(sample_task=True)
            rollouts.obs[0].copy_(torch.Tensor(obs).to(device))
            for task_epoch in range(total_epochs_per_task):
                agent.update_learning_rate(task_epoch, total_epochs_per_task)

                for step in range(args.ppo_num_rollout_steps_per_process):
                    with utils.eval_mode(agent):
                        action, log_pi = agent.act(obs, sample=True, compute_log_pi=True)
                        value = agent.predict_value(obs)

                    obs, reward, done, infos = env.step(action)

                    for done_ in done:
                        if done_:
                            episode += 1

                    for info in infos:
                        if 'episode' in info.keys():
                            recent_success.append(info.get('success', False))
                            recent_episode_reward.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    masks = np.array(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = np.array(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                         for info in infos])
                    rollouts.insert(obs, action, log_pi, value, reward, masks, bad_masks)

                task_steps += args.ppo_num_rollout_steps_per_process * args.ppo_num_processes
                total_steps += args.ppo_num_rollout_steps_per_process * args.ppo_num_processes
                next_value = agent.predict_value(rollouts.obs[-1])
                rollouts.compute_returns(next_value, args.discount,
                                         args.ppo_gae_lambda,
                                         args.ppo_use_proper_time_limits)
                agent.update(rollouts, logger, total_steps)
                rollouts.after_update()

                if task_epoch % args.save_freq == 0:
                    if args.save_model:
                        agent.save(model_dir, total_steps)

                end_time = time.time()
                print("FPS: ", int(task_steps / (end_time - start_time)))

                if task_epoch % args.eval_freq == 0:
                    # obs_rms = utils.get_vec_normalize(envs).obs_rms
                    # evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                    #          args.num_processes, eval_log_dir, device)

                    print('Evaluating:', args.work_dir)
                    logger.log('eval/episode', episode, total_steps)
                    obs_rms = get_vec_normalize(env).obs_rms
                    evaluate(eval_env, agent, video, obs_rms, args.num_eval_episodes, logger, total_steps)

                logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), total_steps)
                log_info = {'train/task_name': infos[0]['task_name']}
                logger.dump(total_steps, ty='train', save=True, info=log_info)

            if 'ewc' in args.algo:
                compute_rewards_kwargs = {
                    'gamma': args.discount,
                    'gae_lambda': args.ppo_gae_lambda,
                    'use_proper_time_limits': args.ppo_use_proper_time_limits
                }
                agent.estimate_fisher(env, est_fisher_rollouts, compute_rewards_kwargs=compute_rewards_kwargs)

    # for total_step in range(args.train_steps):
    #     # (chongyi zheng): we can also evaluate and save model when current episode is not finished
    #     # Evaluate agent periodically
    #     if total_step % args.eval_freq_per_task == 0:
    #         print('Evaluating:', args.work_dir)
    #         logger.log('eval/episode', episode, total_step)
    #         evaluate(eval_env, agent, video, args.num_eval_episodes, logger, total_step)
    #
    #     # Save agent periodically
    #     if total_step % args.save_freq == 0 and total_step > 0:
    #         if args.save_model:
    #             agent.save(model_dir, total_step)
    #
    #     # (chongyi zheng): force reset outside done = True when step reach train_steps_per_task
    #     if task_step >= train_steps_per_task:
    #         obs = environment.reset(sample_task=True)
    #
    #         if 'ewc' in args.algo:
    #             agent.estimate_fisher(replay_buffer)
    #         elif 'si' in args.algo:
    #             agent.update_omegas()
    #
    #         agent.reset_target_critic()
    #         replay_buffer.reset()
    #         task_step = 0
    #
    #     # if done[0]:
    #     if done:
    #         # if step > 0:
    #         #     recent_episode_reward.append(episode_reward)
    #         #     logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), step)
    #         #     logger.log('train/episode_reward', episode_reward, step)
    #         #     logger.dump(step, ty='train', save=(step > args.init_steps))
    #         success = np.any(episode_successes).astype(np.float)
    #         recent_success.append(success)
    #         recent_episode_reward.append(episode_reward)
    #
    #         logger.log(f'train/episode_success', success, total_step)
    #         logger.log(f'train/recent_success_rate', np.mean(recent_success), total_step)
    #         logger.log('train/episode_reward', episode_reward, total_step)
    #         logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), total_step)
    #         logger.log('train/episode', episode, total_step)
    #
    #         if total_step > 0:
    #             # save non-scalar info
    #             log_info = {
    #                 'train/task_name': info['task_name']
    #             }
    #             logger.log('train/duration', time.time() - start_time, total_step)
    #             start_time = time.time()
    #             # logger.dump(step, ty='train', save=(step > args.init_steps), info=log_info)
    #             logger.dump(total_step, ty='train', save=(task_step > args.init_steps), info=log_info)
    #
    #         obs = environment.reset()
    #         episode_reward = 0
    #         episode_step = 0
    #         episode_successes.clear()
    #         episode += 1
    #
    #     # Sample action for data collection
    #     if task_step < args.init_steps:
    #         action = np.array(environment.action_space.sample())
    #     else:
    #         with utils.eval_mode(agent):
    #             action = agent.act(obs, True)
    #
    #     if 'dqn' in args.algo:
    #         agent.on_step(task_step, train_steps_per_task, logger)
    #
    #     # Run training update
    #     if task_step >= args.init_steps and total_step % args.train_freq == 0:
    #         # TODO (chongyi zheng): Do we need multiple updates after initial data collection?
    #         # num_updates = args.init_steps if step == args.init_steps else 1
    #         for _ in range(args.num_train_iters):
    #             agent.update(replay_buffer, logger, total_step)
    #
    #     # Take step
    #     next_obs, reward, done, info = environment.step(action)
    #
    #     if info.get('success') is not None:
    #         episode_successes.append(info.get('success'))
    #
    #     replay_buffer.add(obs, action, reward, next_obs, done)
    #     # replay_buffer.add(obs, next_obs, action, reward, done)
    #     # self.replay_buffer.add(np.expand_dims(obs, axis=0),
    #     #                        np.expand_dims(next_obs, axis=0),
    #     #                        np.expand_dims(action, axis=0),
    #     #                        np.expand_dims(reward, axis=0),
    #     #                        np.expand_dims(done, axis=0))
    #     episode_reward += reward
    #     obs = next_obs
    #     episode_step += 1
    #     task_step += 1

    print('Final evaluating:', args.work_dir)
    obs_rms = get_vec_normalize(env).obs_rms
    evaluate(eval_env, agent, video, obs_rms, args.num_eval_episodes, logger, total_steps)


if __name__ == '__main__':
    args = parse_args()
    main(args)

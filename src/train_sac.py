import torch
import numpy as np
import os
from collections import deque
import copy


from arguments import parse_args
from environment import make_atari_env, make_single_metaworld_env, \
    make_continual_metaworld_env, make_continual_vec_envs
from environment.metaworld_utils import MultiEnvWrapper
from environment.env_utils import get_vec_normalize
from agent import make_agent
import utils
import buffers
import time
from logger import Logger
from video import VideoRecorder


def evaluate(train_env, eval_env, agent, video, num_episodes, logger, step):
    """Evaluate agent"""
    if isinstance(train_env, MultiEnvWrapper) and isinstance(eval_env, MultiEnvWrapper):
        assert (train_env.env_names is not None) and (eval_env.env_names is not None), \
            "Environment name must exist!"

        train_vec_norms = get_vec_normalize(train_env)
        eval_vec_norms = get_vec_normalize(eval_env)
        for train_vec_norm, eval_vec_norm in zip(train_vec_norms, eval_vec_norms):
            if eval_vec_norm is not None:
                eval_vec_norm.eval()
                eval_vec_norm.obs_rms = train_vec_norm.obs_rms

        for task_id, task_name in enumerate(eval_env.env_names):
            episode_rewards = []
            episode_successes = []
            video.init(enabled=True)
            obs = eval_env.reset(sample_task=True)
            video.record(eval_env.env)  # use actually vector env

            while len(episode_rewards) < num_episodes:
                with utils.eval_mode(agent):
                    with utils.eval_mode(agent):
                        if any(x in args.algo for x in ['mh', 'mi', 'individual']):
                            action = agent.act(obs, sample=False, head_idx=task_id)
                        else:
                            action = agent.act(obs, sample=False)

                    obs, _, done, infos = eval_env.step(action)
                    video.record(eval_env.env)  # use actually vector env

                    for done_ in done:
                        if done_ and len(episode_rewards) == 0:
                            video.save('%s_%d.mp4' % (task_name, step))
                            video.init(enabled=False)

                    for info in infos:
                        if 'episode' in info.keys():
                            episode_successes.append(info.get('success', False))
                            episode_rewards.append(info['episode']['r'])

            # if 'ewc_v2' in args.algo:
            #     kl_div = agent.kl_with_optimal_actor(task_id)
            #     logger.log('eval/kl_divergence', kl_div, step)

            # for episode in range(num_episodes):
            #     episode_reward = 0
            #     obs_buf = []
            #     next_obs_buf = []
            #     action_buf = []
            #     is_successes = []
            #     while not done:
            #         with utils.eval_mode(agent):
            #             if 'mh' in args.algo:
            #                 action = agent.act(obs, sample=False, head_idx=task_id)
            #             else:
            #                 action = agent.act(obs, sample=False)
            #         next_obs, reward, done, info = env.step(action)
            #
            #         obs_buf.append(obs)
            #         next_obs_buf.append(next_obs)
            #         action_buf.append(action)
            #         episode_reward += reward
            #         if info.get('success') is not None:
            #             is_successes.append(info.get('success'))
            #
            #         video.record(env)
            #         obs = next_obs
            #     episode_rewards.append(episode_reward)
            #     episode_successes.append(np.any(is_successes).astype(np.float))
            #
            #     video.save('%s_%d.mp4' % (task_name, step))
            # logger.log('eval/episode_reward', np.mean(episode_rewards), step, sw_prefix=task_name + '_')
            # if len(episode_successes) > 0:
            #     logger.log('eval/success_rate', np.mean(episode_successes), step)
            # log_info = {
            #     'eval/task_name': task_name
            # }
            if len(episode_successes) > 0:
                logger.log('eval/success_rate', np.mean(episode_successes), step)
            logger.log('eval/episode_reward', np.mean(episode_rewards), step)
            log_info = {
                'eval/task_name': task_name
            }
            logger.dump(step, ty='eval', info=log_info)


def main(args):
    # Initialize environment
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
    elif args.env_type == 'mujoco':
        train_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'train_env'))
        eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'eval_env'))
        env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            args.discount, train_env_log_dir,
            allow_early_resets=True,
            normalize=False,
            add_onehot=args.add_onehot,
        )
        eval_env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            None, eval_env_log_dir,
            allow_early_resets=True,
            normalize=False,
            add_onehot=args.add_onehot,
        )
    elif args.env_type == 'metaworld':
        train_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'train_env'))
        eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'eval_env'))

        env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            args.discount, train_env_log_dir,
            allow_early_resets=True,
            normalize=False,
            add_onehot=args.add_onehot,
        )

        eval_env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            None, eval_env_log_dir,
            allow_early_resets=True,
            normalize=False,
            add_onehot=args.add_onehot,
        )

    # from PIL import Image
    # for task_env, eval_task_env in zip(env._task_envs, eval_env._task_envs):
    #     task_env.reset()
    #     eval_task_env.reset()
    #
    #     img1 = Image.fromarray(
    #         task_env.render("rgb_array")[:, :, ::-1]
    #     ).resize([480, 480])
    #     img2 = Image.fromarray(
    #         eval_task_env.render("rgb_array")[:, :, ::-1]
    #     ).resize([480, 480])
    #     img1.show()
    #     img2.show()
    #
    #     task_env.reset()
    #     eval_task_env.reset()
    #
    #     img3 = Image.fromarray(
    #         task_env.render("rgb_array")[:, :, ::-1]
    #     ).resize([480, 480])
    #     img4 = Image.fromarray(
    #         eval_task_env.render("rgb_array")[:, :, ::-1]
    #     ).resize([480, 480])
    #     img3.show()
    #     img4.show()

    utils.set_seed_everywhere(args.seed)
    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, args.env_type,
                          height=448, width=448, camera_id=args.video_camera_id)

    # Prepare agent
    # assert torch.cuda.is_available(), 'must have cuda enabled'
    device = torch.device(args.device)

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
    #         obs_space=env.observation_space,
    #         action_space=env.action_space,
    #         capacity=args.replay_buffer_capacity,
    #         device=device,
    #         optimize_memory_usage=True,
    #     )
    # else:
    #     replay_buffer = buffers.ReplayBuffer(
    #         obs_space=env.observation_space,
    #         action_space=env.action_space,
    #         capacity=args.replay_buffer_capacity,
    #         device=device,
    #         optimize_memory_usage=True,
    #     )

    agent = make_agent(
        obs_space=env.observation_space,
        action_space=[env.action_space for _ in range(env.num_tasks)]
        if any(x in args.algo for x in ['mh', 'mi', 'individual'])
        else env.action_space,
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
    # start_time = time.time()
    # train_steps_per_task = args.train_steps_per_task
    if isinstance(env, MultiEnvWrapper):
        total_epochs_per_task = int(args.train_steps_per_task) // args.sac_num_expl_steps_per_process \
                                // args.sac_num_processes

        for task_id in range(env.num_tasks):
            task_steps = 0
            start_time = time.time()
            obs = env.reset(sample_task=True)

            # reset replay buffer
            replay_buffer = buffers.ReplayBuffer(
                obs_space=env.observation_space,
                action_space=env.action_space,
                capacity=args.replay_buffer_capacity,
                device=device,
                optimize_memory_usage=True,
            )

            for task_epoch in range(total_epochs_per_task):
                # Save agent periodically
                if task_epoch % args.save_freq == 0:
                    if args.save_model:
                        agent.save(model_dir, total_steps)

                # Evaluate agent periodically
                if task_epoch % args.eval_freq == 0:
                    print('Evaluating:', args.work_dir)
                    logger.log('eval/episode', episode, total_steps)
                    evaluate(env, eval_env, agent, video, args.num_eval_episodes, logger, total_steps)

                # # (chongyi zheng): force reset outside done = True when step reach train_steps_per_task
                # if task_step >= train_steps_per_task:
                #     obs = env.reset(sample_task=True)
                #
                #     if 'ewc' in args.algo:
                #         agent.estimate_fisher(replay_buffer)
                #     elif 'si' in args.algo:
                #         agent.update_omegas()
                #     elif 'agem' in args.algo:
                #         agent.construct_memory(replay_buffer)
                #
                #     agent.reset_target_critic()
                #     replay_buffer.reset()

                for step in range(args.sac_num_expl_steps_per_process):
                    if task_steps < args.sac_init_steps:
                        action = np.array([env.action_space.sample()
                                           for _ in range(env.unwrapped.num_envs)])
                    else:
                        with utils.eval_mode(agent):
                            if any(x in args.algo for x in ['mh', 'mi', 'individual']):
                                action = agent.act(obs, sample=True, head_idx=task_id)
                            else:
                                action = agent.act(obs, sample=True)

                    next_obs, reward, done, infos = env.step(action)

                    for done_ in done:
                        if done_:
                            episode += 1

                    for info in infos:
                        if 'episode' in info.keys():
                            recent_success.append(info.get('success', False))
                            recent_episode_reward.append(info['episode']['r'])

                    replay_buffer.add(obs, action, reward, next_obs, done, infos)

                    obs = next_obs

                task_steps += args.sac_num_expl_steps_per_process * args.sac_num_processes
                total_steps += args.sac_num_expl_steps_per_process * args.sac_num_processes

                if task_steps >= args.sac_init_steps:
                    for _ in range(args.sac_num_train_iters):
                        if any(x in args.algo for x in ['mh', 'mi', 'individual']):
                            agent.update(replay_buffer, logger, total_steps, head_idx=task_id)
                        else:
                            agent.update(replay_buffer, logger, total_steps)

                end_time = time.time()
                print("FPS: ", int(task_steps / (end_time - start_time)))

                logger.log('train/recent_success', np.mean(recent_success), total_steps)
                logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), total_steps)
                logger.log('train/episode', episode, total_steps)
                log_info = {'train/task_name': infos[0]['task_name']}
                logger.dump(total_steps, ty='train', save=(task_steps > args.sac_init_steps), info=log_info)

                # if done:
                #     success = np.any(episode_successes).astype(np.float)
                #     recent_success.append(success)
                #     recent_episode_reward.append(episode_reward)
                #
                #     logger.log(f'train/episode_success', success, total_steps)
                #     logger.log(f'train/recent_success_rate', np.mean(recent_success), total_steps)
                #     logger.log('train/episode_reward', episode_reward, total_steps)
                #     logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), total_steps)
                #     logger.log('train/episode', episode, total_steps)
                #
                #     if total_steps > 0:
                #         # save non-scalar info
                #         log_info = {
                #             'train/task_name': info['task_name']
                #         }
                #         logger.log('train/duration', time.time() - start_time, total_steps)
                #         start_time = time.time()
                #         # logger.dump(step, ty='train', save=(step > args.init_steps), info=log_info)
                #         logger.dump(total_steps, ty='train', save=(task_step > args.init_steps), info=log_info)

                    # obs = env.reset()
                    # episode_reward = 0
                    # episode_step = 0
                    # episode_successes.clear()
                    # episode += 1

                # # Sample action for data collection
                # if task_step < args.init_steps:
                #     action = np.array(env.action_space.sample())
                # else:
                #     with utils.eval_mode(agent):
                #         if 'mh' in args.algo:
                #             action = agent.act(obs, sample=True, head_idx=task_id)
                #         else:
                #             action = agent.act(obs, sample=True)
                #
                # if 'dqn' in args.algo:
                #     agent.on_step(task_step, train_steps_per_task, logger)
                #
                # # Run training update
                # if task_step >= args.init_steps:
                #     # TODO (chongyi zheng): Do we need multiple updates after initial data collection?
                #     # num_updates = args.init_steps if step == args.init_steps else 1
                #     for _ in range(args.num_train_iters):
                #         if 'mh' in args.algo:
                #             agent.update(replay_buffer, logger, total_steps, head_idx=task_id)
                #         else:
                #             agent.update(replay_buffer, logger, total_steps)
                #
                # # Take step
                # next_obs, reward, done, info = env.step(action)
                #
                # if info.get('success') is not None:
                #     episode_successes.append(info.get('success'))
                #
                # replay_buffer.add(obs, action, reward, next_obs, done)
                # episode_reward += reward
                # obs = next_obs
                # episode_step += 1
                # total_steps += 1

            if 'ewc' in args.algo:
                print(f"Estimating EWC fisher: {infos[0]['task_name']}")
                if any(x in args.algo for x in ['mh', 'mi', 'individual']):
                    agent.estimate_fisher(env=env, replay_buffer=replay_buffer,
                                          head_idx=task_id,
                                          sample_src=args.sac_ewc_estimate_fisher_sample_src)
                else:
                    agent.estimate_fisher(env=env, replay_buffer=replay_buffer,
                                          sample_src=args.sac_ewc_estimate_fisher_sample_src)
            elif 'si' in args.algo:
                print(f"Updating SI omega: {infos[0]['task_name']}")
                agent.update_omegas()
            elif 'agem' in args.algo:
                print(f"Constructing AGEM memory: {infos[0]['task_name']}")
                if any(x in args.algo for x in ['mh', 'mi', 'individual']):
                    agent.construct_memory(env=env, replay_buffer=replay_buffer,
                                           head_idx=task_id,
                                           sample_src=args.sac_agem_memory_sample_src)
                else:
                    agent.construct_memory(env=env, replay_buffer=replay_buffer,
                                           sample_src=args.sac_agem_memory_sample_src)
                # if 'agem_v2' in args.algo:
                #     if any(x in args.algo for x in ['mh', 'mi', 'individual']):
                #         agent.construct_memory(env, head_idx=task_id)
                #     else:
                #         agent.construct_memory(env)
                # else:
                #     agent.construct_memory(replay_buffer)

            agent.reset(reset_critic=args.reset_agent)

    print('Final evaluating:', args.work_dir)
    evaluate(env, eval_env, agent, video, args.num_eval_episodes, logger, total_steps)


if __name__ == '__main__':
    args = parse_args()
    main(args)

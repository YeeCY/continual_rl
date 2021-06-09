import torch
import numpy as np
import os
from collections import deque
import copy
import time


from arguments import parse_args
from environment import make_atari_env, make_single_metaworld_env, make_continual_metaworld_env, \
    make_vec_envs, make_continual_vec_envs
from environment.metaworld_utils import MultiEnvWrapper
from agent import make_agent
import utils
from environment.utils import get_vec_normalize
import storages
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
                    if 'mh' in args.algo:
                        action, _ = agent.act(obs, sample=False, compute_log_pi=False, head_idx=task_id)
                    else:
                        action, _ = agent.act(obs, sample=False, compute_log_pi=False)

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
            if len(episode_successes) > 0:
                logger.log('eval/success_rate', np.mean(episode_successes), step)
            logger.log('eval/episode_reward', np.mean(episode_rewards), step, sw_prefix=task_name + '_')
            log_info = {
                'eval/task_name': task_name
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
    elif args.env_type == 'mujoco':
        # env = make_vec_envs(args.env_name, args.seed, args.ppo_num_processes,
        #                     args.discount, args.work_dir)
        # eval_env = make_vec_envs(args.env_name, args.seed + args.ppo_num_processes,
        #                          args.ppo_num_processes, None, args.work_dir, True)
        train_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'train_env'))
        eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'eval_env'))
        env = make_continual_vec_envs(
            args.env_names, args.seed, args.ppo_num_processes,
            args.discount, train_env_log_dir,
            allow_early_resets=True,
            multi_head=True if 'mh' in args.algo else False,
        )
        eval_env = make_continual_vec_envs(
            args.env_names, args.seed + args.ppo_num_processes,
            args.ppo_num_processes, None, eval_env_log_dir,
            allow_early_resets=True,
            multi_head=True if 'mh' in args.algo else False,
        )
    elif args.env_type == 'metaworld':
        # environment = make_single_metaworld_env(
        #     env_name=args.env_name,
        #     seed=args.seed
        # )
        # eval_env = make_single_metaworld_env(
        #     env_name=args.env_name,
        #     seed=args.seed
        # )
        # env = make_continual_metaworld_env(
        #     env_names=args.env_names,
        #     seed=args.seed
        # )
        # eval_env = copy.deepcopy(env)
        # eval_env = make_continual_metaworld_env(
        #     env_names=args.env_names,
        #     seed=args.seed
        # )
        train_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'train_env'))
        eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'eval_env'))

        env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            args.discount, train_env_log_dir,
            allow_early_resets=True,
            normalize=True,
            multi_head=True if 'mh' in args.algo else False,
        )

        eval_env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            None, eval_env_log_dir,
            allow_early_resets=True,
            normalize=True,
            multi_head=True if 'mh' in args.algo else False,
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
    # elif args.env_type == 'dmc_locomotion' or 'metaworld_utils':
    #     replay_buffer = buffers.ReplayBuffer(
    #         obs_space=environment.observation_space,
    #         action_space=environment.action_space,
    #         capacity=args.replay_buffer_capacity,
    #         device=device,
    #         optimize_memory_usage=True,
    #     )
    if 'mh' not in args.algo:
        rollouts = storages.RolloutStorage(args.ppo_num_rollout_steps_per_process,
                                           args.ppo_num_processes,
                                           env.observation_space.shape,
                                           env.action_space,
                                           device)

        if 'ewc' in args.algo:
            est_fisher_rollouts = storages.RolloutStorage(args.ppo_ewc_rollout_steps_per_process,
                                                          args.ppo_num_processes,
                                                          env.observation_space.shape,
                                                          env.action_space,
                                                          device)

    agent = make_agent(
        obs_space=env.observation_space,
        action_space=env.all_action_spaces if 'mh' in args.algo else env.action_space,
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
    if isinstance(env, MultiEnvWrapper):
        total_epochs_per_task = int(args.train_steps_per_task) // args.ppo_num_rollout_steps_per_process \
                                // args.ppo_num_processes

        for task_id in range(env.num_tasks):
            task_steps = 0
            start_time = time.time()
            obs = env.reset(sample_task=True)

            if 'mh' in args.algo:
                rollouts = storages.RolloutStorage(args.ppo_num_rollout_steps_per_process,
                                                   args.ppo_num_processes,
                                                   env.observation_space.shape,
                                                   env.all_action_spaces[task_id],
                                                   device)

                if 'ewc' in args.algo:
                    est_fisher_rollouts = storages.RolloutStorage(args.ppo_ewc_rollout_steps_per_process,
                                                                  args.ppo_num_processes,
                                                                  env.observation_space.shape,
                                                                  env.all_action_spaces[task_id],
                                                                  device)

            rollouts.obs[0].copy_(torch.Tensor(obs).to(device))
            for task_epoch in range(total_epochs_per_task):
                agent.update_learning_rate(task_epoch, total_epochs_per_task)

                if task_epoch % args.save_freq == 0:
                    if args.save_model:
                        agent.save(model_dir, total_steps)

                if task_epoch % args.eval_freq == 0:
                    print('Evaluating:', args.work_dir)
                    logger.log('eval/episode', episode, total_steps)
                    evaluate(env, eval_env, agent, video, args.num_eval_episodes, logger, total_steps)

                for step in range(args.ppo_num_rollout_steps_per_process):
                    with utils.eval_mode(agent):
                        if 'mh' in args.algo:
                            action, log_pi = agent.act(obs, sample=True, compute_log_pi=True, head_idx=task_id)
                            value = agent.predict_value(obs, head_idx=task_id)
                        else:
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
                if 'mh' in args.algo:
                    next_value = agent.predict_value(rollouts.obs[-1], head_idx=task_id)
                else:
                    next_value = agent.predict_value(rollouts.obs[-1])
                rollouts.compute_returns(next_value, args.discount,
                                         args.ppo_gae_lambda,
                                         args.ppo_use_proper_time_limits)
                if 'mh' in args.algo:
                    agent.update(rollouts, logger, total_steps, head_idx=task_id)
                else:
                    agent.update(rollouts, logger, total_steps)
                rollouts.after_update()

                # log statistics
                end_time = time.time()
                print("FPS: ", int(task_steps / (end_time - start_time)))

                logger.log('train/recent_success', np.mean(recent_success), total_steps)
                logger.log('train/recent_episode_reward', np.mean(recent_episode_reward), total_steps)
                logger.log('train/episode', episode, total_steps)
                log_info = {'train/task_name': infos[0]['task_name']}
                logger.dump(total_steps, ty='train', save=True, info=log_info)

            if 'ewc' in args.algo:
                compute_returns_kwargs = {
                    'gamma': args.discount,
                    'gae_lambda': args.ppo_gae_lambda,
                    'use_proper_time_limits': args.ppo_use_proper_time_limits
                }
                print(f"Estimating EWC fisher: {infos[0]['task_name']}")
                if 'mh' in args.algo:
                    agent.estimate_fisher(env, est_fisher_rollouts, compute_returns_kwargs, head_idx=task_id)
                else:
                    agent.estimate_fisher(env, est_fisher_rollouts, compute_returns_kwargs)
            elif 'si' in args.algo:
                agent.update_omegas()
            elif 'agem' in args.algo:
                compute_returns_kwargs = {
                    'gamma': args.discount,
                    'gae_lambda': args.ppo_gae_lambda,
                    'use_proper_time_limits': args.ppo_use_proper_time_limits
                }
                print(f"Constructing AGEM fisher: {infos[0]['task_name']}")
                if 'mh' in args.algo:
                    agent.construct_memory(env, args.ppo_num_processes, compute_returns_kwargs, head_idx=task_id)
                else:
                    agent.construct_memory(env, args.ppo_num_processes, compute_returns_kwargs)

    print('Final evaluating:', args.work_dir)
    evaluate(env, eval_env, agent, video, args.num_eval_episodes, logger, total_steps)


if __name__ == '__main__':
    args = parse_args()
    main(args)

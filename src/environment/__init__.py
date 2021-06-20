import metaworld
import gym
import os

from gym.wrappers import TimeLimit

from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv)

from src.environment import atari_wrappers
from src.environment.gym_wrapper import TransposeImage, TimeLimitMask, VecNormalize
from src.environment.metaworld_utils import MetaWorldTaskSampler, SingleMT1Wrapper, MultiEnvWrapper, NormalizedEnv
from src.environment.metaworld_utils import uniform_random_strategy, round_robin_strategy
from src.environment.metaworld_utils.wrappers import TaskNameWrapper

import src.utils as utils


def make_atari_env(env_name, seed=None, action_repeat=4, frame_stack=4):
    return atari_wrappers.wrap_deepmind(
        env_id=env_name,
        seed=seed,
        frame_skip=action_repeat,
        frame_stack=frame_stack
    )


def make_single_metaworld_env(env_name, seed=None):
    mt1 = metaworld.MT1(env_name, seed=seed)
    env = mt1.train_classes[env_name]()
    # task = random.choice(mt1.train_tasks)
    # environment.set_task(task)
    env = SingleMT1Wrapper(env, mt1.train_tasks)
    env.seed(seed)

    return env


def make_continual_metaworld_env(env_names, seed=None):
    envs = []
    for env_name in env_names:
        mt1 = metaworld.MT1(env_name)
        train_task_sampler = MetaWorldTaskSampler(
            mt1, 'train',
            lambda env, _: NormalizedEnv(env))
        env_up = train_task_sampler.sample(1)[0]
        envs.append(env_up())
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla',
                          env_names=env_names)
    env.seed(seed)

    return env


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        try:
            env = gym.make(env_id)
        except gym.error.UnregisteredEnv:
            mt1 = metaworld.MT1(env_id, seed=seed + rank)
            # train_task_sampler = MetaWorldTaskSampler(
            #     mt1, 'train',
            #     lambda env, _: NormalizedEnv(env))
            # env = train_task_sampler.sample(1)[0]()
            env = mt1.train_classes[env_id]()
            env.set_task(mt1.train_tasks[0])

            env = TaskNameWrapper(env, task_name=env_id)
            # normalize action
            env = NormalizedEnv(env)
            env = TimeLimit(env, max_episode_steps=env.max_path_length)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(env,
                          os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = EpisodicLifeEnv(env)
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WarpFrame(env, width=84, height=84)
                env = ClipRewardEnv(env)

        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input environment.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  discount,
                  log_dir,
                  allow_early_resets=False,
                  normalize=True):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if normalize:
        if len(envs.observation_space.shape) == 1:
            if discount is None:
                envs = VecNormalize(envs, norm_reward=False)
            else:
                envs = VecNormalize(envs, gamma=discount)

    # envs = VecPyTorch(envs, device)
    #
    # if num_frame_stack is not None:
    #     envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    # elif len(envs.observation_space.shape) == 3:
    #     envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


def make_continual_vec_envs(env_names,
                            seed,
                            num_processes,
                            discount,
                            log_dir,
                            allow_early_resets=False,
                            normalize=True,
                            multi_head=False):
    # TODO (chongyi zheng): We fork many processes here, optimize it
    envs = []
    for env_name in env_names:
        env_log_dir = utils.make_dir(os.path.join(log_dir, env_name)) \
            if log_dir is not None else None
        env = make_vec_envs(env_name, seed, num_processes, discount,
                            env_log_dir, allow_early_resets=allow_early_resets,
                            normalize=normalize)
        env.reward_range = env.get_attr('reward_range')  # prevent wrapper error
        envs.append(env)
    continual_env = MultiEnvWrapper(envs,
                                    sample_strategy=round_robin_strategy,
                                    mode='vanilla',
                                    augment_observation=True,
                                    augment_action=not multi_head,
                                    env_names=env_names)

    return continual_env


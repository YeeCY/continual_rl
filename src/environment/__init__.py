import dmc2gym
import metaworld
import gym
import os


from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv)


from environment import dmc_wrappers
from environment import atari_wrappers
from environment.gym_wrapper import TransposeImage, TimeLimitMask, VecNormalize
from environment.metaworld import MetaWorldTaskSampler, SingleMT1Wrapper, MultiEnvWrapper, NormalizedEnv
from environment.metaworld import uniform_random_strategy, round_robin_strategy


def make_pad_env(
        domain_name,
        task_name,
        seed=0,
        episode_length=1000,
        frame_stack=3,
        action_repeat=4,
        mode='train'
):
    """Make environment for PAD experiments"""
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=True,
        height=100,
        width=100,
        episode_length=episode_length,
        frame_skip=action_repeat
    )
    env.seed(seed)
    env = dmc_wrappers.GreenScreen(env, mode)
    env = dmc_wrappers.FrameStack(env, frame_stack)
    env = dmc_wrappers.ColorWrapper(env, mode)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def make_locomotion_env(
        env_name,
        seed=0,
        episode_length=1000,
        from_pixels=True,
        frame_stack=3,
        action_repeat=4,
        obs_height=100,
        obs_width=100,
        camera_id=0,
        mode='train'):
    """(chongyi zheng) Make dm_control locomotion environments for experiments"""
    env = dmc2gym.make_locomotion(
        env_name=env_name,
        seed=seed,
        from_pixels=from_pixels,
        height=obs_height,
        width=obs_width,
        camera_id=camera_id,
        episode_length=episode_length,
        frame_skip=action_repeat
    )
    env.seed(seed)

    if from_pixels:
        env = dmc_wrappers.VideoBackground(env, mode)
        env = dmc_wrappers.FrameStack(env, frame_stack)
        env = dmc_wrappers.ColorWrapper(env, mode)
    else:
        env = dmc_wrappers.AugmentObs(env, mode)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def make_atari_env(env_name, seed=None, action_repeat=4, frame_stack=4):
    return atari_wrappers.wrap_deepmind(
        env_id=env_name,
        seed=seed,
        frame_skip=action_repeat,
        frame_stack=frame_stack
    )


def make_single_metaworld_env(env_name, seed=None):
    mt1 = metaworld.MT1(env_name)
    env = mt1.train_classes[env_name]()
    # task = random.choice(mt1.train_tasks)
    # environment.set_task(task)
    env = SingleMT1Wrapper(env, mt1.train_tasks)
    env.seed(seed)

    return env


def make_continual_metaworld_env(env_names, seed=None):
    # envs = metaworld_wrappers.MT10Wrapper()
    # envs.seed(seed)

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
        env = gym.make(env_id)

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
                  allow_early_resets=False):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

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

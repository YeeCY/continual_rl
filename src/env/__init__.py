import dmc2gym
import metaworld

from env import dmc_wrappers
from env import atari_wrappers
from env.metaworld import MetaWorldTaskSampler, SingleMT1Wrapper, MultiEnvWrapper, NormalizedEnv
from env.metaworld import uniform_random_strategy, round_robin_strategy


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
    # env.set_task(task)
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

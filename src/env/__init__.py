import dmc2gym
from env import dmc_wrappers
# from env import atari_wrappers


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


# def make_atari_env(env_name, action_repeat=4, frame_stack=4):
#     return atari_wrappers.wrap_deepmind(
#         env_id=env_name,
#         frame_skip=action_repeat,
#         frame_stack=frame_stack
#     )
from src.env.atari_wrappers import make_atari, wrap_deepmind
import gym
from gym.spaces import Box


def make_atari_env(env_id):
    env = make_atari(env_id)
    env = wrap_deepmind(env,
                        episode_life=True,
                        clip_rewards=True,
                        frame_stack=True,
                        scale=False)
    env = TransposeImage(env)

    return env


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

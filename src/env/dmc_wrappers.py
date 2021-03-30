import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import dmc2gym
from dm_control.suite import common
import cv2
from collections import deque

from env.utils import do_green_screen, interpolate_bg, replace_bg


class ColorWrapper(gym.Wrapper):
    """Wrapper for the color experiments"""

    def __init__(self, env, mode):
        assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._mode = mode
        self.time_step = 0
        if 'color' in self._mode:
            self._load_colors()

    def reset(self):
        self.time_step = 0
        # TODO (chongyi zheng): implement color randomization for locomotion environments
        if 'color' in self._mode:
            self.randomize()
        return self.env.reset()

    def step(self, action):
        self.time_step += 1
        return self.env.step(action)

    def randomize(self):
        assert 'color' in self._mode, f'can only randomize in color mode, received {self._mode}'
        self.reload_physics(self.get_random_color())

    def _load_colors(self):
        assert self._mode in {'eval_color_easy', 'eval_color_hard'}
        self._colors = torch.load('src/env/data/{}.pt'.format(self._mode.replace('eval_', '')))

    def get_random_color(self):
        assert len(self._colors) >= 100, 'env must include at least 100 colors'
        return self._colors[randint(len(self._colors))]

    def reload_physics(self, setting_kwargs=None, state=None):
        domain_name = self._get_dmc_wrapper()._domain_name
        if setting_kwargs is None:
            setting_kwargs = {}
        if state is None:
            state = self._get_state()
        self._reload_physics(
            *common.settings.get_model_and_assets_from_setting_kwargs(
                domain_name + '.xml', setting_kwargs
            )
        )
        self._set_state(state)

    def get_state(self):
        return self._get_state()

    def set_state(self, state):
        self._set_state(state)

    def _get_dmc_wrapper(self):
        _env = self.env
        while not isinstance(_env, dmc2gym.wrappers.DMCWrapper) and hasattr(_env, 'env'):
            _env = _env.env
        assert isinstance(_env, dmc2gym.wrappers.DMCWrapper), 'environment is not dmc2gym-wrapped'

        return _env

    def _reload_physics(self, xml_string, assets=None):
        _env = self.env
        while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
            _env = _env.env
        assert hasattr(_env, '_physics'), 'environment does not have physics attribute'
        _env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_physics(self):
        _env = self.env
        while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
            _env = _env.env
        assert hasattr(_env, '_physics'), 'environment does not have physics attribute'

        return _env._physics

    def _get_state(self):
        return self._get_physics().get_state()

    def _set_state(self, state):
        self._get_physics().set_state(state)


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class GreenScreen(gym.Wrapper):
    """Green screen for video experiments"""

    def __init__(self, env, mode):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        if 'video' in mode:
            self._video = mode.replace('eval_', '')
            if not self._video.endswith('.mp4'):
                self._video += '.mp4'
            self._video = os.path.join('src/env/data', self._video)
            self._data = self._load_video(self._video)
        else:
            self._video = None
        self._max_episode_steps = env._max_episode_steps

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        cap = cv2.VideoCapture(video)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                       np.dtype('uint8'))
        i, ret = 0, True
        while (i < n and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def reset(self):
        self._current_frame = 0
        return self._greenscreen(self.env.reset())

    def step(self, action):
        self._current_frame += 1
        obs, reward, done, info = self.env.step(action)
        return self._greenscreen(obs), reward, done, info

    def _interpolate_bg(self, bg, size: tuple):
        """Interpolate background to size of observation"""
        bg = torch.from_numpy(bg).float().unsqueeze(0) / 255
        bg = F.interpolate(bg, size=size, mode='bilinear', align_corners=False)
        return (bg * 255).byte().squeeze(0).numpy()

    def _greenscreen(self, obs):
        """Applies greenscreen if video is selected, otherwise does nothing"""
        if self._video:
            bg = self._data[self._current_frame % len(self._data)]  # select frame
            bg = self._interpolate_bg(bg, obs.shape[1:])  # scale bg to observation size
            return do_green_screen(obs, bg)  # apply greenscreen
        return obs

    def apply_to(self, obs):
        """Applies greenscreen mode of object to observation"""
        obs = obs.copy()
        channels_last = obs.shape[-1] == 3
        if channels_last:
            obs = torch.from_numpy(obs).permute(2, 0, 1).numpy()
        obs = self._greenscreen(obs)
        if channels_last:
            obs = torch.from_numpy(obs).permute(1, 2, 0).numpy()
        return obs


class VideoBackground(gym.Wrapper):
    """Change observation background for video experiments"""

    def __init__(self, env, mode):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        if 'video' in mode:
            self._video = mode.replace('eval_', '')
            if not self._video.endswith('.mp4'):
                self._video += '.mp4'
            self._video = os.path.join('src/env/data', self._video)
            self._data = self._load_video(self._video)
        else:
            self._video = None
        self._max_episode_steps = env._max_episode_steps
        self._current_frame = None

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        cap = cv2.VideoCapture(video)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                       dtype=np.uint8)
        i, ret = 0, True
        while (i < n and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def reset(self):
        self._current_frame = 0
        return self._change_bg(self.env.reset())

    def step(self, action):
        self._current_frame += 1
        obs, reward, done, info = self.env.step(action)
        return self._change_bg(obs), reward, done, info

    def _change_bg(self, obs, camera_id=None):
        """Applies greenscreen if video is selected, otherwise does nothing"""
        if self._video:
            bg = self._data[self._current_frame % len(self._data)]  # select frame
            bg = interpolate_bg(bg, obs.shape[1:])  # scale bg to observation size

            dmc_loco_wrapper = self.env.env
            seg = self.env.render(
                mode='segmentation',
                height=obs.shape[1],
                width=obs.shape[2],
                camera_id=dmc_loco_wrapper.camera_id if not camera_id else camera_id
            )
            seg = seg.transpose(2, 0, 1)  # channels first

            return replace_bg(obs, seg, bg)
        return obs

    def apply_to(self, obs, camera_id=None):
        """Applies greenscreen mode of object to observation"""
        obs = obs.copy()
        channels_last = obs.shape[-1] == 3
        if channels_last:
            obs = torch.from_numpy(obs).permute(2, 0, 1).numpy()
        obs = self._change_bg(obs, camera_id=camera_id)
        if channels_last:
            obs = torch.from_numpy(obs).permute(1, 2, 0).numpy()
        return obs


class AugmentObs(gym.Wrapper):
    """Augment observations with task specific features, i.e. task_id, and pad the dimension"""
    # TODO (chongyi zheng)
    def __init__(self, env, mode):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        self._max_episode_steps = env._max_episode_steps
        self._current_frame = None

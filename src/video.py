import imageio
import os
from PIL import Image
import numpy as np

from environment.gym_wrapper import VecNormalize


class VideoRecorder(object):
    def __init__(self, dir_name, env_type, height=100, width=100, camera_id=0, fps=25):
        self.dir_name = dir_name
        self.env_type = env_type
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, losses=[]):
        if self.enabled:
            if self.env_type == 'atari':
                frame = env.render(mode='rgb_array')
                frame = Image.fromarray(frame).resize([self.width, self.height])
                frame = np.asarray(frame)
            elif self.env_type == 'metaworld':
                frame = env.render(mode='rgb_array')
                frame = Image.fromarray(frame[:, :, ::-1]).resize([self.width, self.height])
                frame = np.asarray(frame)
            elif self.env_type == 'mujoco':
                assert isinstance(env, VecNormalize)
                frame = env.render(mode='rgb_array')
                frame = Image.fromarray(frame).resize([self.width, self.height])
                frame = np.asarray(frame)
            else:
                raise ValueError(f"Unknown environment type {self.env_type}")
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

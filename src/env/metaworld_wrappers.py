from gym import Wrapper
import random

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class SingleMT1Wrapper(Wrapper):
    def __init__(self, env, tasks):
        assert isinstance(env, SawyerXYZEnv), f"Invalid environment type: {type(env)}"
        super().__init__(env)

        self.tasks = tasks

    def _set_random_task(self):
        task = random.choice(self.tasks)
        self.env.set_task(task)

    def reset(self, random_task=True):
        if random_task:
            self._set_random_task()

        return self.env.reset()

    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.env.curr_path_length > self.env.max_path_length or info.get('success'):
            done = True

        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        return self.env.render(mode=mode, **kwargs)

    def close(self):
        self.env.close()


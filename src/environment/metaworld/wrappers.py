import random
import numpy as np
import gym
from gym.spaces import Box

from environment.metaworld.utils import uniform_random_strategy
# from garage import EnvSpec, EnvStep, Wrapper

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class SingleMT1Wrapper(gym.Wrapper):
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

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.env.curr_path_length > self.env.max_path_length or info.get('success'):
            done = True

        return obs, reward, done, info


# TODO (chongyi zheng): truncate episode when success hurt performance, may delete this
# class SuccessTruncatedTimeLimitWrapper(gym.wrappers.TimeLimit):
#     def __init__(self, environment, max_episode_steps):
#         super().__init__(environment, max_episode_steps=max_episode_steps)
#
#     def step(self, action):
#         obs, reward, done, info = super().step(action)
#         success = info.get('success')
#         if success == 1.0:
#             done = True
#
#         return obs, reward, done, info


class NormalizedEnv(gym.Wrapper):
    """An environment wrapper for normalization.
    This wrapper normalizes action, and optionally observation and reward.
    Args:
        env (Environment): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.
    """

    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            expected_action_scale=1.,
            flatten_obs=True,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        super().__init__(env)

        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        # (chongyi zheng): use 'np.prod'
        flat_obs_dim = np.prod(self.env.observation_space.shape)
        self._obs_mean = np.zeros(flat_obs_dim)
        self._obs_var = np.ones(flat_obs_dim)

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

    def reset(self):
        """Call reset on wrapped environment.
        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)
        """
        # (chongyi zheng): remove episode_info
        first_obs = self.env.reset()
        if self._normalize_obs:
            return self._apply_normalize_obs(first_obs)
        else:
            return first_obs

    def step(self, action):
        """Call step on wrapped environment.
        Args:
            action (np.ndarray): An action provided by the agent.
        Returns:
            EnvStep: The environment step resulting from the action.
        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.
        """
        # (chongyi zheng): remove EnvStep
        if isinstance(self.action_space, Box):
            # rescale the action when the bounds are not inf
            lb, ub = self.action_space.low, self.action_space.high
            if np.all(lb != -np.inf) and np.all(ub != -np.inf):
                scaled_action = lb + (action + self._expected_action_scale) * (
                        0.5 * (ub - lb) / self._expected_action_scale)
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action

        obs, reward, done, info = self.env.step(scaled_action)

        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return obs, reward * self._scale_reward, done, info

    def _update_obs_estimate(self, obs):
        flat_obs = self.env.observation_space.flatten(obs)
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(
            flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * \
                            self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
                                   1 - self._reward_alpha
                           ) * self._reward_var + self._reward_alpha * np.square(
            reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        """Compute normalized observation.
        Args:
            obs (np.ndarray): Observation.
        Returns:
            np.ndarray: Normalized observation.
        """
        self._update_obs_estimate(obs)
        flat_obs = self.env.observation_space.flatten(obs)
        normalized_obs = (flat_obs -
                          self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        if not self._flatten_obs:
            normalized_obs = self.env.observation_space.unflatten(
                self.env.observation_space, normalized_obs)
        return normalized_obs

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.
        Args:
            reward (float): Reward.
        Returns:
            float: Normalized reward.
        """
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)


class TaskNameWrapper(gym.Wrapper):
    """Add task_name or task_id to environment infos.
    Args:
        env (gym.Env): The environment to wrap.
        task_name (str or None): Task name to be added, if any.
        task_id (int or None): Task ID to be added, if any.
    """

    def __init__(self, env, task_name=None, task_id=None):
        super().__init__(env)
        self._task_name = task_name
        self._task_id = task_id

    def step(self, action):
        """gym.Env step for the active task environment.
        Args:
            action (np.ndarray): Action performed by the agent in the
                environment.
        Returns:
            tuple:
                np.ndarray: Agent's observation of the current environment.
                float: Amount of reward yielded by previous action.
                bool: True iff the episode has ended.
                dict[str, np.ndarray]: Contains auxiliary diagnostic
                    information about this time-step.
        """
        # (chongyi zheng): remove 'es'
        obs, reward, done, info = super().step(action)
        if self._task_name is not None:
            info['task_name'] = self._task_name
        if self._task_id is not None:
            info['task_id'] = self._task_id
        return obs, reward, done, info


class TaskOnehotWrapper(gym.Wrapper):
    """Append a one-hot task representation to an environment.
    See TaskOnehotWrapper.wrap_env_list for the recommended way of creating
    this class.
    Args:
        env (Environment): The environment to wrap.
        task_index (int): The index of this task among the tasks.
        n_total_tasks (int): The number of total tasks.
    """

    def __init__(self, env, task_index, n_total_tasks):
        assert 0 <= task_index < n_total_tasks
        super().__init__(env)
        self._task_index = task_index
        self._n_total_tasks = n_total_tasks
        env_lb = self.env.observation_space.low
        env_ub = self.env.observation_space.high
        one_hot_ub = np.ones(self._n_total_tasks)
        one_hot_lb = np.zeros(self._n_total_tasks)

        self._observation_space = Box(
            np.concatenate([env_lb, one_hot_lb]),
            np.concatenate([env_ub, one_hot_ub]))
        # self._spec = EnvSpec(
        #     action_space=self.action_space,
        #     observation_space=self.observation_space,
        #     max_episode_length=self._env.spec.max_episode_length)

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    # @property
    # def spec(self):
    #     """Return the environment specification.
    #     Returns:
    #         EnvSpec: The envionrment specification.
    #     """
    #     return self._spec

    def reset(self):
        """Sample new task and call reset on new task environment.
        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)
        """
        # (chongyi zheng): remove episode_info
        first_obs = self.env.reset()
        first_obs = self._obs_with_one_hot(first_obs)

        return first_obs

    def step(self, action):
        """Environment step for the active task environment.
        Args:
            action (np.ndarray): Action performed by the agent in the
                environment.
        Returns:
            EnvStep: The environment step resulting from the action.
        """
        # (chongyi zheng): remove garage EnvStep
        obs, reward, done, info = self.env.step(action)
        oh_obs = self._obs_with_one_hot(obs)

        info['task_id'] = self._task_index

        return oh_obs, reward, done, info

    def _obs_with_one_hot(self, obs):
        """Concatenate observation and task one-hot.
        Args:
            obs (numpy.ndarray): observation
        Returns:
            numpy.ndarray: observation + task one-hot.
        """
        one_hot = np.zeros(self._n_total_tasks)
        one_hot[self._task_index] = 1.0
        return np.concatenate([obs, one_hot])

    @classmethod
    def wrap_env_list(cls, envs):
        """Wrap a list of environments, giving each environment a one-hot.
        This is the primary way of constructing instances of this class.
        It's mostly useful when training multi-task algorithms using a
        multi-task aware sampler.
        For example:
        '''
        .. code-block:: python
            envs = get_mt10_envs()
            wrapped = TaskOnehotWrapper.wrap_env_list(envs)
            sampler = trainer.make_sampler(LocalSampler, environment=wrapped)
        '''
        Args:
            envs (list[Environment]): List of environments to wrap. Note
            that the
                order these environments are passed in determines the value of
                their one-hot encoding. It is essential that this list is
                always in the same order, or the resulting encodings will be
                inconsistent.
        Returns:
            list[TaskOnehotWrapper]: The wrapped environments.
        """
        n_total_tasks = len(envs)
        wrapped = []
        for i, env in enumerate(envs):
            wrapped.append(cls(env, task_index=i, n_total_tasks=n_total_tasks))
        return wrapped

    @classmethod
    def wrap_env_cons_list(cls, env_cons):
        """Wrap a list of environment constructors, giving each a one-hot.
        This function is useful if you want to avoid constructing any
        environments in the main experiment process, and are using a multi-task
        aware remote sampler (i.e. `~RaySampler`).
        For example:
        '''
        .. code-block:: python
            env_constructors = get_mt10_env_cons()
            wrapped = TaskOnehotWrapper.wrap_env_cons_list(env_constructors)
            env_updates = [NewEnvUpdate(wrapped_con)
                           for wrapped_con in wrapped]
            sampler = trainer.make_sampler(RaySampler, environment=env_updates)
        '''
        Args:
            env_cons (list[Callable[Environment]]): List of environment
            constructor
                to wrap. Note that the order these constructors are passed in
                determines the value of their one-hot encoding. It is essential
                that this list is always in the same order, or the resulting
                encodings will be inconsistent.
        Returns:
            list[Callable[TaskOnehotWrapper]]: The wrapped environments.
        """
        n_total_tasks = len(env_cons)
        wrapped = []
        for i, con in enumerate(env_cons):
            # Manually capture this value of i by introducing a new scope.
            wrapped.append(lambda i=i, con=con: cls(
                con(), task_index=i, n_total_tasks=n_total_tasks))
        return wrapped


class MultiEnvWrapper(gym.Wrapper):
    """
    Adapt from https://github.com/rlworkgroup/garage/blob/master/src/garage/envs/multi_env_wrapper.py

    A wrapper class to handle multiple environments.
    This wrapper adds an integer 'task_id' and a string 'task_name' to env_info every timestep.
    Args:
        envs (list(Environment)):
            A list of objects implementing Environment.
        sample_strategy (function(int, int)):
            Sample strategy to be used when sampling a new task.
        mode (str): A string from 'vanilla`, 'add-onehot' and 'del-onehot'.
            The type of observation to use.
            - 'vanilla' provides the observation as it is.
              Use case: metaworld environments with MT* algorithms,
                        gym environments with Task Embedding.
            - 'add-onehot' will append an one-hot task id to observation.
              Use case: gym environments with MT* algorithms.
            - 'del-onehot' assumes an one-hot task id is appended to
              observation, and it excludes that.
              Use case: metaworld environments with Task Embedding.
        env_names (list(str)): The names of the environments corresponding to
            envs. The index of an env_name must correspond to the index of the
            corresponding environment in envs. An env_name in env_names must be unique.
    """

    def __init__(self,
                 envs,
                 sample_strategy=uniform_random_strategy,
                 mode='add-onehot',
                 augment_observation=False,
                 augment_action=False,
                 env_names=None):
        assert mode in ['vanilla', 'add-onehot', 'del-onehot']

        self._sample_strategy = sample_strategy
        self._num_tasks = len(envs)
        self._active_task_index = None
        self._mode = mode
        self._observation_space_index = 0
        self._action_space_index = 0

        super().__init__(envs[0])

        if env_names is not None:
            assert isinstance(env_names, list), 'env_names must be a list'
            msg = ('env_names are not unique or there is not an env_name',
                   'corresponding to each environment in envs')
            assert len(set(env_names)) == len(envs), msg
        self._env_names = env_names
        self._task_envs = []

        max_observation_dim = np.prod(self.env.observation_space.shape)
        max_action_dim = np.prod(self.env.action_space.shape)
        for i, env in enumerate(envs):
            if augment_observation:
                assert len(env.observation_space.shape) == 1
                if np.prod(env.observation_space.shape) > max_observation_dim:
                    self._observation_space_index = i
                    max_observation_dim = np.prod(env.observation_space.shape)
            else:
                if env.observation_space.shape != self.env.observation_space.shape:
                    raise ValueError(
                        'Observation space of all envs should be same.')

            if augment_action:
                assert len(env.action_space.shape) == 1
                if np.prod(env.action_space.shape) > max_action_dim:
                    self._action_space_index = i
                    max_action_dim = np.prod(env.action_space.shape)
            else:
                if env.action_space.shape != self.env.action_space.shape:
                    raise ValueError('Action space of all envs should be same.')
            self._task_envs.append(env)
        self._max_observation_dim = max_observation_dim
        self._max_action_dim = max_action_dim

        self.observation_space = self._update_observation_space()  # avoid crash
        self.action_space = self._task_envs[self._action_space_index].action_space

    def _update_observation_space(self):
        """Observation space.

        Returns:
            akro.Box: Observation space.

        """
        # (chongyi zheng): avoid crash
        if self._mode == 'vanilla':
            return self._task_envs[self._observation_space_index].observation_space
        elif self._mode == 'add-onehot':
            task_lb, task_ub = self.task_space.bounds
            env_lb, env_ub = self._task_envs[self._observation_space_index].observation_space.bounds
            return Box(np.concatenate([env_lb, task_lb]),
                       np.concatenate([env_ub, task_ub]))
        else:  # self._mode == 'del-onehot'
            env_lb, env_ub = self._task_envs[self._observation_space_index].bounds
            num_tasks = self._num_tasks
            return Box(env_lb[:-num_tasks], env_ub[:-num_tasks])

    def _augment_observation(self, obs):
        if obs.shape == self.observation_space.shape:
            obs = np.expand_dims(obs, axis=0)

        # optionally zero-pad observation
        if np.prod(obs.shape[1:]) < self._max_observation_dim:
            zeros = np.zeros([
                obs.shape[0],
                self._max_observation_dim - np.prod(obs.shape[1:])
            ])
            obs = np.concatenate([obs, zeros], axis=-1)

        return obs

    def _curtail_action(self, action):
        if action.shape == self.action_space.shape:
            action = np.expand_dims(action, axis=0)

        # optionally curtail action
        env_action_dim = np.prod(self.env.action_space.shape)
        if np.prod(action.shape[1:]) > env_action_dim:
            action = action[:, :env_action_dim]

        return action

    def seed(self, seed=None):
        for idx, task_env in enumerate(self._task_envs):
            task_env.seed(seed + idx)

        return

    @property
    def num_tasks(self):
        """Total number of tasks.

        Returns:
            int: number of tasks.

        """
        return len(self._task_envs)

    @property
    def env_names(self):
        """Name of tasks.

        Returns:
            list: name of tasks.

        """
        return self._env_names

    @property
    def task_space(self):
        """Task Space.

        Returns:
            akro.Box: Task space.

        """
        one_hot_ub = np.ones(self.num_tasks)
        one_hot_lb = np.zeros(self.num_tasks)
        return Box(one_hot_lb, one_hot_ub)

    @property
    def active_task_index(self):
        """Index of active task environment.

        Returns:
            int: Index of active task.

        """
        if hasattr(self.env, 'active_task_index'):
            return self.env.active_task_index
        else:
            return self._active_task_index

    def reset(self, sample_task=False):
        """Sample new task and call reset on new task environment.
        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)
        """
        # (chongyi zheng): remove episode_info
        if sample_task or self._active_task_index is None:
            self._active_task_index = self._sample_strategy(
                self._num_tasks, self._active_task_index)
        self.env = self._task_envs[self._active_task_index]
        obs = self.env.reset()

        obs = self._augment_observation(obs)

        if self._mode == 'vanilla':
            pass
        elif self._mode == 'add-onehot':
            obs = np.concatenate([obs, self._active_task_one_hot()])
        else:  # self._mode == 'del-onehot'
            obs = obs[:-self._num_tasks]

        return obs

    def step(self, action):
        """Step the active task environment.
        Args:
            action (object): object to be passed in Environment.reset(action)
        Returns:
            EnvStep: The environment step resulting from the action.
        """
        # (chongyi zheng): remove garage EnvStep
        action = self._curtail_action(action)

        obs, reward, done, info = self.env.step(action)

        if self._mode == 'add-onehot':
            obs = np.concatenate([obs, self._active_task_one_hot()])
        elif self._mode == 'del-onehot':
            obs = obs[:-self._num_tasks]
        else:  # self._mode == 'vanilla'
            obs = obs

        obs = self._augment_observation(obs)

        if isinstance(info, dict) and 'task_id' not in info:
            info['task_id'] = self._active_task_index
            if self._env_names is not None:
                info['task_name'] = self._env_names[self._active_task_index]
        elif isinstance(info, list) and 'task_id' not in info[0]:
            for info_ in info:
                info_['task_id'] = self._active_task_index
                if self._env_names is not None:
                    info_['task_name'] = self._env_names[self._active_task_index]

        return obs, reward, done, info

    def close(self):
        """Close all task envs."""
        for env in self._task_envs:
            env.close()

    def _active_task_one_hot(self):
        """One-hot representation of active task.
        Returns:
            numpy.ndarray: one-hot representation of active task
        """
        one_hot = np.zeros(self.task_space.shape)
        index = self.active_task_index or 0
        one_hot[index] = self.task_space.high[index]
        return one_hot

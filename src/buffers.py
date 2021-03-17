import torch
import numpy as np

from utils import random_crop


class ReplayBuffer(object):
    """Buffer to store environment transitions

    (Chongyi Zheng): update replay buffer to stable_baselines style to save memory

    Reference:
    - https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/buffers.py

    """
    def __init__(self, obs_shape, action_shape, capacity):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        # (chongyi zheng): create 0 size buffer instead of max_size buffer to save memory
        # self.obses = np.empty((0, *obs_shape), dtype=obs_dtype)
        # self.next_obses = np.empty((0, *obs_shape), dtype=obs_dtype)
        # self.actions = np.empty((0, *action_shape), dtype=np.float32)
        # self.rewards = np.empty((0, ), dtype=np.float32)
        # self.not_dones = np.empty((0, ), dtype=np.float32)
        # self._storage = []

        self.idx = 0
        self.full = False

    def _sample(self, idxs):
        obses, actions, rewards, next_obses, not_dones = [], [], [], [], []
        for idx in idxs:
            data = self._storage[idx]
            obs, action, reward, next_obs, not_done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_obses.append(np.array(next_obs, copy=False))
            not_dones.append(not_done)

        obses = torch.as_tensor(obses).float().cuda()
        actions = torch.as_tensor(actions).float().cuda()
        rewards = torch.as_tensor(np.expand_dims(rewards, axis=1)).float().cuda()
        next_obses = torch.as_tensor(next_obses).float().cuda()
        not_dones = torch.as_tensor(np.expand_dims(not_dones, axis=1)).float().cuda()

        return obses, actions, rewards, next_obses, not_dones

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        # (chongyi zheng): append transition to buffer when it is not full, replace the last recent one otherwise
        # if not self.full:
        #     self.obses = np.append(
        #         self.obses, np.expand_dims(obs, axis=0).astype(self.obses.dtype), axis=0)
        #     self.actions = np.append(
        #         self.actions, np.expand_dims(action, axis=0).astype(self.actions.dtype), axis=0)
        #     self.rewards = np.append(
        #         self.rewards, np.expand_dims(reward, axis=0).astype(self.rewards.dtype), axis=0)
        #     self.next_obses = np.append(
        #         self.next_obses, np.expand_dims(next_obs, axis=0).astype(self.next_obses.dtype), axis=0)
        #     self.not_dones = np.append(
        #         self.not_dones, np.expand_dims(not done, axis=0).astype(self.not_dones.dtype), axis=0)
        # else:
        #     np.copyto(self.obses[self.idx], obs)
        #     np.copyto(self.actions[self.idx], action)
        #     np.copyto(self.rewards[self.idx], reward)
        #     np.copyto(self.next_obses[self.idx], next_obs)
        #     np.copyto(self.not_dones[self.idx], not done)
        # data = (obs, action, reward, next_obs, not done)
        # if self.full:
        #     self._storage[self.idx] = data
        # else:
        #     self._storage.append(data)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()
        # (chongyi zheng): internal sample function
        # obses, actions, rewards, next_obses, not_dones = self._sample(idxs)

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def sample_curl(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()
        # (chongyi zheng): internal sample function
        # obses, actions, rewards, next_obses, not_dones = self._sample(idxs)

        pos = obses.clone()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)
        pos = random_crop(pos)

        curl_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                           time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, curl_kwargs


class FrameStackReplayBuffer(ReplayBuffer):
    """Only store unique frames to save memory

    """
    def __init__(self, obs_shape, action_shape, capacity, frame_stack):
        super().__init__(obs_shape, action_shape, capacity)
        self.frame_stack = frame_stack

        # (chongyi zheng): We need to set the final not_done = 0.0 to make sure the correct stack when the first
        #   observation is sampled. Note that empty array is initialized to be all zeros.
        # self.not_dones[-1] = 0.0

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs[-1 * self.obs_shape[0]:])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs[-1 * self.obs_shape[0]:])
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        # Reconstruct stacked observations:
        #   If the sampled observation is the first frame of an episode, it is stacked with itself.
        #   Otherwise, we stack it with the previous frames.
        #   The previous not_done indicator must be 0.0 for the first frame of an episode
        not_first_obs = np.squeeze(self.not_dones[(idxs - 1) % self.capacity]).astype(np.bool)
        is_first_obs = np.logical_not(not_first_obs)
        first_obs_idxs = idxs[is_first_obs]
        not_first_obs_idxs = idxs[not_first_obs]
        obses = np.empty([batch_size, self.obs_shape[0] * self.frame_stack] + list(self.obs_shape[1:]),
                         dtype=self.obs_dtype)
        next_obses = np.empty([batch_size, self.obs_shape[0] * self.frame_stack] + list(self.obs_shape[1:]),
                              dtype=self.obs_dtype)
        if len(first_obs_idxs) > 0:  # sanity check
            obses[is_first_obs] = np.concatenate([
                self.obses[first_obs_idxs],
                self.obses[first_obs_idxs],
                self.obses[first_obs_idxs]
            ], axis=1)
            next_obses[is_first_obs] = np.concatenate([
                self.next_obses[first_obs_idxs],
                self.next_obses[first_obs_idxs],
                self.next_obses[first_obs_idxs]
            ], axis=1)
        if len(not_first_obs_idxs) > 0:  # sanity check
            obses[not_first_obs] = np.concatenate([
                self.obses[not_first_obs_idxs - reversed_fs_idx]
                for reversed_fs_idx in reversed(range(self.frame_stack))
            ], axis=1)
            next_obses[not_first_obs] = np.concatenate([
                self.next_obses[not_first_obs_idxs - reversed_fs_idx]
                for reversed_fs_idx in reversed(range(self.frame_stack))
            ], axis=1)

        obses = torch.as_tensor(obses).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        next_obses = torch.as_tensor(next_obses).float().cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)

        return obses, actions, rewards, next_obses, not_dones

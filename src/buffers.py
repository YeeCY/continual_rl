import torch
from torch import nn
import kornia
import numpy as np

from utils import random_crop


class ReplayBuffer(object):
    """Buffer to store environment transitions

    (Chongyi Zheng): update replay buffer to stable_baselines style to save memory

    Reference:
    - https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/buffers.py

    """
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.capacity = capacity

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    # TODO (chongyi zheng): delete this block
    # def _sample(self, idxs):
    #     obses, actions, rewards, next_obses, not_dones = [], [], [], [], []
    #     for idx in idxs:
    #         data = self._storage[idx]
    #         obs, action, reward, next_obs, not_done = data
    #         obses.append(np.array(obs, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append(reward)
    #         next_obses.append(np.array(next_obs, copy=False))
    #         not_dones.append(not_done)
    #
    #     obses = torch.as_tensor(obses).float().cuda()
    #     actions = torch.as_tensor(actions).float().cuda()
    #     rewards = torch.as_tensor(np.expand_dims(rewards, axis=1)).float().cuda()
    #     next_obses = torch.as_tensor(next_obses).float().cuda()
    #     not_dones = torch.as_tensor(np.expand_dims(not_dones, axis=1)).float().cuda()
    #
    #     return obses, actions, rewards, next_obses, not_dones

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        # TODO (chongyi zheng): We don't need to crop image as PAD
        # obses = random_crop(obses)
        # next_obses = random_crop(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def sample_curl(self, batch_size):
        # TODO (chongyi zheng): update this function to drq style
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        # (chongyi zheng): internal sample function
        # obses, actions, rewards, next_obses, not_dones = self._sample(idxs)

        pos = obses.clone()

        # TODO (chongyi zheng): We don't need to crop image as PAD
        # obses = random_crop(obses)
        # next_obses = random_crop(next_obses)
        # pos = random_crop(pos)

        curl_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                           time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, curl_kwargs

    def sample_ensembles(self, batch_size, num_ensembles=1):
        ensem_obses, ensem_actions, ensem_rewards, ensem_next_obses, ensem_not_dones = \
            self.sample(batch_size * num_ensembles)

        # TODO (chongyi zheng): do we need to clone here?
        obses = ensem_obses[:batch_size].detach().clone()
        actions = ensem_actions[:batch_size].detach().clone()
        rewards = ensem_rewards[:batch_size].detach().clone()
        next_obses = ensem_next_obses[:batch_size].detach().clone()
        not_dones = ensem_not_dones[:batch_size].detach().clone()

        ensem_kwargs = dict(obses=ensem_obses, next_obses=ensem_next_obses, actions=ensem_actions)

        return obses, actions, rewards, next_obses, not_dones, ensem_kwargs


class AugmentReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        super().__init__(obs_shape, action_shape, capacity, device)
        self.image_pad = image_pad

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(self.image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx,
            size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones, obses_aug, next_obses_aug

    def sample_curl(self, batch_size):
        # TODO (chongyi zheng)
        pass

    def sample_ensembles(self, batch_size, num_ensembles=1):
        ensem_obses, ensem_actions, ensem_rewards, ensem_next_obses, ensem_not_dones, \
        ensem_obses_aug, ensem_next_obses_aug = self.sample(batch_size * num_ensembles)

        # TODO (chongyi zheng): do we need to clone here?
        obses = ensem_obses[:batch_size].detach().clone()
        next_obses = ensem_next_obses[:batch_size].detach().clone()
        obses_aug = ensem_obses[:batch_size].detach.clone()
        next_obses_aug = ensem_next_obses[:batch_size].detach().clone()
        actions = ensem_actions[:batch_size].detach().clone()
        rewards = ensem_rewards[:batch_size].detach().clone()
        not_dones = ensem_not_dones[:batch_size].detach().clone()

        ensem_kwargs = dict(obses=ensem_obses, next_obses=ensem_next_obses,
                            obses_aug=ensem_obses_aug, next_obses_aug=ensem_next_obses_aug,
                            actions=ensem_actions)

        return obses, actions, rewards, next_obses, not_dones, obses_aug, next_obses_aug, ensem_kwargs


class FrameStackReplayBuffer(ReplayBuffer):
    """Only store unique frames to save memory

    """
    def __init__(self, obs_shape, action_shape, capacity, frame_stack, device):
        super().__init__(obs_shape, action_shape, capacity, device)
        self.frame_stack = frame_stack

        self.stack_frame_dists = np.empty((capacity, self.frame_stack), dtype=np.int32)

        # (chongyi zheng): We need to set the final not_done = 0.0 to make sure the correct stack when the first
        #   observation is sampled. Note that empty array is initialized to be all zeros.
        # self.not_dones[-1] = 0.0

    def add(self, obs, action, reward, next_obs, done, stack_frame_dists=np.empty(0)):
        """
        (chongyi zheng): other_frame_dists are relative indices of the other frame from the current one
        """
        assert len(stack_frame_dists) == self.frame_stack, "Relative indices of stacked frames must be provided!"

        np.copyto(self.obses[self.idx], obs[-1 * self.obs_shape[0]:])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs[-1 * self.obs_shape[0]:])
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.stack_frame_dists[self.idx], stack_frame_dists)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        # Reconstruct stacked observations:
        #   If the sampled observation is the first frame of an episode, it is stacked with itself.
        #   Otherwise, we stack it with the previous frames.
        #   The most recently not_done indicator must be 0.0 for the first 'frame_stack' frames of an episode
        #       first_frame: obs = stack([first_frame, first_frame, first_frame])
        #       second_frame: obs = stack([first_frame, first_frame, second_frame])
        #       third_frame: obs = stack([first_frame, second_frame, third_frame])
        #       forth_frame: obs = stack([second_frame, third_frame, forth_frame])
        #       ...
        obses = []
        next_obses = []
        stack_frame_dists = self.stack_frame_dists[idxs]
        for sf_idx in range(self.frame_stack):
            obses.append(self.obses[stack_frame_dists[:, sf_idx] + idxs])
            next_obses.append(self.next_obses[stack_frame_dists[:, sf_idx] + idxs])
        obses = np.concatenate(obses, axis=1)
        next_obses = np.concatenate(next_obses, axis=1)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        # TODO (chongyi zheng): We don't need to crop image as PAD
        # obses = random_crop(obses)
        # next_obses = random_crop(next_obses)

        return obses, actions, rewards, next_obses, not_dones


class AugmentFrameStackReplayBuffer(FrameStackReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, frame_stack, image_pad, device):
        super().__init__(obs_shape, action_shape, capacity, frame_stack, device)
        self.image_pad = image_pad

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(self.image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

    def add(self, obs, action, reward, next_obs, done, stack_frame_dists=np.empty(0)):
        """
        (chongyi zheng): other_frame_dists are relative indices of the other frame from the current one
        """
        assert len(stack_frame_dists) == self.frame_stack, "Relative indices of stacked frames must be provided!"

        np.copyto(self.obses[self.idx], obs[-1 * self.obs_shape[0]:])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs[-1 * self.obs_shape[0]:])
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.stack_frame_dists[self.idx], stack_frame_dists)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        # Reconstruct stacked observations:
        #   If the sampled observation is the first frame of an episode, it is stacked with itself.
        #   Otherwise, we stack it with the previous frames.
        #   The most recently not_done indicator must be 0.0 for the first 'frame_stack' frames of an episode
        #       first_frame: obs = stack([first_frame, first_frame, first_frame])
        #       second_frame: obs = stack([first_frame, first_frame, second_frame])
        #       third_frame: obs = stack([first_frame, second_frame, third_frame])
        #       forth_frame: obs = stack([second_frame, third_frame, forth_frame])
        #       ...
        obses = []
        next_obses = []
        stack_frame_dists = self.stack_frame_dists[idxs]
        for sf_idx in range(self.frame_stack):
            obses.append(self.obses[stack_frame_dists[:, sf_idx] + idxs])
            next_obses.append(self.next_obses[stack_frame_dists[:, sf_idx] + idxs])
        obses = np.concatenate(obses, axis=1)
        next_obses = np.concatenate(next_obses, axis=1)
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        # TODO (chongyi zheng): We don't need to crop image as PAD
        # obses = random_crop(obses)
        # next_obses = random_crop(next_obses)

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones, obses_aug, next_obses_aug

    def sample_ensembles(self, batch_size, num_ensembles=1):
        # TODO (chongyi zheng)
        pass

import torch
import numpy as np
import os
import random


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


class ReplayBuffer(object):
    """Buffer to store environment transitions

    (Chongyi Zheng): update replay buffer to stable_baselines style to save memory

    Reference:
    - https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/buffers.py

    """

    def __init__(self, capacity):
        self.capacity = capacity

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        # obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        # self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        # self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        # self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        # self.rewards = np.empty((capacity, 1), dtype=np.float32)
        # self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        # (chongyi zheng): create 0 size buffer instead of max_size buffer to save memory
        # self.obses = np.empty((0, *obs_shape), dtype=obs_dtype)
        # self.next_obses = np.empty((0, *obs_shape), dtype=obs_dtype)
        # self.actions = np.empty((0, *action_shape), dtype=np.float32)
        # self.rewards = np.empty((0, ), dtype=np.float32)
        # self.not_dones = np.empty((0, ), dtype=np.float32)
        self._storage = []

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
        # np.copyto(self.obses[self.idx], obs)
        # np.copyto(self.actions[self.idx], action)
        # np.copyto(self.rewards[self.idx], reward)
        # np.copyto(self.next_obses[self.idx], next_obs)
        # np.copyto(self.not_dones[self.idx], not done)
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
        data = (obs, action, reward, next_obs, not done)
        if self.full:
            self._storage[self.idx] = data
        else:
            self._storage.append(data)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        # obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        # actions = torch.as_tensor(self.actions[idxs]).cuda()
        # rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        # next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        # not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()
        # (chongyi zheng): internal sample function
        obses, actions, rewards, next_obses, not_dones = self._sample(idxs)

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def sample_curl(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        # obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        # actions = torch.as_tensor(self.actions[idxs]).cuda()
        # rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        # next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        # not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()
        # (chongyi zheng): internal sample function
        obses, actions, rewards, next_obses, not_dones = self._sample(idxs)

        pos = obses.clone()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)
        pos = random_crop(pos)

        curl_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                           time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, curl_kwargs


def get_curl_pos_neg(obs, replay_buffer):
    """Returns one positive pair + batch of negative samples from buffer"""
    obs = torch.as_tensor(obs).cuda().float().unsqueeze(0)
    pos = obs.clone()

    obs = random_crop(obs)
    pos = random_crop(pos)

    # Sample negatives and insert positive sample
    obs_pos = replay_buffer.sample_curl()[-1]['obs_pos']
    obs_pos[0] = pos

    return obs, obs_pos


def batch_from_obs(obs, batch_size=32):
    """Converts a pixel obs (C,H,W) to a batch (B,C,H,W) of given size"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)


def _rotate_single_with_label(x, label):
    """Rotate an image"""
    if label == 1:
        return x.flip(2).transpose(1, 2)
    elif label == 2:
        return x.flip(2).flip(1)
    elif label == 3:
        return x.transpose(1, 2).flip(2)
    return x


def rotate(x):
    """Randomly rotate a batch of images and return labels"""
    images = []
    labels = torch.randint(4, (x.size(0),), dtype=torch.long).to(x.device)
    for img, label in zip(x, labels):
        img = _rotate_single_with_label(img, label)
        images.append(img.unsqueeze(0))

    return torch.cat(images), labels


def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop"""
    assert isinstance(x, torch.Tensor) and x.is_cuda, \
        'input must be CUDA tensor'

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
        'window_shape must be a tuple with same number of dimensions as x'

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3)
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def random_crop(imgs, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
        'must either specify both w1 and h1 or neither of them'

    is_tensor = isinstance(imgs, torch.Tensor)
    if is_tensor:
        assert imgs.is_cuda, 'input images are tensors but not cuda!'
        return random_crop_cuda(imgs, size=size, w1=w1, h1=h1, return_w1_h1=return_w1_h1)

    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return imgs, None, None
        return imgs

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    if w1 is None:
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)

    windows = view_as_windows(imgs, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[np.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped

"""Loads D4RL dataset from pickle files."""

import d4rl
import gym
import numpy as np
from torch.utils.data import Dataset, DataLoader


class D4RLDataset(Dataset):
    def __init__(self, states, actions, rewards, not_dones, next_states):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.not_dones = not_dones
        self.next_states = next_states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], \
               self.not_dones[idx], self.next_states[idx]


def create_d4rl_env_and_dataset(
    task_name,
    batch_size
):
    """Create gym environment and dataset for d4rl.

    Args:
        task_name: Name of d4rl task.
        batch_size: Mini batch size.

    Returns:
    Gym env and dataset.
    """
    env = gym.make(task_name)
    dataset = d4rl.qlearning_dataset(env)

    states = np.array(dataset['observations'], dtype=np.float32)
    actions = np.array(dataset['actions'], dtype=np.float32)
    rewards = np.array(dataset['rewards'], dtype=np.float32)
    not_dones = np.array(np.logical_not(dataset['terminals']), dtype=np.float32)
    next_states = np.array(dataset['next_observations'], dtype=np.float32)

    # TODO (chongyi zheng): Implement dataloader
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (states, actions, rewards, discounts, next_states)).cache().shuffle(
    #     states.shape[0], reshuffle_each_iteration=True).repeat().batch(
    #     batch_size,
    #     drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = D4RLDataset(states, actions, rewards, not_dones, next_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return env, dataloader

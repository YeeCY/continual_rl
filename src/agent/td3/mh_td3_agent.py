import copy
import numpy as np
import torch

import utils

from agent.td3 import Td3MlpAgent
from agent.network import MultiHeadTd3ActorMlp, MultiHeadTd3CriticMlp


class MultiHeadTd3MlpAgent(Td3MlpAgent):
    def __init__(
            self,
            obs_shape,
            action_shape,
            action_range,
            device,
            actor_hidden_dim=256,
            critic_hidden_dim=256,
            discount=0.99,
            actor_lr=3e-4,
            actor_noise=0.2,
            actor_noise_clip=0.5,
            critic_lr=3e-4,
            expl_noise_std=0.1,
            target_tau=0.005,
            actor_and_target_update_freq=2,
            batch_size=256,
    ):
        assert isinstance(action_shape, list)
        assert isinstance(action_range, list)
        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            actor_lr, actor_noise, actor_noise_clip, critic_lr, expl_noise_std, target_tau,
            actor_and_target_update_freq, batch_size)

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = MultiHeadTd3ActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.action_range
        ).to(self.device)

        self.actor_target = MultiHeadTd3ActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.action_range
        ).to(self.device)

        self.critic = MultiHeadTd3CriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.critic_target = MultiHeadTd3CriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.reset_target()

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # save initial parameters
        self._critic_init_state = copy.deepcopy(self.critic.state_dict())

    def act(self, obs, add_noise=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        with torch.no_grad():
            action = utils.to_np(self.actor(obs, **kwargs))

        low, high = np.array(self.action_range[kwargs['head_idx']][0]), \
                    np.array(self.action_range[kwargs['head_idx']][1])
        if add_noise:
            assert np.alltrue(high == -low), "Action range must be symmetric!"
            noise = np.random.normal(0, high * self.expl_noise_std)
            action += noise
        assert 'head_idx' in kwargs
        action = action.clip(low, high)
        assert action.ndim == 2 and action.shape[0] == 1

        return action

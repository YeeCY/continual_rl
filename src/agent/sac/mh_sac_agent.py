import numpy as np
import torch

import utils

from agent.sac.base_sac_agent import SacMlpAgent
from agent.network import MultiHeadSacActorMlp, MultiHeadSacCriticMlp


class MultiHeadSacMlpAgent(SacMlpAgent):
    def __init__(
            self,
            obs_shape,
            action_shape,
            action_range,
            device,
            hidden_dim=400,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            actor_lr=1e-3,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_tau=0.005,
            critic_target_update_freq=2,
            grad_clip_norm=10.0,
            batch_size=128,
    ):
        assert isinstance(action_shape, list)
        assert isinstance(action_range, list)
        super().__init__(
            obs_shape, action_shape, action_range, device, hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
            actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_tau, critic_target_update_freq,
            grad_clip_norm, batch_size)

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = MultiHeadSacActorMlp(
            self.obs_shape, self.action_shape, self.hidden_dim,
            self.actor_log_std_min, self.actor_log_std_max
        ).to(self.device)

        self.critic = MultiHeadSacCriticMlp(
            self.obs_shape, self.action_shape, self.hidden_dim
        ).to(self.device)

        self.critic_target = MultiHeadSacCriticMlp(
            self.obs_shape, self.action_shape, self.hidden_dim
        ).to(self.device)

        self.reset_target_critic()

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_shape)

        # sac optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def act(self, obs, sample=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        with torch.no_grad():
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False, **kwargs)
            action = pi if sample else mu
            assert 'head_idx' in kwargs
            action = action.clamp(*self.action_range[kwargs['head_idx']])
            assert action.ndim == 2 and action.shape[0] == 1

        return utils.to_np(action)

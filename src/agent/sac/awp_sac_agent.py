import math
import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.nets.awp import AdvWeightPerturbSacActorMlp
from agent.network import SacCriticMlp
from agent.sac.base_sac_agent import SacMlpAgent


class AdversarialWeightPermutationSacMlpAgent(SacMlpAgent):
    def __init__(
            self,
            obs_shape,
            action_shape,
            action_range,
            device,
            actor_hidden_dim=400,
            critic_hidden_dim=256,
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
            batch_size=128,
            awp_coeff=0.01,
    ):
        self.awp_coeff = awp_coeff

        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size)

    def _setup_agent(self):
        self.actor = AdvWeightPerturbSacActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.actor_log_std_min, self.actor_log_std_max, self.awp_coeff,
        ).to(self.device)

        self.critic = SacCriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.critic_target = SacCriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.reset_target_critic()

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_shape[0])

        # sac optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.main_parameters(), lr=self.actor_lr)
        self.actor_proxy_optimizer = torch.optim.Adam(self.actor.proxy_parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # save initial parameters
        self._critic_init_state = copy.deepcopy(self.critic.state_dict())

    def reset(self, **kwargs):
        self.reset_target_critic()
        self.reset_log_alpha()

        self.actor_optimizer = torch.optim.Adam(
            self.actor.main_parameters(), lr=self.actor_lr,
        )
        self.actor_proxy_optimizer = torch.optim.Adam(
            self.actor.proxy_parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def act(self, obs, sample=False, perturb=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        if perturb:
            diff_weights = self.compute_actor_diff_weights(obs)
            self.actor.perturb(diff_weights)

        with torch.no_grad():
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False, **kwargs)
            action = pi if sample else mu
            action = action.clamp(*self.action_range)
            assert action.ndim == 2 and action.shape[0] == obs.shape[0]

        if perturb:
            self.actor.restore(diff_weights)

        return utils.to_np(action)

    def compute_actor_diff_weights(self, obs, **kwargs):
        self.actor.reload_weights()
        _, pi, log_pi, log_std = self.actor(obs, proxy=True, **kwargs)
        actor_Q1, actor_Q2 = self.critic(obs, pi, **kwargs)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # optimize the proxy actor
        self.actor_proxy_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_proxy_optimizer.step()

        return self.actor.diff_weights()

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done, **kwargs)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_update_freq == 0:
            diff_weights = self.compute_actor_diff_weights(obs, **kwargs)
            self.actor.perturb(diff_weights)
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(obs, **kwargs)
            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)


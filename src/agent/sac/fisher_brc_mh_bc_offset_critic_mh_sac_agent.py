import copy
import numpy as np
import torch
import torch.nn.functional as F

from agent.sac.behavioral_cloning import BehavioralCloning
from agent.network import MultiHeadSacActorMlp, MultiHeadSacOffsetCriticMlp
from agent.sac import FisherBRCMHBCMlpCriticMultiHeadSacMlpAgent


class FisherBRCMHBCOffsetCriticMultiHeadSacMlpAgent(FisherBRCMHBCMlpCriticMultiHeadSacMlpAgent):
    """multi-task behavioral cloning policy + single modal offset representation for critic"""
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
            behavior_cloning_hidden_dim=256,
            memory_budget=10000,
            fisher_coeff=1.0,
            reward_bonus=5.0,
    ):
        assert isinstance(action_shape, list)
        assert isinstance(action_range, list)
        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size, behavior_cloning_hidden_dim, memory_budget,
            fisher_coeff, reward_bonus)

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = MultiHeadSacActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.actor_log_std_min, self.actor_log_std_max
        ).to(self.device)

        self.critic = MultiHeadSacOffsetCriticMlp(
            self.behavior_cloning, self.obs_shape, self.action_shape,
            self.critic_hidden_dim
        ).to(self.device)
        self.critic_target = MultiHeadSacOffsetCriticMlp(
            self.behavior_cloning, self.obs_shape, self.action_shape,
            self.critic_hidden_dim
        ).to(self.device)

        self.reset_target_critic()

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_shape[0])

        self.behavior_cloning = BehavioralCloning(
            self.obs_shape, self.action_shape, self.device, self.behavior_cloning_hidden_dim,
            multi_head=False, log_std_min=self.actor_log_std_min, log_std_max=self.actor_log_std_max
        )

        # sac optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # save initial parameters
        self._critic_init_state = copy.deepcopy(self.critic.state_dict())

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done, **kwargs):
        assert 'prev_obs' in kwargs, "We should use observations of previous tasks " \
                                     "to compute critic fisher regularization"
        prev_obses = kwargs['prev_obs']
        assert isinstance(prev_obses, list)

        with torch.no_grad():
            _, next_policy_action, next_log_pi, _ = self.actor(next_obs, **kwargs)
            _, _, target_Q1, target_Q2 = self.critic_target(
                next_obs, next_policy_action, **kwargs)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * next_log_pi
            target_Q = reward + (not_done * self.discount * target_V)
        _, _, current_Q1, current_Q2 = self.critic(
            obs, action, detach_behavioral_cloner=True)

        # regularize with observations of previous tasks
        if prev_obses is not None:
            o_reg = []
            for task_id, prev_obs in enumerate(prev_obses):
                _, policy_action, _, _ = self.actor(prev_obs, **kwargs)
                o1, o2, _, _ = self.critic(prev_obs, policy_action)
                # (cyzheng): create graph for second order derivatives
                o1_grads = torch.autograd.grad(
                    o1.sum(), policy_action, create_graph=True)[0]
                o2_grads = torch.autograd.grad(
                    o2.sum(), policy_action, create_graph=True)[0]
                o1_grad_norm = torch.sum(torch.square(o1_grads), dim=-1)
                o2_grad_norm = torch.sum(torch.square(o2_grads), dim=-1)
                o_reg.append(torch.mean(o1_grad_norm + o2_grad_norm))
            o_reg = torch.mean(torch.stack(o_reg))
        else:
            o_reg = torch.zeros(device=self.device)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + \
                      self.fisher_coeff * o_reg

        return critic_loss

    def compute_actor_and_alpha_loss(self, obs, compute_alpha_loss=True, **kwargs):
        _, pi, log_pi, log_std = self.actor(obs, **kwargs)
        _, _, actor_Q1, actor_Q2 = self.critic(obs, pi, **kwargs)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        alpha_loss = None
        if compute_alpha_loss:
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        return log_pi, actor_loss, alpha_loss

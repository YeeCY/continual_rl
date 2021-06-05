import torch
from itertools import chain
from collections.abc import Iterable

from agent.sac.base_sac_agent import SacMlpAgent


class SiSacMlpAgent(SacMlpAgent):
    """Adapt from https://github.com/GMvandeVen/continual-learning"""
    def __init__(self,
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
                 batch_size=128,
                 si_c=1.0,
                 si_epsilon=0.1,
                 ):
        super().__init__(obs_shape, action_shape, action_range, device, hidden_dim, discount, init_temperature,
                         alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                         critic_tau, critic_target_update_freq, batch_size)

        self.si_c = si_c
        self.si_epsilon = si_epsilon

        self.params_w = {}
        self.omegas = {}
        self.prev_task_params = {}

        # set prev_params and prev_task_params as weight initializations
        self.prev_params = {}
        self.prev_task_params = {}
        for name, param in chain(self.critic.named_parameters(),
                                 self.actor.named_parameters(),
                                 iter([('log_alpha', self.log_alpha)])):
            if param.requires_grad:
                self.prev_params[name] = param.detach().clone()
                self.prev_task_params[name] = param.detach().clone()

    def update_omegas(self):
        for name, param in chain(self.critic.named_parameters(),
                                 self.actor.named_parameters(),
                                 iter([('log_alpha', self.log_alpha)])):
            if param.requires_grad:
                prev_param = self.prev_task_params[name]
                current_param = param.detach().clone()
                delta_param = current_param - prev_param
                current_omega = self.params_w[name] / (delta_param ** 2 + self.si_epsilon)

                self.prev_task_params[name] = current_param
                self.omegas[name] = current_omega + self.omegas.get(name, torch.zeros_like(param))

        # clear importance buffers for the next task
        self.params_w = {}

    def _estimate_importance(self):
        for name, param in chain(self.critic.named_parameters(),
                                 self.actor.named_parameters(),
                                 iter([('log_alpha', self.log_alpha)])):
            if param.requires_grad:
                self.params_w[name] = -param.grad * (param.detach() - self.prev_params[name]) + \
                                      self.params_w.get(name, torch.zeros_like(param))
                self.prev_params[name] = param.detach().clone()

    def _compute_surrogate_loss(self, named_parameters):
        assert isinstance(named_parameters, Iterable), "'named_parameters' must be a iterator"

        si_losses = []
        for name, param in named_parameters:
            if param.requires_grad:
                prev_param = self.prev_task_params[name]
                omega = self.omegas.get(name, torch.zeros_like(param))
                si_loss = torch.sum(omega * (param - prev_param) ** 2)
                si_losses.append(si_loss)

        return torch.sum(torch.stack(si_losses))

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done):
        critic_loss = super().compute_critic_loss(obs, action, reward, next_obs, not_done)

        # critic si surrogate loss
        critic_surrogate_loss = self._compute_surrogate_loss(self.critic.named_parameters())

        return critic_loss + self.si_c * critic_surrogate_loss

    def compute_actor_and_alpha_loss(self, obs, compute_alpha_loss=True):
        log_pi, actor_loss, alpha_loss = super().compute_actor_and_alpha_loss(obs, compute_alpha_loss)

        # actor and alpha si surrogate losses
        actor_surrogate_loss = self._compute_surrogate_loss(self.actor.named_parameters())
        alpha_surrogate_loss = self._compute_surrogate_loss(iter([('log_alpha', self.log_alpha)]))

        return log_pi, actor_loss + self.si_c * actor_surrogate_loss, alpha_loss + self.si_c * alpha_surrogate_loss

    def update(self, replay_buffer, logger, step):
        super().update(replay_buffer, logger, step)

        # estimate weight importance
        self._estimate_importance()

    def save(self, model_dir, step):
        super().save(model_dir, step)
        torch.save(
            self.prev_params, '%s/prev_params_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.params_w, '%s/params_w_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.omegas, '%s/omegas_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.prev_task_params, '%s/prev_task_params_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        super().load(model_dir, step)
        self.prev_params = torch.load(
            '%s/prev_params_%s.pt' % (model_dir, step)
        )
        self.params_w = torch.load(
            '%s/params_w_%s.pt' % (model_dir, step)
        )
        self.omegas = torch.load(
            '%s/omegas_%s.pt' % (model_dir, step)
        )

        self.prev_task_params = torch.load(
            '%s/prev_task_params_%s.pt' % (model_dir, step)
        )

import torch
from collections.abc import Iterable

import utils
from agent.td3 import Td3MlpAgent


class SiTd3MlpAgent(Td3MlpAgent):
    """Adapt from https://github.com/GMvandeVen/continual-learning"""
    def __init__(self,
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
                 si_c=1.0,
                 si_epsilon=0.1,
                 ):
        super().__init__(obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim,
                         discount, actor_lr, actor_noise, actor_noise_clip, critic_lr, expl_noise_std, target_tau,
                         actor_and_target_update_freq, batch_size)

        self.si_c = si_c
        self.si_epsilon = si_epsilon

        self.params_w = {}
        self.omegas = {}
        self.prev_params = {}
        self.prev_task_params = {}

        self._save_init_params()

    def _save_init_params(self):
        # set prev_task_params as weight initializations
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                self.prev_task_params[name] = param.detach().clone()
                self.prev_params[name] = param.detach().clone()

    def update_omegas(self):
        for name, param in self.actor.named_parameters():
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
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                self.params_w[name] = \
                    -param.grad.detach() * (param.detach() - self.prev_params[name]) + \
                    self.params_w.get(name, torch.zeros_like(param))
                self.prev_params[name] = param.detach().clone()

    def _compute_surrogate_loss(self, named_parameters):
        assert isinstance(named_parameters, list), "'named_parameters' must be a list"

        si_losses = []
        for name, param in named_parameters:
            if param.requires_grad:
                prev_param = self.prev_task_params[name]
                omega = self.omegas.get(name, torch.zeros_like(param))
                si_loss = torch.sum(omega * (param - prev_param) ** 2)
                si_losses.append(si_loss)

        return torch.sum(torch.stack(si_losses))

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done, **kwargs)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_and_target_update_freq == 0:
            actor_loss = self.compute_actor_loss(obs, **kwargs)
            actor_si_surrogate_loss = self._compute_surrogate_loss(list(self.actor.named_parameters()))
            actor_loss = actor_loss + self.si_c * actor_si_surrogate_loss
            self.update_actor(actor_loss, logger, step)

            utils.soft_update_params(self.actor, self.actor_target, self.target_tau)
            utils.soft_update_params(self.critic, self.critic_target, self.target_tau)

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

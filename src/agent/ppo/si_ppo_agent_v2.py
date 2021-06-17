import torch
from itertools import chain
from collections.abc import Iterable

from agent.ppo.base_ppo_agent import PpoMlpAgent


class SiPpoMlpAgentV2(PpoMlpAgent):
    """Adapt from https://github.com/GMvandeVen/continual-learning"""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 hidden_dim=64,
                 discount=0.99,
                 clip_param=0.2,
                 ppo_epoch=10,
                 critic_loss_coef=0.5,
                 entropy_coef=0.0,
                 lr=3e-4,
                 eps=1e-5,
                 grad_clip_norm=0.5,
                 use_clipped_critic_loss=True,
                 num_batch=32,
                 si_c=1.0,
                 si_epsilon=0.1,
                 ):
        super().__init__(obs_shape, action_shape, device, hidden_dim, discount, clip_param, ppo_epoch,
                         critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm, use_clipped_critic_loss,
                         num_batch)

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
        assert isinstance(named_parameters, Iterable)

        si_losses = []
        for name, param in named_parameters:
            if param.requires_grad:
                prev_param = self.prev_task_params[name]
                omega = self.omegas.get(name, torch.zeros_like(param))
                si_loss = torch.sum(omega * (param - prev_param) ** 2)
                si_losses.append(si_loss)

        return torch.sum(torch.stack(si_losses))

    def update(self, rollouts, logger, step, **kwargs):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        logger.log('train/batch_normalized_advantages', advantages.mean(), step)

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, \
                return_batch, old_log_pis, adv_targets = sample

                # Reshape to do in a single forward pass for all steps
                actor_loss, entropy = self.compute_actor_loss(
                    obs_batch, actions_batch, old_log_pis, adv_targets, **kwargs)
                critic_loss = self.compute_critic_loss(
                    obs_batch, value_preds_batch, return_batch, **kwargs)
                si_surrogate_loss = self._compute_surrogate_loss(
                    self.actor.named_parameters()
                )
                loss = actor_loss + self.critic_loss_coef * critic_loss - \
                       self.entropy_coef * entropy + self.si_c * si_surrogate_loss

                logger.log('train_actor/loss', actor_loss, step)
                logger.log('train_actor/entropy', entropy, step)
                logger.log('train_critic/loss', critic_loss, step)
                logger.log('train/loss', loss, step)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    chain(self.actor.parameters(), self.critic.parameters()),
                    self.grad_clip_norm)
                self.optimizer.step()

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

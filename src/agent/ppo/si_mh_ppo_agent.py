import torch
from itertools import chain
from collections.abc import Iterable

from src.agent.ppo import MultiHeadPpoMlpAgent, SiPpoMlpAgent


class SiMultiHeadPpoMlpAgent(MultiHeadPpoMlpAgent, SiPpoMlpAgent):
    """Adapt https://github.com/GMvandeVen/continual-learning"""
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
                 si_epsilon=0.1
                 ):
        MultiHeadPpoMlpAgent.__init__(self, obs_shape, action_shape, device, hidden_dim, discount, clip_param,
                                      ppo_epoch, critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm,
                                      use_clipped_critic_loss, num_batch)

        SiPpoMlpAgent.__init__(self, obs_shape, action_shape, device, hidden_dim, discount, clip_param,
                               ppo_epoch, critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm,
                               use_clipped_critic_loss, num_batch, si_c, si_epsilon)

    def _save_init_params(self):
        # set prev_task_params as weight initializations
        for name, param in chain(self.actor.named_common_parameters(),
                                 self.critic.named_common_parameters()):
            if param.requires_grad:
                self.prev_task_params[name] = param.detach().clone()

    def update_omegas(self):
        for name, param in chain(self.actor.named_common_parameters(),
                                 self.critic.named_common_parameters()):
            if param.requires_grad:
                prev_param = self.prev_task_params[name]
                current_param = param.detach().clone()
                delta_param = current_param - prev_param
                current_omega = self.params_w[name] / (delta_param ** 2 + self.si_epsilon)

                self.prev_task_params[name] = current_param
                self.omegas[name] = current_omega + self.omegas.get(name, torch.zeros_like(param))

        # clear importance buffers for the next task
        self.params_w = {}

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
                # values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                #     obs_batch, recurrent_hidden_states_batch, masks_batch,
                #     actions_batch)
                actor_loss, entropy = self.compute_actor_loss(
                    obs_batch, actions_batch, old_log_pis, adv_targets, **kwargs)
                critic_loss = self.compute_critic_loss(
                    obs_batch, value_preds_batch, return_batch, **kwargs)
                si_surrogate_loss = self._compute_surrogate_loss(
                    chain(self.actor.named_common_parameters(),
                          self.critic.named_common_parameters())
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
                self._estimate_importance(
                    chain(self.actor.named_common_parameters(),
                          self.critic.named_common_parameters())
                )

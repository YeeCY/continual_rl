import torch
from itertools import chain
from collections.abc import Iterable

import utils
from agent.ppo.base_ppo_agent import PpoMlpAgent


class EwcPpoMlpAgent(PpoMlpAgent):
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
                 batch_size=32,
                 ewc_lambda=5000,
                 ewc_fisher_sample_size=100,
                 online_ewc=False,
                 online_ewc_gamma=1.0,
                 ):
        super().__init__(obs_shape, action_shape, device, hidden_dim, discount, clip_param, ppo_epoch,
                         critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm, use_clipped_critic_loss,
                         batch_size)

        self.ewc_lambda = ewc_lambda
        self.ewc_fisher_sample_size = ewc_fisher_sample_size
        self.online_ewc = online_ewc
        self.online_ewc_gamma = online_ewc_gamma

        self.ewc_task_count = 0
        self.prev_task_params = {}
        self.prev_task_fishers = {}

    def estimate_fisher(self, replay_buffer):
        with utils.eval_mode(self):
            obs, action, reward, next_obs, not_done = replay_buffer.sample(self.ewc_fisher_sample_size)

            critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            _, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(obs)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()

            for name, param in chain(self.critic.named_parameters(),
                                     self.actor.named_parameters(),
                                     iter([('log_alpha', self.log_alpha)])):
                if param.grad is not None:
                    if self.online_ewc:
                        name = name + '_prev_task'
                        self.prev_task_params[name] = param.detach().clone()
                        self.prev_task_fishers[name] = \
                            param.grad.detach().clone() ** 2 + \
                            self.online_ewc_gamma * self.prev_task_fishers.get(name, torch.zeros_like(param.grad))
                    else:
                        name = name + f'_prev_task{self.ewc_task_count}'
                        self.prev_task_params[name] = param.detach().clone()
                        self.prev_task_fishers[name] = param.grad.detach().clone() ** 2

            self.ewc_task_count += 1

    def compute_ewc_loss(self):
        ewc_losses = []
        # if self.ewc_task_count >= 1:
        if self.online_ewc:
            for name, param in chain(self.actor.named_parameters(),
                                     self.critic.parameters()):
                if param.grad is not None:
                    name = name + '_prev_task'
                    mean = self.prev_task_params.get(name, torch.zeros_like(param))
                    # apply decay-term to the running sum of the Fisher Information matrices
                    fisher = self.online_ewc_gamma * \
                             self.prev_task_fishers.get(name, torch.zeros_like(param))
                    ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                    ewc_losses.append(ewc_loss)
        else:
            for task in range(self.ewc_task_count):
                # compute ewc loss for each parameter
                for name, param in chain(self.actor.named_parameters(),
                                         self.critic.parameters()):
                    if param.grad is not None:
                        name = name + f'_prev_task{task}'
                        mean = self.prev_task_params.get(name, torch.zeros_like(param))
                        fisher = self.prev_task_fishers.get(name, torch.zeros_like(param))
                        ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                        ewc_losses.append(ewc_loss)
        return torch.sum(torch.stack(ewc_losses)) / 2.0
        # else:
        #     _, param = next(named_parameters)
        #     return torch.tensor(0.0, device=param.device)

    # def compute_critic_loss(self, obs, value_pred, ret):
    #     critic_loss = super().compute_critic_loss(obs, action, reward, next_obs, not_done)
    #
    #     # critic ewc loss
    #     critic_ewc_loss = self._compute_ewc_loss(self.critic.named_parameters())
    #
    #     return critic_loss + self.ewc_lambda * critic_ewc_loss
    #
    # def compute_actor_and_alpha_loss(self, obs, compute_alpha_loss=True):
    #     log_pi, actor_loss, alpha_loss = super().compute_actor_and_alpha_loss(obs, compute_alpha_loss)
    #
    #     # actor and alpha ewc loss
    #     actor_ewc_loss = self._compute_ewc_loss(self.actor.named_parameters())
    #     alpha_ewc_loss = self._compute_ewc_loss(iter([('log_alpha', self.log_alpha)]))
    #
    #     return log_pi, actor_loss + self.ewc_lambda * actor_ewc_loss, alpha_loss + self.ewc_lambda * alpha_ewc_loss

    def update(self, rollouts, logger, step):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        logger.log('train/batch_normalized_advantages', advantages.mean(), step)

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.batch_size)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, \
                return_batch, old_log_pis, adv_targets = sample

                # Reshape to do in a single forward pass for all steps
                actor_loss, entropy = self.compute_actor_loss(
                    obs_batch, actions_batch, old_log_pis, adv_targets)
                critic_loss = self.compute_critic_loss(
                    obs_batch, value_preds_batch, return_batch)
                ewc_loss = self.compute_ewc_loss()
                loss = actor_loss + self.critic_loss_coef * critic_loss - \
                       self.entropy_coef * entropy + self.ewc_lambda * ewc_loss

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

    def save(self, model_dir, step):
        super().save(model_dir, step)
        torch.save(
            self.prev_task_params, '%s/prev_task_params_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        super().load(model_dir, step)
        self.prev_task_params = torch.load(
            '%s/prev_task_params_%s.pt' % (model_dir, step)
        )

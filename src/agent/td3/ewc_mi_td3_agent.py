import torch

import utils
from agent.td3 import MultiInputTd3MlpAgent, EwcTd3MlpAgent


class EwcMultiInputTd3MlpAgent(MultiInputTd3MlpAgent, EwcTd3MlpAgent):
    """Adapt https://github.com/GMvandeVen/continual-learning"""
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
                 ewc_lambda=5000,
                 ewc_estimate_fisher_iters=100,
                 ewc_estimate_fisher_batch_size=1024,
                 online_ewc=False,
                 online_ewc_gamma=1.0,
                 ):
        MultiInputTd3MlpAgent.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                       critic_hidden_dim, discount, actor_lr, actor_noise, actor_noise_clip,
                                       critic_lr, expl_noise_std, target_tau, actor_and_target_update_freq,
                                       batch_size)

        EwcTd3MlpAgent.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                critic_hidden_dim, discount, actor_lr, actor_noise, actor_noise_clip, critic_lr,
                                expl_noise_std, target_tau, actor_and_target_update_freq, batch_size, ewc_lambda,
                                ewc_estimate_fisher_iters, ewc_estimate_fisher_batch_size, online_ewc,
                                online_ewc_gamma)

    def estimate_fisher(self, replay_buffer, **kwargs):
        fishers = {}
        for _ in range(self.ewc_estimate_fisher_iters):
            obs, action, reward, next_obs, not_done = replay_buffer.sample(
                self.ewc_estimate_fisher_batch_size)

            actor_loss = self.compute_actor_loss(obs, **kwargs)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            for name, param in self.actor.named_common_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        fishers[name] = param.grad.detach().clone() ** 2 + \
                                        fishers.get(name, torch.zeros_like(param.grad))
                    else:
                        fishers[name] = torch.zeros_like(param)

        for name, param in self.actor.named_common_parameters():
            if param.requires_grad:
                fisher = fishers[name]

                if self.online_ewc:
                    name = name + '_prev_task'
                    self.prev_task_params[name] = param.detach().clone()
                    self.prev_task_fishers[name] = \
                        fisher / self.ewc_estimate_fisher_iters + \
                        self.online_ewc_gamma * self.prev_task_fishers.get(
                            name, torch.zeros_like(param.grad))
                else:
                    name = name + f'_prev_task{self.ewc_task_count}'
                    self.prev_task_params[name] = param.detach().clone()
                    self.prev_task_fishers[name] = \
                        fisher / self.ewc_estimate_fisher_iters

        self.ewc_task_count += 1

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done, **kwargs)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_and_target_update_freq == 0:
            actor_loss = self.compute_actor_loss(obs, **kwargs)
            actor_ewc_loss = self._compute_ewc_loss(list(self.actor.named_common_parameters()))
            actor_loss = actor_loss + self.ewc_lambda * actor_ewc_loss
            self.update_actor(actor_loss, logger, step)

            utils.soft_update_params(self.actor, self.actor_target, self.target_tau)
            utils.soft_update_params(self.critic, self.critic_target, self.target_tau)

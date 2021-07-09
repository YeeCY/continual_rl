import torch

import utils
from agent.sac import MultiHeadSacMlpAgentV2, EwcV2SacMlpAgentV2


class EwcV2MultiHeadSacMlpAgentV2(MultiHeadSacMlpAgentV2, EwcV2SacMlpAgentV2):
    """Adapt https://github.com/GMvandeVen/continual-learning"""
    def __init__(self,
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
                 ewc_lambda=5000,
                 ewc_estimate_fisher_iters=100,
                 ewc_estimate_fisher_rollout_steps=1024,
                 online_ewc=False,
                 online_ewc_gamma=1.0,
                 ):
        MultiHeadSacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                        critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                        actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_tau,
                                        critic_target_update_freq, batch_size)

        EwcV2SacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                    critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                    actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_tau,
                                    critic_target_update_freq, batch_size, ewc_lambda, ewc_estimate_fisher_iters,
                                    ewc_estimate_fisher_rollout_steps, online_ewc, online_ewc_gamma)

    def estimate_fisher(self, env, **kwargs):
        fishers = {}
        obs = env.reset()
        for _ in range(self.ewc_estimate_fisher_iters):
            rollouts = {
                'obs': [],
                'action': [],
                'next_obs': [],
                'done': [],
            }
            for _ in range(self.ewc_estimate_fisher_rollout_steps):
                with utils.eval_mode(self):
                    action = self.act(obs, sample=True, **kwargs)

                next_obs, reward, done, info = env.step(action)

                rollouts['obs'].append(obs)
                rollouts['action'].append(action)
                rollouts['next_obs'].append(next_obs)
                rollouts['done'].append(done)

                obs = next_obs

            _, actor_loss, _ = self.compute_actor_and_alpha_loss(
                torch.as_tensor(rollouts['obs'], device=self.device),
                compute_alpha_loss=False, **kwargs
            )
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
        # TODO (chongyi zheng): delete this block
        # critic_ewc_loss = self._compute_ewc_loss(self.critic.named_common_parameters())
        # critic_loss = critic_loss + self.ewc_lambda * critic_ewc_loss
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_update_freq == 0:
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(obs, **kwargs)
            actor_ewc_loss = self._compute_ewc_loss(self.actor.named_common_parameters())
            actor_loss = actor_loss + self.ewc_lambda * actor_ewc_loss
            # TODO (chongyi zheng): delete this block
            # alpha_ewc_loss = self._compute_ewc_loss(iter([('log_alpha', self.log_alpha)]))
            # alpha_loss = alpha_loss + self.ewc_lambda * alpha_ewc_loss

            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

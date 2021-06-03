import torch
from itertools import chain
from collections.abc import Iterable

import utils
from agent.sac.base_sac_agent import SacMlpAgent


class EwcSacMlpAgent(SacMlpAgent):
    """Adapt https://github.com/GMvandeVen/continual-learning"""
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
                 grad_clip_norm=10.0,
                 batch_size=128,
                 ewc_lambda=5000,
                 ewc_fisher_sample_size=100,
                 online_ewc=False,
                 online_ewc_gamma=1.0,
                 ):
        super().__init__(obs_shape, action_shape, action_range, device, hidden_dim, discount, init_temperature,
                         alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                         critic_tau, critic_target_update_freq, grad_clip_norm, batch_size)

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

    def _compute_ewc_loss(self, named_parameters):
        assert isinstance(named_parameters, Iterable), "'named_parameters' must be a iterator"

        ewc_losses = []
        if self.ewc_task_count >= 1:
            if self.online_ewc:
                for name, param in named_parameters:
                    if param.grad is not None:
                        name = name + '_prev_task'
                        mean = self.prev_task_params[name]
                        # apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.online_ewc_gamma * self.prev_task_fishers[name]
                        ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                        ewc_losses.append(ewc_loss)
            else:
                for task in range(self.ewc_task_count):
                    # compute ewc loss for each parameter
                    for name, param in named_parameters:
                        if param.grad is not None:
                            name = name + f'_prev_task{task}'
                            mean = self.prev_task_params[name]
                            fisher = self.prev_task_fishers[name]
                            ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                            ewc_losses.append(ewc_loss)
            return torch.sum(torch.stack(ewc_losses)) / 2.0
        else:
            _, param = next(named_parameters)
            return torch.tensor(0.0, device=param.device)

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done):
        critic_loss = super().compute_critic_loss(obs, action, reward, next_obs, not_done)

        # critic ewc loss
        critic_ewc_loss = self._compute_ewc_loss(self.critic.named_parameters())

        return critic_loss + self.ewc_lambda * critic_ewc_loss

    def compute_actor_and_alpha_loss(self, obs, compute_alpha_loss=True):
        log_pi, actor_loss, alpha_loss = super().compute_actor_and_alpha_loss(obs, compute_alpha_loss)

        # actor and alpha ewc loss
        actor_ewc_loss = self._compute_ewc_loss(self.actor.named_parameters())
        alpha_ewc_loss = self._compute_ewc_loss(iter([('log_alpha', self.log_alpha)]))

        return log_pi, actor_loss + self.ewc_lambda * actor_ewc_loss, alpha_loss + self.ewc_lambda * alpha_ewc_loss

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

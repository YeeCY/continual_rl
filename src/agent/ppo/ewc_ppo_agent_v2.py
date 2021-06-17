import torch
import numpy as np
from itertools import chain

import utils
from src.agent.ppo.base_ppo_agent import PpoMlpAgent


class EwcPpoMlpAgentV2(PpoMlpAgent):
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
                 ewc_lambda=5000,
                 ewc_estimate_fisher_epochs=100,
                 online_ewc=False,
                 online_ewc_gamma=1.0,
                 ):
        super().__init__(obs_shape, action_shape, device, hidden_dim, discount, clip_param, ppo_epoch,
                         critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm, use_clipped_critic_loss,
                         num_batch)

        self.ewc_lambda = ewc_lambda
        self.ewc_estimate_fisher_epochs = ewc_estimate_fisher_epochs
        self.online_ewc = online_ewc
        self.online_ewc_gamma = online_ewc_gamma

        self.ewc_task_count = 0
        self.prev_task_params = {}
        self.prev_task_fishers = {}

    def estimate_fisher(self, env, rollouts, compute_returns_kwargs, **kwargs):
        fishers = {}
        obs = env.reset()
        rollouts.obs[0].copy_(torch.Tensor(obs).to(self.device))
        for epoch in range(self.ewc_estimate_fisher_epochs):
            for step in range(rollouts.num_steps):
                with utils.eval_mode(self):
                    action, log_pi = self.act(obs, sample=True, compute_log_pi=True, **kwargs)
                    value = self.predict_value(obs, **kwargs)

                obs, reward, done, infos = env.step(action)

                # If done then clean the history of observations.
                masks = np.array(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = np.array(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, action, log_pi, value, reward, masks, bad_masks)

            next_value = self.predict_value(rollouts.obs[-1], **kwargs)
            rollouts.compute_returns(next_value, **compute_returns_kwargs)

            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, \
                return_batch, old_log_pis, adv_targets = sample

                # Reshape to do in a single forward pass for all steps
                actor_loss, entropy = self.compute_actor_loss(
                    obs_batch, actions_batch, old_log_pis, adv_targets, **kwargs)

                # TODO (chongyi zheng): delete this block
                # critic_loss = self.compute_critic_loss(
                #     obs_batch, value_preds_batch, return_batch, **kwargs)
                # loss = actor_loss + self.critic_loss_coef * critic_loss - \
                #        self.entropy_coef * entropy

                self.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.grad_clip_norm
                )

                for name, param in self.actor.named_parameters():
                    if param.requires_grad:
                        # TODO (chongyi zheng): Average this accumulated fishers over num_batch?
                        if param.grad is not None:
                            fishers[name] = param.grad.detach().clone() ** 2 + \
                                            fishers.get(name, torch.zeros_like(param.grad))
                        else:
                            fishers[name] = torch.zeros_like(param)

            rollouts.after_update()

        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                fisher = fishers[name]

                if self.online_ewc:
                    name = name + '_prev_task'
                    self.prev_task_params[name] = param.detach().clone()
                    # (chongyi zheng): Only average over epochs now
                    self.prev_task_fishers[name] = \
                        fisher / self.ewc_estimate_fisher_epochs + \
                        self.online_ewc_gamma * self.prev_task_fishers.get(
                            name, torch.zeros_like(param.grad))
                else:
                    name = name + f'_prev_task{self.ewc_task_count}'
                    self.prev_task_params[name] = param.detach().clone()
                    # (chongyi zheng): Only average over epochs now
                    self.prev_task_fishers[name] = \
                        fisher / self.ewc_estimate_fisher_epochs

        self.ewc_task_count += 1

    def compute_ewc_loss(self):
        ewc_losses = []
        if self.ewc_task_count >= 1:
            if self.online_ewc:
                for name, param in self.actor.named_parameters():
                    if param.requires_grad:
                        name = name + '_prev_task'
                        mean = self.prev_task_params[name]
                        # apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.online_ewc_gamma * self.prev_task_fishers[name]
                        ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                        ewc_losses.append(ewc_loss)
            else:
                for task in range(self.ewc_task_count):
                    # compute ewc loss for each parameter
                    for name, param in self.actor.named_parameters():
                        if param.requires_grad:
                            name = name + f'_prev_task{task}'
                            mean = self.prev_task_params[name]
                            fisher = self.prev_task_fishers[name]
                            ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                            ewc_losses.append(ewc_loss)
            return torch.sum(torch.stack(ewc_losses)) / 2.0
        else:
            return torch.tensor(0.0, device=self.device)

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

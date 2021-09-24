import numpy as np
from collections import Iterable
import torch
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence

import utils
from agent.sac import TaskEmbeddingDistilledActorSacMlpAgent


class EwcTaskEmbeddingDistilledActorSacMlpAgent(TaskEmbeddingDistilledActorSacMlpAgent):
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
            distillation_hidden_dim=256,
            distillation_task_embedding_dim=16,
            distillation_epochs=200,
            distillation_iters_per_epoch=50,
            distillation_batch_size=1000,
            distillation_memory_budget_per_task=50000,
            ewc_lambda=5000,
            ewc_estimate_fisher_iters=50,
            ewc_estimate_fisher_sample_num=1000,
            online_ewc=False,
            online_ewc_gamma=1.0,
    ):
        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size, distillation_hidden_dim,
            distillation_task_embedding_dim, distillation_epochs, distillation_iters_per_epoch,
            distillation_batch_size, distillation_memory_budget_per_task)

        self.ewc_lambda = ewc_lambda
        self.ewc_estimate_fisher_iters = ewc_estimate_fisher_iters
        self.ewc_estimate_fisher_sample_num = ewc_estimate_fisher_sample_num
        self.online_ewc = online_ewc
        self.online_ewc_gamma = online_ewc_gamma

        self.ewc_task_count = 0
        self.prev_task_params = {}
        self.prev_task_fishers = {}

    def estimate_fisher(self, **kwargs):
        sample_src = kwargs.pop('sample_src', 'rollout')
        env = kwargs.pop('env')
        replay_buffer = kwargs.pop('replay_buffer')
        task_idx = kwargs.pop('head_idx')

        fishers = {}
        for _ in range(self.ewc_estimate_fisher_iters):
            obs = env.reset()
            samples = {
                'obses': [],
                'mus': [],
                'log_stds': [],
            }
            if sample_src == 'rollout':
                for _ in range(self.ewc_estimate_fisher_sample_num):
                    with utils.eval_mode(self):
                        action = self.act(obs, sample=True, head_idx=task_idx)

                    next_obs, reward, done, _ = env.step(action)

                    samples['obs'].append(obs)
                    samples['action'].append(action)
                    samples['reward'].append(reward)
                    samples['next_obs'].append(next_obs)
                    not_done = [not done_ for done_ in done]
                    samples['not_done'].append(not_done)

                    obs = next_obs
                samples['obs'] = torch.Tensor(samples['obs']).to(device=self.device)
                samples['action'] = torch.Tensor(samples['action']).to(device=self.device)
                samples['reward'] = torch.Tensor(
                    samples['reward']).to(device=self.device).unsqueeze(-1)
                samples['next_obs'] = torch.Tensor(samples['next_obs']).to(device=self.device)
                samples['not_dones'] = torch.Tensor(
                    samples['not_done']).to(device=self.device).unsqueeze(-1)
            elif sample_src == 'replay_buffer':
                obs, action, reward, next_obs, not_done = replay_buffer.sample(
                    self.ewc_estimate_fisher_sample_num)
                samples['obs'] = obs
                samples['action'] = action
                samples['reward'] = reward
                samples['next_obs'] = next_obs
                samples['not_done'] = not_done
            elif sample_src == 'hybrid':
                for _ in range(self.ewc_estimate_fisher_sample_num // 2):
                    with utils.eval_mode(self):
                        # compute log_pi and Q for later gradient projection
                        mu, action, _, log_std = self.actor(
                            torch.Tensor(obs).to(device=self.device),
                            compute_pi=True, compute_log_pi=True, **kwargs)

                        action = utils.to_np(
                            action.clamp(*self.action_range[task_idx]))

                    next_obs, reward, done, _ = env.step(action)

                    samples['obses'].append(obs)
                    samples['mus'].append(utils.to_np(mu))
                    samples['log_stds'].append(utils.to_np(log_std))

                    obs = next_obs

                rollout_obses = torch.Tensor(samples['obses']).to(device=self.device)
                rollout_mus = torch.Tensor(samples['mus']).to(device=self.device)
                rollout_log_stds = torch.Tensor(samples['log_stds']).to(device=self.device)

                obses, _, _, _, _ = replay_buffer.sample(
                    self.ewc_estimate_fisher_sample_num -
                    self.ewc_estimate_fisher_sample_num // 2)

                with utils.eval_mode(self):
                    mus, _, _, log_stds = self.actor(
                        obses, compute_pi=True, compute_log_pi=True, **kwargs)

                samples['obses'] = torch.cat([rollout_obses, obses], dim=0)
                samples['mus'] = torch.cat([rollout_mus, mus], dim=0)
                samples['log_stds'] = torch.cat([rollout_log_stds, log_stds], dim=0)
            else:
                raise ValueError("Unknown sample source!")

            # compute distillation loss
            mus, _, _, log_stds = self.distilled_actor(
                samples['obses'].squeeze(), task_idx,
                compute_pi=True, compute_log_pi=True)

            actor_dists = Independent(Normal(loc=samples['mus'].squeeze(),
                                             scale=samples['log_stds'].squeeze().exp()), 1)
            distilled_actor_dists = Independent(Normal(loc=mus, scale=log_stds.exp()), 1)
            loss = torch.mean(kl_divergence(actor_dists, distilled_actor_dists))

            self.distilled_actor_optimizer.zero_grad()
            loss.backward()

            # compute Fisher matrix
            for name, param in self.distilled_actor.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        fishers[name] = param.grad.detach().cpu().clone() ** 2 + \
                                        fishers.get(name, torch.zeros_like(param.grad).cpu())
                    else:
                        fishers[name] = torch.zeros_like(param).cpu()

        for name, param in self.distilled_actor.named_parameters():
            if param.requires_grad:
                fisher = fishers[name]

                if self.online_ewc:
                    name = name + '_prev_task'
                    self.prev_task_params[name] = param.detach().cpu().clone()
                    self.prev_task_fishers[name] = \
                        fisher / self.ewc_estimate_fisher_iters + \
                        self.online_ewc_gamma * self.prev_task_fishers.get(
                            name, torch.zeros_like(param.grad).cpu())
                else:
                    name = name + f'_prev_task{self.ewc_task_count}'
                    self.prev_task_params[name] = param.detach().cpu().clone()
                    self.prev_task_fishers[name] = \
                        fisher / self.ewc_estimate_fisher_iters

        self.ewc_task_count += 1

    def _compute_ewc_loss(self, named_parameters):
        assert isinstance(named_parameters, Iterable), "'named_parameters' must be a iterator"

        ewc_losses = []
        if self.ewc_task_count >= 1:
            if self.online_ewc:
                for name, param in named_parameters:
                    if param.grad is not None:
                        name = name + '_prev_task'
                        mean = self.prev_task_params[name].to(self.device)
                        # apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.online_ewc_gamma * self.prev_task_fishers[name].to(self.device)
                        ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                        ewc_losses.append(ewc_loss)
            else:
                for task in range(self.ewc_task_count):
                    # compute ewc loss for each parameter
                    for name, param in named_parameters:
                        if param.grad is not None:
                            name = name + f'_prev_task{task}'
                            mean = self.prev_task_params[name].to(self.device)
                            fisher = self.prev_task_fishers[name].to(self.device)
                            ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                            ewc_losses.append(ewc_loss)
            return torch.sum(torch.stack(ewc_losses)) / 2.0
        else:
            return torch.tensor(0.0, device=self.device)

    def _train_distilled_actor(self, dataset, total_steps, epoch, logger):
        for iter in range(self.distillation_iters_per_epoch):
            random_idxs = np.random.randint(0, self.distillation_memory_budget_per_task,
                                            size=self.distillation_batch_size)
            batch_obses = torch.Tensor(dataset['obses'][random_idxs]).to(self.device).squeeze()
            batch_mus = torch.Tensor(dataset['mus'][random_idxs]).to(self.device).squeeze()
            batch_log_stds = torch.Tensor(dataset['log_stds'][random_idxs]).to(self.device).squeeze()
            task_id = dataset['task_id']

            mus, _, _, log_stds = self.distilled_actor(
                batch_obses, task_id,
                compute_pi=True, compute_log_pi=True)

            actor_dists = Independent(Normal(loc=batch_mus, scale=batch_log_stds.exp()), 1)
            distilled_actor_dists = Independent(Normal(loc=mus, scale=log_stds.exp()), 1)
            distillation_loss = torch.mean(kl_divergence(actor_dists, distilled_actor_dists))
            # regularize with EWC
            ewc_loss = self._compute_ewc_loss(self.distilled_actor.named_parameters())
            distillation_loss = distillation_loss + self.ewc_lambda * ewc_loss

            logger.log('train/distillation_loss', distillation_loss,
                       total_steps + epoch * self.distillation_iters_per_epoch + iter)

            self.distilled_actor_optimizer.zero_grad()
            distillation_loss.backward()
            self.distilled_actor_optimizer.step()

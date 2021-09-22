from collections import Iterable
import torch

import utils
from agent.sac import TaskEmbeddingHyperNetActorSacMlpAgent


class EwcTaskEmbeddingHyperNetActorSacMlpAgent(TaskEmbeddingHyperNetActorSacMlpAgent):
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
            hypernet_hidden_dim=128,
            hypernet_task_embedding_dim=16,
            hypernet_reg_coeff=0.01,
            hypernet_on_the_fly_reg=False,
            hypernet_online_uniform_reg=False,
            hypernet_first_order=True,
            ewc_lambda=5000,
            ewc_estimate_fisher_iters=100,
            ewc_estimate_fisher_sample_num=1024,
            online_ewc=False,
            online_ewc_gamma=1.0,
    ):
        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size, hypernet_hidden_dim, hypernet_task_embedding_dim,
            hypernet_reg_coeff, hypernet_on_the_fly_reg, hypernet_online_uniform_reg, hypernet_first_order)

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
        # TODO (chongyi zheng): save trajectory for KL divergence
        for _ in range(self.ewc_estimate_fisher_iters):
            obs = env.reset()
            samples = {
                'obs': [],
                'action': [],
                'reward': [],
                'next_obs': [],
                'not_done': [],
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
                        action = self.act(obs, sample=True, head_idx=task_idx)

                    next_obs, reward, done, _ = env.step(action)

                    samples['obs'].append(obs)
                    samples['action'].append(action)
                    samples['reward'].append(reward)
                    samples['next_obs'].append(next_obs)
                    not_done = [not done_ for done_ in done]
                    samples['not_done'].append(not_done)

                    obs = next_obs

                rollout_obs = torch.Tensor(samples['obs']).to(device=self.device)
                rollout_action = torch.Tensor(samples['action']).to(device=self.device)
                rollout_reward = torch.Tensor(
                    samples['reward']).to(device=self.device).unsqueeze(-1)
                rollout_next_obs = torch.Tensor(samples['next_obs']).to(device=self.device)
                rollout_not_done = torch.Tensor(
                    samples['not_done']).to(device=self.device).unsqueeze(-1)

                obs, action, reward, next_obs, not_done = replay_buffer.sample(
                    self.ewc_estimate_fisher_sample_num - self.ewc_estimate_fisher_sample_num // 2)
                samples['obs'] = torch.cat([rollout_obs, obs], dim=0)
                samples['action'] = torch.cat([rollout_action, action], dim=0)
                samples['reward'] = torch.cat([rollout_reward, reward], dim=0)
                samples['next_obs'] = torch.cat([rollout_next_obs, next_obs], dim=0)
                samples['not_done'] = torch.cat([rollout_not_done, not_done], dim=0)
            else:
                raise ValueError("Unknown sample source!")

            _, actor_loss, _ = self.compute_actor_and_alpha_loss(
                samples['obs'],
                compute_alpha_loss=False, task_idx=task_idx
            )
            self.hypernet_weight_optimizer.zero_grad()
            actor_loss.backward()

            for name, param in self.hypernet.weights.items():
                if param.requires_grad:
                    if param.grad is not None:
                        fishers[name] = param.grad.detach().clone() ** 2 + \
                                        fishers.get(name, torch.zeros_like(param.grad))
                    else:
                        fishers[name] = torch.zeros_like(param)

        for name, param in self.hypernet.weights.items():
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
            return torch.tensor(0.0, device=self.device)

    def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None,
                               add_reg_loss=False):
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.hypernet_weight_optimizer.zero_grad()
        self.hypernet_emb_optimizer.zero_grad()

        actor_loss.backward(retain_graph=add_reg_loss,
                            create_graph=add_reg_loss and not self.hypernet_first_order)
        self.hypernet_emb_optimizer.step()

        if add_reg_loss:
            # assert len(self.target_weights) == self.task_count
            hypernet_delta_weights = self.compute_hypernet_delta_weights()

            reg_loss = self.hypernet_reg_coeff * self.compute_hypernet_reg(
                hypernet_delta_weights)
            reg_loss.backward()

        self.hypernet_weight_optimizer.step()

        if isinstance(alpha_loss, torch.Tensor):
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        assert 'head_idx' in kwargs
        task_idx = kwargs.pop('head_idx')

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done,
                                               task_idx=task_idx)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_update_freq == 0:
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
                obs, task_idx=task_idx)
            actor_ewc_loss = self._compute_ewc_loss(self.hypernet.weights.items())
            actor_loss = actor_loss + self.ewc_lambda * actor_ewc_loss
            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss,
                                        add_reg_loss=False)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

import copy
import numpy as np
import torch

import utils

from agent.sac.base_sac_agent import SacMlpAgent
from agent.network import MultiHeadSacActorMlp, SacCriticMlp


class MultiHeadSacMlpAgentV2(SacMlpAgent):
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
            distill_epochs=10,  # (cyzheng): refresh data every epoch
            distill_iters_per_epoch=250,
            distill_batch_size=1000,
    ):
        assert isinstance(action_shape, list)
        assert isinstance(action_range, list)
        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size)

        self.distill_epochs = distill_epochs
        self.distill_iters_per_epoch = distill_iters_per_epoch
        self.distill_batch_size = distill_batch_size

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = MultiHeadSacActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.actor_log_std_min, self.actor_log_std_max
        ).to(self.device)

        self.distilled_actor = MultiHeadSacActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.actor_log_std_min, self.actor_log_std_max
        ).to(self.device)

        self.critic = SacCriticMlp(
            self.obs_shape, self.action_shape[0], self.critic_hidden_dim
        ).to(self.device)
        self.critic_target = SacCriticMlp(
            self.obs_shape, self.action_shape[0], self.critic_hidden_dim
        ).to(self.device)

        self.reset_target_critic()

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_shape[0])

        # sac optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.distilled_actor_optimizer = torch.optim.Adam(
            self.distilled_actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # save initial parameters
        self._critic_init_state = copy.deepcopy(self.critic.state_dict())

    def act(self, obs, sample=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        with torch.no_grad():
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False, **kwargs)
            action = pi if sample else mu
            assert 'head_idx' in kwargs
            action = action.clamp(*self.action_range[kwargs['head_idx']])
            assert action.ndim == 2 and action.shape[0] == 1

        return utils.to_np(action)

    def distill(self, **kwargs):
        sample_src = kwargs.pop('sample_src', 'rollout')
        env = kwargs.pop('env')
        replay_buffer = kwargs.pop('replay_buffer')

    def distill(self, **kwargs):
        sample_src = kwargs.pop('sample_src', 'rollout')
        env = kwargs.pop('env')
        replay_buffer = kwargs.pop('replay_buffer')

        # memory_size_per_task = self.agem_memory_budget // (self.agem_task_count + 1)
        # self._adjust_memory_size(memory_size_per_task)

        obs = env.reset()
        # self.agem_memories[self.agem_task_count] = {
        #     'obses': [],
        #     'actions': [],
        #     'rewards': [],
        #     'next_obses': [],
        #     'not_dones': [],
        #     'log_pis': [],
        #     'qs': [],
        # }
        if sample_src == 'rollout':
            for _ in range(memory_size_per_task):
                with utils.eval_mode(self):
                    # compute log_pi and Q for later gradient projection
                    _, action, log_pi, _ = self.actor(
                        torch.Tensor(obs).to(device=self.device),
                        compute_pi=True, compute_log_pi=True, **kwargs)
                    actor_Q1, actor_Q2 = self.critic(
                        torch.Tensor(obs).to(device=self.device), action, **kwargs)
                    actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pi

                    if 'head_idx' in kwargs:
                        action = utils.to_np(
                            action.clamp(*self.action_range[kwargs['head_idx']]))
                    else:
                        action = utils.to_np(action.clamp(*self.action_range))

                next_obs, reward, done, _ = env.step(action)

                self.agem_memories[self.agem_task_count]['obses'].append(obs)
                self.agem_memories[self.agem_task_count]['actions'].append(action)
                self.agem_memories[self.agem_task_count]['rewards'].append(reward)
                self.agem_memories[self.agem_task_count]['next_obses'].append(next_obs)
                not_done = np.array([not done_ for done_ in done], dtype=np.float32)
                self.agem_memories[self.agem_task_count]['not_dones'].append(not_done)
                self.agem_memories[self.agem_task_count]['log_pis'].append(log_pi)
                self.agem_memories[self.agem_task_count]['qs'].append(actor_Q)

                obs = next_obs

            self.agem_memories[self.agem_task_count]['obses'] = torch.Tensor(
                self.agem_memories[self.agem_task_count]['obses']).to(device=self.device)
            self.agem_memories[self.agem_task_count]['actions'] = torch.Tensor(
                self.agem_memories[self.agem_task_count]['actions']).to(device=self.device)
            self.agem_memories[self.agem_task_count]['rewards'] = torch.Tensor(
                self.agem_memories[self.agem_task_count]['rewards']).to(device=self.device).unsqueeze(-1)
            self.agem_memories[self.agem_task_count]['next_obses'] = torch.Tensor(
                self.agem_memories[self.agem_task_count]['next_obses']).to(device=self.device)
            self.agem_memories[self.agem_task_count]['not_dones'] = torch.Tensor(
                self.agem_memories[self.agem_task_count]['not_dones']).to(device=self.device).unsqueeze(-1)
            self.agem_memories[self.agem_task_count]['log_pis'] = torch.cat(
                self.agem_memories[self.agem_task_count]['log_pis']).unsqueeze(-1)
            self.agem_memories[self.agem_task_count]['qs'] = torch.cat(
                self.agem_memories[self.agem_task_count]['qs']).unsqueeze(-1)
        elif sample_src == 'replay_buffer':
            obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(
                memory_size_per_task)
            with utils.eval_mode(self):
                log_pis = self.actor.compute_log_probs(obses, actions, **kwargs)
                actor_Q1, actor_Q2 = self.critic(
                    obses, actions, **kwargs)
                actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pis

            self.agem_memories[self.agem_task_count]['obses'] = obses
            self.agem_memories[self.agem_task_count]['actions'] = actions
            self.agem_memories[self.agem_task_count]['rewards'] = rewards
            self.agem_memories[self.agem_task_count]['next_obses'] = next_obses
            self.agem_memories[self.agem_task_count]['not_dones'] = not_dones
            self.agem_memories[self.agem_task_count]['log_pis'] = log_pis
            self.agem_memories[self.agem_task_count]['qs'] = actor_Q
        elif sample_src == 'hybrid':
            for _ in range(memory_size_per_task // 2):
                with utils.eval_mode(self):
                    # compute log_pi and Q for later gradient projection
                    _, action, log_pi, _ = self.actor(
                        torch.Tensor(obs).to(device=self.device),
                        compute_pi=True, compute_log_pi=True, **kwargs)
                    actor_Q1, actor_Q2 = self.critic(
                        torch.Tensor(obs).to(device=self.device), action, **kwargs)
                    actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pi

                    if 'head_idx' in kwargs:
                        action = utils.to_np(
                            action.clamp(*self.action_range[kwargs['head_idx']]))
                    else:
                        action = utils.to_np(action.clamp(*self.action_range))

                next_obs, reward, done, _ = env.step(action)

                self.agem_memories[self.agem_task_count]['obses'].append(obs)
                self.agem_memories[self.agem_task_count]['actions'].append(action)
                self.agem_memories[self.agem_task_count]['rewards'].append(reward)
                self.agem_memories[self.agem_task_count]['next_obses'].append(next_obs)
                not_done = np.array([not done_ for done_ in done], dtype=np.float32)
                self.agem_memories[self.agem_task_count]['not_dones'].append(not_done)
                self.agem_memories[self.agem_task_count]['log_pis'].append(log_pi)
                self.agem_memories[self.agem_task_count]['qs'].append(actor_Q)

                obs = next_obs

            rollout_obses = torch.Tensor(self.agem_memories[self.agem_task_count]['obses']).to(device=self.device)
            rollout_actions = torch.Tensor(
                self.agem_memories[self.agem_task_count]['actions']).to(device=self.device)
            rollout_rewards = torch.Tensor(
                self.agem_memories[self.agem_task_count]['rewards']).to(device=self.device).unsqueeze(-1)
            rollout_next_obses = torch.Tensor(
                self.agem_memories[self.agem_task_count]['next_obses']).to(device=self.device)
            rollout_not_dones = torch.Tensor(
                self.agem_memories[self.agem_task_count]['not_dones']).to(device=self.device).unsqueeze(-1)
            rollout_log_pis = torch.cat(
                self.agem_memories[self.agem_task_count]['log_pis']).unsqueeze(-1)
            rollout_qs = torch.cat(
                self.agem_memories[self.agem_task_count]['qs']).unsqueeze(-1)

            obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(
                memory_size_per_task)
            with utils.eval_mode(self):
                log_pis = self.actor.compute_log_probs(obses, actions, **kwargs)
                actor_Q1, actor_Q2 = self.critic(
                    obses, actions, **kwargs)
                actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pis

            self.agem_memories[self.agem_task_count]['obses'] = \
                torch.cat([rollout_obses, obses], dim=0)
            self.agem_memories[self.agem_task_count]['actions'] = \
                torch.cat([rollout_actions, actions], dim=0)
            self.agem_memories[self.agem_task_count]['rewards'] = \
                torch.cat([rollout_rewards, rewards], dim=0)
            self.agem_memories[self.agem_task_count]['next_obses'] = \
                torch.cat([rollout_next_obses, next_obses], dim=0)
            self.agem_memories[self.agem_task_count]['not_dones'] = \
                torch.cat([rollout_not_dones, not_dones], dim=0)
            self.agem_memories[self.agem_task_count]['log_pis'] = \
                torch.cat([rollout_log_pis, log_pis], dim=0)
            self.agem_memories[self.agem_task_count]['qs'] = \
                torch.cat([rollout_qs, actor_Q], dim=0)
        else:
            raise ValueError("Unknown sample source!")

        self.agem_task_count += 1

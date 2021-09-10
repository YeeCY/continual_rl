import copy

import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.sac import AgemContinualActorCriticMultiInputSacMlpAgent


class AgemContinualActorCriticGradNormRegCriticPrioritizedMemoryMultiInputSacMlpAgent(
    AgemContinualActorCriticMultiInputSacMlpAgent):
    """Adapt from https://github.com/GMvandeVen/continual-learning"""
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
                 agem_memory_budget=2000,
                 agem_ref_grad_batch_size=500,
                 critic_grad_norm_reg_coeff=1.0,
                 ):
        AgemContinualActorCriticMultiInputSacMlpAgent.__init__(self, obs_shape, action_shape, action_range, device,
                                                               actor_hidden_dim, critic_hidden_dim, discount,
                                                               init_temperature, alpha_lr, actor_lr, actor_log_std_min,
                                                               actor_log_std_max, actor_update_freq, critic_lr,
                                                               critic_tau, critic_target_update_freq, batch_size,
                                                               agem_memory_budget, agem_ref_grad_batch_size)

        # TODO (cyzheng): agem_memory_budget is the memory budget for each task here

        self.critic_grad_norm_reg_coeff = critic_grad_norm_reg_coeff

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None, None

        ref_critic_grad = []
        ref_actor_grad = []
        for task_id, memory in enumerate(self.agem_memories.values()):
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obs, action, reward, next_obs, not_done = \
                torch.Tensor(memory['obses'][idxs]).to(self.device), \
                torch.Tensor(memory['actions'][idxs]).to(self.device), \
                torch.Tensor(memory['rewards'][idxs]).to(self.device), \
                torch.Tensor(memory['next_obses'][idxs]).to(self.device), \
                torch.Tensor(memory['not_dones'][idxs]).to(self.device)

            critic_loss = self.compute_critic_loss(
                obs, action, reward, next_obs, not_done, head_idx=task_id)
            self.critic_optimizer.zero_grad()  # clear current gradient
            critic_loss.backward()

            single_ref_critic_grad = []
            for param in self.critic.common_parameters():
                if param.requires_grad:
                    single_ref_critic_grad.append(param.grad.detach().clone().flatten())
            single_ref_critic_grad = torch.cat(single_ref_critic_grad)
            self.critic_optimizer.zero_grad()

            _, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
                obs, compute_alpha_loss=False, head_idx=task_id)
            self.actor_optimizer.zero_grad()  # clear current gradient
            actor_loss.backward()

            single_ref_actor_grad = []
            for param in self.actor.common_parameters():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            self.actor_optimizer.zero_grad()

            ref_critic_grad.append(single_ref_critic_grad)
            ref_actor_grad.append(single_ref_actor_grad)
        ref_critic_grad = torch.stack(ref_critic_grad).mean(dim=0)
        ref_actor_grad = torch.stack(ref_actor_grad).mean(dim=0)

        return ref_critic_grad, ref_actor_grad

    def construct_memory(self, **kwargs):
        sample_src = kwargs.pop('sample_src', 'rollout')
        env = kwargs.pop('env')
        replay_buffer = kwargs.pop('replay_buffer')

        # memory_size_per_task = self.agem_memory_budget // (self.agem_task_count + 1)
        # self._adjust_memory_size(memory_size_per_task)

        obs = env.reset()
        self.agem_memories[self.agem_task_count] = {
            'obses': [],
            'actions': [],
            'rewards': [],
            'next_obses': [],
            'not_dones': [],
            'log_pis': [],
            'qs': [],
        }
        if sample_src == 'rollout':
            rollout_obses, rollout_actions, rollout_rewards, rollout_next_obses, \
            rollout_not_dones, rollout_log_pis, rollout_qs = [], [], [], [], [], [], []
            for _ in range(10):
                rollouts = {
                    'obses': [],
                    'actions': [],
                    'rewards': [],
                    'next_obses': [],
                    'not_dones': [],
                    'log_pis': [],
                    'qs': [],
                }
                grad_norms = []
                for _ in range(self.agem_memory_budget):
                    with utils.eval_mode(self):
                        # compute log_pi and Q for later gradient projection
                        _, action, log_pi, _ = self.actor(
                            torch.Tensor(obs).to(device=self.device),
                            compute_pi=True, compute_log_pi=True, **kwargs)
                        actor_Q1, actor_Q2 = self.critic(
                            torch.Tensor(obs).to(device=self.device), action, **kwargs)
                        actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pi

                        # compute critic gradient norm
                        Q1_grads = torch.autograd.grad(
                            actor_Q1.sum(), action)[0]
                        Q2_grads = torch.autograd.grad(
                            actor_Q2.sum(), action)[0]
                        grad1_norm = torch.sum(torch.square(Q1_grads), dim=-1)
                        grad2_norm = torch.sum(torch.square(Q2_grads), dim=-1)
                        grad_norm = torch.mean(grad1_norm + grad2_norm)

                        action = utils.to_np(action)
                        log_pi = utils.to_np(log_pi)
                        actor_Q = utils.to_np(actor_Q)
                        grad_norm = utils.to_np(grad_norm)

                    next_obs, reward, done, _ = env.step(action)

                    # (cyzheng): convert to threshold
                    grad_norms.append(grad_norm)
                    rollouts['obses'].append(obs)
                    rollouts['actions'].append(action)
                    rollouts['rewards'].append([reward])
                    rollouts['next_obses'].append(next_obs)
                    not_done = np.array([[not done_ for done_ in done]], dtype=np.float32)
                    rollouts['not_dones'].append(not_done)
                    rollouts['log_pis'].append(log_pi)
                    rollouts['qs'].append(actor_Q)

                    obs = next_obs

                # sort according to grad_norms
                idxs = np.argsort(grad_norms)[:self.agem_memory_budget // 10]

                rollout_obses.append(np.asarray(rollouts['obses'])[idxs])
                rollout_actions.append(np.asarray(rollouts['actions'])[idxs])
                rollout_rewards.append(np.asarray(rollouts['rewards'])[idxs])
                rollout_next_obses.append(np.asarray(rollouts['next_obses'])[idxs])
                rollout_not_dones.append(np.asarray(rollouts['not_dones'])[idxs])
                rollout_log_pis.append(np.asarray(rollouts['log_pis'])[idxs])
                rollout_qs.append(np.asarray(rollouts['qs'])[idxs])

            self.agem_memories[self.agem_task_count]['obses'] = np.concatenate(rollout_obses, axis=0)
            self.agem_memories[self.agem_task_count]['actions'] = np.concatenate(rollout_actions, axis=0)
            self.agem_memories[self.agem_task_count]['rewards'] = np.concatenate(rollout_rewards, axis=0)
            self.agem_memories[self.agem_task_count]['next_obses'] = np.concatenate(rollout_next_obses, axis=0)
            self.agem_memories[self.agem_task_count]['not_dones'] = np.concatenate(rollout_not_dones, axis=0)
            self.agem_memories[self.agem_task_count]['log_pis'] = np.concatenate(rollout_log_pis, axis=0)
            self.agem_memories[self.agem_task_count]['qs'] = np.concatenate(rollout_qs, axis=0)

        elif sample_src == 'replay_buffer':
            obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(
                self.agem_memory_budget * 10)

            actions.requires_grad = True
            actions.retain_grad()
            with utils.eval_mode(self):
                log_pis = self.actor.compute_log_probs(obses, actions, **kwargs)
                actor_Q1, actor_Q2 = self.critic(
                    obses, actions, **kwargs)
                actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pis

                # compute critic gradient norm
                Q1_grads = torch.autograd.grad(actor_Q1.sum(), actions)[0]
                Q2_grads = torch.autograd.grad(actor_Q2.sum(), actions)[0]
                grad1_norm = torch.sum(torch.square(Q1_grads), dim=-1)
                grad2_norm = torch.sum(torch.square(Q2_grads), dim=-1)
                grad_norm = torch.mean(grad1_norm + grad2_norm, dim=-1)

                obses = utils.to_np(obses)
                actions = utils.to_np(actions)
                rewards = utils.to_np(rewards)
                next_obses = utils.to_np(next_obses)
                not_dones = utils.to_np(not_dones)
                log_pis = utils.to_np(log_pis)
                actor_Q = utils.to_np(actor_Q)
                grad_norm = utils.to_np(grad_norm)

            # sort according to grad_norms
            idxs = np.argsort(grad_norm)[:self.agem_memory_budget]

            self.agem_memories[self.agem_task_count]['obses'] = obses[idxs]
            self.agem_memories[self.agem_task_count]['actions'] = actions[idxs]
            self.agem_memories[self.agem_task_count]['rewards'] = rewards[idxs]
            self.agem_memories[self.agem_task_count]['next_obses'] = next_obses[idxs]
            self.agem_memories[self.agem_task_count]['not_dones'] = not_dones[idxs]
            self.agem_memories[self.agem_task_count]['log_pis'] = log_pis[idxs]
            self.agem_memories[self.agem_task_count]['qs'] = actor_Q[idxs]
        elif sample_src == 'hybrid':
            rollout_obses, rollout_actions, rollout_rewards, rollout_next_obses, \
            rollout_not_dones, rollout_log_pis, rollout_qs = [], [], [], [], [], [], []
            for _ in range(5):
                rollouts = {
                    'obses': [],
                    'actions': [],
                    'rewards': [],
                    'next_obses': [],
                    'not_dones': [],
                    'log_pis': [],
                    'qs': [],
                }
                grad_norms = []
                for _ in range(self.agem_memory_budget):
                    with utils.eval_mode(self):
                        # compute log_pi and Q for later gradient projection
                        _, action, log_pi, _ = self.actor(
                            torch.Tensor(obs).to(device=self.device),
                            compute_pi=True, compute_log_pi=True, **kwargs)
                        actor_Q1, actor_Q2 = self.critic(
                            torch.Tensor(obs).to(device=self.device), action, **kwargs)
                        actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pi

                        # compute critic gradient norm
                        Q1_grads = torch.autograd.grad(
                            actor_Q1.sum(), action)[0]
                        Q2_grads = torch.autograd.grad(
                            actor_Q2.sum(), action)[0]
                        grad1_norm = torch.sum(torch.square(Q1_grads), dim=-1)
                        grad2_norm = torch.sum(torch.square(Q2_grads), dim=-1)
                        grad_norm = torch.mean(grad1_norm + grad2_norm)

                        action = utils.to_np(action)
                        log_pi = utils.to_np(log_pi)
                        actor_Q = utils.to_np(actor_Q)
                        grad_norm = utils.to_np(grad_norm)

                    next_obs, reward, done, _ = env.step(action)

                    # (cyzheng): convert to threshold
                    grad_norms.append(grad_norm)
                    rollouts['obses'].append(obs)
                    rollouts['actions'].append(action)
                    rollouts['rewards'].append([reward])
                    rollouts['next_obses'].append(next_obs)
                    not_done = np.array([[not done_ for done_ in done]], dtype=np.float32)
                    rollouts['not_dones'].append(not_done)
                    rollouts['log_pis'].append(log_pi)
                    rollouts['qs'].append(actor_Q)

                    obs = next_obs

                # sort according to grad_norms
                idxs = np.argsort(grad_norms)[:self.agem_memory_budget // 10]

                rollout_obses.append(np.asarray(rollouts['obses'])[idxs])
                rollout_actions.append(np.asarray(rollouts['actions'])[idxs])
                rollout_rewards.append(np.asarray(rollouts['rewards'])[idxs])
                rollout_next_obses.append(np.asarray(rollouts['next_obses'])[idxs])
                rollout_not_dones.append(np.asarray(rollouts['not_dones'])[idxs])
                rollout_log_pis.append(np.asarray(rollouts['log_pis'])[idxs])
                rollout_qs.append(np.asarray(rollouts['qs'])[idxs])

            rollout_obses = np.concatenate(rollout_obses, axis=0)
            rollout_actions = np.concatenate(rollout_actions, axis=0)
            rollout_rewards = np.concatenate(rollout_rewards, axis=0)
            rollout_next_obses = np.concatenate(rollout_next_obses, axis=0)
            rollout_not_dones = np.concatenate(rollout_not_dones, axis=0)
            rollout_log_pis = np.concatenate(rollout_log_pis, axis=0)
            rollout_qs = np.concatenate(rollout_qs, axis=0)

            obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(
                self.agem_memory_budget * 5)

            actions.requires_grad = True
            actions.retain_grad()
            with utils.eval_mode(self):
                log_pis = self.actor.compute_log_probs(obses, actions, **kwargs)
                actor_Q1, actor_Q2 = self.critic(
                    obses, actions, **kwargs)
                actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pis

                # compute critic gradient norm
                Q1_grads = torch.autograd.grad(actor_Q1.sum(), actions)[0]
                Q2_grads = torch.autograd.grad(actor_Q2.sum(), actions)[0]
                grad1_norm = torch.sum(torch.square(Q1_grads), dim=-1)
                grad2_norm = torch.sum(torch.square(Q2_grads), dim=-1)
                grad_norm = torch.mean(grad1_norm + grad2_norm, dim=-1)

                obses = utils.to_np(obses)
                actions = utils.to_np(actions)
                rewards = utils.to_np(rewards)
                next_obses = utils.to_np(next_obses)
                not_dones = utils.to_np(not_dones)
                log_pis = utils.to_np(log_pis)
                actor_Q = utils.to_np(actor_Q)
                grad_norm = utils.to_np(grad_norm)

            idxs = np.argsort(
                grad_norm)[:self.agem_memory_budget - self.agem_memory_budget // 2]

            self.agem_memories[self.agem_task_count]['obses'] = \
                np.concatenate([rollout_obses, obses[idxs]], axis=0)
            self.agem_memories[self.agem_task_count]['actions'] = \
                np.concatenate([rollout_actions, actions[idxs]], axis=0)
            self.agem_memories[self.agem_task_count]['rewards'] = \
                np.concatenate([rollout_rewards, rewards[idxs]], axis=0)
            self.agem_memories[self.agem_task_count]['next_obses'] = \
                np.concatenate([rollout_next_obses, next_obses[idxs]], axis=0)
            self.agem_memories[self.agem_task_count]['not_dones'] = \
                np.concatenate([rollout_not_dones, not_dones[idxs]], axis=0)
            self.agem_memories[self.agem_task_count]['log_pis'] = \
                np.concatenate([rollout_log_pis, log_pis[idxs]], axis=0)
            self.agem_memories[self.agem_task_count]['qs'] = \
                np.concatenate([rollout_qs, actor_Q[idxs]], axis=0)
        else:
            raise ValueError("Unknown sample source!")

        self.agem_task_count += 1

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done, **kwargs):
        with torch.no_grad():
            _, next_policy_action, next_log_pi, _ = self.actor(next_obs, **kwargs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_policy_action, **kwargs)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * next_log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, **kwargs)

        # gradient norm regularization
        _, policy_action, _, _ = self.actor(obs, **kwargs)
        reg_Q1, reg_Q2 = self.critic(obs, policy_action, **kwargs)

        # (cyzheng) reference: equation 10 in http://arxiv.org/abs/2103.08050.
        # (cyzheng): create graph for second order derivatives
        reg_Q1_grads = torch.autograd.grad(
            reg_Q1.sum(), policy_action, create_graph=True)[0]
        reg_Q2_grads = torch.autograd.grad(
            reg_Q2.sum(), policy_action, create_graph=True)[0]
        grad1_norm = torch.sum(torch.square(reg_Q1_grads), dim=-1)
        grad2_norm = torch.sum(torch.square(reg_Q2_grads), dim=-1)
        grad_norm_reg = torch.mean(grad1_norm + grad2_norm)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + \
                      self.critic_grad_norm_reg_coeff * grad_norm_reg

        return critic_loss

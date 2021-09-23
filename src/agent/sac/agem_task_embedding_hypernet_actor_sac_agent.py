import numpy as np
import torch

import utils
from agent.sac import TaskEmbeddingHyperNetActorSacMlpAgent


class AgemTaskEmbeddingHyperNetActorSacMlpAgent(TaskEmbeddingHyperNetActorSacMlpAgent):
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
            agem_memory_budget=4500,
            agem_ref_grad_batch_size=500,
    ):
        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size, hypernet_hidden_dim, hypernet_task_embedding_dim,
            hypernet_reg_coeff, hypernet_on_the_fly_reg, hypernet_online_uniform_reg, hypernet_first_order)

        self.agem_memory_budget = agem_memory_budget
        self.agem_ref_grad_batch_size = agem_ref_grad_batch_size

        self.agem_task_count = 0
        self.agem_memories = {}

    # FIXME (cyzheng): not used now
    # def _adjust_memory_size(self, size):
    #     for mem in self.agem_memories.values():
    #         mem['obses'] = mem['obses'][:size]
    #         mem['actions'] = mem['actions'][:size]
    #         mem['rewards'] = mem['rewards'][:size]
    #         mem['next_obses'] = mem['next_obses'][:size]
    #         mem['not_dones'] = mem['not_dones'][:size]

    def construct_memory(self, **kwargs):
        sample_src = kwargs.pop('sample_src', 'rollout')
        env = kwargs.pop('env')
        replay_buffer = kwargs.pop('replay_buffer')
        task_idx = kwargs.pop('head_idx')

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

            with utils.eval_mode(self):
                weights = self.hypernet(task_idx)

            for _ in range(self.agem_memory_budget // 2):
                with utils.eval_mode(self):
                    # compute log_pi and Q for later gradient projection
                    _, action, log_pi, _ = self.actor(
                        torch.Tensor(obs).to(device=self.device),
                        compute_pi=True, compute_log_pi=True, weights=weights)
                    actor_Q1, actor_Q2 = self.critic(
                        torch.Tensor(obs).to(device=self.device), action, **kwargs)
                    actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pi

                    action = utils.to_np(action)
                    log_pi = utils.to_np(log_pi)
                    actor_Q = utils.to_np(actor_Q)

                next_obs, reward, done, _ = env.step(action)

                rollout_obses.append(obs)
                rollout_actions.append(action)
                rollout_rewards.append([reward])
                rollout_next_obses.append(next_obs)
                not_done = np.array([[not done_ for done_ in done]], dtype=np.float32)
                rollout_not_dones.append(not_done)
                rollout_log_pis.append(log_pi)
                rollout_qs.append(actor_Q)

                obs = next_obs
            self.agem_memories[self.agem_task_count]['obses'] = np.asarray(rollout_obses)
            self.agem_memories[self.agem_task_count]['actions'] = np.asarray(rollout_actions)
            self.agem_memories[self.agem_task_count]['rewards'] = np.asarray(rollout_rewards)
            self.agem_memories[self.agem_task_count]['next_obses'] = np.asarray(rollout_next_obses)
            self.agem_memories[self.agem_task_count]['not_dones'] = np.asarray(rollout_not_dones)
            self.agem_memories[self.agem_task_count]['log_pis'] = np.asarray(rollout_log_pis)
            self.agem_memories[self.agem_task_count]['qs'] = np.asarray(rollout_qs)

        elif sample_src == 'replay_buffer':
            obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(
                self.agem_memory_budget)

            with utils.eval_mode(self):
                weights = self.hypernet(task_idx)
                log_pis = self.actor.compute_log_probs(obses, actions, **kwargs, weights=weights)
                actor_Q1, actor_Q2 = self.critic(
                    obses, actions, **kwargs)
                actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pis

                obses = utils.to_np(obses)
                actions = utils.to_np(actions)
                rewards = utils.to_np(rewards)
                next_obses = utils.to_np(next_obses)
                not_dones = utils.to_np(not_dones)
                log_pis = utils.to_np(log_pis)
                actor_Q = utils.to_np(actor_Q)

            self.agem_memories[self.agem_task_count]['obses'] = obses
            self.agem_memories[self.agem_task_count]['actions'] = actions
            self.agem_memories[self.agem_task_count]['rewards'] = rewards
            self.agem_memories[self.agem_task_count]['next_obses'] = next_obses
            self.agem_memories[self.agem_task_count]['not_dones'] = not_dones
            self.agem_memories[self.agem_task_count]['log_pis'] = log_pis
            self.agem_memories[self.agem_task_count]['qs'] = actor_Q
        elif sample_src == 'hybrid':
            rollout_obses, rollout_actions, rollout_rewards, rollout_next_obses, \
            rollout_not_dones, rollout_log_pis, rollout_qs = [], [], [], [], [], [], []

            with utils.eval_mode(self):
                weights = self.hypernet(task_idx)

            for _ in range(self.agem_memory_budget // 2):
                with utils.eval_mode(self):
                    # compute log_pi and Q for later gradient projection
                    _, action, log_pi, _ = self.actor(
                        torch.Tensor(obs).to(device=self.device),
                        compute_pi=True, compute_log_pi=True, weights=weights)
                    actor_Q1, actor_Q2 = self.critic(
                        torch.Tensor(obs).to(device=self.device), action, **kwargs)
                    actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pi

                    action = utils.to_np(action)
                    log_pi = utils.to_np(log_pi)
                    actor_Q = utils.to_np(actor_Q)

                next_obs, reward, done, _ = env.step(action)

                rollout_obses.append(obs)
                rollout_actions.append(action)
                rollout_rewards.append([reward])
                rollout_next_obses.append(next_obs)
                not_done = np.array([[not done_ for done_ in done]], dtype=np.float32)
                rollout_not_dones.append(not_done)
                rollout_log_pis.append(log_pi)
                rollout_qs.append(actor_Q)

                obs = next_obs
            rollout_obses = np.asarray(rollout_obses)
            rollout_actions = np.asarray(rollout_actions)
            rollout_rewards = np.asarray(rollout_rewards)
            rollout_next_obses = np.asarray(rollout_next_obses)
            rollout_not_dones = np.asarray(rollout_not_dones)
            rollout_log_pis = np.asarray(rollout_log_pis)
            rollout_qs = np.asarray(rollout_qs)

            obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(
                self.agem_memory_budget - self.agem_memory_budget // 2)

            with utils.eval_mode(self):
                log_pis = self.actor.compute_log_probs(obses, actions, weights=weights)
                actor_Q1, actor_Q2 = self.critic(
                    obses, actions, **kwargs)
                actor_Q = torch.min(actor_Q1, actor_Q2) - self.alpha.detach() * log_pis

                obses = utils.to_np(obses)
                actions = utils.to_np(actions)
                rewards = utils.to_np(rewards)
                next_obses = utils.to_np(next_obses)
                not_dones = utils.to_np(not_dones)
                log_pis = utils.to_np(log_pis)
                actor_Q = utils.to_np(actor_Q)

            self.agem_memories[self.agem_task_count]['obses'] = \
                np.concatenate([rollout_obses, obses], axis=0)
            self.agem_memories[self.agem_task_count]['actions'] = \
                np.concatenate([rollout_actions, actions], axis=0)
            self.agem_memories[self.agem_task_count]['rewards'] = \
                np.concatenate([rollout_rewards, rewards], axis=0)
            self.agem_memories[self.agem_task_count]['next_obses'] = \
                np.concatenate([rollout_next_obses, next_obses], axis=0)
            self.agem_memories[self.agem_task_count]['not_dones'] = \
                np.concatenate([rollout_not_dones, not_dones], axis=0)
            self.agem_memories[self.agem_task_count]['log_pis'] = \
                np.concatenate([rollout_log_pis, log_pis], axis=0)
            self.agem_memories[self.agem_task_count]['qs'] = \
                np.concatenate([rollout_qs, actor_Q], axis=0)
        else:
            raise ValueError("Unknown sample source!")

        self.agem_task_count += 1

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None

        ref_actor_grad = []
        for task_idx, memory in enumerate(self.agem_memories.values()):
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obses, actions, rewards, next_obses, not_dones, old_log_pis, qs = \
                memory['obses'][idxs], memory['actions'][idxs], memory['rewards'][idxs], \
                memory['next_obses'][idxs], memory['not_dones'][idxs], memory['log_pis'], \
                memory['qs']

            obses = torch.Tensor(obses).to(self.device)
            actions = torch.Tensor(actions).to(self.device)
            old_log_pis = torch.Tensor(old_log_pis).to(self.device)
            qs = torch.Tensor(qs).to(self.device)

            # (chongyi zheng): use PPO style gradient projection loss for actor
            weights = self.hypernet(task_idx)
            log_pis = self.actor.compute_log_probs(obses, actions, weights=weights)
            ratio = torch.exp(log_pis - old_log_pis)  # importance sampling ratio
            proj_actor_loss = (ratio * qs).mean()

            self.hypernet_weight_optimizer.zero_grad()  # clear current gradient
            proj_actor_loss.backward()

            single_ref_actor_grad = []
            for param in self.hypernet.weights.values():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            self.hypernet_weight_optimizer.zero_grad()

            ref_actor_grad.append(single_ref_actor_grad)
        ref_actor_grad = torch.stack(ref_actor_grad).mean(dim=0)

        return ref_actor_grad

    def _project_grad(self, parameters, ref_grad):
        assert isinstance(parameters, list), "'parameters' must be a list"

        if ref_grad is None:
            return

        grad = []
        for param in parameters:
            if param.requires_grad:
                grad.append(param.grad.flatten())
        grad = torch.cat(grad)

        # inequality constrain
        angle = (grad * ref_grad).sum()
        if angle < 0:
            # project the gradient of the current transitions onto the gradient of the memory transitions ...
            proj_grad = grad - (angle / (ref_grad * ref_grad).sum()) * ref_grad
            # replace all the gradients within the model with this projected gradient
            idx = 0
            for param in parameters:
                if param.requires_grad:
                    num_param = param.numel()  # number of parameters in [p]
                    param.grad.copy_(proj_grad[idx:idx + num_param].reshape(param.shape))
                    idx += num_param

    def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None,
                               add_reg_loss=False, ref_actor_grad=None):
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

        self._project_grad(list(self.hypernet.weights.values()), ref_actor_grad)

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
            ref_actor_grad = self._compute_ref_grad()
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
                obs, task_idx=task_idx)
            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss,
                                        add_reg_loss=False, ref_actor_grad=ref_actor_grad)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

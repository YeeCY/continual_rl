import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence

import utils
from agent.sac import TaskEmbeddingDistilledActorSacMlpAgent


class AgemTaskEmbeddingDistilledActorSacMlpAgent(TaskEmbeddingDistilledActorSacMlpAgent):
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
            agem_memory_budget=5000,
            agem_ref_grad_batch_size=500,
            agem_clip_param=0.2,
    ):
        self.agem_memory_budget = agem_memory_budget
        self.agem_ref_grad_batch_size = agem_ref_grad_batch_size
        self.agem_clip_param = agem_clip_param

        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size, distillation_hidden_dim,
            distillation_task_embedding_dim, distillation_epochs, distillation_iters_per_epoch,
            distillation_batch_size, distillation_memory_budget_per_task)

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
            'mus': [],
            'log_stds': [],
        }
        if sample_src == 'rollout':  # FIXME (cyzheng)
            rollout_obses, rollout_actions, rollout_rewards, rollout_next_obses, \
            rollout_not_dones, rollout_log_pis, rollout_qs = [], [], [], [], [], [], []

            for _ in range(self.agem_memory_budget):
                with utils.eval_mode(self):
                    # compute log_pi and Q for later gradient projection
                    _, action, log_pi, _ = self.actor(
                        torch.Tensor(obs).to(device=self.device),
                        compute_pi=True, compute_log_pi=True)
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

        elif sample_src == 'replay_buffer':  # FIXME (cyzheng)
            obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(
                self.agem_memory_budget)

            with utils.eval_mode(self):
                log_pis = self.actor.compute_log_probs(obses, actions, **kwargs)
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
            rollout_obses, rollout_mus, rollout_log_stds = [], [], []

            for _ in range(self.agem_memory_budget // 2):
                with utils.eval_mode(self):
                    mu, action, _, log_std = self.actor(
                        torch.Tensor(obs).to(device=self.device),
                        compute_pi=True, compute_log_pi=True, **kwargs)

                    action = utils.to_np(
                        action.clamp(*self.action_range[task_idx]))
                    mu = utils.to_np(mu)
                    log_std = utils.to_np(log_std)

                next_obs, reward, done, _ = env.step(action)

                rollout_obses.append(obs)
                rollout_mus.append(mu)
                rollout_log_stds.append(log_std)

                obs = next_obs
            rollout_obses = np.asarray(rollout_obses)
            rollout_mus = np.asarray(rollout_mus)
            rollout_log_stds = np.asarray(rollout_log_stds)

            obses, _, _, _, _ = replay_buffer.sample(
                self.agem_memory_budget -
                self.agem_memory_budget // 2)
            with utils.eval_mode(self):
                mus, _, _, log_stds = self.actor(
                    obses, compute_pi=True, compute_log_pi=True,
                    **kwargs)

                obses = utils.to_np(obses)
                mus = utils.to_np(mus)
                log_stds = utils.to_np(log_stds)

            self.agem_memories[self.agem_task_count]['obses'] = \
                np.concatenate([rollout_obses, obses], axis=0)
            self.agem_memories[self.agem_task_count]['mus'] = \
                np.concatenate([rollout_mus, mus], axis=0)
            self.agem_memories[self.agem_task_count]['log_stds'] = \
                np.concatenate([rollout_log_stds, log_stds], axis=0)
        else:
            raise ValueError("Unknown sample source!")

        self.agem_task_count += 1

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None

        ref_actor_grad = []
        for task_idx, memory in enumerate(self.agem_memories.values()):
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size
            )

            batch_obses, batch_mus, batch_log_stds = \
                memory['obses'][idxs], memory['mus'][idxs], memory['log_stds'][idxs]

            batch_obses = torch.Tensor(batch_obses).to(self.device).squeeze()
            batch_mus = torch.Tensor(batch_mus).to(self.device).squeeze()
            batch_log_stds = torch.Tensor(batch_log_stds).to(self.device).squeeze()

            # compute distillation loss
            mus, _, _, log_stds = self.distilled_actor(
                batch_obses, task_idx,
                compute_pi=True, compute_log_pi=True)
            actor_dists = Independent(Normal(loc=batch_mus, scale=batch_log_stds.exp()), 1)
            distilled_actor_dists = Independent(Normal(loc=mus, scale=log_stds.exp()), 1)
            loss = torch.mean(kl_divergence(actor_dists, distilled_actor_dists))

            self.distilled_actor_weight_optimizer.zero_grad()  # clear current gradient
            loss.backward()

            single_ref_actor_grad = []
            for param in self.distilled_actor.weights.values():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            self.distilled_actor_weight_optimizer.zero_grad()

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

    def _train_distilled_actor(self, dataset, total_steps, epoch, logger):
        for iter in range(self.distillation_iters_per_epoch):
            random_idxs = np.random.randint(0, self.distillation_memory_budget_per_task,
                                            size=self.distillation_batch_size)
            batch_obses = torch.Tensor(dataset['obses'][random_idxs]).to(self.device).squeeze()
            batch_mus = torch.Tensor(dataset['mus'][random_idxs]).to(self.device).squeeze()
            batch_log_stds = torch.Tensor(dataset['log_stds'][random_idxs]).to(self.device).squeeze()
            task_id = dataset['task_id']

            # regularize with AGEM
            ref_grad = self._compute_ref_grad()

            mus, _, _, log_stds = self.distilled_actor(
                batch_obses, task_id,
                compute_pi=True, compute_log_pi=True)

            actor_dists = Independent(Normal(loc=batch_mus, scale=batch_log_stds.exp()), 1)
            distilled_actor_dists = Independent(Normal(loc=mus, scale=log_stds.exp()), 1)
            distillation_loss = torch.mean(kl_divergence(actor_dists, distilled_actor_dists))

            logger.log('train/distillation_loss', distillation_loss,
                       total_steps + epoch * self.distillation_iters_per_epoch + iter)

            self.distilled_actor_weight_optimizer.zero_grad()
            self.distilled_actor_emb_optimizer.zero_grad()
            distillation_loss.backward()
            self._project_grad(list(self.distilled_actor.weights.values()), ref_grad)
            self.distilled_actor_weight_optimizer.step()
            self.distilled_actor_emb_optimizer.zero_grad()

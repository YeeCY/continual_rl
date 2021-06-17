import torch
import numpy as np
from itertools import chain

import utils
import storages
from agent.ppo.base_ppo_agent import PpoMlpAgent


class AgemPpoMlpAgentV2(PpoMlpAgent):
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
                 agem_memory_budget=10240,
                 agem_ref_grad_batch_size=1024,
                 ):
        super().__init__(obs_shape, action_shape, device, hidden_dim, discount, clip_param, ppo_epoch,
                         critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm, use_clipped_critic_loss,
                         num_batch)

        self.agem_memory_budget = agem_memory_budget
        self.agem_ref_grad_batch_size = agem_ref_grad_batch_size

        self.agem_task_count = 0
        self.agem_memories = {}

    def _adjust_memory_size(self, size):
        for mem in self.agem_memories.values():
            mem.update_num_steps(size)

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None

        ref_grad = []
        for memory in self.agem_memories.values():
            advantages = memory.returns[:-1] - memory.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)
            data_generator = memory.feed_forward_generator(
                advantages, mini_batch_size=self.agem_ref_grad_batch_size // self.agem_task_count)

            # (chongyi zheng): we only use one batch of rollouts to compute ref_grad
            obs_batch, actions_batch, value_preds_batch, \
            return_batch, old_log_pis, adv_targets = next(data_generator)

            actor_loss, entropy = self.compute_actor_loss(
                obs_batch, actions_batch, old_log_pis, adv_targets)
            critic_loss = self.compute_critic_loss(
                obs_batch, value_preds_batch, return_batch)
            loss = actor_loss + self.critic_loss_coef * critic_loss - \
                   self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()

            # compute reference gradient
            single_ref_grad = []
            for param in chain(self.actor.parameters(), self.critic.parameters()):
                if param.requires_grad:
                    single_ref_grad.append(param.grad.detach().clone().flatten())
            single_ref_grad = torch.cat(single_ref_grad)
            self.optimizer.zero_grad()

            ref_grad.append(single_ref_grad)

        ref_grad = torch.stack(ref_grad).mean(dim=0)

        return ref_grad

    def _project_grad(self, ref_grad):
        if ref_grad is None:
            return

        grad = []
        for param in chain(self.actor.parameters(), self.critic.parameters()):
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
            for param in chain(self.actor.parameters(), self.critic.parameters()):
                if param.requires_grad:
                    num_param = param.numel()  # number of parameters in [p]
                    param.grad.copy_(proj_grad[idx:idx + num_param].reshape(param.shape))
                    idx += num_param

    def construct_memory(self, env, num_processes, compute_returns_kwargs, **kwargs):
        memory_size_per_task = self.agem_memory_budget // (self.agem_task_count + 1)
        self._adjust_memory_size(memory_size_per_task)

        memory = storages.RolloutStorage(memory_size_per_task, num_processes,
                                         env.observation_space.shape, env.action_space,
                                         self.device)

        obs = env.reset()
        memory.obs[0].copy_(torch.Tensor(obs).to(self.device))

        for _ in range(memory_size_per_task):
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
            memory.insert(obs, action, log_pi, value, reward, masks, bad_masks)

        next_value = self.predict_value(memory.obs[-1], **kwargs)
        memory.compute_returns(next_value, **compute_returns_kwargs)

        self.agem_memories[self.agem_task_count] = memory

        self.agem_task_count += 1

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

                ref_grad = self._compute_ref_grad()

                # Reshape to do in a single forward pass for all steps
                actor_loss, entropy = self.compute_actor_loss(
                    obs_batch, actions_batch, old_log_pis, adv_targets, **kwargs)
                critic_loss = self.compute_critic_loss(
                    obs_batch, value_preds_batch, return_batch, **kwargs)
                loss = actor_loss + self.critic_loss_coef * critic_loss - \
                       self.entropy_coef * entropy

                logger.log('train_actor/loss', actor_loss, step)
                logger.log('train_actor/entropy', entropy, step)
                logger.log('train_critic/loss', critic_loss, step)
                logger.log('train/loss', loss, step)
                self.optimizer.zero_grad()
                loss.backward()

                self._project_grad(ref_grad)

                torch.nn.utils.clip_grad_norm_(
                    chain(self.actor.parameters(), self.critic.parameters()),
                    self.grad_clip_norm)
                self.optimizer.step()

    def save(self, model_dir, step):
        super().save(model_dir, step)
        torch.save(
            self.agem_memories, '%s/agem_memories_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        super().load(model_dir, step)
        self.agem_memories = torch.load(
            '%s/agem_memories_%s.pt' % (model_dir, step)
        )

import torch
import numpy as np
import src.utils as utils
from collections.abc import Iterable

from src.agent.sac.base_sac_agent import SacMlpAgent


class AgemSacMlpAgent(SacMlpAgent):
    """Adapt from https://github.com/GMvandeVen/continual-learning"""
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
                 agem_memory_budget=3000,
                 agem_ref_grad_batch_size=128,
                 ):
        super().__init__(obs_shape, action_shape, action_range, device, hidden_dim, discount, init_temperature,
                         alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                         critic_tau, critic_target_update_freq, grad_clip_norm, batch_size)

        self.agem_memory_budget = agem_memory_budget
        self.agem_ref_grad_batch_size = agem_ref_grad_batch_size

        self.agem_task_count = 0
        self.agem_memories = {}

    def _adjust_memory_size(self, size):
        for mem in self.agem_memories.values():
            mem['obses'] = mem['obses'][:size]
            mem['actions'] = mem['actions'][:size]
            mem['rewards'] = mem['rewards'][:size]
            mem['next_obses'] = mem['next_obses'][:size]
            mem['not_dones'] = mem['not_dones'][:size]

    def _compute_ref_grad(self, compute_alpha_ref_grad=True):
        # (chongyi zheng): We compute reference gradients for actor and critic separately
        # assert isinstance(named_parameters, Iterable), "'named_parameters' must be a iterator"
        #
        # si_losses = []
        # for name, param in named_parameters:
        #     if param.requires_grad:
        #         prev_param = self.prev_task_params[name]
        #         omega = self.omegas.get(name, torch.zeros_like(param))
        #         si_loss = torch.sum(omega * (param - prev_param) ** 2)
        #         si_losses.append(si_loss)
        #
        # return torch.sum(torch.stack(si_losses))

        if not self.agem_memories:
            return None, None, None

        # sample memory transitions
        concat_obses = []
        concat_actions = []
        concat_rewards = []
        concat_next_obses = []
        concat_not_dones = []
        for mem in self.agem_memories.values():
            concat_obses.append(mem['obses'])
            concat_actions.append(mem['actions'])
            concat_rewards.append(mem['rewards'])
            concat_next_obses.append(mem['next_obses'])
            concat_not_dones.append(mem['not_dones'])

        concat_obses = torch.cat(concat_obses)
        concat_actions = torch.cat(concat_actions)
        concat_rewards = torch.cat(concat_rewards)
        concat_next_obses = torch.cat(concat_next_obses)
        concat_not_dones = torch.cat(concat_not_dones)

        perm_idxs = np.random.permutation(concat_obses.shape[0])
        sample_idxs = np.random.randint(0, len(concat_obses), size=self.agem_ref_grad_batch_size)
        obs = concat_obses[perm_idxs][sample_idxs]
        action = concat_actions[perm_idxs][sample_idxs]
        reward = concat_rewards[perm_idxs][sample_idxs]
        next_obs = concat_next_obses[perm_idxs][sample_idxs]
        not_done = concat_not_dones[perm_idxs][sample_idxs]

        # reference critic gradients
        ref_critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done)
        self.critic_optimizer.zero_grad()  # clear current gradient
        ref_critic_loss.backward()

        # reorganize the gradient of the memory transitions as a single vector
        ref_critic_grad = []
        for param in self.critic.parameters():
            if param.requires_grad:
                ref_critic_grad.append(param.grad.detach().clone().flatten())
        ref_critic_grad = torch.cat(ref_critic_grad)
        # reset gradients (with A-GEM, gradients of memory transitions should only be used as inequality constraint)
        self.critic_optimizer.zero_grad()

        # reference actor and alpha gradients
        _, ref_actor_loss, ref_alpha_loss = self.compute_actor_and_alpha_loss(
            obs, compute_alpha_loss=compute_alpha_ref_grad)
        self.actor_optimizer.zero_grad()  # clear current gradient
        ref_actor_loss.backward()

        ref_actor_grad = []
        for param in self.actor.parameters():
            if param.requires_grad:
                ref_actor_grad.append(param.grad.detach().clone().flatten())
        ref_actor_grad = torch.cat(ref_actor_grad)
        self.actor_optimizer.zero_grad()

        ref_alpha_grad = None
        if compute_alpha_ref_grad:
            self.log_alpha_optimizer.zero_grad()  # clear current gradient
            ref_alpha_loss.backward()
            ref_alpha_grad = self.log_alpha.grad.detach().clone()
            self.log_alpha_optimizer.zero_grad()

        return ref_critic_grad, ref_actor_grad, ref_alpha_grad

    def _project_grad(self, named_parameters, ref_grad):
        assert isinstance(named_parameters, Iterable), "'named_parameters' must be a iterator"

        if ref_grad is None:
            return

        grad = []
        for name, param in named_parameters:
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
            for _, param in named_parameters:
                if param.requires_grad:
                    num_param = param.numel()  # number of parameters in [p]
                    param.grad.copy_(proj_grad[idx:idx + num_param].reshape(param.shape))
                    idx += num_param

    def construct_memory(self, replay_buffer):
        memory_size_per_task = self.agem_memory_budget // (self.agem_task_count + 1)
        self._adjust_memory_size(memory_size_per_task)

        assert memory_size_per_task <= len(replay_buffer)
        # random sample transitions
        obses, actions, rewards, next_obses, not_dones = replay_buffer.sample(memory_size_per_task)
        self.agem_memories[self.agem_task_count] = {
            'obses': obses,
            'actions': actions,
            'rewards': rewards,
            'next_obses': next_obses,
            'not_dones': not_dones,
        }

        self.agem_task_count += 1

    def update_critic(self, critic_loss, logger, step, ref_critic_grad=None):
        # Optimize the critic
        logger.log('train_critic/loss', critic_loss, step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        self._project_grad(self.critic.named_parameters(), ref_critic_grad)

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None, ref_actor_grad=None,
                               ref_alpha_grad=None):
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(self.actor.named_parameters(), ref_actor_grad)

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        if isinstance(alpha_loss, torch.Tensor):
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()

            self._project_grad(iter([('log_alpha', self.log_alpha)]), ref_alpha_grad)

            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.grad_clip_norm)
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        ref_critic_grad, ref_actor_grad, ref_alpha_grad = self._compute_ref_grad()

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done)
        self.update_critic(critic_loss, logger, step, ref_critic_grad=ref_critic_grad)

        if step % self.actor_update_freq == 0:
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(obs)
            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss,
                                        ref_actor_grad=ref_actor_grad, ref_alpha_grad=ref_alpha_grad)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

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

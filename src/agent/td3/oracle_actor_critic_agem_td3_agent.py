import copy
import torch
import numpy as np

import utils
from agent.td3 import Td3MlpAgent


class OracleActorCriticAgemTd3MlpAgent(Td3MlpAgent):
    """Adapt from https://github.com/GMvandeVen/continual-learning"""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 action_range,
                 device,
                 actor_hidden_dim=256,
                 critic_hidden_dim=256,
                 discount=0.99,
                 actor_lr=3e-4,
                 actor_noise=0.2,
                 actor_noise_clip=0.5,
                 critic_lr=3e-4,
                 expl_noise_std=0.1,
                 target_tau=0.005,
                 actor_and_target_update_freq=2,
                 batch_size=256,
                 agem_memory_budget=4500,
                 agem_ref_grad_batch_size=500,
                 ):
        super().__init__(obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim,
                         discount, actor_lr, actor_noise, actor_noise_clip, critic_lr, expl_noise_std, target_tau,
                         actor_and_target_update_freq, batch_size)

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

    def _compute_ref_grad(self):
        # (chongyi zheng): We compute reference gradients for actor and critic separately
        if not self.agem_memories:
            return None

        ref_actor_grad = []
        for memory in self.agem_memories.values():
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obs, action, reward, next_obs, not_done, old_actor, old_critic = \
                memory['obses'][idxs], memory['actions'][idxs], memory['rewards'][idxs], \
                memory['next_obses'][idxs], memory['not_dones'][idxs], \
                memory['actor'], memory['critic']

            actor_action = old_actor(obs)
            actor_proj_loss = -old_critic.Q1(obs, actor_action).mean()
            old_actor.zero_grad()  # clear current gradient
            actor_proj_loss.backward()

            single_ref_actor_grad = []
            for param in old_actor.parameters():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            old_actor.zero_grad()

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

    def construct_memory(self, env, **kwargs):
        memory_size_per_task = self.agem_memory_budget // (self.agem_task_count + 1)
        self._adjust_memory_size(memory_size_per_task)

        obs = env.reset()
        self.agem_memories[self.agem_task_count] = {
            'obses': [],
            'actions': [],
            'rewards': [],
            'next_obses': [],
            'not_dones': [],
        }
        for _ in range(memory_size_per_task):
            with utils.eval_mode(self):
                action = self.actor(obs, **kwargs)

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
        self.agem_memories[self.agem_task_count]['critic'] = copy.deepcopy(self.critic)
        self.agem_memories[self.agem_task_count]['actor'] = copy.deepcopy(self.actor)

        self.agem_task_count += 1

    def update_actor(self, actor_loss, logger, step, ref_actor_grad=None):
        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(list(self.actor.parameters()), ref_actor_grad)

        self.actor_optimizer.step()

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        ref_actor_grad = self._compute_ref_grad()

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done, **kwargs)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_and_target_update_freq == 0:
            actor_loss = self.compute_actor_loss(obs, **kwargs)
            self.update_actor(actor_loss, logger, step, ref_actor_grad=ref_actor_grad)

            utils.soft_update_params(self.actor, self.actor_target, self.target_tau)
            utils.soft_update_params(self.critic, self.critic_target, self.target_tau)

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

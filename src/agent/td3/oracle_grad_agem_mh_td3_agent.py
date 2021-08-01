import torch
import numpy as np

import utils
from agent.td3 import MultiHeadTd3MlpAgent, OracleGradAgemTd3MlpAgent


class OracleGradAgemMultiHeadTd3MlpAgent(MultiHeadTd3MlpAgent, OracleGradAgemTd3MlpAgent):
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
        MultiHeadTd3MlpAgent.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                      critic_hidden_dim, discount, actor_lr, actor_noise, actor_noise_clip,
                                      critic_lr, expl_noise_std, target_tau, actor_and_target_update_freq,
                                      batch_size)

        OracleGradAgemTd3MlpAgent.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                           critic_hidden_dim, discount, actor_lr, actor_noise, actor_noise_clip,
                                           critic_lr, expl_noise_std, target_tau, actor_and_target_update_freq,
                                           batch_size, agem_memory_budget, agem_ref_grad_batch_size)

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
                action = self.actor(
                    torch.Tensor(obs).to(device=self.device),
                    **kwargs)

                if 'head_idx' in kwargs:
                    action = utils.to_np(action).clip(
                        *self.action_range[kwargs['head_idx']])
                else:
                    action = utils.to_np(action).clip(*self.action_range)

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

        # save oracle gradient in memory
        actor_loss = self.compute_actor_loss(
            self.agem_memories[self.agem_task_count]['obses'], **kwargs)
        self.actor_optimizer.zero_grad()  # clear current gradient
        actor_loss.backward()

        single_ref_actor_grad = []
        for param in self.actor.common_parameters():
            if param.requires_grad:
                single_ref_actor_grad.append(param.grad.detach().clone().flatten())
        single_ref_actor_grad = torch.cat(single_ref_actor_grad)
        self.actor_optimizer.zero_grad()

        self.agem_memories[self.agem_task_count]['ref_grad'] = single_ref_actor_grad

        self.agem_task_count += 1

    def update_actor(self, actor_loss, logger, step, ref_actor_grad=None):
        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(list(self.actor.common_parameters()), ref_actor_grad)

        self.actor_optimizer.step()

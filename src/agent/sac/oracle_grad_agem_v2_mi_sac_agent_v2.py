import copy
import torch
import numpy as np

import utils
from agent.sac import MultiInputSacMlpAgentV2, OracleGradAgemV2SacMlpAgentV2


class OracleGradAgemV2MultiInputSacMlpAgentV2(MultiInputSacMlpAgentV2, OracleGradAgemV2SacMlpAgentV2):
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
                 agem_memory_budget=4500,
                 agem_ref_grad_batch_size=500,
                 ):
        MultiInputSacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                         critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                         actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                                         critic_tau, critic_target_update_freq, batch_size)

        OracleGradAgemV2SacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                               critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                               actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                                               critic_tau, critic_target_update_freq, batch_size, agem_memory_budget,
                                               agem_ref_grad_batch_size)

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
                # compute log_pi and Q for later gradient projection
                _, action, log_pi, _ = self.actor(
                    torch.Tensor(obs).to(device=self.device),
                    compute_pi=True, compute_log_pi=True, **kwargs)

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

        # save oracle gradient in memory
        _, actor_loss, _ = self.compute_actor_and_alpha_loss(
            self.agem_memories[self.agem_task_count]['obses'], compute_alpha_loss=False)
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

    def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None, ref_actor_grad=None):
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(self.actor.common_parameters(), ref_actor_grad)

        self.actor_optimizer.step()

        if isinstance(alpha_loss, torch.Tensor):
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            # self._project_grad(iter([self.log_alpha]), ref_alpha_grad)
            self.log_alpha_optimizer.step()

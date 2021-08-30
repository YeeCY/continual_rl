import torch
import numpy as np
from collections.abc import Iterable

import utils
from agent.sac.base_sac_agent import SacMlpAgent


class AgemV2SacMlpAgentV2(SacMlpAgent):
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
        super().__init__(obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim,
                         discount, init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max,
                         actor_update_freq, critic_lr, critic_tau, critic_target_update_freq, batch_size)

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
            mem['log_pis'] = mem['log_pis'][:size]
            mem['qs'] = mem['qs'][:size]

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None

        ref_actor_grad = []
        for memory in self.agem_memories.values():
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obses, actions, rewards, next_obses, not_dones, old_log_pis, qs = \
                memory['obses'][idxs], memory['actions'][idxs], memory['rewards'][idxs], \
                memory['next_obses'][idxs], memory['not_dones'][idxs], memory['log_pis'], \
                memory['qs']

            # (chongyi zheng): use PPO style gradient projection loss for actor
            log_pis = self.actor.compute_log_probs(obses, actions)
            ratio = torch.exp(log_pis - old_log_pis.detach())  # importance sampling ratio
            proj_actor_loss = (ratio * qs.detach()).mean()

            self.actor_optimizer.zero_grad()  # clear current gradient
            proj_actor_loss.backward()

            single_ref_actor_grad = []
            for param in self.actor.parameters():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            self.actor_optimizer.zero_grad()

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

    def construct_memory(self, **kwargs):
        sample_src = kwargs.pop('sample_src', 'rollout')
        env = kwargs.pop('env')
        replay_buffer = kwargs.pop('replay_buffer')

        memory_size_per_task = self.agem_memory_budget // (self.agem_task_count + 1)
        self._adjust_memory_size(memory_size_per_task)

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
            self.agem_memories[self.agem_task_count]['actor_Q'] = actor_Q
        elif sample_src == 'hybrid':
            pass
        else:
            raise ValueError("Unknown sample source!")

        self.agem_task_count += 1

    def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None, ref_actor_grad=None):
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(list(self.actor.parameters()), ref_actor_grad)

        self.actor_optimizer.step()

        if isinstance(alpha_loss, torch.Tensor):
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done, **kwargs)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_update_freq == 0:
            ref_actor_grad = self._compute_ref_grad()
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(obs, **kwargs)
            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss,
                                        ref_actor_grad=ref_actor_grad)

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

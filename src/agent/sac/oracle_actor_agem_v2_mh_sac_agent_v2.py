import copy
import torch
import numpy as np

import utils
from agent.sac import MultiHeadSacMlpAgentV2, OracleActorAgemV2SacMlpAgentV2


class OracleActorAgemV2MultiHeadSacMlpAgentV2(MultiHeadSacMlpAgentV2, OracleActorAgemV2SacMlpAgentV2):
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
        MultiHeadSacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                        critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                        actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                                        critic_tau, critic_target_update_freq, batch_size)

        OracleActorAgemV2SacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                                critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                                actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                                                critic_tau, critic_target_update_freq, batch_size, agem_memory_budget,
                                                agem_ref_grad_batch_size)

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None

        ref_actor_grad = []
        for task_id, memory in enumerate(self.agem_memories.values()):
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obses, actions, rewards, next_obses, not_dones, old_actor, old_critic, old_log_alpha = \
                memory['obses'][idxs], memory['actions'][idxs], memory['rewards'][idxs], \
                memory['next_obses'][idxs], memory['not_dones'][idxs], memory['actor'], \
                memory['critic'], memory['log_alpha']

            _, pi, log_pi, log_std = old_actor(obses, head_idx=task_id)
            actor_Q1, actor_Q2 = old_critic(obses, pi, head_idx=task_id)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_proj_loss = (old_log_alpha.exp().detach() * log_pi - actor_Q).mean()

            # clear current gradient with net since we didn't save optimizer
            old_actor.zero_grad()
            actor_proj_loss.backward()

            single_ref_actor_grad = []
            for param in old_actor.common_parameters():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            old_actor.zero_grad()

            ref_actor_grad.append(single_ref_actor_grad)
        ref_actor_grad = torch.stack(ref_actor_grad).mean(dim=0)

        return ref_actor_grad

    def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None, ref_actor_grad=None):
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(list(self.actor.common_parameters()), ref_actor_grad)

        self.actor_optimizer.step()

        if isinstance(alpha_loss, torch.Tensor):
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

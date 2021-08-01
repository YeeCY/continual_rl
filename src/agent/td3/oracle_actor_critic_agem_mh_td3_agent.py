import torch
import numpy as np

from agent.td3 import MultiHeadTd3MlpAgent, OracleActorCriticAgemTd3MlpAgent


class OracleActorCriticAgemMultiHeadTd3MlpAgent(MultiHeadTd3MlpAgent, OracleActorCriticAgemTd3MlpAgent):
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

        OracleActorCriticAgemTd3MlpAgent.__init__(self, obs_shape, action_shape, action_range, device,
                                                  actor_hidden_dim, critic_hidden_dim, discount, actor_lr,
                                                  actor_noise, actor_noise_clip, critic_lr, expl_noise_std,
                                                  target_tau, actor_and_target_update_freq, batch_size,
                                                  agem_memory_budget, agem_ref_grad_batch_size)

    def _compute_ref_grad(self):
        # (chongyi zheng): We compute reference gradients for actor and critic separately
        if not self.agem_memories:
            return None

        ref_actor_grad = []
        for task_id, memory in enumerate(self.agem_memories.values()):
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obs, action, reward, next_obs, not_done, old_actor, old_critic = \
                memory['obses'][idxs], memory['actions'][idxs], memory['rewards'][idxs], \
                memory['next_obses'][idxs], memory['not_dones'][idxs], \
                memory['actor'], memory['critic']

            actor_action = old_actor(obs, head_idx=task_id)
            actor_proj_loss = -old_critic.Q1(obs, actor_action, head_idx=task_id).mean()
            old_actor.zero_grad()  # clear current gradient
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

    def update_actor(self, actor_loss, logger, step, ref_actor_grad=None):
        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(list(self.actor.common_parameters()), ref_actor_grad)

        self.actor_optimizer.step()

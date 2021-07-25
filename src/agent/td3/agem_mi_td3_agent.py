import torch
import numpy as np
import copy

from agent.td3 import MultiHeadTd3MlpAgent, AgemTd3MlpAgent
from agent.network import MultiInputTd3ActorMlp, MultiInputTd3CriticMlp


class AgemMultiInputTd3MlpAgent(MultiHeadTd3MlpAgent, AgemTd3MlpAgent):
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

        AgemTd3MlpAgent.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                 critic_hidden_dim, discount, actor_lr, actor_noise, actor_noise_clip,
                                 critic_lr, expl_noise_std, target_tau, actor_and_target_update_freq,
                                 batch_size, agem_memory_budget, agem_ref_grad_batch_size)

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = MultiInputTd3ActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.action_range
        ).to(self.device)

        self.actor_target = MultiInputTd3ActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.action_range
        ).to(self.device)

        self.critic = MultiInputTd3CriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.critic_target = MultiInputTd3CriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.reset_target()

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # save initial parameters
        self._critic_init_state = copy.deepcopy(self.critic.state_dict())

    def _compute_ref_grad(self):
        # (chongyi zheng): We compute reference gradients for actor and critic separately
        if not self.agem_memories:
            return None, None, None

        ref_critic_grad = []
        ref_actor_grad = []
        for task_id, memory in enumerate(self.agem_memories.values()):
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obs, action, reward, next_obs, not_done = \
                memory['obses'][idxs], memory['actions'][idxs], memory['rewards'][idxs], \
                memory['next_obses'][idxs], memory['not_dones'][idxs]

            critic_loss = self.compute_critic_loss(
                obs, action, reward, next_obs, not_done, head_idx=task_id)
            self.critic_optimizer.zero_grad()  # clear current gradient
            critic_loss.backward()

            single_ref_critic_grad = []
            for param in self.critic.common_parameters():
                if param.requires_grad:
                    single_ref_critic_grad.append(param.grad.detach().clone().flatten())
            single_ref_critic_grad = torch.cat(single_ref_critic_grad)
            self.critic_optimizer.zero_grad()

            actor_loss = self.compute_actor_loss(obs, head_idx=task_id)
            self.actor_optimizer.zero_grad()  # clear current gradient
            actor_loss.backward()

            single_ref_actor_grad = []
            for param in self.actor.common_parameters():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            self.actor_optimizer.zero_grad()

            ref_critic_grad.append(single_ref_critic_grad)
            ref_actor_grad.append(single_ref_actor_grad)
        ref_critic_grad = torch.stack(ref_critic_grad).mean(dim=0)
        ref_actor_grad = torch.stack(ref_actor_grad).mean(dim=0)

        return ref_critic_grad, ref_actor_grad

    def update_critic(self, critic_loss, logger, step, ref_critic_grad=None):
        # Optimize the critic
        logger.log('train_critic/loss', critic_loss, step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        self._project_grad(list(self.critic.common_parameters()), ref_critic_grad)

        self.critic_optimizer.step()

    def update_actor(self, actor_loss, logger, step, ref_actor_grad=None):
        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self._project_grad(list(self.actor.common_parameters()), ref_actor_grad)

        self.actor_optimizer.step()

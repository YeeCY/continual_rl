import torch
import numpy as np

from agent.sac import MultiHeadSacMlpAgentV2, AgemSacMlpAgentV2


class AgemMultiHeadSacMlpAgentV2(MultiHeadSacMlpAgentV2, AgemSacMlpAgentV2):
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
                 agem_memory_budget=5000,
                 agem_ref_grad_batch_size=500,
                 ):
        MultiHeadSacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                        critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                        actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
                                        critic_tau, critic_target_update_freq, batch_size)

        AgemSacMlpAgentV2.__init__(self, obs_shape, action_shape, action_range, device, actor_hidden_dim,
                                   critic_hidden_dim, discount, init_temperature, alpha_lr, actor_lr,
                                   actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_tau,
                                   critic_target_update_freq, batch_size, agem_memory_budget, agem_ref_grad_batch_size)

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None

        ref_actor_grad = []
        for task_id, memory in enumerate(self.agem_memories.values()):
            idxs = np.random.randint(
                0, len(memory['obses']), size=self.agem_ref_grad_batch_size // self.agem_task_count
            )

            obs, action, reward, next_obs, not_done = \
                memory['obses'][idxs], memory['actions'][idxs], memory['rewards'][idxs], \
                memory['next_obses'][idxs], memory['not_dones'][idxs]

            # TODO (chongyi zheng): delete this block
            # critic_loss = self.compute_critic_loss(
            #     obs, action, reward, next_obs, not_done, head_idx=task_id)
            # self.critic_optimizer.zero_grad()  # clear current gradient
            # critic_loss.backward()

            # single_ref_critic_grad = []
            # for param in self.critic.common_parameters():
            #     if param.requires_grad:
            #         single_ref_critic_grad.append(param.grad.detach().clone().flatten())
            # single_ref_critic_grad = torch.cat(single_ref_critic_grad)
            # self.critic_optimizer.zero_grad()

            _, actor_loss, _ = self.compute_actor_and_alpha_loss(
                obs, compute_alpha_loss=False, head_idx=task_id)
            self.actor_optimizer.zero_grad()  # clear current gradient
            actor_loss.backward()

            single_ref_actor_grad = []
            for param in self.actor.common_parameters():
                if param.requires_grad:
                    single_ref_actor_grad.append(param.grad.detach().clone().flatten())
            single_ref_actor_grad = torch.cat(single_ref_actor_grad)
            self.actor_optimizer.zero_grad()

            # if compute_alpha_ref_grad:
            #     self.log_alpha_optimizer.zero_grad()  # clear current gradient
            #     alpha_loss.backward()
            #     single_ref_alpha_grad = self.log_alpha.grad.detach().clone()
            #     self.log_alpha_optimizer.zero_grad()
            # else:
            #     single_ref_alpha_grad = None

            # ref_critic_grad.append(single_ref_critic_grad)
            ref_actor_grad.append(single_ref_actor_grad)
            # if single_ref_alpha_grad is not None:
            #     ref_alpha_grad.append(single_ref_alpha_grad)
            # else:
            #     ref_alpha_grad = None

        # ref_critic_grad = torch.stack(ref_critic_grad).mean(dim=0)
        ref_actor_grad = torch.stack(ref_actor_grad).mean(dim=0)
        # if ref_alpha_grad is not None:
        #     ref_alpha_grad = torch.stack(ref_alpha_grad).mean(dim=0)

        # return ref_critic_grad, ref_actor_grad, ref_alpha_grad
        return ref_actor_grad

    # TODO (chongyi zheng): delete this block
    # def update_critic(self, critic_loss, logger, step, ref_critic_grad=None):
    #     # Optimize the critic
    #     logger.log('train_critic/loss', critic_loss, step)
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #
    #     self._project_grad(self.critic.common_parameters(), ref_critic_grad)
    #
    #     self.critic_optimizer.step()

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

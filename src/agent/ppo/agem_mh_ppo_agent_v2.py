import torch
from itertools import chain

from agent.ppo.mh_ppo_agent import MultiHeadPpoMlpAgent
from agent.ppo.agem_ppo_agent import AgemPpoMlpAgent


class AgemMultiHeadPpoMlpAgentV2(MultiHeadPpoMlpAgent, AgemPpoMlpAgent):
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
                 agem_memory_budget=3072,
                 agem_ref_grad_batch_size=1024,
                 ):
        MultiHeadPpoMlpAgent.__init__(self, obs_shape, action_shape, device, hidden_dim, discount, clip_param,
                                      ppo_epoch, critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm,
                                      use_clipped_critic_loss, num_batch)

        AgemPpoMlpAgent.__init__(self, obs_shape, action_shape, device, hidden_dim, discount, clip_param,
                                 ppo_epoch, critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm,
                                 use_clipped_critic_loss, num_batch, agem_memory_budget, agem_ref_grad_batch_size)

    def _compute_ref_grad(self):
        if not self.agem_memories:
            return None

        ref_grad = []
        for task_id, memory in enumerate(self.agem_memories.values()):
            advantages = memory.returns[:-1] - memory.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)
            data_generator = memory.feed_forward_generator(
                advantages, mini_batch_size=self.agem_ref_grad_batch_size // self.agem_task_count)

            # (chongyi zheng): we only use one batch of rollouts to compute ref_grad
            obs_batch, actions_batch, value_preds_batch, \
            return_batch, old_log_pis, adv_targets = next(data_generator)

            actor_loss, entropy = self.compute_actor_loss(
                obs_batch, actions_batch, old_log_pis, adv_targets, head_idx=task_id)
            critic_loss = self.compute_critic_loss(
                obs_batch, value_preds_batch, return_batch, head_idx=task_id)
            loss = actor_loss + self.critic_loss_coef * critic_loss - \
                   self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()

            # compute reference gradient
            single_ref_grad = []
            for param in chain(self.actor.common_parameters(),
                               self.critic.common_parameters()):
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
        for param in chain(self.actor.common_parameters(),
                           self.critic.common_parameters()):
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
            for param in chain(self.actor.common_parameters(),
                               self.critic.common_parameters()):
                if param.requires_grad:
                    num_param = param.numel()  # number of parameters in [p]
                    param.grad.copy_(proj_grad[idx:idx + num_param].reshape(param.shape))
                    idx += num_param

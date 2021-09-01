import torch
import torch.nn.functional as F

from agent.sac import AgemContinualActorCriticMultiHeadSacMlpAgent


class AgemContinualActorCriticGradNormRegCriticMultiHeadSacMlpAgent(
    AgemContinualActorCriticMultiHeadSacMlpAgent):
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
                 critic_grad_norm_reg_coeff=1.0,
                 ):
        AgemContinualActorCriticMultiHeadSacMlpAgent.__init__(self, obs_shape, action_shape, action_range, device,
                                                              actor_hidden_dim, critic_hidden_dim, discount,
                                                              init_temperature, alpha_lr, actor_lr, actor_log_std_min,
                                                              actor_log_std_max, actor_update_freq, critic_lr,
                                                              critic_tau, critic_target_update_freq, batch_size,
                                                              agem_memory_budget, agem_ref_grad_batch_size)

        self.critic_grad_norm_reg_coeff = critic_grad_norm_reg_coeff

    ### (cyzheng): we use ppo style critic loss for AGEM gradient projection here.

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done, **kwargs):
        with torch.no_grad():
            _, next_policy_action, next_log_pi, _ = self.actor(next_obs, **kwargs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_policy_action, **kwargs)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * next_log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, **kwargs)

        # gradient norm regularization
        _, policy_action, _, _ = self.actor(obs, **kwargs)
        reg_Q1, reg_Q2 = self.critic(obs, policy_action, **kwargs)

        # (cyzheng) reference: equation 10 in http://arxiv.org/abs/2103.08050.
        # (cyzheng): create graph for second order derivatives
        reg_Q1_grads = torch.autograd.grad(
            reg_Q1.sum(), policy_action, create_graph=True)[0]
        reg_Q2_grads = torch.autograd.grad(
            reg_Q2.sum(), policy_action, create_graph=True)[0]
        grad1_norm = torch.sum(torch.square(reg_Q1_grads), dim=-1)
        grad2_norm = torch.sum(torch.square(reg_Q2_grads), dim=-1)
        grad_norm_reg = torch.mean(grad1_norm + grad2_norm)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + \
                      self.critic_grad_norm_reg_coeff * grad_norm_reg

        return critic_loss

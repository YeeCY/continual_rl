"""Implementation of DDPG."""

import numpy as np
import torch
import torch.nn.functional as F

import utils
from behavioral_cloning import BehavioralCloning
from networks import DiagGaussianPolicy, OffsetCritic


class FBRC:
    """Class performing BRAC training."""

    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 hidden_dims=(256, 256, 256),
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 discount=0.99,
                 tau=0.005,
                 fisher_coeff=1.0,
                 reward_bonus=5.0):
        """Creates networks."""
        assert len(observation_space.shape) == 1
        state_dim = observation_space.shape[0]

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.hidden_dims = hidden_dims
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.discount = discount
        self.tau = tau
        self.fisher_coeff = fisher_coeff
        self.reward_bonus = reward_bonus

        self.actor = DiagGaussianPolicy(
            state_dim, action_space, hidden_dims=hidden_dims
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        # self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
        # self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)
        self.log_alpha = torch.tensor(np.log(1.0)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.action_space.shape)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # (cyzheng): mixture of diagonal Gaussians for behavioral cloner
        self.bc = BehavioralCloning(
            observation_space, action_space, device, mixture=True)

        action_dim = action_space.shape[0]
        self.critic = OffsetCritic(
            self.bc, state_dim, action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_target = OffsetCritic(
            self.bc, state_dim, action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_target = Critic(state_dim, action_dim, hidden_dims=hidden_dims)
        # critic.soft_update(self.critic, self.critic_target, tau=1.0)

        # self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    # def dist_critic(self, states, actions, target=False, stop_gradient=False):
    #     if target:
    #         q1, q2 = self.critic_target(states, actions)
    #     else:
    #         q1, q2 = self.critic(states, actions)
    #     log_probs = self.bc.policy.log_probs(states, actions)
    #     if stop_gradient:  # (cyzheng): only stop gradients from behavior policy to actor
    #         log_probs = tf.stop_gradient(log_probs)
    #     return q1 + log_probs, q2 + log_probs

    def act(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.Tensor(states).to(self.device)

        with torch.no_grad():
            actions = self.actor(states, sample=False)

            assert actions.ndim == 1

        return utils.to_np(actions)

    def update_critic(self, states, actions, next_states, rewards, not_dones):
        """Updates critic parameters.

        Args:
          states: Batch of states.
          actions: Batch of actions.
          next_states: Batch of next states.
          rewards: Batch of rewards.
          not_dones: Batch of masks indicating the end of the episodes.

        Returns:
          Dictionary with information to track.
        """
        # (cyzheng): use entropy augmented target value as SAC which was not used in the official
        # FisherBRC implementation
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states, sample=True, with_log_probs=True)
            _, _, next_target_q1, next_target_q2 = self.critic_target(
                next_states, next_actions)
            target_v = torch.min(
                next_target_q1, next_target_q2) - self.alpha.detach() * next_log_probs
            target_q = rewards + self.discount * not_dones * target_v

        # next_actions = self.actor(next_states, sample=True)
        # policy_actions = self.actor(states, sample=True)

        # next_target_q1, next_target_q2 = self.dist_critic(
        #     next_states, next_actions, target=True)
        # target_q = rewards + self.discount * discounts * tf.minimum(
        #     next_target_q1, next_target_q2)

        # critic_variables = self.critic.trainable_variables

        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch(critic_variables)
        #     q1, q2 = self.dist_critic(states, actions, stop_gradient=True)  # (cyzheng): only stop gradients from behavior policy to actor, but why?
        #     with tf.GradientTape(
        #             watch_accessed_variables=False, persistent=True) as tape2:
        #         tape2.watch([policy_actions])
        #
        #         q1_reg, q2_reg = self.critic(states, policy_actions)
        #
        #     q1_grads = tape2.gradient(q1_reg, policy_actions)
        #     q2_grads = tape2.gradient(q2_reg, policy_actions)
        #
        #     q1_grad_norm = tf.reduce_sum(tf.square(q1_grads), axis=-1)
        #     q2_grad_norm = tf.reduce_sum(tf.square(q2_grads), axis=-1)
        #
        #     del tape2
        #
        #     q_reg = tf.reduce_mean(q1_grad_norm + q2_grad_norm)
        #
        #     critic_loss = tf.losses.mean_squared_error(target_q, q1) + \
        #                   tf.losses.mean_squared_error(target_q, q2) + self.fisher_coeff * q_reg
        #
        # critic_grads = tape.gradient(critic_loss, critic_variables)
        #
        # self.critic_optimizer.apply_gradients(zip(critic_grads, critic_variables))
        #
        # critic.soft_update(self.critic, self.critic_target, tau=self.tau)

        _, _, q1, q2 = self.critic(states, actions, detach_behavioral_cloner=True)

        policy_actions = self.actor(states, sample=True)
        if not policy_actions.requires_grad:
            policy_actions.requires_grad = True
        o1, o2, _, _ = self.critic(states, policy_actions)
        # (cyzheng): create graph for second order derivatives
        o1_grads = torch.autograd.grad(
            o1.sum(), policy_actions, create_graph=True)[0]
        o2_grads = torch.autograd.grad(
            o2.sum(), policy_actions, create_graph=True)[0]
        o1_grad_norm = torch.sum(torch.square(o1_grads), dim=-1)
        o2_grad_norm = torch.sum(torch.square(o2_grads), dim=-1)
        o_reg = torch.mean(o1_grad_norm + o2_grad_norm)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) + \
                      self.fisher_coeff * o_reg

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        utils.soft_update_params(self.critic, self.critic_target,
                                 self.tau)

        return {
            'q1': q1.mean(),
            'q2': q2.mean(),
            'critic_loss': critic_loss,
            'o1_grad': o1_grad_norm.mean(),
            'o2_grad': o2_grad_norm.mean(),
        }

    def update_actor_and_alpha(self, states):
        """Updates actor parameters and alpha.

        Args:
          states: A batch of states.

        Returns:
          Actor loss.
        """
        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch(self.actor.trainable_variables)
        #     actions, log_probs = self.actor(states, sample=True, with_log_probs=True)
        #     q1, q2 = self.dist_critic(states, actions)
        #     q = tf.minimum(q1, q2)
        #     actor_loss = tf.reduce_mean(self.alpha * log_probs - q)
        # TODO (cyzheng): what should we do if actions aren't sampled using reparameterization trick?
        actions, log_probs = self.actor(states, sample=True, with_log_probs=True)
        _, _, q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_probs - q).mean()

        # actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # self.actor_optimizer.apply_gradients(
        #     zip(actor_grads, self.actor.trainable_variables))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch([self.log_alpha])
        #     alpha_loss = tf.reduce_mean(self.alpha *
        #                                 (-log_probs - self.target_entropy))
        alpha_loss = (self.alpha * (-log_probs - self.target_entropy).detach()).mean()

        # alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        # self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            'actor_loss': actor_loss,
            'alpha': self.alpha,
            'alpha_loss': alpha_loss
        }

    def update(self, states, actions, rewards, not_dones, next_states):
        """Performs a single training step for critic and actor."""

        # states, actions, rewards, discounts, next_states = next(dataset_iter)
        rewards = rewards + self.reward_bonus  # TODO (cyzheng): why we add a reward bonus here?

        critic_dict = self.update_critic(states, actions, next_states, rewards,
                                         not_dones)

        actor_dict = self.update_actor_and_alpha(states)

        return {**actor_dict, **critic_dict}

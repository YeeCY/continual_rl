"""Behavioral Clonning training."""

import torch
import numpy as np

from networks import MixtureGaussianPolicy, DiagGaussianPolicy


class BehavioralCloning:
    """Training class for behavioral clonning."""

    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 mixture=False,
                 piecewise_lrs=(1e-3, 1e-4, 1e-5),
                 piecewise_lr_boundaries=(800000, 900000)):
        assert len(observation_space.shape) == 1
        state_dim = observation_space.shape[0]

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        if mixture:
            self.policy = MixtureGaussianPolicy(state_dim, action_space)
        else:
            self.policy = DiagGaussianPolicy(state_dim, action_space)

        assert len(piecewise_lrs) == len(piecewise_lr_boundaries) + 1
        self.piecewise_lrs = piecewise_lrs
        self.piecewise_lr_boundaries = piecewise_lr_boundaries
        # boundaries = [800_000, 900_000]
        # values = [1e-3, 1e-4, 1e-5]
        # learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #     boundaries, values)

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=piecewise_lrs[0])

        # self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
        # self.alpha_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=learning_rate_fn)
        self.log_alpha = torch.tensor(np.log(1.0)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.action_space.shape)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=piecewise_lrs[0])

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def act(self, states):
        return self.policy(states, sample=False)

    def update_learning_rate(self, iters):
        lr = self.piecewise_lrs[0]
        for idx, boundary in enumerate(self.piecewise_lr_boundaries):
            if iters > boundary:
                lr = self.piecewise_lrs[idx + 1]
                break

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, states, actions):
        """Performs a single training step.

        Args:
          states: Batch of states.
          actions: Batch of actions.

        Returns:
          Dictionary with losses to track.
        """
        # states, actions, _, _, _ = next(dataset_iter)

        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch(self.policy.trainable_variables)
        #     log_probs, entropy = self.policy.log_probs(
        #         states, actions, with_entropy=True)
        #
        #     loss = -tf.reduce_mean(self.alpha * entropy + log_probs)
        #
        # grads = tape.gradient(loss, self.policy.trainable_variables)
        #
        # self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        log_probs, entropy = self.policy.log_probs(states, actions, with_entropy=True)
        loss = -torch.mean(self.alpha * entropy + log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch([self.log_alpha])
        #     alpha_loss = tf.reduce_mean(self.alpha * (entropy - self.target_entropy))
        #
        # alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        # self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        alpha_loss = (self.alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            'bc_actor_loss': loss,
            'bc_alpha': self.alpha,
            'bc_alpha_loss': alpha_loss,
            'bc_log_probs': log_probs.mean(),
            'bc_entropy': entropy.mean(),
        }

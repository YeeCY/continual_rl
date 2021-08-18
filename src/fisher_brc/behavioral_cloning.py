"""Behavioral Clonning training."""

import tensorflow as tf
from tf_agents.specs.tensor_spec import BoundedTensorSpec
from tf_agents.specs.tensor_spec import TensorSpec

# from fisher_brc import policies
import policies


class BehavioralCloning:
    """Training class for behavioral clonning."""

    def __init__(self,
                 observation_space,
                 action_space,
                 mixture=False):
        assert len(observation_space.shape) == 1
        state_dim = observation_space.shape[0]

        self.action_space = action_space
        if mixture:
            self.policy = policies.MixtureGuassianPolicy(state_dim, action_space)
        else:
            self.policy = policies.DiagGuassianPolicy(state_dim, action_space)

        boundaries = [800_000, 900_000]
        values = [1e-3, 1e-4, 1e-5]
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_fn)

        self.target_entropy = -action_space.shape[0]

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @tf.function
    def update_step(self, dataset_iter):
        """Performs a single training step.

        Args:
          dataset_iter: Iterator over dataset samples.

        Returns:
          Dictionary with losses to track.
        """
        states, actions, _, _, _ = next(dataset_iter)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy.trainable_variables)
            log_probs, entropy = self.policy.log_probs(
                states, actions, with_entropy=True)

            loss = -tf.reduce_mean(self.alpha * entropy + log_probs)

        grads = tape.gradient(loss, self.policy.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([self.log_alpha])
            alpha_loss = tf.reduce_mean(self.alpha * (entropy - self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        return {
            'bc_actor_loss': loss,
            'bc_alpha': self.alpha,
            'bc_alpha_loss': alpha_loss,
            'bc_log_probs': tf.reduce_mean(log_probs),
            'bc_entropy': tf.reduce_mean(entropy)
        }

    @tf.function
    def act(self, states):
        return self.policy(states, sample=False)

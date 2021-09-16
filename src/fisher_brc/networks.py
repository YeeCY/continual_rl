"""Implementation of twin_sac, a mix of TD3 (https://arxiv.org/abs/1802.09477) and SAC (https://arxiv.org/abs/1801.01290, https://arxiv.org/abs/1812.05905).

Overall structure and hyperparameters are taken from TD3. However, the algorithm
itself represents a version of SAC.
"""

import torch
from torch import nn
from torch.distributions import Normal, Independent, MixtureSameFamily, Categorical, \
    TransformedDistribution, AffineTransform, TanhTransform

from utils import weight_init
from utils import gaussian_logprob, squash

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class BasePolicy(nn.Module):
    """Base class for policies."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_space,
                 hidden_dims=(256, 256),
                 eps=1e-6):
        """Creates an actor.

        Args:
          state_dim: State size.
          action_dim: Actiom size.
          action_space: Action space.
          hidden_dims: List of hidden dimensions.
          eps: Epsilon for numerical stability.
        """
        super().__init__()

        # relu_gain = tf.math.sqrt(2.0)
        # relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
        # near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.trunk = nn.Sequential(*layers)

        # inputs = tf.keras.Input(shape=(state_dim,))
        # outputs = tf.keras.Sequential(
        #     layers + [tf.keras.layers.Dense(
        #         action_dim, kernel_initializer=near_zero_orthogonal)]
        # )(inputs)
        # self.trunk = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, action_dim)
        # )

        self.apply(weight_init)

        # self.trunk = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.action_space = action_space
        self.action_mean = (action_space.high[0] + action_space.low[0]) / 2.0
        self.action_scale = (action_space.high[0] - action_space.low[0]) / 2.0
        self.eps = eps


class MixtureGaussianPolicy(BasePolicy):
    """Gaussian policy with TanH squashing."""

    def __init__(self,
                 state_dim,
                 action_space,
                 hidden_dims=(256, 256),
                 num_components=5,
                 log_std_min=LOG_STD_MIN,
                 log_std_max=LOG_STD_MAX):
        super().__init__(
            state_dim,
            num_components * action_space.shape[0] * 3,
            action_space,
            hidden_dims=hidden_dims)
        self._num_components = num_components
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    def _get_dist_and_mode(self, states, stddev=1.0):
        """Returns a tf.Distribution for given states modes of this distribution.

        Args:
          states: Batch of states.
          stddev: Standard deviation of sampling distribution.
        """
        # out = self.trunk(states)
        # logits, mu, log_std = tf.split(out, num_or_size_splits=3, axis=1)
        logits, mu, log_std = self.trunk(states).chunk(3, dim=-1)

        # log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # std = tf.exp(log_std)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        std = torch.exp(log_std)

        shape = [std.shape[0], -1, self._num_components]
        # logits = tf.reshape(logits, shape)
        # mu = tf.reshape(mu, shape)
        # std = tf.reshape(std, shape)
        logits = logits.view(*shape)
        mu = mu.view(*shape)
        std = std.view(*shape)

        # components_distribution = tfd.TransformedDistribution(
        #     tfd.Normal(loc=mu, scale=std),
        #     tfp.bijectors.Chain([
        #         tfp.bijectors.Shift(shift=self.action_mean),
        #         tfp.bijectors.Scale(scale=self.action_scale),
        #         tfp.bijectors.Tanh(),
        #     ]))
        component_distribution = TransformedDistribution(
            Normal(loc=mu, scale=std * stddev),
            [AffineTransform(loc=self.action_mean, scale=self.action_scale),
             TanhTransform()]
        )

        # distribution = tfd.MixtureSameFamily(
        #     mixture_distribution=tfd.Categorical(logits=logits),
        #     components_distribution=components_distribution)
        distribution = MixtureSameFamily(
            mixture_distribution=Categorical(logits=logits),
            component_distribution=component_distribution
        )

        # return tfd.Independent(distribution)
        return Independent(distribution, 1)

    def forward(self, states, sample=False, with_log_probs=False):
        """Computes actions for given inputs.

        Args:
          states: Batch of states.
          sample: Whether to sample actions.
          with_log_probs: Whether to return log probability of sampled actions.

        Returns:
          Sampled actions.
        """
        if sample:
            dist = self._get_dist_and_mode(states)
        else:
            dist = self._get_dist_and_mode(states, stddev=0.0)
        # TODO (chongyi zheng): want to use rsample(), but it is not implemented
        #  for MixtureSameFamily distribution
        actions = dist.sample()
        # (cyzheng): clip actions with epsilon to avoid nan logprobs.
        actions = torch.clamp(actions,
                              self.action_space.low[0] + self.eps,
                              self.action_space.high[0] - self.eps)

        if with_log_probs:
            return actions, dist.log_prob(actions)
        else:
            return actions

    def log_probs(self, states, actions, with_entropy=False):
        # actions = tf.clip_by_value(actions, self.action_spec.minimum + self.eps,
        #                            self.action_spec.maximum - self.eps)
        actions = torch.clamp(actions, self.action_space.low[0] + self.eps,
                              self.action_space.high[0] - self.eps)
        dist = self._get_dist_and_mode(states)

        # TODO (chongyi zheng): want to use rsample(), but it is not implemented
        #  for MixtureSameFamily distribution
        sampled_actions = dist.sample()
        # sampled_actions = tf.clip_by_value(sampled_actions,
        #                                    self.action_spec.minimum + self.eps,
        #                                    self.action_spec.maximum - self.eps)
        sampled_actions = torch.clamp(sampled_actions,
                                      self.action_space.low[0] + self.eps,
                                      self.action_space.high[0] - self.eps)

        if with_entropy:
            return dist.log_prob(actions), -dist.log_prob(sampled_actions)
        else:
            return dist.log_prob(actions)


class DiagGaussianPolicy(BasePolicy):
    """Gaussian policy with TanH squashing."""

    def __init__(self,
                 state_dim,
                 action_space,
                 hidden_dims=(256, 256),
                 log_std_min=LOG_STD_MIN,
                 log_std_max=LOG_STD_MAX):
        super().__init__(state_dim, action_space.shape[0] * 2, action_space,
                         hidden_dims=hidden_dims)

        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    # def _get_dist_and_mode(self, states, stddev=1.0):
    #     """Returns a tf.Distribution for given states modes of this distribution.
    #
    #     Args:
    #       states: Batch of states.
    #       stddev: Standard deviation of sampling distribution.
    #     """
    #     # out = self.trunk(states)
    #     # mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
    #     mu, log_std = self.trunk(states).chunk(2, dim=-1)
    #
    #     # log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    #     # std = tf.exp(log_std)
    #     log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
    #     std = torch.exp(log_std)
    #
    #     # dist = tfd.TransformedDistribution(
    #     #     tfd.MultivariateNormalDiag(loc=mu, scale_diag=std * stddev),
    #     #     tfp.bijectors.Chain([
    #     #         tfp.bijectors.Shift(shift=self.action_mean),
    #     #         tfp.bijectors.Scale(scale=self.action_scale),
    #     #         tfp.bijectors.Tanh(),
    #     #     ]))
    #     distribution = TransformedDistribution(
    #         Independent(Normal(loc=mu, scale=std * stddev), 1),
    #         [AffineTransform(loc=self.action_mean, scale=self.action_scale),
    #          TanhTransform()]
    #     )
    #
    #     return distribution

    def forward(self, states, sample=False, with_log_probs=False):
        """Computes actions for given inputs.

        Args:
          states: Batch of states.
          sample: Whether to sample actions.
          with_log_probs: Whether to return log probability of sampled actions.

        Returns:
          Sampled actions.
        """
        # if sample:
        #     dist = self._get_dist_and_mode(states)
        # else:
        #     dist = self._get_dist_and_mode(states, stddev=0.0)
        # actions = dist.rsample()
        # # (cyzheng): clip actions with epsilon to avoid nan logprobs.
        # actions = torch.clamp(actions,
        #                       self.action_space.low[0] + self.eps,
        #                       self.action_space.high[0] - self.eps)

        # TODO (cyzheng): Does SAC style distribution fix NAN bugs?
        mu, log_std = self.trunk(states).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self._log_std_min + 0.5 * (
                self._log_std_max - self._log_std_min
        ) * (log_std + 1)

        if sample:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if with_log_probs:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        if log_pi is not None:
            log_pi = torch.squeeze(log_pi, dim=-1)

        return mu, pi, log_pi

    def log_probs(self, states, actions, with_entropy=False):
        # # actions = tf.clip_by_value(actions, self.action_spec.minimum + self.eps,
        # #                            self.action_spec.maximum - self.eps)
        # actions = torch.clamp(actions, self.action_space.low[0] + self.eps,
        #                       self.action_space.high[0] - self.eps)
        # dist = self._get_dist_and_mode(states)
        #
        # sampled_actions = dist.rsample()
        # # sampled_actions = tf.clip_by_value(sampled_actions,
        # #                                    self.action_spec.minimum + self.eps,
        # #                                    self.action_spec.maximum - self.eps)
        # sampled_actions = torch.clamp(sampled_actions,
        #                               self.action_space.low[0] + self.eps,
        #                               self.action_space.high[0] - self.eps)
        #
        # if with_entropy:
        #     return dist.log_prob(actions), -dist.log_prob(sampled_actions)
        # else:
        #     return dist.log_prob(actions)

        # TODO (cyzheng): The following block hasn't been debugged
        mu, log_std = self.trunk(states).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        pi = (actions - mu) / (std + 1e-6)
        log_pi = gaussian_logprob(pi, log_std)

        mu, _, log_pi = squash(mu, None, log_pi)
        log_pi = torch.squeeze(log_pi, dim=-1)

        if with_entropy:
            noise = torch.randn_like(mu)
            sampled_pi = mu + noise * std
            sampled_log_pi = gaussian_logprob(sampled_pi, log_std)

            mu, _, sampled_log_pi = squash(mu, None, sampled_log_pi)
            sampled_log_pi = torch.squeeze(sampled_log_pi, dim=-1)
            entropy = -sampled_log_pi
        else:
            entropy = None

        return log_pi, entropy


class OffsetNet(nn.Module):
    """A critic offset network."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dims=(256, 256)):
        """Creates a neural net.

        Args:
          state_dim: State size.
          action_dim: Action size.
          hidden_dims: List of hidden dimensions.
        """
        super().__init__()
        # relu_gain = tf.math.sqrt(2.0)
        # relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
        # near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)

        # inputs = tf.keras.Input(shape=(state_dim + action_dim,))

        # layers = []
        # for hidden_dim in hidden_dims:
        #     layers.append(
        #         tf.keras.layers.Dense(
        #             hidden_dim,
        #             activation=tf.nn.relu,
        #             kernel_initializer=relu_orthogonal))
        # outputs = tf.keras.Sequential(
        #     layers + [tf.keras.layers.Dense(
        #         1, kernel_initializer=near_zero_orthogonal)]
        # )(inputs)

        # self.main = tf.keras.Model(inputs=inputs, outputs=outputs)

        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.trunk = nn.Sequential(*layers)

        self.apply(weight_init)

    def forward(self, states, actions):
        """Returns Q-value estimates for given states and actions.

        Args:
          states: A batch of states.
          actions: A batch of actions.

        Returns:
          Two estimates of Q-values.
        """
        x = torch.cat([states, actions], dim=-1)
        return torch.squeeze(self.trunk(x), dim=-1)


class OffsetCritic(nn.Module):
    """A critic network that estimates a dual Q-function."""

    def __init__(self,
                 behavioral_cloner,
                 state_dim,
                 action_dim,
                 hidden_dims=(256, 256)):
        """Creates nets.

        Args:
          behavioral_cloner: Behavioral cloning network.
          state_dim: State size.
          action_dim: Action size.
          hidden_dims: List of hidden dimensions.
        """
        super().__init__()

        self.behavioral_cloner = behavioral_cloner

        self.offset1 = OffsetNet(state_dim, action_dim, hidden_dims=hidden_dims)
        self.offset2 = OffsetNet(state_dim, action_dim, hidden_dims=hidden_dims)

    def forward(self, states, actions, detach_behavioral_cloner=False):
        """Returns Q-value estimates for given states and actions.

        Args:
          states: A batch of states.
          actions: A batch of actions.
          detach_behavioral_cloner: Stop gradients from behavioral cloning network.

        Returns:
          Two estimates of Q-values.
        """
        o1 = self.offset1(states, actions)
        o2 = self.offset2(states, actions)

        log_probs = self.behavioral_cloner.policy.log_probs(states, actions)
        if detach_behavioral_cloner:
            log_probs = log_probs.detach()

        return o1, o2, o1 + log_probs, o2 + log_probs


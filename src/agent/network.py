import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, \
    TransformedDistribution, AffineTransform, TanhTransform, \
    MixtureSameFamily, Categorical

from itertools import chain
from collections import OrderedDict

from src.agent.encoder import PixelEncoder, DqnEncoder
from src.utils import weight_init, SquashedNormal, gaussian_logprob, squash, DiagGaussian


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        # self.trunk = nn.Sequential(
        #     nn.Linear(obs_dim + action_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )

        # (chongyi zheng): add another layer
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        return self.trunk(obs_action)


class MultiHeadQFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dims, hidden_dim):
        super().__init__()
        assert isinstance(action_dims, list)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dims[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = torch.nn.ModuleList()
        for _ in action_dims:
            self.heads.append(nn.Linear(hidden_dim, 1))

    def forward(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        hidden = self.trunk(obs_action)
        return self.heads[head_idx](hidden)


class MultiInputQFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dims, hidden_dim):
        super().__init__()
        assert isinstance(action_dims, list)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dims[0], hidden_dim),
            nn.ReLU(),
        )

        self.heads = torch.nn.ModuleList()
        for _ in action_dims:
            self.heads.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ))

    def forward(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        hidden = self.trunk(obs_action)
        return self.heads[head_idx](hidden)


class RotFunction(nn.Module):
    """MLP for rotation prediction."""
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, h):
        return self.trunk(h)


class DQNCnn(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim):
        super().__init__()
        assert obs_shape == (4, 84, 84), "invalid observation shape"

        self.encoder = DqnEncoder(obs_shape)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_dim = np.prod(
                self.encoder(torch.zeros(1, *obs_shape)).shape[1:])

        self.trunk = nn.Sequential(
            nn.Linear(flatten_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_shape)
        )

        self.apply(weight_init)

    def forward(self, obs):
        h = self.encoder(obs)
        q_values = self.trunk(h)

        return q_values


class DQNDuelingCnn(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim):
        super().__init__()
        assert obs_shape == (4, 84, 84), "invalid observation shape"

        self.encoder = DqnEncoder(obs_shape)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_dim = np.prod(
                self.encoder(torch.zeros(1, *obs_shape)).shape[1:])

        self.v_trunk = nn.Sequential(
            nn.Linear(flatten_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

        self.adv_trunk = nn.Sequential(
            nn.Linear(flatten_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_shape)
        )

        # self.apply(weight_init)

    def forward(self, obs):
        h = self.encoder(obs)
        values = self.v_trunk(h)
        advantages = self.adv_trunk(h)
        q_values = values.expand_as(advantages) + (
            advantages - advantages.mean(-1, keepdim=True)
        )

        return q_values


class ActorCnn(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with CNN"""
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters):
        super().__init__()

        # TODO (chongyi zheng): delete this block
        # self.encoder = make_encoder(
        #     obs_shape, encoder_feature_dim, num_layers,
        #     num_filters, num_shared_layers
        # )
        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(weight_init)

    # TODO (chongyi zheng): delete this version of 'forward'
    # def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
    #     # detach_encoder allows to stop gradient propogation to encoder
    #     obs = self.encoder(obs, detach=detach_encoder)
    #
    #     mu, log_std = self.trunk(obs).chunk(2, dim=-1)
    #
    #     # constrain log_std inside [log_std_min, log_std_max]
    #     log_std = torch.tanh(log_std)
    #     log_std = self.log_std_min + 0.5 * (
    #             self.log_std_max - self.log_std_min
    #     ) * (log_std + 1)
    #
    #     if compute_pi:
    #         std = log_std.exp()
    #         noise = torch.randn_like(mu)
    #         pi = mu + noise * std
    #     else:
    #         pi = None
    #         entropy = None
    #
    #     if compute_log_pi:
    #         log_pi = gaussian_logprob(noise, log_std)
    #     else:
    #         log_pi = None
    #
    #     mu, pi, log_pi = squash(mu, pi, log_pi)
    #
    #     return mu, pi, log_pi, log_std

    def forward(self, obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)

        return dist


class CriticCnn(nn.Module):
    """Critic network with CNN, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        # TODO (chongyi zheng): delete this block
        # self.encoder = make_encoder(
        #     obs_shape, encoder_feature_dim, num_layers,
        #     num_filters, num_shared_layers
        # )
        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.Q1 = QFunction(encoder_feature_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(encoder_feature_dim, action_shape[0], hidden_dim)
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # (chongyi zheng): propagate critic gradients into encoder's convolutional layers
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2


class SacActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.trunk = nn.Sequential(
        #     nn.Linear(obs_shape[0], hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 2 * action_shape[0])
        # )

        # (chongyi zheng): add another layer
        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def compute_log_probs(self, obs, action):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        pi = (action - mu) / (std + 1e-6)
        log_pi = gaussian_logprob(pi, log_std)

        mu, _, log_pi = squash(mu, None, log_pi)

        return log_pi


class SacTaskOneHotHyperNetMlp(nn.Module):
    def __init__(self, onehot_dim, output_dim, hidden_dim):
        super().__init__()

        self.onehot_dim = onehot_dim

        # self.log_std_min = log_std_min
        # self.log_std_max = log_std_max

        # self.trunk = nn.Sequential(
        #     nn.Linear(obs_shape[0], hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 2 * action_shape[0])
        # )

        # (chongyi zheng): add another layer
        self.trunk = nn.Sequential(
            nn.Linear(onehot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(weight_init)

    def forward(self, task_idx):
        onehot = torch.zeros(self.onehot_dim)
        onehot[task_idx] = 1.0

        return self.trunk(onehot)


class MultiHeadSacActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shapes, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        assert isinstance(action_shapes, list)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.dist_heads = torch.nn.ModuleList()
        for action_shape in action_shapes:
            self.dist_heads.append(
                nn.Linear(hidden_dim, 2 * action_shape[0])
            )

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, recurse=True):
        for elem in self.trunk.named_parameters(prefix='trunk', recurse=recurse):
            yield elem

    def forward(self, obs, head_idx, compute_pi=True, compute_log_pi=True):
        hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](hidden).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def compute_log_probs(self, obs, action, head_idx):
        hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](hidden).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        noise = (action - mu) / (std + 1e-6)
        log_pi = gaussian_logprob(noise, log_std)

        # squash log_pi
        log_pi -= torch.log(torch.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)

        return log_pi


class MultiInputSacActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shapes, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        assert isinstance(action_shapes, list)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
        )

        self.dist_heads = torch.nn.ModuleList()
        for action_shape in action_shapes:
            self.dist_heads.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * action_shape[0]),
            ))

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, recurse=True):
        for elem in self.trunk.named_parameters(prefix='trunk', recurse=recurse):
            yield elem

    def forward(self, obs, head_idx, compute_pi=True, compute_log_pi=True):
        hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](hidden).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def compute_log_probs(self, obs, action, head_idx):
        hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](hidden).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        noise = (action - mu) / (std + 1e-6)
        log_pi = gaussian_logprob(noise, log_std)

        # squash log_pi
        log_pi -= torch.log(torch.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)

        return log_pi


class IndividualSacActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shapes, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        assert isinstance(action_shapes, list)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.dist_heads = torch.nn.ModuleList()
        for action_shape in action_shapes:
            self.dist_heads.append(nn.Sequential(
                nn.Linear(obs_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * action_shape[0])
            ))

        self.apply(weight_init)

    def forward(self, obs, head_idx, compute_pi=True, compute_log_pi=True):
        # hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def compute_log_probs(self, obs, action, head_idx):
        # hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        noise = (action - mu) / (std + 1e-6)
        log_pi = gaussian_logprob(noise, log_std)

        # squash log_pi
        log_pi -= torch.log(torch.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)

        return log_pi


class SacCriticMlp(nn.Module):
    """Critic network with MLP, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.Q1 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.Q2 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.apply(weight_init)

    def forward(self, obs, action, **kwargs):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2


class MultiHeadSacCriticMlp(nn.Module):
    """Critic network with MLP, employes two q-functions."""
    def __init__(self, obs_shape, action_shapes, hidden_dim):
        super().__init__()
        assert isinstance(action_shapes, list)

        action_dims = [action_shape[0] for action_shape in action_shapes]
        self.Q1 = MultiHeadQFunction(obs_shape[0], action_dims, hidden_dim)
        self.Q2 = MultiHeadQFunction(obs_shape[0], action_dims, hidden_dim)

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in chain(self.Q1.trunk.named_parameters(recurse=recurse),
                                 self.Q2.trunk.named_parameters(recurse=recurse)):
            yield param

    def named_common_parameters(self, prefix='', recurse=True):
        for elem in chain(self.Q1.trunk.named_parameters(prefix=prefix, recurse=recurse),
                          self.Q2.trunk.named_parameters(prefix=prefix, recurse=recurse)):
            yield elem

    def forward(self, obs, action, head_idx):
        q1 = self.Q1(obs, action, head_idx)
        q2 = self.Q2(obs, action, head_idx)

        return q1, q2


class MultiInputSacCriticMlp(nn.Module):
    """Critic network with MLP, employes two q-functions."""
    def __init__(self, obs_shape, action_shapes, hidden_dim):
        super().__init__()
        assert isinstance(action_shapes, list)

        action_dims = [action_shape[0] for action_shape in action_shapes]
        self.Q1 = MultiInputQFunction(obs_shape[0], action_dims, hidden_dim)
        self.Q2 = MultiInputQFunction(obs_shape[0], action_dims, hidden_dim)

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in chain(self.Q1.trunk.named_parameters(recurse=recurse),
                                 self.Q2.trunk.named_parameters(recurse=recurse)):
            yield param

    def named_common_parameters(self, prefix='', recurse=True):
        for elem in chain(self.Q1.trunk.named_parameters(prefix=prefix, recurse=recurse),
                          self.Q2.trunk.named_parameters(prefix=prefix, recurse=recurse)):
            yield elem

    def forward(self, obs, action, head_idx):
        q1 = self.Q1(obs, action, head_idx)
        q2 = self.Q2(obs, action, head_idx)

        return q1, q2


class SacOffsetCriticMlp(nn.Module):
    """Critic network with MLP, employes two q-functions."""
    def __init__(self, behavioral_cloning, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.behavioral_cloning = behavioral_cloning

        self.offset1 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.offset2 = QFunction(obs_shape[0], action_shape[0], hidden_dim)

        self.apply(weight_init)

    def forward(self, obs, action, detach_behavioral_cloning=False, **kwargs):
        o1 = self.offset1(obs, action)
        o2 = self.offset2(obs, action)

        log_mu, _ = self.behavioral_cloning.policy.log_probs(obs, action)
        if detach_behavioral_cloning:
            log_mu = log_mu.detach()

        return o1, o2, o1 + log_mu, o2 + log_mu


class MultiHeadSacOffsetCriticMlp(nn.Module):
    """Critic network with MLP, employes two q-functions."""
    def __init__(self, behavioral_cloning, obs_shape, action_shapes, hidden_dim):
        super().__init__()
        assert isinstance(action_shapes, list)

        self.behavioral_cloning = behavioral_cloning

        action_dims = [action_shape[0] for action_shape in action_shapes]
        self.offset1 = MultiHeadQFunction(obs_shape[0], action_dims, hidden_dim)
        self.offset2 = MultiHeadQFunction(obs_shape[0], action_dims, hidden_dim)

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in chain(self.Q1.trunk.named_parameters(recurse=recurse),
                                 self.Q2.trunk.named_parameters(recurse=recurse)):
            yield param

    def named_common_parameters(self, prefix='', recurse=True):
        for elem in chain(self.Q1.trunk.named_parameters(prefix=prefix, recurse=recurse),
                          self.Q2.trunk.named_parameters(prefix=prefix, recurse=recurse)):
            yield elem

    def forward(self, obs, action, head_idx, detach_behavioral_cloning=False, **kwargs):
        o1 = self.offset1(obs, action, head_idx)
        o2 = self.offset2(obs, action, head_idx)

        log_mu, _ = self.behavioral_cloning.policy.log_probs(obs, action)
        if detach_behavioral_cloning:
            log_mu = log_mu.detach()

        return o1, o2, o1 + log_mu, o2 + log_mu


class MixtureGaussianBehavioralCloningMlp(nn.Module):
    """Mixture of Gaussian policy with TanH squashing."""

    def __init__(self,
                 obs_shape,
                 action_shape,
                 log_std_min,
                 log_std_max,
                 hidden_dim,
                 num_components,
                 action_range=(-1.0, 1.0)):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_components = num_components

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_components * action_shape[0] * 3)
        )

        self.apply(weight_init)

        self.action_range = action_range
        self.action_mean = (action_range[0] + action_range[1]) / 2.0
        self.action_scale = (action_range[0] - action_range[1]) / 2.0

    def _get_dist_and_mode(self, states, stddev=1.0):
        """Returns a tf.Distribution for given states modes of this distribution.

        Args:
          states: Batch of states.
          stddev: Standard deviation of sampling distribution.
        """
        logits, mu, log_std = self.trunk(states).chunk(3, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        shape = [std.shape[0], 1, -1, self.num_components]
        logits = logits.view(*shape)
        mu = mu.view(*shape)
        std = std.view(*shape)

        component_distribution = TransformedDistribution(
            Normal(loc=mu, scale=std * stddev),
            [AffineTransform(loc=self.action_mean, scale=self.action_scale),
             TanhTransform()]
        )

        distribution = MixtureSameFamily(
            mixture_distribution=Categorical(logits=logits),
            component_distribution=component_distribution
        )

        return Independent(distribution, 1)

    def forward(self, obs, sample=False, with_log_probs=False, **kwargs):
        if sample:
            dist = self._get_dist_and_mode(obs)
        else:
            dist = self._get_dist_and_mode(obs, stddev=0.0)
        # TODO (chongyi zheng): want to use rsample(), but it is not implemented
        #  for MixtureSameFamily distribution
        action = dist.sample()
        action = torch.clamp(action,
                             self.action_range[0] + 1e-6,
                             self.action_range[1] - 1e-6)

        if with_log_probs:
            return action, dist.log_prob(action)
        else:
            return action, None

    def log_probs(self, obs, action, with_entropy=False, **kwargs):
        action = torch.clamp(action,
                             self.action_range[0] + 1e-6,
                             self.action_range[1] - 1e-6)

        dist = self._get_dist_and_mode(obs)

        # TODO (chongyi zheng): want to use rsample(), but it is not implemented
        #  for MixtureSameFamily distribution
        sampled_action = dist.sample()
        sampled_action = torch.clamp(sampled_action,
                                     self.action_range[0] + 1e-6,
                                     self.action_range[1] - 1e-6)

        if with_entropy:
            return torch.unsqueeze(dist.log_prob(action), dim=-1), \
                   torch.unsqueeze(-dist.log_prob(sampled_action), dim=-1)
        else:
            return torch.unsqueeze(dist.log_prob(action), dim=-1), None


class MultiHeadGaussianBehavioralCloningMlp(nn.Module):
    """Multi-headed Gaussian policy with TanH squashing."""
    def __init__(self,
                 obs_shape,
                 action_shapes,
                 log_std_min,
                 log_std_max,
                 hidden_dim):
        super().__init__()
        assert isinstance(action_shapes, list)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.dist_heads = torch.nn.ModuleList()
        for action_shape in action_shapes:
            self.dist_heads.append(
                nn.Linear(hidden_dim, 2 * action_shape[0])
            )

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, recurse=True):
        for elem in self.trunk.named_parameters(prefix='trunk', recurse=recurse):
            yield elem

    def forward(self, obs, head_idx, sample=False, with_log_probs=False,
                **kwargs):
        hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](hidden).chunk(2, dim=-1)

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

        if sample:
            action = pi
        else:
            action = mu

        return action, log_pi

    def log_probs(self, obs, action, head_idx, with_entropy=False, **kwargs):
        hidden = self.trunk(obs)
        mu, log_std = self.dist_heads[head_idx](hidden).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        noise = (action - mu) / (std + 1e-6)
        log_pi = gaussian_logprob(noise, log_std)

        # squash log_pi
        log_pi -= torch.log(
            torch.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)

        if with_entropy:
            sampled_noise = torch.randn_like(mu)
            sampled_action = mu + sampled_noise * std
            sampled_log_pi = gaussian_logprob(sampled_action, log_std)

            sampled_log_pi -= torch.log(
                torch.relu(1 - sampled_action.pow(2)) + 1e-6).sum(-1, keepdim=True)

            return log_pi, -sampled_log_pi
        else:
            return log_pi, None


class Td3ActorMlp(nn.Module):
    """Adapt from https://github.com/rail-berkeley/rlkit and https://github.com/sfujim/TD3"""
    def __init__(self, obs_shape, action_shape, hidden_dim, action_range):
        super().__init__()

        assert isinstance(action_range, list), "Action range must be a list!"
        self.action_range = action_range

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0]),
            nn.Tanh(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        normalized_out = self.trunk(obs)
        low, high = torch.as_tensor(self.action_ranges[0], device=obs.device), \
                    torch.as_tensor(self.action_ranges[1], device=obs.device)
        out = 0.5 * (normalized_out + 1.0) * (high - low) + low

        return out


class MultiHeadTd3ActorMlp(nn.Module):
    """Adapt from https://github.com/rail-berkeley/rlkit and https://github.com/sfujim/TD3"""
    def __init__(self, obs_shape, action_shapes, hidden_dim, action_ranges):
        super().__init__()

        assert isinstance(action_ranges[0], list), "Action range must be a list!"
        self.action_ranges = action_ranges

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = torch.nn.ModuleList()
        for action_shape in action_shapes:
            self.heads.append(nn.Sequential(
                nn.Linear(hidden_dim, action_shape[0]),
                nn.Tanh(),
            ))

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, recurse=True):
        for elem in self.trunk.named_parameters(prefix='trunk', recurse=recurse):
            yield elem

    def forward(self, obs, head_idx):
        hidden = self.trunk(obs)
        normalized_out = self.heads[head_idx](hidden)

        low, high = torch.as_tensor(self.action_ranges[head_idx][0], device=obs.device), \
                    torch.as_tensor(self.action_ranges[head_idx][1], device=obs.device)
        out = 0.5 * (normalized_out + 1.0) * (high - low) + low

        return out


class MultiInputTd3ActorMlp(nn.Module):
    """Adapt from https://github.com/rail-berkeley/rlkit and https://github.com/sfujim/TD3"""
    def __init__(self, obs_shape, action_shapes, hidden_dim, action_ranges):
        super().__init__()

        assert isinstance(action_ranges[0], list), "Action range must be a list!"
        self.action_ranges = action_ranges

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
        )

        self.heads = torch.nn.ModuleList()
        for action_shape in action_shapes:
            self.heads.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0]),
                nn.Tanh(),
            ))

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, recurse=True):
        for elem in self.trunk.named_parameters(prefix='trunk', recurse=recurse):
            yield elem

    def forward(self, obs, head_idx):
        hidden = self.trunk(obs)
        normalized_out = self.heads[head_idx](hidden)

        low, high = torch.as_tensor(self.action_ranges[head_idx][0], device=obs.device), \
                    torch.as_tensor(self.action_ranges[head_idx][1], device=obs.device)
        out = 0.5 * (normalized_out + 1.0) * (high - low) + low

        return out


class Td3CriticMlp(nn.Module):
    """Adapt from https://github.com/rail-berkeley/rlkit and https://github.com/sfujim/TD3"""
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.q1_trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weight_init)

    def forward(self, obs, action, **kwargs):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_trunk(obs_action)
        q2 = self.q2_trunk(obs_action)

        return q1, q2

    def Q1(self, obs, action, **kwargs):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_trunk(obs_action)

        return q1

    def Q2(self, obs, action, **kwargs):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q2 = self.q2_trunk(obs_action)

        return q2


class MultiHeadTd3CriticMlp(nn.Module):
    """Adapt from https://github.com/rail-berkeley/rlkit and https://github.com/sfujim/TD3"""
    def __init__(self, obs_shape, action_shapes, hidden_dim):
        super().__init__()

        self.q1_trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shapes[0][0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q1_heads = torch.nn.ModuleList()
        for _ in action_shapes:
            self.q1_heads.append(nn.Linear(hidden_dim, 1))

        self.q2_trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shapes[0][0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q2_heads = torch.nn.ModuleList()
        for _ in action_shapes:
            self.q2_heads.append(nn.Linear(hidden_dim, 1))

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in chain(self.q1_trunk.named_parameters(recurse=recurse),
                                 self.q2_trunk.named_parameters(recurse=recurse)):
            yield param

    def named_common_parameters(self, prefix='', recurse=True):
        for elem in chain(self.q1_trunk.named_parameters(prefix=prefix, recurse=recurse),
                          self.q2_trunk.named_parameters(prefix=prefix, recurse=recurse)):
            yield elem

    def forward(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)

        hidden1 = self.q1_trunk(obs_action)
        q1 = self.q1_heads[head_idx](hidden1)
        hidden2 = self.q2_trunk(obs_action)
        q2 = self.q2_heads[head_idx](hidden2)

        return q1, q2

    def Q1(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        hidden1 = self.q1_trunk(obs_action)
        q1 = self.q1_heads[head_idx](hidden1)

        return q1

    def Q2(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        hidden2 = self.q2_trunk(obs_action)
        q2 = self.q2_heads[head_idx](hidden2)

        return q2


class MultiInputTd3CriticMlp(nn.Module):
    """Adapt from https://github.com/rail-berkeley/rlkit and https://github.com/sfujim/TD3"""
    def __init__(self, obs_shape, action_shapes, hidden_dim):
        super().__init__()

        self.q1_trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shapes[0][0], hidden_dim),
            nn.ReLU(),
        )
        self.q1_heads = torch.nn.ModuleList()
        for _ in action_shapes:
            self.q1_heads.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ))

        self.q2_trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shapes[0][0], hidden_dim),
            nn.ReLU(),
        )
        self.q2_heads = torch.nn.ModuleList()
        for _ in action_shapes:
            self.q2_heads.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ))

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in chain(self.q1_trunk.named_parameters(recurse=recurse),
                                 self.q2_trunk.named_parameters(recurse=recurse)):
            yield param

    def named_common_parameters(self, prefix='', recurse=True):
        for elem in chain(self.q1_trunk.named_parameters(prefix=prefix, recurse=recurse),
                          self.q2_trunk.named_parameters(prefix=prefix, recurse=recurse)):
            yield elem

    def forward(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)

        hidden1 = self.q1_trunk(obs_action)
        q1 = self.q1_heads[head_idx](hidden1)
        hidden2 = self.q2_trunk(obs_action)
        q2 = self.q2_heads[head_idx](hidden2)

        return q1, q2

    def Q1(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        hidden1 = self.q1_trunk(obs_action)
        q1 = self.q1_heads[head_idx](hidden1)

        return q1

    def Q2(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        hidden2 = self.q2_trunk(obs_action)
        q2 = self.q2_heads[head_idx](hidden2)

        return q2


class PpoActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.dist = DiagGaussian(hidden_dim, action_shape[0])

        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        hidden = self.trunk(obs)
        dist = self.dist(hidden)

        mu = dist.mode()
        if compute_pi:
            pi = dist.sample()
        else:
            pi = None

        if compute_log_pi:
            log_pi = dist.log_probs(pi)
        else:
            log_pi = None

        return mu, pi, log_pi

    def compute_log_probs(self, obs, action):
        hidden = self.trunk(obs)
        dist = self.dist(hidden)

        log_pi = dist.log_probs(action)
        entropy = dist.entropy().mean()

        return log_pi, entropy


class CmamlPpoActorMlp(nn.Module):
    """Adapt from https://github.com/tristandeleu/pytorch-maml-rl"""
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.layer_sizes = (obs_shape[0], hidden_dim, hidden_dim)
        self.num_layers = len(self.layer_sizes) + 1
        for i in range(1, self.num_layers - 1):
            self.add_module('layer{}'.format(i - 1),
                            nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]))

        self.mu = nn.Linear(hidden_dim, action_shape[0])
        self.logstd = nn.Parameter(torch.Tensor(action_shape[0]))
        self.logstd.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs, params=None, compute_pi=True, compute_log_pi=True, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = obs
        for i in range(1, self.num_layers - 1):
            output = F.linear(output,
                              weight=params['layer{}.weight'.format(i - 1)],
                              bias=params['layer{}.bias'.format(i - 1)])
            output = torch.tanh(output)

        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        std = torch.exp(params['logstd'])
        dist = Independent(Normal(mu, std), 1)

        mu = dist.mean
        if compute_pi:
            pi = dist.sample()
        else:
            pi = None

        if compute_log_pi:
            log_pi = dist.log_prob(pi)
        else:
            log_pi = None

        return mu, pi, log_pi

    def compute_log_probs(self, obs, action, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = obs
        for i in range(1, self.num_layers - 1):
            output = F.linear(output,
                              weight=params['layer{}.weight'.format(i - 1)],
                              bias=params['layer{}.bias'.format(i - 1)])
            output = torch.tanh(output)

        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        std = torch.exp(params['logstd'])
        dist = Independent(Normal(mu, std), 1)

        log_pi = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().mean()

        return log_pi, entropy


class MultiHeadPpoActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shapes, hidden_dim):
        super().__init__()
        assert isinstance(action_shapes, list)

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.dist_heads = torch.nn.ModuleList()
        for action_shape in action_shapes:
            self.dist_heads.append(
                DiagGaussian(hidden_dim, action_shape[0])
            )

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, recurse=True):
        for elem in self.trunk.named_parameters(prefix='trunk', recurse=recurse):
            yield elem

    def forward(self, obs, head_idx, compute_pi=True, compute_log_pi=True):
        hidden = self.trunk(obs)
        dist = self.dist_heads[head_idx](hidden)

        mu = dist.mode()
        if compute_pi:
            pi = dist.sample()
        else:
            pi = None

        if compute_log_pi:
            log_pi = dist.log_probs(pi)
        else:
            log_pi = None

        return mu, pi, log_pi

    def compute_log_probs(self, obs, action, head_idx):
        hidden = self.trunk(obs)
        dist = self.dist_heads[head_idx](hidden)

        log_pi = dist.log_probs(action)
        entropy = dist.entropy().mean()

        return log_pi, entropy


class PpoCriticMlp(nn.Module):
    """PPO critic network with MLP"""
    def __init__(self, obs_shape, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, **kwargs):
        return self.trunk(obs)


class MultiHeadPpoCriticMlp(nn.Module):
    """PPO critic network with MLP"""
    def __init__(self, obs_shape, hidden_dim, head_num):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.heads = torch.nn.ModuleList()
        for _ in range(head_num):
            self.heads.append(
                nn.Linear(hidden_dim, 1)
            )

        self.apply(weight_init)

    def common_parameters(self, recurse=True):
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, prefix='', recurse=True):
        for elem in self.trunk.named_parameters(prefix=prefix, recurse=recurse):
            yield elem

    def forward(self, obs, head_idx):
        hidden = self.trunk(obs)

        return self.heads[head_idx](hidden)


class CURL(nn.Module):
    """Implements CURL, a contrastive learning method"""
    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class FwdFunction(nn.Module):
    """MLP for forward dynamics model."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, h, action):
        h_act = torch.cat([h, action], dim=-1)
        return self.trunk(h_act)


class InvFunction(nn.Module):
    """MLP for inverse dynamics model."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, h, h_next):
        joint_h = torch.cat([h, h_next], dim=-1)
        return self.trunk(joint_h)


class SelfSupervisedCnnInvPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.trunk = nn.Sequential(
            nn.Linear(2 * encoder_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

    def forward(self, obs, next_obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)
        next_h = self.encoder(next_obs, detach=detach_encoder)

        joint_h = torch.cat([h, next_h], dim=-1)
        pred_action = self.trunk(joint_h)

        return pred_action


class SelfSupervisedCnnInvPredictorEnsem(SelfSupervisedCnnInvPredictor):
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters, num_comps):
        super().__init__(obs_shape, action_shape, hidden_dim,
                         encoder_feature_dim, num_layers, num_filters)
        self.num_comps = num_comps

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(2 * encoder_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0])
            )
            trunks.append(trunk)
        self.trunks = nn.ModuleList(trunks)
        self.apply(weight_init)

    def forward(self, obs, next_obs, detach_encoder=False, split_hidden=False):
        """
            split_hidden: split encoder outputs uniformly for each components
        """
        num_samples_each_slice = obs.shape[0] // self.num_comps if split_hidden else None

        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)
        next_h = self.encoder(next_obs, detach=detach_encoder)

        joint_h = torch.cat([h, next_h], dim=-1)
        pred_actions = []
        for idx, trunk in enumerate(self.trunks):
            if split_hidden:
                joint_h_slice = joint_h[idx * num_samples_each_slice:(idx + 1) * num_samples_each_slice]
                pred_action = trunk(joint_h_slice)
            else:
                pred_action = trunk(joint_h)
            pred_actions.append(pred_action)

        pred_actions = torch.cat(pred_actions, dim=0)

        return pred_actions


class SelfSupervisedCnnFwdPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_feature_dim)
        )
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)

        joint_h_act = torch.cat([h, action], dim=-1)
        pred_h_next = self.trunk(joint_h_act)


        return pred_h_next


class SelfSupervisedCnnFwdPredictorEnsem(SelfSupervisedCnnFwdPredictor):
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters, num_comps):
        super().__init__(obs_shape, action_shape, hidden_dim,
                         encoder_feature_dim, num_layers, num_filters)
        self.num_comps = num_comps

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoder_feature_dim)
            )
            trunks.append(trunk)
        self.trunks = nn.ModuleList(trunks)
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False, split_hidden=False):
        """
            split_hidden: split encoder outputs uniformly for each components
        """
        num_samples_each_slice = obs.shape[0] // self.num_comps if split_hidden else None

        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)

        joint_h_act = torch.cat([h, action], dim=-1)
        pred_next_hs = []
        for idx, trunk in enumerate(self.trunks):
            if split_hidden:
                joint_h_act_slice = joint_h_act[idx * num_samples_each_slice:(idx + 1) * num_samples_each_slice]
                pred_next_h = trunk(joint_h_act_slice)
            else:
                pred_next_h = trunk(joint_h_act)
            pred_next_hs.append(pred_next_h)

        pred_next_hs = torch.cat(pred_next_hs, dim=0)

        return pred_next_hs


class SelfSupervisedMlpInvPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2 * obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        joint_obs = torch.cat([obs, next_obs], dim=-1)
        pred_action = self.trunk(joint_obs)

        return pred_action


class SelfSupervisedMlpInvPredictorEnsem(SelfSupervisedMlpInvPredictor):
    def __init__(self, obs_shape, action_shape, hidden_dim, num_comps):
        super().__init__(obs_shape, action_shape, hidden_dim)
        self.num_comps = num_comps

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(2 * obs_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0])
            )
            trunks.append(trunk)
        self.trunks = nn.ModuleList(trunks)
        self.apply(weight_init)

    def forward(self, obs, next_obs, split_input=False):
        """
            split_input: split inputs uniformly for each components
        """
        if split_input:
            assert obs.shape[0] % self.num_comps == 0, 'input is not splitable'

        joint_obs = torch.cat([obs, next_obs], dim=-1)
        if split_input:
            joint_obs = joint_obs.chunk(self.num_comps, dim=0)
        pred_actions = []
        for idx, trunk in enumerate(self.trunks):
            if split_input:
                pred_action = trunk(joint_obs[idx])
            else:
                pred_action = trunk(joint_obs)
            pred_actions.append(pred_action)

        pred_actions = torch.cat(pred_actions, dim=0)

        return pred_actions


class SelfSupervisedMlpFwdPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_shape[0])
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        joint_obs_act = torch.cat([obs, action], dim=-1)
        pred_next_obs = self.trunk(joint_obs_act)

        return pred_next_obs


class SelfSupervisedMlpFwdPredictorEnsem(SelfSupervisedMlpFwdPredictor):
    def __init__(self, obs_shape, action_shape, hidden_dim, num_comps):
        super().__init__(obs_shape, action_shape, hidden_dim)
        self.num_comps = num_comps

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_shape[0])
            )
            trunks.append(trunk)
        self.trunks = nn.ModuleList(trunks)
        self.apply(weight_init)

    def forward(self, obs, action, split_input=False):
        """
            split_input: split inputs uniformly for each components
        """
        if split_input:
            assert obs.shape[0] % self.num_comps == 0, 'input is not splitable'

        joint_obs_act = torch.cat([obs, action], dim=-1)
        if split_input:
            joint_obs_act = joint_obs_act.chunk(self.num_comps, dim=0)
        pred_next_obss = []
        for idx, trunk in enumerate(self.trunks):
            if split_input:
                pred_next_obs = trunk(joint_obs_act[idx])
            else:
                pred_next_obs = trunk(joint_obs_act)
            pred_next_obss.append(pred_next_obs)

        pred_next_obss = torch.cat(pred_next_obss, dim=0)

        return pred_next_obss


class DqnCnnSSInvPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim):
        super().__init__()

        self.encoder = DqnEncoder(obs_shape)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_dim = np.prod(
                self.encoder(torch.zeros(1, *obs_shape)).shape[1:])

        self.trunk = nn.Sequential(
            nn.Linear(2 * flatten_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_shape)
        )

        self.apply(weight_init)

    def forward(self, obs, next_obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)
        next_h = self.encoder(next_obs, detach=detach_encoder)

        joint_h = torch.cat([h, next_h], dim=-1)
        pred_logit = self.trunk(joint_h)

        return pred_logit


class DqnCnnSSInvPredictorEnsem(DqnCnnSSInvPredictor):
    def __init__(self, obs_shape, action_shape, feature_dim, num_comps):
        super().__init__(obs_shape, action_shape, feature_dim)
        self.num_comps = num_comps

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_dim = np.prod(
                self.encoder(torch.zeros(1, *obs_shape)).shape[1:])

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(2 * flatten_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, action_shape)
            )
            trunks.append(trunk)

        self.trunks = nn.ModuleList(trunks)
        self.apply(weight_init)

    def forward(self, obs, next_obs, detach_encoder=False, split_hidden=False):
        """
            split_hidden: split encoder outputs uniformly for each components
        """
        num_samples_each_slice = obs.shape[0] // self.num_comps if split_hidden else None

        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)
        next_h = self.encoder(next_obs, detach=detach_encoder)

        joint_h = torch.cat([h, next_h], dim=-1)
        pred_logits = []
        for idx, trunk in enumerate(self.trunks):
            if split_hidden:
                joint_h_slice = joint_h[idx * num_samples_each_slice:(idx + 1) * num_samples_each_slice]
                pred_logit = trunk(joint_h_slice)
            else:
                pred_logit = trunk(joint_h)
            pred_logits.append(pred_logit)

        pred_logits = torch.cat(pred_logits, dim=0)

        return pred_logits


class DqnCnnSSFwdPredictor(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        self.encoder = DqnEncoder(obs_shape)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_dim = np.prod(
                self.encoder(torch.zeros(1, *obs_shape)).shape[1:])

        self.trunk = nn.Sequential(
            nn.Linear(flatten_dim + 1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, flatten_dim)
        )
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)

        joint_h_act = torch.cat([h, action], dim=-1)
        pred_next_h = self.trunk(joint_h_act)

        return pred_next_h


class DqnCnnSSFwdPredictorEnsem(SelfSupervisedCnnFwdPredictor):
    def __init__(self, obs_shape, feature_dim, num_comps):
        super().__init__(obs_shape, feature_dim)
        self.num_comps = num_comps

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_dim = np.prod(
                self.encoder(torch.zeros(1, *obs_shape)).shape[1:])

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(flatten_dim + 1, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, flatten_dim)
            )
            trunks.append(trunk)
        self.trunks = nn.ModuleList(trunks)
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False, split_hidden=False):
        """
            split_hidden: split encoder outputs uniformly for each components
        """
        num_samples_each_slice = obs.shape[0] // self.num_comps if split_hidden else None

        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)

        joint_h_act = torch.cat([h, action], dim=-1)
        pred_next_hs = []
        for idx, trunk in enumerate(self.trunks):
            if split_hidden:
                joint_h_act_slice = joint_h_act[idx * num_samples_each_slice:(idx + 1) * num_samples_each_slice]
                pred_next_h = trunk(joint_h_act_slice)
            else:
                pred_next_h = trunk(joint_h_act)
            pred_next_hs.append(pred_next_h)

        pred_next_hs = torch.cat(pred_next_hs, dim=0)

        return pred_next_hs

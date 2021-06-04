import numpy as np
import torch
import torch.nn as nn

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

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class MultiHeadQFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dims, hidden_dim):
        super().__init__()
        assert isinstance(action_dims, list)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = torch.nn.ModuleList()
        for action_dim in action_dims:
            self.heads.append(
                nn.Linear(hidden_dim + action_dim, 1)
            )

    def forward(self, obs, action, head_idx):
        assert obs.size(0) == action.size(0)

        hidden = self.trunk(obs)
        hidden_action = torch.cat([hidden, action], dim=-1)
        return self.heads[head_idx](hidden_action)


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

        # self.apply(weight_init)

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

    def forward(self, obs, head_idx, compute_pi=True, compute_log_pi=True):
        hidden = self.trunk(obs)
        mu, log_std, = self.dist_heads[head_idx](hidden).chunk(2, dim=-1)

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


class SacCriticMlp(nn.Module):
    """Critic network with MLP, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.Q1 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.Q2 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.apply(weight_init)

    def forward(self, obs, action):
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
        for name, param in self.trunk.named_parameters(recurse=recurse):
            yield param

    def named_common_parameters(self, prefix='', recurse=True):
        for elem in self.trunk.named_parameters(prefix=prefix, recurse=recurse):
            yield elem

    def forward(self, obs, action, head_idx):
        q1 = self.Q1(obs, action, head_idx)
        q2 = self.Q2(obs, action, head_idx)

        return q1, q2


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

    def forward(self, obs):
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

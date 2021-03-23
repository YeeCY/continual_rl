import torch
import torch.nn as nn

from agent.encoder import PixelEncoder
from utils import weight_init, SquashedNormal


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


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


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
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
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
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

        self.output['mu'] = mu
        self.output['std'] = std

        dist = SquashedNormal(mu, std)

        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
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

        self.outputs = dict()  # log placeholder

    def forward(self, obs, action, detach_encoder=False):
        # (chongyi zheng): propagate critic gradients into encoder's convolutional layers
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1.trunk) == len(self.Q2.trunk)
        for i, (m1, m2) in enumerate(zip(self.Q1.trunk, self.Q2.trunk)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


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
        h_act = torch.cat([h, action], dim=1)
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
        joint_h = torch.cat([h, h_next], dim=1)
        return self.trunk(joint_h)

import numpy as np
import torch
import torch.nn as nn

from agent.encoder import PixelEncoder, DqnEncoder
from utils import weight_init, SquashedNormal, gaussian_logprob, squash


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
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


class DQNCnn(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim):
        super().__init__()
        assert obs_shape == (4, 84, 84), "invalid observation shape"

        # self.preprocess = nn.Sequential(
        #     NormalizeImg()
        # )

        self.encoder = DqnEncoder(obs_shape)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_dim = np.prod(
                self.encoder(torch.zeros(1, *obs_shape)).shape[1:])

        self.trunk = nn.Sequential(
            nn.Linear(flatten_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, action_shape)
        )

        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs):
        h = self.encoder(obs)
        q_values = self.trunk(h)

        self.outputs['q_values'] = q_values

        return q_values

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_dqn/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_dqn/fc{i}', m, step)


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
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1)
        )

        self.adv_trunk = nn.Sequential(
            nn.Linear(flatten_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, action_shape)
        )

        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs):
        h = self.encoder(obs)
        values = self.v_trunk(h)
        advantages = self.adv_trunk(h)
        q_values = values.expand_as(advantages) + (
            advantages - advantages.mean(-1, keepdim=True)
        )

        self.outputs['q_values'] = q_values

        return q_values

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_dueling_dqn/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_dueling_dqn/fc{i}', m, step)


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
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

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

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)

        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


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


class ActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

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

        self.outputs['mu'] = mu
        self.outputs['log_std'] = log_std

        return mu, pi, log_pi, log_std

    # def forward(self, obs):
    #     mu, log_std = self.trunk(obs).chunk(2, dim=-1)
    #
    #     # constrain log_std inside [log_std_min, log_std_max]
    #     log_std = torch.tanh(log_std)
    #     log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
    #     std = log_std.exp()
    #
    #     self.outputs['mu'] = mu
    #     self.outputs['std'] = std
    #
    #     dist = SquashedNormal(mu, std)
    #
    #     return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class CriticMlp(nn.Module):
    """Critic network with MLP, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.Q1 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.Q2 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs, action):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
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
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
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
            nn.Linear(2 * obs_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, h, h_next):
        joint_h = torch.cat([h, h_next], dim=1)
        return self.trunk(joint_h)


class SelfSupervisedCnnInvPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.trunk = nn.Sequential(
            nn.Linear(2 * encoder_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs, next_obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)
        next_h = self.encoder(next_obs, detach=detach_encoder)

        joint_h = torch.cat([h, next_h], dim=-1)
        pred_action = self.trunk(joint_h)

        self.outputs['pred_action'] = pred_action

        return pred_action

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_inv/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_ss_inv/fc{i}', m, step)


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
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
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
            self.outputs[f'pred_action{idx}'] = pred_action
            pred_actions.append(pred_action)

        pred_actions = torch.cat(pred_actions, dim=0)

        return pred_actions

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_inv/{k}_hist', v, step)

        for i, trunk in enumerate(self.trunks):
            for j, m in enumerate(trunk):
                if type(m) == nn.Linear:
                    logger.log_param(f'train_ss_inv/ensem{i}/fc{j}', m, step)


class SelfSupervisedCnnFwdPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = PixelEncoder(obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_feature_dim)
        )
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)

        joint_h_act = torch.cat([h, action], dim=-1)
        pred_h_next = self.trunk(joint_h_act)

        self.outputs['pred_h_next'] = pred_h_next

        return pred_h_next

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_fwd/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_ss_fwd/fc{i}', m, step)


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
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
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
            self.outputs[f'pred_next_h{idx}'] = pred_next_h
            pred_next_hs.append(pred_next_h)

        pred_next_hs = torch.cat(pred_next_hs, dim=0)

        return pred_next_hs

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_fwd/{k}_hist', v, step)

        for i, trunk in enumerate(self.trunks):
            for j, m in enumerate(trunk):
                if type(m) == nn.Linear:
                    logger.log_param(f'train_ss_fwd/ensem{i}/fc{j}', m, step)


class SelfSupervisedMlpInvPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2 * obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs, next_obs):
        joint_obs = torch.cat([obs, next_obs], dim=-1)
        pred_action = self.trunk(joint_obs)

        self.outputs['pred_action'] = pred_action

        return pred_action

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_inv/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_ss_inv/fc{i}', m, step)


class SelfSupervisedMlpInvPredictorEnsem(SelfSupervisedMlpInvPredictor):
    def __init__(self, obs_shape, action_shape, hidden_dim, num_comps):
        super().__init__(obs_shape, action_shape, hidden_dim)
        self.num_comps = num_comps

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(2 * obs_shape[0], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
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
            self.outputs[f'pred_action{idx}'] = pred_action
            pred_actions.append(pred_action)

        pred_actions = torch.cat(pred_actions, dim=0)

        return pred_actions

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_inv/{k}_hist', v, step)

        for i, trunk in enumerate(self.trunks):
            for j, m in enumerate(trunk):
                if type(m) == nn.Linear:
                    logger.log_param(f'train_ss_inv/ensem{i}/fc{j}', m, step)


class SelfSupervisedMlpFwdPredictor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, obs_shape[0])
        )
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs, action):
        joint_obs_act = torch.cat([obs, action], dim=-1)
        pred_next_obs = self.trunk(joint_obs_act)

        self.outputs['pred_next_obs'] = pred_next_obs

        return pred_next_obs

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_fwd/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_ss_fwd/fc{i}', m, step)


class SelfSupervisedMlpFwdPredictorEnsem(SelfSupervisedMlpFwdPredictor):
    def __init__(self, obs_shape, action_shape, hidden_dim, num_comps):
        super().__init__(obs_shape, action_shape, hidden_dim)
        self.num_comps = num_comps

        trunks = [self.trunk]
        for _ in range(self.num_comps - 1):
            trunk = nn.Sequential(
                nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
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
            self.outputs[f'pred_next_obs{idx}'] = pred_next_obs
            pred_next_obss.append(pred_next_obs)

        pred_next_obss = torch.cat(pred_next_obss, dim=0)

        return pred_next_obss

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_fwd/{k}_hist', v, step)

        for i, trunk in enumerate(self.trunks):
            for j, m in enumerate(trunk):
                if type(m) == nn.Linear:
                    logger.log_param(f'train_ss_fwd/ensem{i}/fc{j}', m, step)


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
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, action_shape)
        )

        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs, next_obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)
        next_h = self.encoder(next_obs, detach=detach_encoder)

        joint_h = torch.cat([h, next_h], dim=-1)
        pred_logit = self.trunk(joint_h)

        self.outputs['pred_logit'] = pred_logit

        return pred_logit

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_inv/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_ss_inv/fc{i}', m, step)


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
                nn.ReLU(inplace=True),
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
            self.outputs[f'pred_logit{idx}'] = pred_logit
            pred_logits.append(pred_logit)

        pred_logits = torch.cat(pred_logits, dim=0)

        return pred_logits

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_inv/{k}_hist', v, step)

        for i, trunk in enumerate(self.trunks):
            for j, m in enumerate(trunk):
                if type(m) == nn.Linear:
                    logger.log_param(f'train_ss_inv/ensem{i}/fc{j}', m, step)


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
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, flatten_dim)
        )
        self.apply(weight_init)

        self.outputs = dict()  # log placeholder

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder's convolutional layers
        h = self.encoder(obs, detach=detach_encoder)

        joint_h_act = torch.cat([h, action], dim=-1)
        pred_next_h = self.trunk(joint_h_act)

        self.outputs['pred_next_h'] = pred_next_h

        return pred_next_h

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_fwd/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_ss_fwd/fc{i}', m, step)


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
                nn.ReLU(inplace=True),
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
            self.outputs[f'pred_next_h{idx}'] = pred_next_h
            pred_next_hs.append(pred_next_h)

        pred_next_hs = torch.cat(pred_next_hs, dim=0)

        return pred_next_hs

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_ss_fwd/{k}_hist', v, step)

        for i, trunk in enumerate(self.trunks):
            for j, m in enumerate(trunk):
                if type(m) == nn.Linear:
                    logger.log_param(f'train_ss_fwd/ensem{i}/fc{j}', m, step)

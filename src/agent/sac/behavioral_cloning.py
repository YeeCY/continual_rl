"""Behavioral Clonning training."""

import torch
import numpy as np

import utils
from agent.network import MixtureGaussianBehavioralCloningMlp, \
    MultiHeadGaussianBehavioralCloningMlp


class BehavioralCloning:
    """Training class for behavioral clonning."""

    def __init__(self,
                 obs_shape,
                 action_shapes,
                 device,
                 hidden_dim=256,
                 multi_head=False,
                 mixture_components=10,
                 log_std_min=-10,
                 log_std_max=2,
                 piecewise_lrs=(1e-3, 1e-4, 1e-5),
                 piecewise_lr_boundaries=(80000, 90000)):
        self.obs_shape = obs_shape
        self.action_shapes = action_shapes
        self.device = device
        self.hidden_dim = hidden_dim
        self.mixture_components = mixture_components
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        assert len(piecewise_lrs) == len(piecewise_lr_boundaries) + 1
        self.piecewise_lrs = piecewise_lrs
        self.piecewise_lr_boundaries = piecewise_lr_boundaries

        # mixture of gaussian policy
        if multi_head:
            self.policy = MultiHeadGaussianBehavioralCloningMlp(
                obs_shape, action_shapes, log_std_min, log_std_max,
                hidden_dim).to(device)
        else:
            self.policy = MixtureGaussianBehavioralCloningMlp(
                obs_shape, action_shapes[0], log_std_min, log_std_max,
                hidden_dim, mixture_components).to(device)

        # trainable entropy coefficient
        self.log_alpha = torch.tensor(np.log(1.0)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.action_shapes[0])

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=piecewise_lrs[0])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=piecewise_lrs[0])

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def act(self, obs, sample=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        with torch.no_grad():
            actions = self.policy(obs, sample=sample, **kwargs)

            assert actions.ndim == 1

        return utils.to_np(actions)

    def update_learning_rate(self, step):
        lr = self.piecewise_lrs[0]
        for idx, boundary in enumerate(self.piecewise_lr_boundaries):
            if step > boundary:
                lr = self.piecewise_lrs[idx + 1]
                break

        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, obses, actions, logger, step, **kwargs):
        # update policy
        log_probs, entropy = self.policy.log_probs(obses, actions, with_entropy=True, **kwargs)
        loss = -torch.mean(self.alpha * entropy + log_probs)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        logger.log('train/bc_loss', loss, step)

        # update entropy coefficient
        alpha_loss = (self.alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

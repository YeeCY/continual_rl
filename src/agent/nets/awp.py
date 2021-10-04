import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from src.utils import gaussian_logprob, squash, weight_init


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        # if len(old_w.size()) <= 1:
        #     continue
        # if 'weight' in old_k:
        diff_w = new_w - old_w
        diff_dict[old_k] = old_w.norm() / (diff_w.norm() + 1e-12) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturbSacActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shape, hidden_dim, log_std_min, log_std_max,
                 awp_coeff):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.awp_coeff = awp_coeff

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

        self.proxy_trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.apply(weight_init)

    def reload_weights(self):
        self.proxy_trunk.load_state_dict(self.trunk.state_dict())

    def diff_weights(self):
        return diff_in_weights(self.trunk, self.proxy_trunk)

    def perturb(self, diff):
        add_into_weights(self.trunk, diff, coeff=self.awp_coeff)

    def restore(self, diff):
        add_into_weights(self.trunk, diff, coeff=-self.awp_coeff)

    def main_parameters(self):
        return self.trunk.parameters()

    def main_named_parameters(self):
        return self.trunk.named_parameters()

    def proxy_parameters(self):
        return self.proxy_trunk.parameters()

    def proxy_named_parameters(self):
        return self.proxy_trunk.named_parameters()

    def forward(self, obs, compute_pi=True, compute_log_pi=True, proxy=False, **kwargs):
        # FIXME (cyzheng): kwargs not used
        if proxy:
            mu, log_std = self.proxy_trunk(obs).chunk(2, dim=-1)
        else:
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

    def compute_log_probs(self, obs, action, proxy=False, **kwargs):
        # FIXME (cyzheng): kwargs not used
        if proxy:
            mu, log_std = self.proxy_trunk(obs).chunk(2, dim=-1)
        else:
            mu, log_std = self.trunk(obs).chunk(2, dim=-1)

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


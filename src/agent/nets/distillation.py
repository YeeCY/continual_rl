# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F

# from collections import OrderedDict

from src.utils import weight_init, gaussian_logprob, squash


class SacTaskEmbeddingDistilledActorMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, num_tasks, obs_shape, action_shape, hidden_dim,
                 task_embedding_dim, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + task_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(weight_init)

        # task embeddings
        self.task_embs = nn.ParameterList()
        for _ in range(num_tasks):
            self.task_embs.append(
                nn.Parameter(data=torch.Tensor(task_embedding_dim),
                             requires_grad=True)
            )
            torch.nn.init.normal_(self.task_embs[-1], mean=0., std=1.)

    def construct_input(self, obs, task_idx):
        batch_size = obs.shape[0]
        task_emb = self.task_embs[task_idx]
        task_emb = task_emb.expand(batch_size,
                                   task_emb.shape[-1])
        obs_emb = torch.cat([obs, task_emb], dim=-1)

        return obs_emb

    def forward(self, obs, task_idx, compute_pi=True, compute_log_pi=True):
        obs_emb = self.construct_input(obs, task_idx)
        mu, log_std = self.trunk(obs_emb).chunk(2, dim=-1)

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

    def compute_log_probs(self, obs, action, task_idx):
        obs_emb = self.construct_input(obs, task_idx)
        mu, log_std = self.trunk(obs_emb).chunk(2, dim=-1)

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

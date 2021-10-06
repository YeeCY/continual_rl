import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

from collections import OrderedDict

from src.utils import gaussian_logprob, squash


class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)

        self.mean = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean(x)
        covar = self.kernel(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)


class SacActorMainNetMlp(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy with MLP"""
    def __init__(self, obs_shape, action_shape, hidden_dim, log_std_min, log_std_max,
                 act_func=torch.relu):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_func = act_func

        # FIXME (cyzheng): configurable architecture
        # FIXME (cyzheng): create weight placeholder instead
        self.weights = None
        self.weight_shapes = OrderedDict()

        self.hidden_layers = [hidden_dim, hidden_dim, hidden_dim]
        input_dim = obs_shape[0]
        for i, layer_dim in enumerate(self.hidden_layers):
            # self.add_module('hidden_layer{}'.format(i),
            #                 nn.Linear(input_dim, layer_dim))
            self.weight_shapes['hidden_layer{}.weight'.format(i)] = \
                [layer_dim, input_dim]
            self.weight_shapes['hidden_layer{}.bias'.format(i)] = \
                [layer_dim]
            input_dim = layer_dim

        # self.output_layer = nn.Linear(hidden_dim, action_shape[0] * 2)
        self.weight_shapes['output_layer0.weight'] = \
            [action_shape[0] * 2, hidden_dim]
        self.weight_shapes['output_layer0.bias'] = \
            [action_shape[0] * 2]

    def forward_trunk(self, obs, weights):
        hidden = obs
        for i in range(len(self.hidden_layers)):
            hidden = F.linear(hidden,
                              weight=weights['hidden_layer{}.weight'.format(i)],
                              bias=weights['hidden_layer{}.bias'.format(i)])
            hidden = self.act_func(hidden)

        mu, log_std = F.linear(hidden,
                               weight=weights['output_layer0.weight'],
                               bias=weights['output_layer0.bias']).chunk(2, dim=-1)

        return mu, log_std

    def forward(self, obs, compute_pi=True, compute_log_pi=True, weights=None):
        assert weights is not None or self.weights is not None

        if weights is None:
            weights = self.weights

        mu, log_std = self.forward_trunk(obs, weights)

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

    def compute_log_probs(self, obs, action, weights=None):
        assert weights is not None or self.weights is not None

        if weights is None:
            weights = self.weights

        # mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        mu, log_std = self.forward_trunk(obs, weights)

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


class SacSparseGPActorHyperNetMlp(nn.Module):
    def __init__(self, num_tasks, actor_shapes, num_inducing_points, act_func=torch.relu):
        super().__init__()

        assert isinstance(actor_shapes, OrderedDict)

        self.num_tasks = num_tasks
        self.actor_shapes = actor_shapes
        self.act_fun = act_func

        inducing_points = norm_actor_params[rand_idxs]
        self.model = ApproximateGPModel(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # # hypernet trunk
        # # FIXME (cyzheng): configurable architecture
        # self.weights = nn.ParameterDict()
        #
        # self.hidden_layers = [hidden_dim, hidden_dim]
        # input_dim = task_embedding_dim
        # for i, layer_dim in enumerate(self.hidden_layers):
        #     # self.add_module('hidden_layer{}'.format(i),
        #     #                 nn.Linear(input_dim, layer_dim))
        #     # self.weights['hidden_layer{}/weight'.format(i)] = \
        #     #     nn.Parameter(data=torch.Tensor(layer_dim, input_dim),
        #     #                  requires_grad=True)
        #     # self.weights['hidden_layer{}/bias'.format(i)] = \
        #     #     nn.Parameter(data=torch.Tensor(layer_dim),
        #     #                  requires_grad=True)
        #     weight = nn.Parameter(data=torch.Tensor(layer_dim, input_dim),
        #                           requires_grad=True)
        #     nn.init.orthogonal_(weight.data)
        #     self.weights['hidden_layer{}/weight'.format(i)] = weight
        #
        #     bias = nn.Parameter(data=torch.Tensor(layer_dim),
        #                         requires_grad=True)
        #     bias.data.fill_(0.0)
        #     self.weights['hidden_layer{}/bias'.format(i)] = bias
        #
        #     input_dim = layer_dim
        #
        # # hypernet output layers
        # self.output_layers = []
        # input_dim = hidden_dim
        # for i, shape in enumerate(actor_shapes.values()):
        #     output_dim = int(np.prod(shape))
        #     self.output_layers.append(output_dim)
        #     # self.add_module('output_layer{}'.format(i),
        #     #                 nn.Linear(input_dim, output_dim))
        #     weight = nn.Parameter(data=torch.Tensor(output_dim, input_dim),
        #                           requires_grad=True)
        #     nn.init.orthogonal_(weight.data)
        #     self.weights['output_layer{}/weight'.format(i)] = weight
        #
        #     bias = nn.Parameter(data=torch.Tensor(output_dim),
        #                         requires_grad=True)
        #     bias.data.fill_(0.0)
        #     self.weights['output_layer{}/bias'.format(i)] = bias
        #
        # # task embeddings
        # self.task_embs = nn.ParameterList()
        # for _ in range(num_tasks):
        #     self.task_embs.append(
        #         nn.Parameter(data=torch.Tensor(task_embedding_dim),
        #                      requires_grad=True)
        #     )
        #     torch.nn.init.normal_(self.task_embs[-1], mean=0., std=1.)

    def construct_input(self, task_idx):
        batch_size = 1
        task_emb = self.task_embs[task_idx]
        task_emb = task_emb.expand(batch_size,
                                   self.task_embedding_dim)

        return task_emb

    def forward(self, task_idx, weights=None):
        # (cyzheng): add weights parameter for output regularization
        if weights is None:
            weights = self.weights

        hidden = self.construct_input(task_idx)

        for i in range(len(self.hidden_layers)):
            hidden = F.linear(hidden,
                              weight=weights['hidden_layer{}/weight'.format(i)],
                              bias=weights['hidden_layer{}/bias'.format(i)])
            hidden = self.act_fun(hidden)

        actor_weights = OrderedDict()
        for i, (actor_layer_name, actor_layer_shape) in enumerate(
                self.actor_shapes.items()):
            actor_weight = F.linear(hidden,
                                    weight=weights['output_layer{}/weight'.format(i)],
                                    bias=weights['output_layer{}/bias'.format(i)])
            actor_weight = actor_weight.reshape(-1, *actor_layer_shape)
            actor_weight = torch.squeeze(actor_weight, dim=0)
            actor_weights[actor_layer_name] = actor_weight

        return actor_weights

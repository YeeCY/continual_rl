import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

from collections import OrderedDict
from itertools import chain

from src.utils import gaussian_logprob, squash


class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        # standard
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)

        # orthogonally decoupled
        # mean_inducing_points = inducing_points
        # covar_inducing_points = inducing_points[
        #     np.random.randint(len(inducing_points), size=len(inducing_points) // 10)]
        # covar_variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
        #     self, covar_inducing_points,
        #     gpytorch.variational.MeanFieldVariationalDistribution(covar_inducing_points.size(0)),
        #     learn_inducing_locations=True
        # )
        # variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        #     covar_variational_strategy, mean_inducing_points,
        #     gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(0)),
        # )
        super().__init__(variational_strategy)

        self.mean = gpytorch.means.ConstantMean()
        # self.kernel = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(lengthscale_prior=None),
        #     outputscale_prior=None
        # )
        self.kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(1e-6, 1e-3)),
            outputscale_constraint=gpytorch.constraints.Interval(1e-6, 1e-3)
        )

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

        self.num_actor_params = 0
        for actor_layer_shapes in self.actor_shapes.values():
            self.num_actor_params += np.prod(actor_layer_shapes)
        actor_param_dims = torch.linspace(0, self.num_actor_params,
                                          self.num_actor_params)
        # normalize to [-1, 1]
        self.norm_actor_param_dims = 2 * actor_param_dims / self.num_actor_params - 1
        self.models = []
        for _ in range(num_tasks):
            rand_idxs = np.random.randint(0, len(actor_param_dims), num_inducing_points)
            inducing_points = self.norm_actor_param_dims[rand_idxs]
            self.models.append(ApproximateGPModel(inducing_points))

        # initialize model
        # self.model.variational_strategy

        # TODO (cyzheng): Do we need likelihood model?
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

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

    # def construct_input(self, task_idx):
    #     batch_size = 1
    #     task_emb = self.task_embs[task_idx]
    #     task_emb = task_emb.expand(batch_size,
    #                                self.task_embedding_dim)
    #
    #     return task_emb

    def to(self, device):
        for idx, model in enumerate(self.models):
            self.models[idx] = model.to(device)
            # self.model = self.model.to(device)
        # self.likelihood = self.likelihood.to(device)
        self.norm_actor_param_dims = self.norm_actor_param_dims.to(device)

        return self

    def parameters(self, task_idx=0, recurse=True):
        # return chain(self.model.parameters(), self.likelihood.parameters())
        return self.models[task_idx].parameters()

    def named_parameters(self, task_idx=0, prefix='', recurse=True):
        # return chain(self.model.named_parameters(), self.likelihood.named_parameters())
        return self.models[task_idx].named_parameters()

    # def warmup(self, epochs, lr=1e-3):
    #     # GP regress to random parameters
    #
    #     mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model,
    #                                         num_data=self.num_actor_params)
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    #
    #     num_iters = np.ceil(self.num_actor_params / 4000).astype(int)
    #     for _ in range(epochs):
    #         for i in range(num_iters):
    #             norm_actor_param_dims = self.norm_actor_param_dims[i * 4000:(i + 1) * 4000]
    #             rand_actor_param = torch.zeros(*norm_actor_param_dims.shape,
    #                                            device=self.norm_actor_param_dims.device)
    #             torch.nn.init.normal_(rand_actor_param.data, mean=0.0, std=0.2)
    #
    #             output = self.model(norm_actor_param_dims)
    #             loss = -mll(output, rand_actor_param)
    #
    #             loss.backward()
    #             optimizer.step()
    #
    #             del rand_actor_param
    #             torch.cuda.empty_cache()
    #
    #     del mll
    #     del optimizer

    def forward(self, task_idx):
        # hidden = self.construct_input(task_idx)
        #
        # for i in range(len(self.hidden_layers)):
        #     hidden = F.linear(hidden,
        #                       weight=weights['hidden_layer{}/weight'.format(i)],
        #                       bias=weights['hidden_layer{}/bias'.format(i)])
        #     hidden = self.act_fun(hidden)

        actor_weights = OrderedDict()

        idx = 0
        for actor_layer_name, actor_layer_shape in self.actor_shapes.items():
            num_actor_layer_params = np.prod(actor_layer_shape)
            norm_layer_actor_param_dims = self.norm_actor_param_dims[idx:idx + num_actor_layer_params]

            # actor_weight = F.linear(hidden,
            #                         weight=weights['output_layer{}/weight'.format(i)],
            #                         bias=weights['output_layer{}/bias'.format(i)])
            actor_weight = self.models[task_idx](norm_layer_actor_param_dims).rsample()

            # actor_weight = []
            # for i in range(np.ceil(num_actor_layer_params / 2000).astype(int)):
            #     weight_chunk = self.models[task_idx](
            #         norm_layer_actor_param_dims[i * 2000:(i + 1) * 2000]).rsample()
            #     actor_weight.append(weight_chunk)
            # actor_weight = torch.cat(actor_weight)

            # weight_chunks = self.model(
            #     norm_layer_actor_param_dims[:num_actor_layer_params // 1000 * 1000].reshape(-1, 1000)
            # ).rsample()
            # remained_weight_chunk = self.model(
            #     norm_layer_actor_param_dims[num_actor_layer_params // 1000 * 1000:]
            # ).rsample()
            # actor_weight = torch.cat(weight_chunks.reshape(-1), remained_weight_chunk)

            actor_weight = actor_weight.reshape(-1, *actor_layer_shape)
            actor_weight = torch.squeeze(actor_weight, dim=0)
            actor_weights[actor_layer_name] = actor_weight

            idx += num_actor_layer_params

        return actor_weights

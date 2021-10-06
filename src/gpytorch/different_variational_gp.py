import urllib.request
import os
from scipy.io import loadmat
from math import floor
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import tqdm


class StandardApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MeanFieldApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main():
    if not os.path.isfile('elevators.mat'):
        print('Downloading \'elevators\' UCI dataset...')
        urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk',
                                   'elevators.mat')

    data = torch.Tensor(loadmat('elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]

    # f, ax = plt.subplots(1, 1, figsize=(20, 10))
    # ax.plot(X[:, 0], y, 'k*')
    # ax.set_ylim([-3, 3])
    # ax.legend(['song'])
    # plt.show()

    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    num_epochs = 10

    def train_and_test_approximate_gp(model_cls):
        inducing_points = torch.randn(128, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
        model = model_cls(inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Training
        model.train()
        likelihood.train()
        epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"Training {model_cls.__name__}")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                epochs_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()

        # Testing
        model.eval()
        likelihood.eval()
        means = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = model(x_batch)
                means = torch.cat([means, preds.mean.cpu()])
        means = means[1:]
        error = torch.mean(torch.abs(means - test_y.cpu()))
        print(f"Test {model_cls.__name__} MAE: {error.item()}")

    train_and_test_approximate_gp(StandardApproximateGP)


if __name__ == "__main__":
    main()

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

    f, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(X[:, 0], y, 'k*')
    ax.set_ylim([-3, 3])
    ax.legend(['song'])
    plt.show()

    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    inducing_points = train_x[:500, :]
    model = ApproximateGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    nn_model = nn.Sequential(
        nn.Linear(train_x.size(1), 90),
        nn.ReLU(),
        nn.Linear(90, 90),
        nn.ReLU(),
        nn.Linear(90, 1)
    )

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        nn_model = nn_model.cuda()

    num_epochs = 10
    model.train()
    likelihood.train()
    nn_model.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)
    nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    epoch_iters = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for _ in epoch_iters:
        minibatch_iters = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch, in minibatch_iters:
            optimizer.zero_grad()
            nn_optimizer.zero_grad()

            output = model(x_batch)
            loss = -mll(output, y_batch)
            nn_output = nn_model(x_batch).squeeze()
            nn_loss = F.mse_loss(nn_output, y_batch)

            minibatch_iters.set_postfix(loss=loss.item(), nn_loss=nn_loss.item())

            loss.backward()
            optimizer.step()
            nn_loss.backward()
            nn_optimizer.step()

    model.eval()
    likelihood.eval()
    nn_model.eval()
    means = []
    nn_means = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means.append(preds.mean.cpu())

            nn_preds = nn_model(x_batch)
            nn_means.append(nn_preds.squeeze().cpu())

    means = torch.cat(means)
    nn_means = torch.cat(nn_means)
    print('Test MAE: {}, NN MAE: {}'.format(
        torch.mean(torch.abs(means - test_y.cpu())),
        torch.mean(torch.abs(nn_means - test_y.cpu()))
    ))


if __name__ == "__main__":
    main()

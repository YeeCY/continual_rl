import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean(x)
        covar = self.kernel(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)


def main():
    train_x = torch.linspace(0, 1, 100)
    # train_y = torch.sin(2 * np.pi * train_x) + \
    #           torch.randn(train_x.size()) * 0.2
    train_y = torch.rand(train_x.size()) * 10 - 5

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    training_iters = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # loss for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iters):
        optimizer.zero_grad()

        predict_y = model(train_x)
        loss = -mll(predict_y, train_y)
        loss.backward()

        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iters, loss.item(),
            model.kernel.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    model.eval()
    likelihood.eval()

    # test model
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        observed_pred = likelihood(model(test_x))

    # plot regression result
    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        lower, upper = observed_pred.confidence_region()

        # training dataset
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

        # test mean
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')

        # confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-7, 7])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        plt.show()


if __name__ == "__main__":
    main()

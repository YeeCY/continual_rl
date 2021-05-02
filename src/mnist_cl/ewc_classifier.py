import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from src.mnist_cl import utils


class EwcClassifier(nn.Module):
    def __init__(self, image_size, image_channels, classes, hidden_units=400, lr=0.001,
                 lam=5000, fisher_sample_size=None,
                 online=False, gamma=1.0, device=None):

        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.hidden_units = hidden_units
        self.lr = lr
        self.lam = lam
        self.fisher_sample_size = fisher_sample_size
        self.online = online
        self.gamma = gamma

        # flatten image to 2D-tensor
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_channels * self.image_size ** 2, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.classes)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.to(device)

        self.ewc_task_count = 0
        self.prev_task_params = {}
        self.prev_task_fishers = {}

    def device(self):
        return next(self.parameters()).device

    def is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        return self.trunk(x)

    def estimate_fisher(self, dataset, allowed_classes=None):
        mode = self.training
        self.eval()

        fisher_sample_size = self.fisher_sample_size if self.fisher_sample_size is not None else len(dataset)
        data_loader = utils.get_data_loader(dataset, batch_size=fisher_sample_size, cuda=self.is_on_cuda())
        x, y = list(data_loader)[0]

        # run forward pass of model
        x = x.to(self.device())
        y_hat = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
        label = y_hat.max(1)[1]  # use predicted label to calculate loglikelihood
        negloglikelihood = F.nll_loss(F.log_softmax(y_hat, dim=1), label)

        self.zero_grad()
        negloglikelihood.backward()

        # critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done)
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        #
        # _, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(obs)
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.log_alpha_optimizer.zero_grad()
        # alpha_loss.backward()

        for name, param in self.named_parameters():
            if param.grad is not None:
                if self.online:
                    name = name + '_prev_task'
                    self.prev_task_params[name] = param.detach().clone()
                    self.prev_task_fishers[name] = \
                        param.grad.detach().clone() ** 2 + \
                        self.online_ewc_gamma * self.prev_task_fishers.get(name, torch.zeros_like(param.grad))
                else:
                    name = name + f'_prev_task{self.ewc_task_count}'
                    self.prev_task_params[name] = param.detach().clone()
                    self.prev_task_fishers[name] = param.grad.detach().clone() ** 2

        self.ewc_task_count += 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def _ewc_loss(self):
        ewc_losses = []
        if self.ewc_task_count >= 1:
            if self.online:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        name = name + '_prev_task'
                        mean = self.prev_task_params[name]
                        # apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma * self.prev_task_fishers[name]
                        ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                        ewc_losses.append(ewc_loss)
            else:
                for task in range(self.ewc_task_count):
                    # compute ewc loss for each parameter
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            name = name + f'_prev_task{task}'
                            mean = self.prev_task_params[name]
                            fisher = self.prev_task_fishers[name]
                            ewc_loss = torch.sum(fisher * (param - mean) ** 2)
                            ewc_losses.append(ewc_loss)
            return torch.sum(torch.stack(ewc_losses)) / 2.0
        else:
            param = next(self.parameters())
            return torch.tensor(0.0, device=param.device)

    def train_a_batch(self, x, y, active_classes=None):
        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Run model
        y_hat = self(x)
        # -if needed, remove predictions for classes not in current task
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
            y_hat = y_hat[:, class_entries]

        # Calculate prediction loss
        loss_total = F.cross_entropy(input=y_hat, target=y, reduction='mean')

        # Calculate training-precision
        precision = (y == y_hat.max(1)[1]).sum().item() / x.size(0)

        # Add EWC-loss
        ewc_loss = self._ewc_loss()
        if self.lam > 0:
            loss_total += self.lam * ewc_loss

        # Backpropagate errors (if not yet done)
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'ewc_loss': ewc_loss.item(),
            'precision': precision if precision is not None else 0.,
        }


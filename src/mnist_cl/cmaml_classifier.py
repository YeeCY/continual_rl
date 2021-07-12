import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from src.mnist_cl import utils


class CmamlClassfier(nn.Module):
    def __init__(self, image_size, image_channels, classes, hidden_units=400, lr=0.001,
                 memory_budget=2000, device=None):

        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.hidden_units = hidden_units
        self.lr = lr
        self.memory_budget = memory_budget
        # self.lam = lam
        # self.fisher_sample_size = fisher_sample_size
        # self.online = online
        # self.gamma = gamma

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

    def inner_update(self, x, y, task):

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

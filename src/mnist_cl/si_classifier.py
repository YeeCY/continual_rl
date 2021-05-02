import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import utils


class SiClassifier(nn.Module):
    def __init__(self, image_size, image_channels, classes, hidden_units=400, lr=0.001,
                 c=1.0, epsilon=1.0):

        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.hidden_units = hidden_units
        self.lr = lr
        self.c = c
        self.epsilon = epsilon

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

        self.params_w = {}
        self.omegas = {}
        self.prev_task_params = {}

        # set prev_params and prev_task_params as weight initializations
        self.prev_params = {}
        self.prev_task_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.detach().clone()
                self.prev_task_params[name] = param.detach().clone()

    def device(self):
        return next(self.parameters()).device

    def is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        return self.trunk(x)

    def update_omegas(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                prev_param = self.prev_task_params[name]
                current_param = param.detach().clone()
                delta_param = current_param - prev_param
                current_omega = self.params_w[name] / (delta_param ** 2 + self.si_epsilon)

                self.prev_task_params[name] = current_param
                self.omegas[name] = current_omega + self.omegas.get(name, torch.zeros_like(param))

        # clear importance buffers for the next task
        self.params_w = {}

    def _estimate_importance(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.params_w[name] = -param.grad * (param.detach() - self.prev_params[name]) + \
                                      self.params_w.get(name, torch.zeros_like(param))
                self.prev_params[name] = param.detach().clone()

    def _surrogate_loss(self):
        si_losses = []
        for name, param in self.named_parameters():
            if param.grad is not None:
                prev_param = self.prev_task_params[name]
                omega = self.omegas.get(name, torch.zeros_like(param))
                si_loss = torch.sum(omega * (param - prev_param) ** 2)
                si_losses.append(si_loss)

        return torch.sum(torch.stack(si_losses))

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

        # Add SI-loss
        si_loss = self._surrogate_loss()
        if self.c > 0:
            loss_total += self.c * si_loss

        # Backpropagate errors (if not yet done)
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Estimate weight importance
        self._estimate_importance()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'si_loss': si_loss.item(),
            'precision': precision if precision is not None else 0.,
        }


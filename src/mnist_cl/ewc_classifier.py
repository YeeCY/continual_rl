import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class EwcClassifier(nn.Module):
    def __init__(self, image_size, image_channels, classes, hidden_units=400,
                 lam=5000, fisher_sample_size=1024,
                 online=False, gamma=1.0, emp_fi=False):

        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.hidden_units = hidden_units
        self.lam = lam
        self.fisher_sample_size = fisher_sample_size
        self.online = online
        self.gamma = gamma
        self.emp_fi = emp_fi

        # flatten image to 2D-tensor
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_channels * self.image_size ** 2, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.classes)
        )

        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999))

    def device(self):
        return next(self.parameters()).device

    def is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        return self.trunk(x)

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
        loss = F.cross_entropy(input=y_hat, target=y, reduction='mean')

        # Calculate training-precision
        precision = (y == y_hat.max(1)[1]).sum().item() / x.size(0)

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.lam > 0:
            loss += self.lam * ewc_loss

        # Backpropagate errors (if not yet done)
        loss.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss': loss.item(),
            'ewc': ewc_loss.item(),
            'precision': precision if precision is not None else 0.,
        }


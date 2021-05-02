import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import utils


class EwcClassifier(nn.Module):
    def __init__(self, image_size, image_channels, classes,
                 lam=5000, fisher_sample_size=1024,
                 online=False, gamma=1.0, emp_fi=False):

        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.lam = lam
        self.fisher_sample_size = fisher_sample_size
        self.online = online
        self.gamma = gamma
        self.emp_fi = emp_fi

        # flatten image to 2D-tensor
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_channels * self.image_size ** 2, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.classes)
        )

        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999))

    def forward(self, x):
        return self.trunk(x)

    def train_a_batch(self, x, y, active_classes=None):
        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        if x is not None:
            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = predL

            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        else:
            precision = predL = None
            # -> it's possible there is only "replay" [e.g., for offline with task-incremental learning]


        # Combine loss from current and replayed batch
        loss_total = loss_cur


        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c>0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss


        # Backpropagate errors (if not yet done)
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'pred': predL.item() if predL is not None else 0,
            'ewc': ewc_loss.item(),
            'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }


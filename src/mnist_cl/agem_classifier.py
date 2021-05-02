import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

import utils


class AgemClassifier(nn.Module):
    def __init__(self, image_size, image_channels, classes, hidden_units=400, lr=0.001,
                 memory_budget=2000, device=None):

        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.hidden_units = hidden_units
        self.lr = lr
        self.memory_budget = memory_budget

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

        self.exemplar_sets = []

    def device(self):
        return next(self.parameters()).device

    def is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        return self.trunk(x)

    def reduce_exemplar_sets(self, m):
        for y, p_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = p_y[:m]

    def construct_exemplar_set(self, dataset, n):
        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []

        # if self.herding:
        #     # compute features for each example in [dataset]
        #     first_entry = True
        #     dataloader = utils.get_data_loader(dataset, 128, cuda=self._is_on_cuda())
        #     for (image_batch, _) in dataloader:
        #         image_batch = image_batch.to(self._device())
        #         with torch.no_grad():
        #             feature_batch = self.feature_extractor(image_batch).cpu()
        #         if first_entry:
        #             features = feature_batch
        #             first_entry = False
        #         else:
        #             features = torch.cat([features, feature_batch], dim=0)
        #     if self.norm_exemplars:
        #         features = F.normalize(features, p=2, dim=1)
        #
        #     # calculate mean of all features
        #     class_mean = torch.mean(features, dim=0, keepdim=True)
        #     if self.norm_exemplars:
        #         class_mean = F.normalize(class_mean, p=2, dim=1)
        #
        #     # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
        #     exemplar_features = torch.zeros_like(features[:min(n, n_max)])
        #     list_of_selected = []
        #     for k in range(min(n, n_max)):
        #         if k>0:
        #             exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
        #             features_means = (features + exemplar_sum)/(k+1)
        #             features_dists = features_means - class_mean
        #         else:
        #             features_dists = features - class_mean
        #         index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
        #         if index_selected in list_of_selected:
        #             raise ValueError("Exemplars should not be repeated!!!!")
        #         list_of_selected.append(index_selected)
        #
        #         exemplar_set.append(dataset[index_selected][0].numpy())
        #         exemplar_features[k] = copy.deepcopy(features[index_selected])
        #
        #         # make sure this example won't be selected again
        #         features[index_selected] = features[index_selected] + 10000
        # else:
        indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
        for k in indeces_selected:
            exemplar_set.append(dataset[k][0].numpy())

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))

        # set mode of model back
        self.train(mode=mode)

    def train_a_batch(self, x, y, x_=None, y_=None, active_classes=None):
        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()


        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
            y_ = [y_]
            active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else None

            # Prepare lists to store losses for each replay
            loss_replay = [None] * n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            y_hat_all = self(x_)

            # Loop to evalute predictions on replay according to each previous task
            for replay_id in range(n_replays):
                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in replayed task
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    loss_replay[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays

        # If using A-GEM, calculate and store averaged gradient of replayed data
        if x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            loss_replay.backward()

            # Reorganize the gradient of the replayed batch as a single vector
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.view(-1))
            grad_rep = torch.cat(grad_rep)
            # Reset gradients (with A-GEM, gradients of replayed batch should only be used as inequality constraint)
            self.optimizer.zero_grad()

        ##--(2)-- CURRENT DATA --##
        if x is not None:
            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            # -multiclass prediction loss
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
        # Backpropagate errors (if not yet done)
        loss_total.backward()

        # If using A-GEM, potentially change gradient:
        if x_ is not None:
            # -reorganize gradient (of current batch) as single vector
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_rep).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_rep*grad_rep).sum()
                grad_proj = grad_cur-(angle/length_rep)*grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'pred': predL.item() if predL is not None else 0,
            'precision': precision if precision is not None else 0.,
        }


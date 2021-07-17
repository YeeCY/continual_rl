import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from collections import OrderedDict

from src import utils


class CmamlClassfier(nn.Module):
    def __init__(self, image_size, image_channels, classes, hidden_units=400, fast_lr=0.0003,
                 meta_lr=0.001, memory_budget=200, grad_clip_norm=2.0, first_order=True,
                 device=None):

        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.hidden_units = hidden_units
        self.fast_lr = fast_lr
        self.meta_lr = meta_lr
        self.memory_budget = memory_budget
        self.grad_clip_norm = grad_clip_norm
        self.first_order = first_order

        # TODO (chongyi zheng): update to adaptive layer size
        self.layer_sizes = (self.image_channels * self.image_size ** 2,
                            self.hidden_units,
                            self.hidden_units,
                            self.classes)
        self.num_layers = len(self.layer_sizes)
        for i in range(1, self.num_layers):
            self.add_module('layer{}'.format(i - 1),
                            nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]))

        self.meta_optimizer = optim.Adam(self.parameters(), lr=self.meta_lr)

        self.to(device)

        self.memory = {
            'x': np.empty((self.memory_budget, self.image_channels,
                           self.image_size, self.image_size),
                          dtype=np.float32),
            'y': np.empty((self.memory_budget, 1), dtype=np.int64),
            'class_entries': [],
            'sample_num': 0
        }
        self.total_sample_num = 0

    def _augment_prev_samples(self, x, y, active_classes=None):
        class_entries = None
        if active_classes is not None:
            class_entries = torch.as_tensor(
                [active_classes[-1]] * x.size(0), dtype=torch.int64,
                device=self.device()) \
                if type(active_classes[0]) == list else active_classes

        aug_size = min(x.size(0), self.memory['sample_num'])
        aug_class_entries = None
        if aug_size > 0:
            sample_idxs = np.random.randint(
                0, self.memory['sample_num'], size=aug_size)
            mem_x = self.memory['x'][sample_idxs]
            mem_y = self.memory['y'][sample_idxs].squeeze()
            if class_entries is not None:
                mem_class_entries = torch.as_tensor(
                    [self.memory['class_entries'][idx] for idx in sample_idxs],
                    dtype=torch.int64, device=self.device())

            # (chongyi zheng): clone original data to prevent erroneous backpropagation
            aug_x = torch.cat([x, torch.as_tensor(
                mem_x, dtype=x.dtype, device=self.device())])
            aug_y = torch.cat([y, torch.as_tensor(
                mem_y, dtype=y.dtype, device=self.device())])
            if class_entries is not None:
                aug_class_entries = torch.cat([class_entries, mem_class_entries])
        else:
            # (chongyi zheng): clone original data to prevent erroneous backpropagation
            aug_x = x.clone()
            aug_y = y.clone()
            if class_entries is not None:
                aug_class_entries = class_entries

        return aug_x, aug_y, aug_class_entries

    def _reservoir_sampling(self, x, y, active_classes=None):
        if active_classes is not None:
            class_entries = [active_classes[-1]] * x.size(0) \
                if type(active_classes[0]) == list else active_classes
        else:
            class_entries = [None] * x.size(0)

        for single_x, single_y, single_class_entries in zip(x, y, class_entries):
            self.total_sample_num += 1
            sample_num = self.memory['sample_num']
            if sample_num < self.memory_budget:
                np.copyto(self.memory['x'][sample_num], utils.to_np(single_x))
                np.copyto(self.memory['y'][sample_num], utils.to_np(single_y))
                if single_class_entries is not None:
                    self.memory['class_entries'].append(single_class_entries)
                self.memory['sample_num'] += 1
            else:
                idx = np.random.randint(0, self.total_sample_num)
                if idx < self.memory_budget:
                    np.copyto(self.memory['x'][idx], utils.to_np(single_x))
                    np.copyto(self.memory['y'][idx], utils.to_np(single_y))
                    if single_class_entries is not None:
                        self.memory['class_entries'][idx] = single_class_entries

    def _inner_update(self, x, y, active_classes=None, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        y_hat = self(x, params)
        if active_classes is not None:
            class_entries = active_classes[-1] \
                if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]

        loss = F.cross_entropy(y_hat, y)

        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=not self.first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            torch.clamp(grad, -self.grad_clip_norm, self.grad_clip_norm)
            updated_params[name] = param - self.fast_lr * grad

        return updated_params

    def device(self):
        return next(self.parameters()).device

    def is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # flatten image first
        output = x.view(x.size(0), -1)
        for i in range(1, self.num_layers - 1):
            output = F.linear(output,
                              weight=params['layer{}.weight'.format(i - 1)],
                              bias=params['layer{}.bias'.format(i - 1)])
            output = torch.relu(output)

        logits = F.linear(output,
                          weight=params['layer{}.weight'.format(self.num_layers - 2)],
                          bias=params['layer{}.bias'.format(self.num_layers - 2)])

        return logits

    def train_a_batch(self, x, y, active_classes=None):
        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.meta_optimizer.zero_grad()

        aug_x, aug_y, class_entries = self._augment_prev_samples(x, y, active_classes)
        perm = torch.randperm(aug_x.size(0))
        aug_x = aug_x[perm]
        aug_y = aug_y[perm]
        class_entries = class_entries[perm]

        # Reservoir samping
        self._reservoir_sampling(x, y, active_classes)

        # Run inner gradient steps
        fast_params = None
        for single_x, single_y in zip(x, y):
            fast_params = self._inner_update(single_x, single_y.unsqueeze(0),
                                             active_classes, params=fast_params)

        # Compute meta loss
        aug_y_hat = self(aug_x, fast_params)
        aug_y_hat = aug_y_hat.gather(-1, class_entries)

        meta_loss = F.cross_entropy(aug_y_hat, aug_y)

        # Calculate training-precision
        precision = (aug_y == aug_y_hat.max(1)[1]).sum().item() / aug_y.size(0)

        # Update meta parameters
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.grad_clip_norm)
        self.meta_optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': meta_loss.item(),
            'precision': precision if precision is not None else 0.,
        }

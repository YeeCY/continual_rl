from collections import Iterable
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence

from agent.sac import TaskEmbeddingDistilledActorSacMlpAgent


class SiTaskEmbeddingDistilledActorSacMlpAgent(TaskEmbeddingDistilledActorSacMlpAgent):
    def __init__(
            self,
            obs_shape,
            action_shape,
            action_range,
            device,
            actor_hidden_dim=400,
            critic_hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            actor_lr=1e-3,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_tau=0.005,
            critic_target_update_freq=2,
            batch_size=128,
            distillation_hidden_dim=256,
            distillation_task_embedding_dim=16,
            distillation_epochs=200,
            distillation_iters_per_epoch=50,
            distillation_batch_size=1000,
            distillation_memory_budget_per_task=50000,
            si_c=1.0,
            si_epsilon=0.1,
    ):
        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size, distillation_hidden_dim,
            distillation_task_embedding_dim, distillation_epochs, distillation_iters_per_epoch,
            distillation_batch_size, distillation_memory_budget_per_task)

        self.si_c = si_c
        self.si_epsilon = si_epsilon

        self.params_w = {}
        self.omegas = {}
        self.prev_params = {}
        self.prev_task_params = {}

        self._save_init_params()

    def _save_init_params(self):
        # set prev_task_params as weight initializations
        for name, param in self.distilled_actor.named_parameters():
            if param.requires_grad:
                self.prev_task_params[name] = param.detach().cpu().clone()
                self.prev_params[name] = param.detach().cpu().clone()

    def update_omegas(self):
        for name, param in self.distilled_actor.named_parameters():
            if param.requires_grad:
                prev_param = self.prev_task_params[name]
                current_param = param.detach().cpu().clone()
                delta_param = current_param - prev_param
                current_omega = self.params_w[name] / (delta_param ** 2 + self.si_epsilon)

                self.prev_task_params[name] = current_param
                self.omegas[name] = current_omega + self.omegas.get(
                    name, torch.zeros_like(param).cpu())

        # clear importance buffers for the next task
        self.params_w = {}

    def _estimate_importance(self):
        for name, param in self.distilled_actor.named_parameters():
            if param.requires_grad:
                self.params_w[name] = \
                    -param.grad.detach().cpu() * (param.detach().cpu() - self.prev_params[name]) + \
                    self.params_w.get(name, torch.zeros_like(param).cpu())
                self.prev_params[name] = param.detach().cpu().clone()

    def _compute_surrogate_loss(self, named_parameters):
        assert isinstance(named_parameters, Iterable), "'named_parameters' must be a iterator"

        si_losses = []
        for name, param in named_parameters:
            if param.requires_grad:
                prev_param = self.prev_task_params[name].to(self.device)
                omega = self.omegas.get(name, torch.zeros_like(param)).to(self.device)
                si_loss = torch.sum(omega * (param - prev_param) ** 2)
                si_losses.append(si_loss)

        return torch.sum(torch.stack(si_losses))

    def _train_distilled_actor(self, dataset, total_steps, epoch, logger):
        for iter in range(self.distillation_iters_per_epoch):
            random_idxs = np.random.randint(0, self.distillation_memory_budget_per_task,
                                            size=self.distillation_batch_size)
            batch_obses = torch.Tensor(dataset['obses'][random_idxs]).to(self.device).squeeze()
            batch_mus = torch.Tensor(dataset['mus'][random_idxs]).to(self.device).squeeze()
            batch_log_stds = torch.Tensor(dataset['log_stds'][random_idxs]).to(self.device).squeeze()
            task_id = dataset['task_id']

            mus, _, _, log_stds = self.distilled_actor(
                batch_obses, task_id,
                compute_pi=True, compute_log_pi=True)

            actor_dists = Independent(Normal(loc=batch_mus, scale=batch_log_stds.exp()), 1)
            distilled_actor_dists = Independent(Normal(loc=mus, scale=log_stds.exp()), 1)
            distillation_loss = torch.mean(kl_divergence(actor_dists, distilled_actor_dists))
            # regularize with SI
            si_surrogate_loss = self._compute_surrogate_loss(self.distilled_actor.named_parameters())
            distillation_loss = distillation_loss + self.si_c * si_surrogate_loss

            logger.log('train/distillation_loss', distillation_loss,
                       total_steps + epoch * self.distillation_iters_per_epoch + iter)

            self.distilled_actor_optimizer.zero_grad()
            distillation_loss.backward()
            self.distilled_actor_optimizer.step()

            # estimate weight importance
            self._estimate_importance()

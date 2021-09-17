import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.sac.base_sac_agent import SacMlpAgent
from agent.nets.hypernet import SacActorMainNetMlp, SacTaskEmbeddingActorHyperNetMlp
from agent.network import SacCriticMlp


class TaskEmbeddingHyperNetActorSacMlpAgent(SacMlpAgent):
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
            hypernet_hidden_dim=128,
            hypernet_task_embedding_dim=16,
            hypernet_reg_coeff=0.01,
            hypernet_on_the_fly_reg=False,
            hypernet_first_order=True,
    ):
        assert isinstance(action_shape, list)
        assert isinstance(action_range, list)

        self.hypernet_hidden_dim = hypernet_hidden_dim
        self.hypernet_task_embedding_dim = hypernet_task_embedding_dim
        self.hypernet_reg_coeff = hypernet_reg_coeff
        self.hypernet_on_the_fly_reg = hypernet_on_the_fly_reg
        self.hypernet_first_order = hypernet_first_order

        self.task_count = 0
        self.weights = None
        self.target_weights = []

        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size)

        # self.distill_epochs = distill_epochs
        # self.distill_iters_per_epoch = distill_iters_per_epoch
        # self.distill_batch_size = distill_batch_size
        # self.distill_memory_budget_per_task = distill_memory_budget_per_task

        # self.task_count = 0
        # self.memories = []

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = SacActorMainNetMlp(

            self.obs_shape, self.action_shape[0], self.actor_hidden_dim,
            self.actor_log_std_min, self.actor_log_std_max
        ).to(self.device)

        num_tasks = len(self.action_shape)
        actor_shapes = self.actor.weight_shapes
        self.hypernet = SacTaskEmbeddingActorHyperNetMlp(
            num_tasks, actor_shapes, self.hypernet_hidden_dim,
            self.hypernet_task_embedding_dim
        ).to(self.device)

        self.critic = SacCriticMlp(
            self.obs_shape, self.action_shape[0], self.critic_hidden_dim
        ).to(self.device)

        self.critic_target = SacCriticMlp(
            self.obs_shape, self.action_shape[0], self.critic_hidden_dim
        ).to(self.device)

        self.reset_target_critic()

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_shape[0])

        # sac optimizers
        # (cyzheng): note that we will create new optimizers when one task finished,
        # and we don't optimize actor main network directly
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # self.actor_hypernet_optimizer = torch.optim.Adam(
        #     self.actor_hypernet.parameters(), lr=self.actor_lr)
        self.hypernet_weight_optimizer = torch.optim.Adam(
            self.hypernet.weights.values(), lr=self.actor_lr,
        )
        self.hypernet_emb_optimizer = torch.optim.Adam(
            [self.hypernet.task_embs[self.task_count]],
            lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def reset(self, **kwargs):
        self.reset_target_critic()
        self.reset_log_alpha()

        self.hypernet_weight_optimizer = torch.optim.Adam(
            # self.hypernet.weights.values(), lr=self.actor_lr,
            self.hypernet.weights.values(), lr=self.actor_lr,
        )
        self.hypernet_emb_optimizer = torch.optim.Adam(
            [self.hypernet.task_embs[self.task_count]]
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def act(self, obs, sample=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        assert 'head_idx' in kwargs
        task_idx = kwargs['head_idx']

        with torch.no_grad():
            if self.weights is not None:
                weights = self.weights
            else:
                weights = self.hypernet(task_idx)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False, weights=weights)
            action = pi if sample else mu
            action = action.clamp(*self.action_range[task_idx])
            assert action.ndim == 2 and action.shape[0] == 1

        return utils.to_np(action)

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done, **kwargs):
        assert 'task_idx' in kwargs
        task_idx = kwargs.pop('task_idx')

        with torch.no_grad():
            weights = self.hypernet(task_idx)
            _, policy_action, log_pi, _ = self.actor(next_obs, weights=weights)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, **kwargs)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        return critic_loss

    def compute_actor_and_alpha_loss(self, obs, compute_alpha_loss=True, **kwargs):
        assert 'task_idx' in kwargs
        task_idx = kwargs.pop('task_idx')

        weights = self.hypernet(task_idx)
        _, pi, log_pi, log_std = self.actor(obs, weights=weights)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        alpha_loss = None
        if compute_alpha_loss:
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        return log_pi, actor_loss, alpha_loss

    def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None,
                               add_reg_loss=False):
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.hypernet_weight_optimizer.zero_grad()
        self.hypernet_emb_optimizer.zero_grad()

        actor_loss.backward(retain_graph=add_reg_loss,
                            create_graph=add_reg_loss and not self.hypernet_first_order)
        self.hypernet_emb_optimizer.step()

        if add_reg_loss:
            assert len(self.target_weights) == self.task_count
            hypernet_delta_weights = self.compute_hypernet_delta_weights()

            reg_loss = self.hypernet_reg_coeff * self.compute_hypernet_reg(
                hypernet_delta_weights)
            reg_loss.backward()

        self.hypernet_weight_optimizer.step()

        if isinstance(alpha_loss, torch.Tensor):
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def compute_hypernet_delta_weights(self):
        """Performs a single optimization step using the Adam optimizer. The code
        has been copied from:

            https://git.io/fjYP3

        Note, this function does not change the inner state of the given
        optimizer object.

        Note, gradients are cloned and detached by default.

        Args:
            optimizer: An instance of class :class:`torch.optim.Adam`.
            detach_dp: Whether gradients are detached from the computational
                graph. Note, :code:`False` only makes sense if
                func:`torch.autograd.backward` was called with the argument
                `create_graph` set to :code:`True`.

        Returns:
            A list of gradient changes `d_p` that would be applied by this
            optimizer to all parameters when calling :meth:`torch.optim.Adam.step`.
        """
        assert isinstance(self.hypernet_weight_optimizer, torch.optim.Adam)

        d_ps = []

        optimizer = self.hypernet_weight_optimizer
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.hypernet_first_order:
                    grad = p.grad.detach().clone()
                else:
                    grad = p.grad.clone()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                if amsgrad and not self.hypernet_first_order:
                    raise ValueError('Cannot backprop through optimizer step if ' +
                                     '"amsgrad" is enabled.')

                orig_state = dict(optimizer.state[p])
                state = dict()

                # State initialization
                if len(orig_state) == 0:
                    orig_state['step'] = 0
                    # Exponential moving average of gradient values
                    orig_state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    orig_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        orig_state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # Copy original state.
                state['step'] = int(orig_state['step'])
                state['exp_avg'] = orig_state['exp_avg'].clone()
                state['exp_avg_sq'] = orig_state['exp_avg_sq'].clone()
                if amsgrad:
                    state['max_exp_avg_sq'] = orig_state['max_exp_avg_sq'].clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    #grad.add_(group['weight_decay'], p.data)
                    grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    #denom = exp_avg_sq.sqrt().add_(group['eps'])
                    denom = exp_avg_sq.sqrt() + group['eps']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                d_ps.append(-step_size * (exp_avg / denom))

        return d_ps

    def compute_hypernet_reg(self, hypernet_delta_weights):
        num_regs = self.task_count

        hypernet_weights = OrderedDict()
        for (name, weight), delta_weight in zip(self.hypernet.weights.items(),
                                                hypernet_delta_weights):
            assert weight.shape == delta_weight.shape
            hypernet_weights[name] = weight + delta_weight

        reg_loss = []
        for i in range(num_regs):
            predicted_weights = self.hypernet(i, weights=hypernet_weights)
            if self.hypernet_on_the_fly_reg:
                with utils.eval_mode(self):
                    with torch.no_grad():
                        target_weights = self.hypernet(i)
            else:
                target_weights = self.target_weights[i]
            predicted_ws = torch.cat([w.view(-1) for w in predicted_weights.values()])
            target_ws = torch.cat([w.view(-1) for w in target_weights.values()])
            reg_loss.append(
                (target_ws - predicted_ws).pow(2).sum()
            )
        reg_loss = torch.mean(torch.stack(reg_loss))

        return reg_loss

    def infer_weights(self, task_id):
        self.weights = self.hypernet(task_id)

    def clear_weights(self):
        self.weights = None

    def construct_hypernet_targets(self):
        self.task_count += 1
        self.target_weights.clear()

        if not self.hypernet_on_the_fly_reg:
            with utils.eval_mode(self):
                with torch.no_grad():
                    for task_idx in range(self.task_count):
                        weights = self.hypernet(task_idx)
                        self.target_weights.append(weights)

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        if self.task_count > 0 and self.hypernet_reg_coeff > 0:
            add_reg_loss = True
        else:
            add_reg_loss = False

        assert 'head_idx' in kwargs
        task_idx = kwargs.pop('head_idx')

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done,
                                               task_idx=task_idx)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_update_freq == 0:
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
                obs, task_idx=task_idx)
            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss,
                                        add_reg_loss=add_reg_loss)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

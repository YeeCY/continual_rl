# import math
# import copy
# from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.sac.base_sac_agent import SacMlpAgent
from agent.nets.gp_lvm import SacActorMainNetMlp, SacGPLVMActorHyperNetMlp
from agent.network import SacCriticMlp


class GPLatentVariableModelHyperNetActorSacMlpAgent(SacMlpAgent):
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
            gp_chunk_size=1000,
            gp_latent_dim=64,
            gp_num_inducing_points=20,
    ):
        assert isinstance(action_shape, list)
        assert isinstance(action_range, list)

        self.gp_chunk_size = gp_chunk_size
        self.gp_latent_dim = gp_latent_dim
        self.gp_num_inducing_points = gp_num_inducing_points

        self.task_count = 0
        self.weights = None

        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.hypernet.train(training)
        self.critic.train(training)
        self.critic_target.train(training)

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
        self.hypernet = SacGPLVMActorHyperNetMlp(
            num_tasks, actor_shapes, self.gp_chunk_size,
            self.gp_latent_dim, self.gp_num_inducing_points
        ).to(self.device)

        # TODO (cyzheng): warmup hypernet
        # self.hypernet.warmup(self.gp_warmup_epochs, lr=self.actor_lr)

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
        # FIXME (cyzheng)
        self.hypernet_optimizer = torch.optim.Adam(
            self.hypernet.parameters(), lr=self.actor_lr,
        )
        # self.hypernet_emb_optimizer = torch.optim.Adam(
        #     [self.hypernet.task_embs[self.task_count]],
        #     lr=self.actor_lr
        # )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def reset(self, **kwargs):
        self.reset_target_critic()
        self.reset_log_alpha()

        # FIXME (cyzheng)
        self.hypernet_optimizer = torch.optim.Adam(
            self.hypernet.parameters(), lr=self.actor_lr,
        )
        # self.hypernet_emb_optimizer = torch.optim.Adam(
        #     [self.hypernet.task_embs[self.task_count]], lr=self.actor_lr,
        # )
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
            assert action.ndim == 2

        return utils.to_np(action)

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done, **kwargs):
        assert 'task_idx' in kwargs
        task_idx = kwargs.pop('task_idx')

        with torch.no_grad():
            # if self.weights is None:
            #     weights = self.hypernet(task_idx)
            # else:
            #     weights = self.weights
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

        # if self.weights is None:
        #     weights = self.hypernet(task_idx)
        # else:
        #     weights = self.weights
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
        self.hypernet_optimizer.zero_grad()
        # self.hypernet_emb_optimizer.zero_grad()

        actor_loss.backward()
        # self.hypernet_emb_optimizer.step()

        # if add_reg_loss:
        #     # assert len(self.target_weights) == self.task_count
        #     hypernet_delta_weights = self.compute_hypernet_delta_weights()
        #
        #     reg_loss = self.hypernet_reg_coeff * self.compute_hypernet_reg(
        #         hypernet_delta_weights)
        #     reg_loss.backward()

        self.hypernet_optimizer.step()

        if isinstance(alpha_loss, torch.Tensor):
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def infer_weights(self, task_id):
        self.weights = self.hypernet(task_id)

    def clear_weights(self):
        del self.weights
        torch.cuda.empty_cache()
        self.weights = None

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        assert 'head_idx' in kwargs
        task_idx = kwargs.pop('head_idx')

        # self.infer_weights(task_idx)
        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done,
                                               task_idx=task_idx)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_update_freq == 0:
            log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
                obs, task_idx=task_idx)
            self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        # self.clear_weights()

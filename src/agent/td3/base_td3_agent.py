import copy
import torch
import numpy as np
import torch.nn.functional as F

import utils
from agent.network import Td3ActorMlp, Td3CriticMlp


class Td3MlpAgent:
    """Adapt from https://github.com/rail-berkeley/rlkit and https://github.com/sfujim/TD3"""
    def __init__(
            self,
            obs_shape,
            action_shape,
            action_range,
            device,
            actor_hidden_dim=256,
            critic_hidden_dim=256,
            discount=0.99,
            actor_lr=3e-4,
            actor_noise=0.2,
            actor_noise_clip=0.5,
            critic_lr=3e-4,
            expl_noise_std=0.1,
            target_tau=0.005,
            actor_and_target_update_freq=2,
            batch_size=256,
    ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_range = action_range
        self.device = device
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.discount = discount
        self.actor_lr = actor_lr
        self.actor_noise = actor_noise
        self.actor_noise_clip = actor_noise_clip
        self.critic_lr = critic_lr
        self.expl_noise_std = expl_noise_std
        self.target_tau = target_tau
        self.actor_and_target_update_freq = actor_and_target_update_freq
        self.batch_size = batch_size

        self.training = False

        self._setup_agent()

        self.train()

    def _setup_agent(self):
        self.actor = Td3ActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.action_range
        ).to(self.device)

        self.actor_target = Td3ActorMlp(
            self.obs_shape, self.action_shape, self.actor_hidden_dim,
            self.action_range
        ).to(self.device)

        self.critic = Td3CriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.critic_target = Td3CriticMlp(
            self.obs_shape, self.action_shape, self.critic_hidden_dim
        ).to(self.device)

        self.reset_target()

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # save initial parameters
        self._critic_init_state = copy.deepcopy(self.critic.state_dict())

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.actor_target.train(training)
        self.critic.train(training)
        self.critic_target.train(training)

    def reset(self, reset_critic=False):
        if reset_critic:
            self.critic.load_state_dict(self._critic_init_state)

        self.reset_target()

    def reset_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, obs, add_noise=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        with torch.no_grad():
            action = self.actor(obs, **kwargs)
            if add_noise:
                low, high = np.array(self.action_range[0]), np.array(self.action_range[1])
                assert np.alltrue(high == -low), "Action range must be symmetric!"
                noise = np.random.normal(0, high * self.expl_noise_std)
                action += noise
            action = action.clamp(*self.action_range)
            assert action.ndim == 2 and action.shape[0] == 1

        return utils.to_np(action)

    def compute_critic_loss(self, obs, action, reward, next_obs, not_done, **kwargs):
        with torch.no_grad():
            actor_next_action = self.actor_target(next_obs, **kwargs)
            noise = torch.rand(actor_next_action.shape, device=self.device) * self.actor_noise
            noise = torch.clamp(noise, -self.actor_noise_clip, self.actor_noise_clip)
            noisy_actor_next_action = actor_next_action + noise
            target_Q1, target_Q2 = self.critic_target(
                next_obs, noisy_actor_next_action, **kwargs)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, **kwargs)

        # critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        return critic_loss

    def update_critic(self, critic_loss, logger, step):
        logger.log('train_critic/loss', critic_loss, step)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def compute_actor_loss(self, obs, **kwargs):
        actor_action = self.actor(obs, **kwargs)
        actor_loss = -self.critic.Q1(obs, actor_action, **kwargs).mean()

        return actor_loss

    def update_actor(self, actor_loss, logger, step):
        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update(self, replay_buffer, logger, step, **kwargs):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done, **kwargs)
        self.update_critic(critic_loss, logger, step)

        if step % self.actor_and_target_update_freq == 0:
            actor_loss = self.compute_actor_loss(obs, **kwargs)
            self.update_actor(actor_loss, logger, step)

            utils.soft_update_params(self.actor, self.actor_target, self.target_tau)
            utils.soft_update_params(self.critic, self.critic_target, self.target_tau)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

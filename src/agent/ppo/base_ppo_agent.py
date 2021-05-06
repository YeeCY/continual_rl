import torch
import numpy as np
from itertools import chain
import utils
import torch.nn.functional as F

from agent.network import PpoActorMlp, PpoCriticMlp


class PpoMlpAgent:
    """Adapt from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail"""
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            hidden_dim=64,
            discount=0.99,
            clip_param=0.2,
            ppo_epoch=10,
            critic_loss_coef=0.5,
            entropy_coef=0.0,
            lr=3e-4,
            eps=1e-5,
            grad_clip_norm=0.5,
            use_clipped_critic_loss=True,
            batch_size=32,
    ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.hidden_dim = hidden_dim
        self.discount = discount
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.eps = eps
        self.grad_clip_norm = grad_clip_norm
        self.use_clipped_critic_loss = use_clipped_critic_loss
        self.batch_size = batch_size

        self.training = False

        self._setup_agent()

        self.train()

    def _setup_agent(self):
        self.actor = PpoActorMlp(
            self.obs_shape, self.action_shape, self.hidden_dim
        ).to(self.device)

        self.critic = PpoCriticMlp(
            self.obs_shape, self.hidden_dim
        ).to(self.device)

        # TODO (chongyi zheng): delete this line
        # tie encoders between actor and criticp
        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        # self.log_alpha.requires_grad = True
        # # set target entropy to -|A|
        # self.target_entropy = -np.prod(self.action_shape)
        #
        # # sac optimizers
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        #
        # self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.optimizer = torch.optim.Adam(chain(self.actor.parameters(), self.critic.parameters()),
                                          lr=self.lr, eps=self.eps)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def predict_value(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        with torch.no_grad():
            value = self.critic(obs)

        return utils.to_np(value)

    def act(self, obs, sample=False, compute_log_pi=True):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        with torch.no_grad():
            mu, pi, log_pi = self.actor(obs, compute_log_pi=compute_log_pi)
            action = pi if sample else mu

        return utils.to_np(action), utils.to_np(log_pi)

    def compute_critic_loss(self, obs, value_pred, ret):
        # with torch.no_grad():
        #     _, policy_action, log_pi, _ = self.actor(next_obs)
        #     target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
        #     target_V = torch.min(target_Q1,
        #                          target_Q2) - self.alpha.detach() * log_pi
        #     target_Q = reward + (not_done * self.discount * target_V)
        #
        # # get current Q estimates
        # current_Q1, current_Q2 = self.critic(obs, action)
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        #
        # return critic_loss
        value = self.critic(obs)

        if self.use_clipped_critic_loss:
            value_pred_clipped = value_pred + (value - value_pred).clamp(-self.clip_param, self.clip_param)
            critic_losses = (value - ret).pow(2)
            critic_losses_clipped = (value_pred_clipped - ret).pow(2)
            critic_loss = 0.5 * torch.max(critic_losses, critic_losses_clipped).mean()
        else:
            critic_loss = 0.5 * (ret - value).pow(2).mean()

        return critic_loss

    def compute_actor_loss(self, obs, action, old_log_pi, adv_target):
        log_pi, entropy = self.actor.compute_log_probs(obs, action)

        ratio = torch.exp(log_pi - old_log_pi)
        surr1 = ratio * adv_target
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_target
        actor_loss = -torch.min(surr1, surr2).mean()

        return actor_loss, entropy

    # def update_actor_(self, critic_loss, logger, step):
    #     # Optimize the critic
    #     logger.log('train_critic/loss', critic_loss, step)
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
    #     self.critic_optimizer.step()

    # def update_actor_and_alpha(self, log_pi, actor_loss, logger, step, alpha_loss=None):
    #     logger.log('train_actor/loss', actor_loss, step)
    #     logger.log('train_actor/target_entropy', self.target_entropy, step)
    #     logger.log('train_actor/entropy', -log_pi.mean(), step)
    #
    #     # optimize the actor
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
    #     self.actor_optimizer.step()
    #
    #     if isinstance(alpha_loss, torch.Tensor):
    #         logger.log('train_alpha/loss', alpha_loss, step)
    #         logger.log('train_alpha/value', self.alpha, step)
    #
    #         self.log_alpha_optimizer.zero_grad()
    #         alpha_loss.backward()
    #         torch.nn.utils.clip_grad_norm_([self.log_alpha], self.grad_clip_norm)
    #         self.log_alpha_optimizer.step()

    def update_learning_rate(self, epoch, total_epochs):
        lr = self.lr - (self.lr * (epoch / float(total_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, rollouts, logger, step):
        # obs, action, reward, next_obs, not_done, ensem_kwargs = replay_buffer.sample_ensembles(
        #     self.batch_size, num_ensembles=self.num_ensem_comps)
        # obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)
        # obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
        #     self.batch_size)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        logger.log('train/batch_normalized_advantages', advantages.mean(), step)

        # critic_loss = self.compute_critic_loss(obs, action, reward, next_obs, not_done)
        # self.update_critic(critic_loss, logger, step)
        #
        # if step % self.actor_update_freq == 0:
        #     log_pi, actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(obs)
        #     self.update_actor_and_alpha(log_pi, actor_loss, logger, step, alpha_loss=alpha_loss)
        #
        # if step % self.critic_target_update_freq == 0:
        #     utils.soft_update_params(self.critic, self.critic_target,
        #                              self.critic_tau)

        # value_loss_epoch = 0
        # action_loss_epoch = 0
        # dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.batch_size)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, \
                return_batch, old_log_pis, adv_targets = sample

                # Reshape to do in a single forward pass for all steps
                # values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                #     obs_batch, recurrent_hidden_states_batch, masks_batch,
                #     actions_batch)
                actor_loss, entropy = self.compute_actor_loss(
                    obs_batch, actions_batch, old_log_pis, adv_targets)
                critic_loss = self.compute_critic_loss(
                    obs_batch, value_preds_batch, return_batch)
                loss = actor_loss + self.critic_loss_coef * critic_loss - \
                       self.entropy_coef * entropy

                logger.log('train_actor/loss', actor_loss, step)
                logger.log('train_actor/entropy', entropy, step)
                logger.log('train_critic/loss', critic_loss, step)
                logger.log('train/loss', loss, step)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    chain(self.actor.parameters(), self.critic.parameters()),
                    self.grad_clip_norm)
                self.optimizer.step()

                # value_loss_epoch += value_loss.item()
                # action_loss_epoch += action_loss.item()
                # dist_entropy_epoch += dist_entropy.item()

        # num_updates = self.ppo_epoch * self.batch_size

        # value_loss_epoch /= num_updates
        # action_loss_epoch /= num_updates
        # dist_entropy_epoch /= num_updates
        #
        # return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

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

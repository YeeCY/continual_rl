import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.utils import get_linear_fn

from agent.network import ActorCnn, ActorMlp, CriticCnn, CriticMlp, CURL, \
    SelfSupervisedCnnInvPredictorEnsem, SelfSupervisedMlpInvPredictorEnsem, \
    SelfSupervisedCnnFwdPredictorEnsem, SelfSupervisedMlpFwdPredictorEnsem, \
    DQNCnn, DQNDuelingCnn


class DqnCnnSSEnsembleAgent(object):
    """
    DQN with an auxiliary self-supervised task.
    Based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py
    Enable double and dueling q learning

    """
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            double_q=True,
            dueling=True,
            feature_dim=512,
            discount=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=10000,
            max_grad_norm=10,
            q_net_lr=1e-4,
            q_net_tau=1.0,
            use_fwd=False,
            use_inv=False,
            ss_lr=1e-3,
            ss_update_freq=1,
            batch_size=32,
    ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.double_q = double_q
        self.dueling = dueling
        self.discount = discount
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.q_net_lr = q_net_lr
        self.q_net_tau = q_net_tau
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

        self.ss_lr = ss_lr
        self.ss_update_freq = ss_update_freq
        self.batch_size = batch_size
        self.use_fwd = use_fwd
        self.use_inv = use_inv

        if self.dueling:
            self.q_net = DQNDuelingCnn(
                obs_shape, action_shape, feature_dim).to(self.device)
            self.target_q_net = DQNDuelingCnn(
                obs_shape, action_shape, feature_dim).to(self.device)
        else:
            self.q_net = DQNCnn(
                obs_shape, action_shape, feature_dim).to(self.device)
            self.target_q_net = DQNCnn(
                obs_shape, action_shape, feature_dim).to(self.device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # # tie encoders between actor and critic
        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        #
        # self.log_alpha = torch.tensor(np.log(init_temperature)).cuda()
        # self.log_alpha.requires_grad = True
        # # set target entropy to -|A|
        # self.target_entropy = -np.prod(action_shape)

        # self-supervision
        self.rot = None
        self.inv = None
        self.ss_encoder = None

        # if use_rot or use_inv:
        #     self.ss_encoder = make_encoder(
        #         obs_shape, encoder_feature_dim, num_layers,
        #         num_filters, num_shared_layers
        #     ).cuda()
        #     self.ss_encoder.copy_conv_weights_from(self.critic.encoder, num_shared_layers)
        #
        #     # rotation
        #     if use_rot:
        #         self.rot = RotFunction(encoder_feature_dim, hidden_dim).cuda()
        #         self.rot.apply(utils.weight_init)
        #
        #     # inverse dynamics
        #     if use_inv:
        #         self.inv = InvFunction(encoder_feature_dim, action_shape[0], hidden_dim).cuda()
        #         self.inv.apply(utils.weight_init)

        # ss optimizers
        # self.init_ss_optimizers(encoder_lr, ss_lr)

        # dqn optimizers
        self.q_net_optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.q_net_lr)

        self.train()

    def init_ss_optimizers(self, encoder_lr=1e-3, ss_lr=1e-3):
        if self.ss_encoder is not None:
            self.encoder_optimizer = torch.optim.Adam(
                self.ss_encoder.parameters(), lr=encoder_lr
            )
        if self.use_inv:
            self.inv_optimizer = torch.optim.Adam(
                self.inv.parameters(), lr=ss_lr
            )

    def train(self, training=True):
        self.training = training
        self.q_net.train(training)
        if self.ss_encoder is not None:
            self.ss_encoder.train(training)
        if self.rot is not None:
            self.rot.train(training)
        if self.inv is not None:
            self.inv.train(training)

    # def select_action(self, obs):
    #     with torch.no_grad():
    #         obs = torch.FloatTensor(obs).cuda()
    #         obs = obs.unsqueeze(0)
    #         mu, _, _, _ = self.actor(
    #             obs, compute_pi=False, compute_log_pi=False
    #         )
    #         return mu.cpu().data.numpy().flatten()
    #
    # def sample_action(self, obs):
    #     with torch.no_grad():
    #         obs = torch.FloatTensor(obs).cuda()
    #         obs = obs.unsqueeze(0)
    #         mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
    #         return pi.cpu().data.numpy().flatten()

    def act(self, obs, explore=False):
        # TODO (chongyi zheng)
        if explore and np.random.rand() < self.exploration_rate:
            action = np.random.randint(low=0, high=self.action_shape[0])
        else:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                q_values = self.q_net(obs)
                # greed action
                action = q_values.argmax(dim=1).reshape(-1)
                assert action.shape[0] == 1

                action = utils.to_np(action[0])

        return action

    def update_q_net(self, obs, action, reward, next_obs, not_done, logger, step):
        with torch.no_grad():
            # compute the next Q-values using the target network
            next_q_values = self.target_q_net(next_obs)
            # follow greedy policy: use the one with the highest value
            if self.double_q:
                best_next_actions = torch.argmax(self.q_net(next_obs), dim=-1)
                next_q_values = next_q_values.gather(1, best_next_actions.unsqueeze(-1))
            else:
                next_q_values = next_q_values.max(dim=1, keepdims=True)[0]
            # 1-step TD target
            target_q_values = reward + not_done * self.discount * next_q_values

        # get current Q estimates
        current_q_values = self.q_net(obs)

        # retrieve the q-values for the actions from the replay buffer
        current_q_values = torch.gather(current_q_values, dim=1, index=action.long())

        # Huber loss (less sensitive to outliers)
        q_net_loss = F.smooth_l1_loss(current_q_values, target_q_values)

        logger.log('train_q_net/loss', q_net_loss, step)

        # optimize the Q network
        self.q_net_optimizer.zero_grad()
        q_net_loss.backward()
        # TODO (chongyi zheng): Do we need to clip gradient norm?
        # clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_net_optimizer.step()

    # def update_inv(self, obs, next_obs, action, L=None, step=None):
    #     assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84
    #
    #     h = self.ss_encoder(obs)
    #     h_next = self.ss_encoder(next_obs)
    #
    #     pred_action = self.inv(h, h_next)
    #     inv_loss = F.mse_loss(pred_action, action)
    #
    #     self.encoder_optimizer.zero_grad()
    #     self.inv_optimizer.zero_grad()
    #     inv_loss.backward()
    #
    #     self.encoder_optimizer.step()
    #     self.inv_optimizer.step()
    #
    #     if L is not None:
    #         L.log('train_inv/inv_loss', inv_loss, step)
    #
    #     return inv_loss.item()

    def update(self, replay_buffer, logger, step, total_steps):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_q_net(obs, action, reward, next_obs, not_done, logger, step)

        if step % self.target_update_interval == 0:
            utils.soft_update_params(self.q_net, self.target_q_net, self.q_net_tau)

        self.exploration_rate = self.exploration_schedule(1.0 - float(step) / float(total_steps))

        logger.log('train/exploration_rate', self.exploration_rate, step)

        # if self.inv is not None and step % self.ss_update_freq == 0:
        #     self.update_inv(obs, next_obs, action, logger, step)

    def save(self, model_dir, step):
        torch.save(
            self.q_net.state_dict(), '%s/q_net_%s.pt' % (model_dir, step)
        )
        # if self.inv is not None:
        #     torch.save(
        #         self.inv.state_dict(),
        #         '%s/inv_%s.pt' % (model_dir, step)
        #     )
        # if self.ss_encoder is not None:
        #     torch.save(
        #         self.ss_encoder.state_dict(),
        #         '%s/ss_encoder_%s.pt' % (model_dir, step)
        #     )

    def load(self, model_dir, step):
        self.q_net.load_state_dict(
            torch.load('%s/q_net_%s.pt' % (model_dir, step))
        )
        # if self.inv is not None:
        #     self.inv.load_state_dict(
        #         torch.load('%s/inv_%s.pt' % (model_dir, step))
        #     )
        # if self.ss_encoder is not None:
        #     self.ss_encoder.load_state_dict(
        #         torch.load('%s/ss_encoder_%s.pt' % (model_dir, step))
        #     )

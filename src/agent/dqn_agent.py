import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.utils import get_linear_fn
from stable_baselines3.common.utils import polyak_update

from agent.network import DqnCnnSSFwdPredictorEnsem, DqnCnnSSInvPredictorEnsem, \
    DQNCnn, DQNDuelingCnn


class DqnCnnAgent:
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
            double_q=False,
            dueling=False,
            feature_dim=512,
            discount=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=10000,
            max_grad_norm=10,
            q_net_lr=1e-4,
            q_net_tau=1.0,
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
        self.batch_size = batch_size

        self.training = False

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

        # dqn optimizers
        self.q_net_optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.q_net_lr)

        # self.train()
        # self.target_q_net.train()

    # def train(self, training=True):
    #     self.training = training
    #     self.q_net.train(training)

    def act(self, obs, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_shape)
        else:
            with torch.no_grad():
                # observation = th.as_tensor(observation).to(self.device)
                # observation = th.FloatTensor(observation).to(self.device)
                obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                q_values = self.q_net(obs)
                # Greedy action
                action = q_values.argmax(dim=1).reshape(-1)
                action = utils.to_np(action)[0]

        return action

    def on_step(self, step, total_steps, logger):
        # if step % self.target_update_interval == 0:
        #     utils.soft_update_params(self.q_net, self.target_q_net, self.q_net_tau)
        if step % self.target_update_interval == 0:
            # polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.q_net_tau)

        self.exploration_rate = self.exploration_schedule(1.0 - float(step) / float(total_steps))
        logger.log('train/exploration_rate', self.exploration_rate, step)

    def update_q_net(self, obs, action, reward, next_obs, not_done, logger, step):
        with torch.no_grad():
            # compute the next Q-values using the target network
            next_q_values = self.target_q_net(next_obs)
            # follow greedy policy: use the one with the highest value
            if self.double_q:
                best_next_actions = torch.argmax(self.q_net(next_obs), dim=-1)
                next_q_values = next_q_values.gather(1, best_next_actions.unsqueeze(-1))
            else:
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = reward + not_done * self.discount * next_q_values

        # get current Q estimates
        current_q_values = self.q_net(obs)

        # retrieve the q-values for the actions from the replay buffer
        current_q_values = torch.gather(current_q_values, dim=1, index=action.long())

        # Huber loss (less sensitive to outliers)
        q_net_loss = F.smooth_l1_loss(current_q_values, target_q_values)

        logger.log('train/q_net_loss', q_net_loss, step)

        # optimize the Q network
        self.q_net.zero_grad()
        q_net_loss.backward()
        # TODO (chongyi zheng): Do we need to clip gradient norm?
        # clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_net_optimizer.step()

    def update(self, replay_buffer, logger, step):
        # obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)
        samples = replay_buffer.sample(self.batch_size)
        obs = samples.observations
        action = samples.actions
        next_obs = samples.next_observations
        not_done = 1.0 - samples.dones
        reward = samples.rewards

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_q_net(obs, action, reward, next_obs, not_done, logger, step)

        # if step % self.target_update_interval == 0:
        #     utils.soft_update_params(self.q_net, self.target_q_net, self.q_net_tau)

    def save(self, model_dir, step):
        torch.save(
            self.q_net.state_dict(), '%s/q_net_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.q_net.load_state_dict(
            torch.load('%s/q_net_%s.pt' % (model_dir, step))
        )


class DqnCnnSSEnsembleAgent(DqnCnnAgent):
    def __init__(self,
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
                 num_ensem_comps=4,
                 batch_size=32):
        super().__init__(obs_shape,
                         action_shape,
                         device,
                         double_q=double_q,
                         dueling=dueling,
                         feature_dim=feature_dim,
                         discount=discount,
                         exploration_fraction=exploration_fraction,
                         exploration_initial_eps=exploration_initial_eps,
                         exploration_final_eps=exploration_final_eps,
                         target_update_interval=target_update_interval,
                         max_grad_norm=max_grad_norm,
                         q_net_lr=q_net_lr,
                         q_net_tau=q_net_tau,
                         batch_size=batch_size)
        self.use_fwd = use_fwd
        self.use_inv = use_inv
        self.ss_lr = ss_lr
        self.ss_update_freq = ss_update_freq
        self.num_ensem_comps = num_ensem_comps

        if self.use_fwd:
            self.ss_fwd_pred_ensem = DqnCnnSSFwdPredictorEnsem(
                obs_shape, feature_dim, num_ensem_comps).to(self.device)
            # TODO (chongyi zheng): use tie weights
            self.ss_fwd_pred_ensem.encoder.copy_conv_weights_from(self.q_net.encoder)
            self.ss_fwd_optimizer = torch.optim.Adam(
                self.ss_fwd_pred_ensem.parameters(), lr=ss_lr)

            # self.ss_fwd_pred_ensem.train()

        if self.use_inv:
            self.ss_inv_pred_ensem = DqnCnnSSInvPredictorEnsem(
                obs_shape, action_shape, feature_dim, num_ensem_comps).to(self.device)
            self.ss_inv_pred_ensem.encoder.copy_conv_weights_from(self.q_net.encoder)
            self.ss_inv_optimizer = torch.optim.Adam(
                self.ss_inv_pred_ensem.parameters(), lr=ss_lr)

            # self.ss_inv_pred_ensem.train()

    def ss_preds_var(self, obs, next_obs, action):
        # TODO (chongyi zheng):
        #  do we need next_obs (forward) or action (inverse) - measure the prediction error,
        #  or we just need to predictions - measure the prediction variance?
        #  task identity inference - threshold or statistical hypothesis testing like: https://arxiv.org/abs/1902.09434
        assert obs.shape == next_obs.shape and obs.shape[0] == next_obs.shape[0] == action.shape[0], \
            "invalid transitions shapes!"

        # TODO (chongyi zheng): Do we need to set agent mode to evaluation before prediction?
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device) \
                if not isinstance(obs, torch.Tensor) else obs.to(self.device)
            next_obs = torch.FloatTensor(next_obs).to(self.device) \
                if not isinstance(next_obs, torch.Tensor) else next_obs.to(self.device)
            action = torch.FloatTensor(action).to(self.device) \
                if not isinstance(action, torch.Tensor) else action.to(self.device)

            if len(obs.size()) == 3 or len(obs.size()) == 1:
                obs = obs.unsqueeze(0)
                next_obs = next_obs.unsqueeze(0)
                action = action.unsqueeze(0)

            # prediction variances
            if self.use_fwd:
                preds = self.ss_fwd_pred_ensem(obs, action)

            if self.use_inv:
                # (chongyi zheng): we compute logits variance here
                preds = self.ss_inv_pred_ensem(obs, next_obs)

            # (chongyi zheng): the same as equation (1) in https://arxiv.org/abs/1906.04161
            preds = torch.stack(preds.chunk(self.num_ensem_comps, dim=0))
            preds_var = torch.var(preds, dim=0).sum(dim=-1)

            return utils.to_np(preds_var)

    def update_ss_preds(self, obs, next_obs, action, logger, step):
        assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84

        # TODO (chongyi zheng): Do we need to stop the gradients from self-supervision loss?
        if self.use_fwd:
            pred_h_next = self.ss_fwd_pred_ensem(obs, action,
                                                 detach_encoder=True, split_hidden=True)
            h_next = self.ss_fwd_pred_ensem.encoder(next_obs).detach()  # stop gradient for the target
            fwd_loss = F.mse_loss(pred_h_next, h_next)

            self.ss_fwd_optimizer.zero_grad()
            fwd_loss.backward()
            # TODO (chongyi zheng): Do we need to clip gradient norm?
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.ss_fwd_pred_ensem.parameters(), self.max_grad_norm)
            self.ss_fwd_optimizer.step()

            self.ss_fwd_pred_ensem.log(logger, step)
            logger.log('train/ss_fwd_loss', fwd_loss, step)

        if self.use_inv:
            pred_logit = self.ss_inv_pred_ensem(obs, next_obs,
                                                detach_encoder=True, split_hidden=True)
            inv_loss = F.cross_entropy(pred_logit, action.squeeze(-1).long())

            self.ss_inv_optimizer.zero_grad()
            inv_loss.backward()
            # TODO (chongyi zheng): Do we need to clip gradient norm?
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.ss_inv_pred_ensem.parameters(), self.max_grad_norm)
            self.ss_inv_optimizer.step()

            self.ss_inv_pred_ensem.log(logger, step)
            logger.log('train/ss_inv_loss', inv_loss, step)

    def update(self, replay_buffer, logger, step):
        # TODO (chongyi zheng): fix duplication
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)
        # samples = replay_buffer.sample(self.batch_size)
        # obs = samples.observations
        # action = samples.actions
        # next_obs = samples.next_observations
        # not_done = 1.0 - samples.dones
        # reward = samples.rewards

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_q_net(obs, action, reward, next_obs, not_done, logger, step)

        if (self.use_fwd or self.use_inv) and step % self.ss_update_freq == 0:
            self.update_ss_preds(obs, next_obs, action, logger, step)
            ss_preds_var = self.ss_preds_var(obs, next_obs, action)
            logger.log('train/batch_ss_preds_var', ss_preds_var.mean(), step)

    def save(self, model_dir, step):
        super().save(model_dir, step)
        if self.use_fwd:
            torch.save(
                self.ss_fwd_pred_ensem.state_dict(),
                '%s/ss_fwd_pred_ensem_%s.pt' % (model_dir, step)
            )
        if self.use_inv:
            torch.save(
                self.ss_inv_pred_ensem.state_dict(),
                '%s/ss_inv_pred_ensem_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        super().load(model_dir, step)
        if self.use_fwd:
            self.ss_inv_pred_ensem.load_state_dict(
                torch.load('%s/ss_fwd_pred_ensem_%s.pt' % (model_dir, step))
            )
        if self.use_inv:
            self.ss_inv_pred_ensem.load_state_dict(
                torch.load('%s/ss_inv_pred_ensem_%s.pt' % (model_dir, step))
            )

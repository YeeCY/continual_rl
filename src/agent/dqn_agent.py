import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.utils import LinearSchedule

from agent.network import DqnCnnSSFwdPredictorEnsem, DqnCnnSSInvPredictorEnsem, \
    DqnCnn, DqnDuelingCnn, DqnCategoricalCnn


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
            exploration_anneal_steps=int(1e6),
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            target_update_interval=40000,
            max_grad_norm=0.5,
            categorical_n_atoms=51,
            categorical_v_min=-10,
            categorical_v_max=10,
            q_net_opt_lr=2.5e-4,
            q_net_opt_eps=0.01 / 32,
            batch_size=32,
    ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.double_q = double_q
        self.dueling = dueling
        self.discount = discount
        self.exploration_anneal_steps = exploration_anneal_steps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.categorical_n_atoms = categorical_n_atoms
        self.categorical_v_min = categorical_v_min
        self.categorical_v_max = categorical_v_max
        self.q_net_opt_lr = q_net_opt_lr
        self.q_net_opt_eps = q_net_opt_eps
        self.exploration_rate = 1.0  # "epsilon" for the epsilon-greedy exploration
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_anneal_steps
        )
        self.batch_size = batch_size

        self.training = False
        self.batch_indices = torch.arange(self.batch_size).long().to(self.device)
        self.atoms = torch.linspace(
            categorical_v_min, categorical_v_max, categorical_n_atoms).to(self.device)
        self.delta_atom = (categorical_v_max - categorical_v_min) / float(categorical_n_atoms - 1)


        if self.dueling:
            self.q_net = DqnDuelingCnn(
                obs_shape, action_shape, feature_dim).to(self.device)
            self.target_q_net = DqnDuelingCnn(
                obs_shape, action_shape, feature_dim).to(self.device)
        else:
            self.q_net = DqnCategoricalCnn(
                obs_shape, action_shape, feature_dim, categorical_n_atoms).to(self.device)
            self.target_q_net = DqnCategoricalCnn(
                obs_shape, action_shape, feature_dim, categorical_n_atoms).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # dqn optimizers
        self.q_net_optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.q_net_opt_lr,
            eps=self.q_net_opt_eps,
        )

        self.train()
        self.target_q_net.train()

    def train(self, training=True):
        self.training = training
        self.q_net.train(training)

    def act(self, obs, exploration=False):
        # TODO (chongyi zheng)
        if exploration and np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_shape)
        else:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                prob, _ = self.q_net(obs)
                q_values = (prob * self.atoms).sum(-1)
                # greed action
                action = utils.to_np(q_values.argmax(dim=-1))

        return action

    def schedule_exploration_rate(self, step, logger):
        # TODO (chongyi zheng): update exploration rate
        self.exploration_rate = self.exploration_schedule()
        logger.log('train/exploration_rate', self.exploration_rate, step)

    def update_q_net(self, obs, action, reward, next_obs, not_done, logger, step):
        # with torch.no_grad():
        #     # compute the next Q-values using the target network
        #     next_q_values = self.target_q_net(next_obs).detach()
        #     # follow greedy policy: use the one with the highest value
        #     if self.double_q:
        #         best_next_actions = torch.argmax(self.q_net(next_obs), dim=-1)
        #         next_q_values = next_q_values.gather(1, best_next_actions.unsqueeze(-1))
        #     else:
        #         # Follow greedy policy: use the one with the highest value
        #         next_q_values = next_q_values.max(dim=1)[0]
        #         # Avoid potential broadcast issue
        #         next_q_values = next_q_values.reshape(-1, 1)
        #     # 1-step TD target
        #     target_q_values = reward + not_done * self.discount * next_q_values
        #
        # # get current Q estimates
        # current_q_values = self.q_net(obs)
        #
        # # retrieve the q-values for the actions from the replay buffer
        # current_q_values = current_q_values.gather(1, index=action.long())
        #
        # # Huber loss (less sensitive to outliers)
        # q_net_loss = F.smooth_l1_loss(current_q_values, target_q_values)
        # target z probability
        with torch.no_grad():
            next_prob = self.target_q_net(next_obs)[0]
            next_q_values = (next_prob * self.atoms).sum(-1)
            if self.double_q:
                next_action = torch.argmax((self.q_net(next_obs)[0] * self.atoms).sum(-1), dim=-1)
            else:
                next_action = torch.argmax(next_q_values, dim=-1)
            next_prob = next_prob[self.batch_indices, next_action, :]

            target_atoms = reward + self.discount * not_done * self.atoms.view(1, -1)
            target_atoms.clamp(self.categorical_v_min, self.categorical_v_max).unsqueeze(1)
            target_prob = (1 - (target_atoms - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                          next_prob.unsqueeze(1)
            target_prob = target_prob.sum(-1)

        # current z probability
        log_prob = self.q_net(obs)[1]
        log_prob = log_prob[self.batch_indices, action.long(), :]

        # KL divergence
        q_net_loss = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        q_net_loss = q_net_loss.mean()

        logger.log('train/q_net_loss', q_net_loss, step)

        # optimize the Q network
        self.q_net_optimizer.zero_grad()
        q_net_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_net_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_q_net(obs, action, reward, next_obs, not_done, logger, step)

        # update target q network
        if step % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

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
                 double_q=False,
                 dueling=False,
                 feature_dim=512,
                 discount=0.99,
                 exploration_anneal_steps=int(1e6),
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.01,
                 target_update_interval=40000,
                 max_grad_norm=0.5,
                 categorical_n_atoms=51,
                 categorical_v_min=-10,
                 categorical_v_max=10,
                 q_net_opt_lr=2.5e-4,
                 q_net_opt_eps=0.01 / 32,
                 batch_size=32,
                 use_fwd=False,
                 use_inv=False,
                 ss_lr=1e-3,
                 ss_update_freq=1,
                 num_ensem_comps=4):
        super().__init__(obs_shape,
                         action_shape,
                         device,
                         double_q=double_q,
                         dueling=dueling,
                         feature_dim=feature_dim,
                         discount=discount,
                         exploration_anneal_steps=exploration_anneal_steps,
                         exploration_initial_eps=exploration_initial_eps,
                         exploration_final_eps=exploration_final_eps,
                         target_update_interval=target_update_interval,
                         max_grad_norm=max_grad_norm,
                         categorical_n_atoms=51,
                         categorical_v_min=-10,
                         categorical_v_max=10,
                         q_net_opt_lr=2.5e-4,
                         q_net_opt_eps=0.01 / 32,
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

            self.ss_fwd_pred_ensem.train()

        if self.use_inv:
            self.ss_inv_pred_ensem = DqnCnnSSInvPredictorEnsem(
                obs_shape, action_shape, feature_dim, num_ensem_comps).to(self.device)
            self.ss_inv_pred_ensem.encoder.copy_conv_weights_from(self.q_net.encoder)
            self.ss_inv_optimizer = torch.optim.Adam(
                self.ss_inv_pred_ensem.parameters(), lr=ss_lr)

            self.ss_inv_pred_ensem.train()

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

        # update target q network
        if step % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

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

import torch
import numpy as np
import copy
from itertools import chain
from collections import OrderedDict


from agent.ppo.base_ppo_agent import PpoMlpAgent
from agent.network import CmamlPpoActorMlp, PpoCriticMlp


class CmamlPpoMlpAgentV2(PpoMlpAgent):
    """Adapt https://github.com/GMvandeVen/continual-learning"""
    def __init__(self,
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
                 num_batch=32,
                 cmaml_inner_grad_steps=8,
                 cmaml_fast_lr=1e-5,
                 cmaml_meta_lr=3e-4,
                 cmaml_memory_budget=10240,
                 cmaml_first_order=True,
                 ):
        self.cmaml_inner_grad_steps = cmaml_inner_grad_steps
        self.cmaml_fast_lr = cmaml_fast_lr
        self.cmaml_meta_lr = cmaml_meta_lr
        self.cmaml_memory_budget = cmaml_memory_budget
        self.cmaml_first_order = cmaml_first_order

        super().__init__(obs_shape, action_shape, device, hidden_dim, discount, clip_param, ppo_epoch,
                         critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm, use_clipped_critic_loss,
                         num_batch)

        self.cmaml_total_sample_num = 0

        # initialize memory
        self.cmaml_memory = {
            'obses': torch.zeros(self.cmaml_memory_budget, *obs_shape).to(self.device),
            'actions': torch.zeros(self.cmaml_memory_budget, *action_shape).to(self.device),
            'value_preds': torch.zeros(self.cmaml_memory_budget, 1).to(self.device),
            'returns': torch.zeros(self.cmaml_memory_budget, 1).to(self.device),
            'old_log_pis': torch.zeros(self.cmaml_memory_budget, 1).to(self.device),
            'adv_targets': torch.zeros(self.cmaml_memory_budget, 1).to(self.device),
            'sample_num': 0
        }

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'actor_meta_optimizer') \
                and hasattr(self, 'critic_optimizer'):
            return

        self.actor = CmamlPpoActorMlp(
            self.obs_shape, self.action_shape, self.hidden_dim
        ).to(self.device)

        self.critic = PpoCriticMlp(
            self.obs_shape, self.hidden_dim
        ).to(self.device)

        self.actor_meta_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                     lr=self.cmaml_meta_lr, eps=self.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr, eps=self.eps)

        self._critic_init_state = copy.deepcopy(self.critic.state_dict())
        self._critic_optimizer_init_state = copy.deepcopy(self.critic_optimizer.state_dict())

    def _augment_prev_samples(self, obses, actions, value_preds,
                              returns, old_log_pis, adv_targets):
        aug_size = min(obses.size(0), self.cmaml_memory['sample_num'])
        if aug_size > 0:
            sample_idxs = np.random.randint(
                0, self.cmaml_memory['sample_num'], size=aug_size)
            aug_obses = torch.cat([obses, self.cmaml_memory['obses'][sample_idxs]])
            aug_actions = torch.cat([actions, self.cmaml_memory['actions'][sample_idxs]])
            aug_value_preds = torch.cat([value_preds, self.cmaml_memory['value_preds'][sample_idxs]])
            aug_returns = torch.cat([returns, self.cmaml_memory['returns'][sample_idxs]])
            aug_old_log_pis = torch.cat([old_log_pis, self.cmaml_memory['old_log_pis'][sample_idxs]])
            aug_adv_targets = torch.cat([adv_targets, self.cmaml_memory['adv_targets'][sample_idxs]])
        else:
            aug_obses = obses
            aug_actions = actions
            aug_value_preds = value_preds
            aug_returns = returns
            aug_old_log_pis = old_log_pis
            aug_adv_targets = adv_targets

        return aug_obses, aug_actions, aug_value_preds, \
               aug_returns, aug_old_log_pis, aug_adv_targets

    def _reservoir_sampling(self, obses, actions, value_preds,
                            returns, old_log_pis, adv_targets):
        for obs, action, value_pred, ret, old_log_pi, adv_target \
                in zip(obses, actions, value_preds, returns, old_log_pis, adv_targets):
            self.cmaml_total_sample_num += 1
            sample_num = self.cmaml_memory['sample_num']
            if sample_num < self.cmaml_memory_budget:
                self.cmaml_memory['obses'][sample_num].copy_(obs)
                self.cmaml_memory['actions'][sample_num].copy_(action)
                self.cmaml_memory['value_preds'][sample_num].copy_(value_pred)
                self.cmaml_memory['returns'][sample_num].copy_(ret)
                self.cmaml_memory['old_log_pis'][sample_num].copy_(old_log_pi)
                self.cmaml_memory['adv_targets'][sample_num].copy_(adv_target)
                self.cmaml_memory['sample_num'] += 1
            else:
                idx = np.random.randint(0, self.cmaml_total_sample_num)
                if idx < self.cmaml_memory_budget:
                    self.cmaml_memory['obses'][idx].copy_(obs)
                    self.cmaml_memory['actions'][idx].copy_(action)
                    self.cmaml_memory['value_preds'][idx].copy_(value_pred)
                    self.cmaml_memory['returns'][idx].copy_(ret)
                    self.cmaml_memory['old_log_pis'][idx].copy_(old_log_pi)
                    self.cmaml_memory['adv_targets'][idx].copy_(adv_target)

    def reset(self):
        self.critic.load_state_dict(self._critic_init_state)
        self.critic_optimizer.load_state_dict(self._critic_optimizer_init_state)

    def actor_inner_update(self, obs, action, old_log_pi, adv_target, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.actor.named_parameters())

        log_pi, entropy = self.actor.compute_log_probs(obs, action, params=params, **kwargs)

        ratio = torch.exp(log_pi - old_log_pi)
        surr1 = ratio * adv_target
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_target
        actor_loss = -torch.min(surr1, surr2).mean()

        grads = torch.autograd.grad(actor_loss, params.values(),
                                    create_graph=not self.cmaml_first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            torch.clamp(grad, -self.grad_clip_norm, self.grad_clip_norm)
            updated_params[name] = param - self.cmaml_fast_lr * grad

        return updated_params

    def update_learning_rate(self, epoch, total_epochs):
        lr = self.lr - (self.lr * (epoch / float(total_epochs)))
        meta_lr = self.cmaml_meta_lr - (self.cmaml_meta_lr * (epoch / float(total_epochs)))
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.actor_meta_optimizer.param_groups:
            param_group['lr'] = meta_lr

    def update(self, rollouts, logger, step, **kwargs):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        logger.log('train/batch_normalized_advantages', advantages.mean(), step)

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, \
                return_batch, old_log_pis, adv_targets = sample

                # Augment samples
                aug_obs_batch, aug_actions_batch, aug_value_preds_batch, \
                aug_return_batch, aug_old_log_pis, aug_adv_targets = \
                    self._augment_prev_samples(obs_batch, actions_batch,
                                               value_preds_batch, return_batch,
                                               old_log_pis, adv_targets)

                # Reservoir samping
                self._reservoir_sampling(obs_batch, actions_batch,
                                         value_preds_batch, return_batch,
                                         old_log_pis, adv_targets)

                # Actor inner gradient update
                assert obs_batch.size(0) >= self.cmaml_inner_grad_steps, \
                    "Batch size should be greater than or equal to inner gradient steps!"
                min_batch_size = obs_batch.size(0) // self.cmaml_inner_grad_steps
                fast_params = None
                for inner_step in range(self.cmaml_inner_grad_steps):
                    fast_params = self.actor_inner_update(
                        obs_batch[inner_step * min_batch_size:(inner_step + 1) * min_batch_size],
                        actions_batch[inner_step * min_batch_size:(inner_step + 1) * min_batch_size],
                        old_log_pis[inner_step * min_batch_size:(inner_step + 1) * min_batch_size],
                        adv_targets[inner_step * min_batch_size:(inner_step + 1) * min_batch_size],
                        params=fast_params, **kwargs)

                actor_meta_loss, entropy = self.compute_actor_loss(
                    aug_obs_batch, aug_actions_batch, aug_old_log_pis, aug_adv_targets,
                    params=fast_params, **kwargs)

                critic_loss = self.compute_critic_loss(
                    obs_batch, value_preds_batch, return_batch, **kwargs)
                loss = actor_meta_loss + self.critic_loss_coef * critic_loss - \
                       self.entropy_coef * entropy

                logger.log('train_actor/meta_loss', actor_meta_loss, step)
                logger.log('train_actor/entropy', entropy, step)
                logger.log('train_critic/loss', critic_loss, step)
                logger.log('train/loss', loss, step)
                self.actor_meta_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    chain(self.actor.parameters(), self.critic.parameters()),
                    self.grad_clip_norm)
                self.actor_meta_optimizer.step()
                self.critic_optimizer.step()

    def save(self, model_dir, step):
        # TODO (chongyi zheng)
        super().save(model_dir, step)
        torch.save(
            self.cmaml_memories, '%s/cmaml_memories_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        super().load(model_dir, step)
        self.cmaml_memories = torch.load(
            '%s/cmaml_memories_%s.pt' % (model_dir, step)
        )

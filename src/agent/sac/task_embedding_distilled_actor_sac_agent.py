import copy
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence

import utils

from agent.sac.base_sac_agent import SacMlpAgent
from agent.network import SacActorMlp, SacCriticMlp
from agent.nets.distillation import SacTaskEmbeddingDistilledActorMlp


class TaskEmbeddingDistilledActorSacMlpAgent(SacMlpAgent):
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
    ):
        assert isinstance(action_shape, list)
        assert isinstance(action_range, list)

        self.distillation_hidden_dim = distillation_hidden_dim
        self.distillation_task_embedding_dim = distillation_task_embedding_dim
        self.distillation_epochs = distillation_epochs
        self.distillation_iters_per_epoch = distillation_iters_per_epoch
        self.distillation_batch_size = distillation_batch_size
        self.distillation_memory_budget_per_task = distillation_memory_budget_per_task

        self.task_count = 0
        self.memories = []

        super().__init__(
            obs_shape, action_shape, action_range, device, actor_hidden_dim, critic_hidden_dim, discount,
            init_temperature, alpha_lr, actor_lr, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr,
            critic_tau, critic_target_update_freq, batch_size)

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = SacActorMlp(
            self.obs_shape, self.action_shape[0], self.actor_hidden_dim,
            self.actor_log_std_min, self.actor_log_std_max
        ).to(self.device)

        num_tasks = len(self.action_shape)
        self.distilled_actor = SacTaskEmbeddingDistilledActorMlp(
            num_tasks, self.obs_shape, self.action_shape[0],
            self.distillation_hidden_dim, self.distillation_task_embedding_dim,
            self.actor_log_std_min, self.actor_log_std_max
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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.distilled_actor_weight_optimizer = torch.optim.Adam(
            self.distilled_actor.weights.values(), lr=self.actor_lr)
        self.distilled_actor_emb_optimizer = torch.optim.Adam(
            [self.distilled_actor.task_embs[self.task_count]],
            lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # save initial parameters
        self._critic_init_state = copy.deepcopy(self.critic.state_dict())

    def reset(self, **kwargs):
        self.reset_target_critic()
        self.reset_log_alpha()

        self.distilled_actor_weight_optimizer = torch.optim.Adam(
            self.distilled_actor.weights.values(), lr=self.actor_lr,
        )
        self.distilled_actor_emb_optimizer = torch.optim.Adam(
            [self.distilled_actor.task_embs[self.task_count]]
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def _train_distilled_actor(self, dataset, total_steps, epoch, logger):
        for iter in range(self.distillation_iters_per_epoch):
            # TODO (cyzheng): convert previous task memory batch size to an argument
            if self.task_count > 0:
                prev_task_dataset = np.random.choice(self.memories, 3).tolist()
            else:
                prev_task_dataset = []

            losses = []
            for subset in [dataset] + prev_task_dataset:
                random_idxs = np.random.randint(0, self.distillation_memory_budget_per_task,
                                                size=self.distillation_batch_size)
                batch_obses = torch.Tensor(subset['obses'][random_idxs]).to(self.device).squeeze()
                batch_mus = torch.Tensor(subset['mus'][random_idxs]).to(self.device).squeeze()
                batch_log_stds = torch.Tensor(subset['log_stds'][random_idxs]).to(self.device).squeeze()
                task_id = subset['task_id']

                mus, _, _, log_stds = self.distilled_actor(
                    batch_obses, task_id,
                    compute_pi=True, compute_log_pi=True)

                actor_dists = Independent(Normal(loc=batch_mus, scale=batch_log_stds.exp()), 1)
                distilled_actor_dists = Independent(Normal(loc=mus, scale=log_stds.exp()), 1)
                losses.append(torch.mean(kl_divergence(actor_dists, distilled_actor_dists)))
            loss = torch.mean(torch.stack(losses))

            logger.log('train/distillation_loss', loss,
                       total_steps + epoch * self.distillation_iters_per_epoch + iter)

            # (cyzheng): don't optimize embedding vector of previous tasks
            self.distilled_actor_weight_optimizer.zero_grad()
            self.distilled_actor_emb_optimizer.zero_grad()
            loss.backward()
            self.distilled_actor_weight_optimizer.step()
            self.distilled_actor_emb_optimizer.step()

    def act(self, obs, sample=False, use_distilled_actor=False, **kwargs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        assert 'head_idx' in kwargs
        task_idx = kwargs['head_idx']

        with torch.no_grad():
            if use_distilled_actor:
                mu, pi, _, _ = self.distilled_actor(obs, task_idx, compute_log_pi=False)
            else:
                mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            action = pi if sample else mu
            assert 'head_idx' in kwargs
            action = action.clamp(*self.action_range[kwargs['head_idx']])
            assert action.ndim == 2 and action.shape[0] == 1

        return utils.to_np(action)

    def distill(self, **kwargs):
        sample_src = kwargs.pop('sample_src', 'rollout')
        env = kwargs.pop('env')
        replay_buffer = kwargs.pop('replay_buffer')
        total_steps = kwargs.pop('total_steps')
        logger = kwargs.pop('logger')

        # (cyzheng): refresh data every epoch
        for epoch in range(self.distillation_epochs):
            # collect samples
            dataset = {
                'obses': [],
                'actions': [],
                'mus': [],
                'log_stds': [],
                'task_id': self.task_count,
            }
            obs = env.reset()
            if sample_src == 'rollout':
                for _ in range(self.distillation_memory_budget_per_task):
                    with utils.eval_mode(self):
                        # compute log_pi and Q for later gradient projection
                        mu, action, _, log_std = self.actor(
                            torch.Tensor(obs).to(device=self.device),
                            compute_pi=True, compute_log_pi=True, **kwargs)

                        if 'head_idx' in kwargs:
                            action = utils.to_np(
                                action.clamp(*self.action_range[kwargs['head_idx']]))
                        else:
                            action = utils.to_np(action.clamp(*self.action_range))

                    next_obs, reward, done, _ = env.step(action)

                    # TODO (cyzheng): note we save numpy array instead of tensor to save GPU memory
                    dataset['obses'].append(obs)
                    dataset['actions'].append(action)
                    dataset['mus'].append(utils.to_np(mu))
                    dataset['log_stds'].append(utils.to_np(log_std))

                    obs = next_obs

                dataset['obses'] = np.stack(dataset['obses'])
                dataset['actions'] = np.stack(dataset['actions'])
                dataset['mus'] = np.stack(dataset['mus'])
                dataset['log_stds'] = np.stack(dataset['log_stds'])
            elif sample_src == 'replay_buffer':
                obses, _, _, _, _ = replay_buffer.sample(
                    self.distillation_memory_budget_per_task)
                with utils.eval_mode(self):
                    mus, actions, _, log_stds = self.actor(
                        obses, compute_pi=True, compute_log_pi=True, **kwargs)

                dataset['obses'] = utils.to_np(obses)
                dataset['actions'] = utils.to_np(actions)
                dataset['mus'] = utils.to_np(mus)
                dataset['log_stds'] = utils.to_np(log_stds)
            elif sample_src == 'hybrid':
                for _ in range(self.distillation_memory_budget_per_task // 2):
                    with utils.eval_mode(self):
                        # compute log_pi and Q for later gradient projection
                        mu, action, _, log_std = self.actor(
                            torch.Tensor(obs).to(device=self.device),
                            compute_pi=True, compute_log_pi=True, **kwargs)

                        if 'head_idx' in kwargs:
                            action = utils.to_np(
                                action.clamp(*self.action_range[kwargs['head_idx']]))
                        else:
                            action = utils.to_np(action.clamp(*self.action_range))

                    next_obs, reward, done, _ = env.step(action)

                    # TODO (cyzheng): note we save numpy array instead of tensor to save GPU memory
                    dataset['obses'].append(obs)
                    dataset['actions'].append(action)
                    dataset['mus'].append(utils.to_np(mu))
                    dataset['log_stds'].append(utils.to_np(log_std))

                    obs = next_obs

                rollout_obses = np.stack(dataset['obses'])
                rollout_actions = np.stack(dataset['actions'])
                rollout_mus = np.stack(dataset['mus'])
                rollout_log_stds = np.stack(dataset['log_stds'])

                obses, _, _, _, _ = replay_buffer.sample(
                    self.distillation_memory_budget_per_task -
                    self.distillation_memory_budget_per_task // 2)
                with utils.eval_mode(self):
                    mus, actions, _, log_stds = self.actor(
                        obses, compute_pi=True, compute_log_pi=True, **kwargs)

                dataset['obses'] = np.concatenate([rollout_obses, utils.to_np(obses)])
                dataset['actions'] = np.concatenate([rollout_actions, utils.to_np(actions)])
                dataset['mus'] = np.concatenate([rollout_mus, utils.to_np(mus)])
                dataset['log_stds'] = np.concatenate([rollout_log_stds, utils.to_np(log_stds)])
            else:
                raise ValueError("Unknown sample source!")

            self._train_distilled_actor(dataset, total_steps, epoch, logger)

            # TODO (cyzheng): log loss for every epoch
            logger.dump(total_steps + epoch * self.distillation_iters_per_epoch,
                        ty='train', save=True)

        self.memories.append(dataset)
        self.task_count += 1

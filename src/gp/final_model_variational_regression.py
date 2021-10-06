import os
import matplotlib.pyplot as plt
import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
import gpytorch

import utils
from video import VideoRecorder

from arguments import parse_args
from environment import make_continual_vec_envs
from agent import make_agent


class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)

        self.mean = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(1e-6, 1e-4)),
            outputscale_constraint=gpytorch.constraints.Interval(1e-6, 1e-4)
        )

    def forward(self, x):
        mean = self.mean(x)
        covar = self.kernel(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)


def evaluate_task(env, task_name, agent, video, num_episodes, **act_kwargs):
    """Evaluate agent"""

    # for task_id, task_name in enumerate(env.get_attr('env_names')[0]):
    episode_rewards = []
    episode_successes = []
    video.init(enabled=True)
    # env.env_method('sample_task')
    # FIXME (cyzheng): Fix the following method
    env.env_method('set_task', task_name)
    task_id = env.get_attr('active_task_index')[0]
    obs = env.reset()
    video.record(env)

    if 'task_embedding_hypernet' in args.algo:
        agent.infer_weights(task_id)

    while len(episode_rewards) < num_episodes:
        with utils.eval_mode(agent):
            with utils.eval_mode(agent):
                if any(x in args.algo for x in ['mh', 'mi', 'individual', 'hypernet', 'distilled']):
                    action = agent.act(obs, sample=False, head_idx=task_id, **act_kwargs)
                else:
                    action = agent.act(obs, sample=False, **act_kwargs)

            obs, _, done, infos = env.step(action)
            video.record(env)

            for done_ in done:
                if done_ and len(episode_rewards) == 0:
                    video.save('%s.mp4' % task_name)
                    video.init(enabled=False)

            for info in infos:
                if 'episode' in info.keys():
                    episode_successes.append(info.get('success', 0.0))
                    episode_rewards.append(info['episode']['r'])

    episode_rewards = episode_rewards[:num_episodes]
    episode_successes = episode_successes[:num_episodes]

    if 'task_embedding_hypernet' in args.algo:
        agent.clear_weights()

    if len(episode_successes) > 0:
        return np.mean(episode_successes), np.mean(episode_rewards)
    else:
        return None, np.mean(episode_rewards)


def main(args):
    if args.env_type == 'mujoco':
        eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'dummy'))
        env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            None, eval_env_log_dir,
            allow_early_resets=True,
            normalize=False,
            add_onehot=args.add_onehot,
        )
    elif args.env_type == 'metaworld':
        eval_env_log_dir = utils.make_dir(os.path.join(args.work_dir, 'dummy'))
        env = make_continual_vec_envs(
            args.env_names, args.seed, args.sac_num_processes,
            None, eval_env_log_dir,
            allow_early_resets=True,
            normalize=False,
            add_onehot=args.add_onehot,
        )

    utils.set_seed_everywhere(args.seed)
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, args.env_type,
                          height=448, width=448, camera_id=args.video_camera_id)

    device = torch.device(args.device)
    agent = make_agent(
        obs_space=env.observation_space,
        action_space=[env.action_space for _ in range(env.num_tasks)]
        if any(x in args.algo for x in ['mh', 'mi', 'individual', 'hypernet', 'distilled'])
        else env.action_space,
        device=device,
        args=args
    )

    for task_name in args.env_names:
        task_load_dir = os.path.join(args.load_dir, task_name)

        assert os.path.exists(task_load_dir)

        agent.load(task_load_dir)

        # # plot actor weights
        actor_params = []
        for param in agent.actor.main_parameters():
            actor_params.append(param.detach().clone().flatten())
        actor_params = torch.cat([actor_params[2], actor_params[3]])
        actor_param_dims = torch.linspace(0, actor_params.shape[0] - 1,
                                          actor_params.shape[0], device=args.device)

        # f, ax = plt.subplots(1, 1, figsize=(20, 10))
        # ax.plot(utils.to_np(actor_param_dims), utils.to_np(actor_params), 'k*')
        # ax.set_ylim([-10, 10])
        # ax.legend(['Actor Weights'])
        # plt.show()

        # # TODO (cyzheng): normalize?
        # actor_param_mean = torch.mean(actor_params)
        # actor_param_std = torch.std(actor_params)
        # X = X - X.min(0)[0]
        # X = 2 * (X / X.max(0)[0]) - 1
        # norm_actor_params = (actor_params - actor_param_mean) / (actor_param_std + 1e-6)
        norm_actor_param_dims = 2 * (actor_param_dims - actor_param_dims.min(0)[0]) / actor_param_dims.max(0)[0] - 1
        norm_actor_params = actor_params

        f, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(utils.to_np(norm_actor_param_dims), utils.to_np(norm_actor_params), 'k*')
        ax.set_ylim([-3, 3])
        ax.legend(['Actor Weights'])
        # plt.show()

        train_dataset = TensorDataset(norm_actor_param_dims, norm_actor_params)
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

        test_dataset = TensorDataset(norm_actor_param_dims, norm_actor_params)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        rand_idxs = np.random.randint(0, len(actor_param_dims), 1000)
        # inducing_points = torch.rand(5000, device=args.device) * actor_param_dims[-1]
        inducing_points = norm_actor_param_dims[rand_idxs]
        model = ApproximateGPModel(inducing_points).to(args.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(args.device)
        nn_model = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(args.device)

        num_epochs = 15
        model.train()
        likelihood.train()
        nn_model.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=1e-2)
        nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-2)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=norm_actor_params.shape[0])

        epoch_iters = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for _ in epoch_iters:
            minibatch_iters = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch, in minibatch_iters:
                optimizer.zero_grad()
                nn_optimizer.zero_grad()

                output = model(x_batch)
                loss = -mll(output, y_batch)
                nn_output = nn_model(x_batch.unsqueeze(1)).squeeze()
                nn_loss = F.mse_loss(nn_output, y_batch)

                minibatch_iters.set_postfix(loss=loss.item(), nn_loss=nn_loss.item())

                loss.backward()
                nn_loss.backward()
                optimizer.step()
                nn_optimizer.step()

        model.eval()
        likelihood.eval()
        nn_model.eval()
        means = []
        nn_means = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = likelihood(model(x_batch))
                means.append(preds.mean)

                nn_preds = nn_model(x_batch.unsqueeze(1)).squeeze()
                nn_means.append(nn_preds)
        means = torch.cat(means)
        nn_means = torch.cat(nn_means)
        print('Test MAE: {}, NN MAE: {}'.format(
            utils.to_np(torch.mean(torch.abs(means - norm_actor_params))),
            utils.to_np(torch.mean(torch.abs(nn_means - norm_actor_params)))
        ))
        # print('Test Relative MAE: {}, NN Relative MAE: {}'.format(
        #     utils.to_np(torch.mean(torch.abs((means - norm_actor_params) / norm_actor_params))),
        #     utils.to_np(torch.mean(torch.abs((nn_means - norm_actor_params) / norm_actor_params)))
        # ))

        with torch.no_grad():
            idx = 0
            pred_params = []
            nn_pred_params = []
            while idx < actor_params.shape[0]:
                sample_param = likelihood(model(norm_actor_param_dims[idx:idx + 1000])).sample()
                # pred_param = (mean + 1) * actor_param_max / 2 + actor_param_min
                pred_param = sample_param
                pred_params.append(pred_param)

                nn_param = nn_model(norm_actor_param_dims[idx:idx + 1000].unsqueeze(1)).squeeze()
                nn_pred_params.append(nn_param)

                idx += 1000
            pred_params = torch.cat(pred_params)
            nn_pred_params = torch.cat(nn_pred_params)

            idx = 0
            for param in [list(agent.actor.main_parameters())[2], list(agent.actor.main_parameters())[3]]:
                num_param = param.numel()  # number of parameters in [p]
                param.copy_(pred_params[idx:idx + num_param].reshape(param.shape))
                idx += num_param

        print(f'Evaluating {task_name} for {args.num_eval_episodes} episodes...')
        task_successes, task_rewards = evaluate_task(env, task_name, agent, video, args.num_eval_episodes)
        print(f'Success Rate: {task_successes}, Return: {task_rewards}')

        with torch.no_grad():
            idx = 0
            for param in [list(agent.actor.main_parameters())[2], list(agent.actor.main_parameters())[3]]:
                num_param = param.numel()  # number of parameters in [p]
                param.copy_(nn_pred_params[idx:idx + num_param].reshape(param.shape))
                idx += num_param

        print(f'Evaluating NN {task_name} for {args.num_eval_episodes} episodes...')
        task_successes, task_rewards = evaluate_task(env, task_name, agent, video, args.num_eval_episodes)
        print(f'Success Rate: {task_successes}, Return: {task_rewards}')


if __name__ == '__main__':
    args = parse_args()

    main(args)

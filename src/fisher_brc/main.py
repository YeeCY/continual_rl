"""Run training loop for batch rl."""

import os.path as osp
import argparse
import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import utils
import d4rl_utils
from agent import FBRC


def evaluate(env, agent, num_episodes=10):
    """Evaluates the policy.

  Args:
    env: Environment to evaluate the policy on.
    agent: Agent to evaluate.
    num_episodes: A number of episodes to average the policy on.

  Returns:
    Averaged reward and a total number of steps.
  """
    total_timesteps = 0
    total_returns = 0.0

    for _ in range(num_episodes):
        episode_return = 0
        timestep = env.reset()

        while not timestep.is_last():
            action = agent.act(timestep.observation)
            timestep = env.step(action)

            total_returns += timestep.reward[0]
            episode_return += timestep.reward[0]
            total_timesteps += 1

    return total_returns / num_episodes, total_timesteps / num_episodes


def main(args):
    device = torch.device(args.device)

    env, dataloader = d4rl_utils.create_d4rl_env_and_dataset(
        task_name=args.task_name, batch_size=args.batch_size)

    # env = gym_wrapper.GymWrapper(gym_env)
    # env = tf_py_environment.TFPyEnvironment(env)

    dataset_iter = iter(dataloader)

    # tf.random.set_seed(FLAGS.seed)
    utils.set_seed_everywhere(args.seed)

    # hparam_str = f'{FLAGS.algo_name}_{FLAGS.task_name}_seed={FLAGS.seed}'
    hparam_str = f'fisher-brc_{args.task_name}_seed{args.seed}'

    # summary_writer = tf.summary.create_file_writer(
    #     os.path.join(FLAGS.save_dir, 'tb', hparam_str))
    # result_writer = tf.summary.create_file_writer(
    #     os.path.join(FLAGS.save_dir, 'results', hparam_str))
    summary_writer = SummaryWriter(osp.join(args.save_dir, hparam_str, 'summary'))
    result_writer = SummaryWriter(osp.join(args.save_dir, hparam_str, 'result'))

    # if FLAGS.algo_name == 'bc':
    #     model = behavioral_cloning.BehavioralCloning(
    #         env.observation_spec(),
    #         env.action_spec())
    # else:
    #     model = fisher_brc.FBRC(
    #         env.observation_spec(),
    #         env.action_spec(),
    #         target_entropy=-env.action_spec().shape[0],
    #         f_reg=FLAGS.f_reg,
    #         reward_bonus=FLAGS.reward_bonus)
    agent = FBRC(
        env.observation_space,
        env.action_space,
        device,
        fisher_coeff=args.fisher_coeff,
        reward_bonus=args.reward_bonus,
    )

    for i in tqdm.tqdm(range(args.bc_pretraining_steps)):
        agent.bc.update_learning_rate(i)

        try:
            # Samples the batch
            states, actions, _, _, _ = next(dataset_iter)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            dataset_iter = iter(dataloader)
            states, actions, _, _, _ = next(dataset_iter)

        info_dict = agent.bc.update(states, actions)

        if i % args.log_interval == 0:
            for k, v in info_dict.items():
                summary_writer.add_scalar(f'training/{k}', v, i - args.bc_pretraining_steps)
            # with summary_writer.as_default():
            #     for k, v in info_dict.items():
            #         tf.summary.scalar(
            #             f'training/{k}', v, step=i - FLAGS.bc_pretraining_steps)

    for i in tqdm.tqdm(range(args.num_updates)):
        try:
            # Samples the batch
            states, actions, rewards, not_dones, next_states = next(dataset_iter)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            dataset_iter = iter(dataloader)
            states, actions, rewards, not_dones, next_states = next(dataset_iter)

        info_dict = agent.update(states, actions, rewards, not_dones, next_states)

        if i % args.log_interval == 0:
            # with summary_writer.as_default():
            #     for k, v in info_dict.items():
            #         tf.summary.scalar(f'training/{k}', v, step=i)
            for k, v in info_dict.items():
                summary_writer.add_scalar(f'training/{k}', v, i)

        if (i + 1) % args.eval_interval == 0:
            average_returns, average_length = evaluate(env, agent)
            average_returns = env.get_normalized_score(average_returns) * 100.0  # (cyzheng): normalize by oracle returns

            # with result_writer.as_default():
            #     tf.summary.scalar('evaluation/returns', average_returns, step=i + 1)
            #     tf.summary.scalar('evaluation/length', average_length, step=i + 1)
            result_writer.add_scalar('evaluation/returns', average_returns, i + 1)
            result_writer.add_scalar('evaluation/length', average_length, i + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_updates', type=int, default=1000000)
    parser.add_argument('--bc_pretraining_steps', type=int, default=1000000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--fisher_coeff', type=float, default=0.1)
    parser.add_argument('--reward_bonus', type=float, default=5.0)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)

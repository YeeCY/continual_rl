from my_dqn import DQN
# from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from src.env.atari_wrappers import wrap_deepmind

from src.agent.dqn_agent import DQNCnn

import argparse


def main(args):
    env = wrap_deepmind(
        env_id=args.env_name,
        frame_skip=args.action_repeat,
        frame_stack=args.frame_stack
    )

    agent = DQN(
        CnnPolicy,
        env,
        learning_rate=args.q_net_lr,
        buffer_size=args.replay_buffer_capacity,
        learning_starts=args.init_steps,
        batch_size=args.batch_size,
        tau=args.q_net_tau,
        gamma=args.discount,
        train_freq=args.train_freq,
        gradient_steps=args.num_train_iters,
        optimize_memory_usage=True,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
    )

    agent.learn(args.train_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='PongNoFrameskip-v4')
    parser.add_argument('--frame_stack', default=4, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--init_steps', default=100000, type=int)
    parser.add_argument('--num_train_iters', default=1, type=int)
    parser.add_argument('--train_freq', default=4, type=int)
    parser.add_argument('--train_steps', default=10000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--replay_buffer_capacity', default=10000, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--encoder_feature_dim', default=512, type=int)
    parser.add_argument('--exploration_fraction', default=0.1, type=float)
    parser.add_argument('--exploration_initial_eps', default=1.0, type=float)
    parser.add_argument('--exploration_final_eps', default=0.01, type=float)
    parser.add_argument('--target_update_interval', default=1000, type=int)
    parser.add_argument('--max_grad_norm', default=10, type=float)
    parser.add_argument('--q_net_lr', default=1e-4, type=float)
    parser.add_argument('--q_net_tau', default=1.0, type=float)
    args = parser.parse_args()

    main(args)

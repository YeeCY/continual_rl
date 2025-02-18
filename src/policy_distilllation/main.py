# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------


# basic utils
from itertools import count
from time import time, strftime, localtime
import os
import numpy as np
# import scipy.optimize

# deeplearning utils
from tensorboardX import SummaryWriter
from utils import *

# LDE utils
# import gym
# from utils.agent_pd_baselines import load_env_and_model
# from utils2 import ALGOS

# teacher policy & student policy
from agent import Teacher, Student


# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

'''
1. train single or multiple teacher policies
2. collect samples from teacher policy
3. use KL or W2 distance as metric to train student policy
4. test student policy
'''

def main(args):
    # policy and envs for sampling
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    exp_date = strftime('%Y.%m.%d', localtime(time()))
    writer = SummaryWriter(log_dir='./exp_data/{}/{}_{}'.format(exp_date, args.env, time()))


    ###########################################
    # load teacher models
    ##########################################
    envs = []
    teacher_policies = []
    for i in range(args.num_teachers):
        env, model = load_env_and_model(args.env, args.algo, args.folder)
        envs.append(env)
        teacher_policies.append(model)
    ##########################################

    teachers = Teacher(envs, teacher_policies, args)
    student = Student(args)
    print('Training student policy...')
    time_beigin = time()
    exp_id = '%s_%s_%s' % (args.env, args.algo, time_beigin)
    path_to_save = '%s/distilled-agents/%s' % (os.getcwd(), exp_id)
    os.makedirs(path_to_save, exist_ok=True)


    ################################
    # train student policy
    ################################

    for epoch in range(args.total_training_epochs):
        expert_data, expert_reward = teachers.get_expert_sample()
        for iter in range(args.iters_per_epoch):
            loss = student.train(expert_data)
            if iter % args.log_interval == 0:  # for logging
                writer.add_scalar('{} loss'.format(args.loss_metric), loss, iter)
                print('Epoch {} | Itr {} | {} loss: {:.2f}'.format(epoch, iter,
                                                                   args.loss_metric, loss.item()))
            if iter % args.eval_interval == 0:
                average_reward = student.eval()
                student.save('{}/student_{}_{:.2f}.pkl'.format(path_to_save, iter, average_reward))
                writer.add_scalar('Students_average_reward', average_reward, iter)
                writer.add_scalar('teacher_reward', expert_reward, iter)
                print("Students_average_reward: {:.3f} (teacher_reward:{:3f})".format(average_reward, expert_reward))

    time_train = time() - time_beigin
    print('Training student policy finished, using time {}'.format(time_train))


if __name__ == '__main__':
    import argparse

    ##########################
    # Arguments Setting
    ##########################
    parser = argparse.ArgumentParser(description='Policy distillation')

    # Environment Setting
    parser.add_argument("--env", help="environment ID", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="td3", type=str,
                        required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--device", default="cuda", type=str)


    # Network, env, seed
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')

    # For Teacher policy
    parser.add_argument('--agent-count', type=int, default=3, metavar='N',
                        help='number of agents (default: 100)')
    parser.add_argument('--num-teachers', type=int, default=1, metavar='N',
                        help='number of teacher policies (default: 1)')
    parser.add_argument('--sample-batch-size', type=int, default=10000, metavar='N',
                        help='expert batch size for each teacher (default: 10000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')

    # For Student policy
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='adam learnig rate (default: 1e-3)')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--learner-batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for student (default: 1000)')
    # parser.add_argument('--sample-interval', type=int, default=2500, metavar='N',
    #                     help='frequency to update expert data (default: 10)')
    parser.add_argument('--learner-eval-steps', type=int, default=5000, metavar='N',
                        help='batch size for testing student policy (default: 10000)')
    parser.add_argument('--total-training-epochs', type=int, default=4)
    parser.add_argument('--iters-per-epoch', type=int, default=2500)
    # parser.add_argument('--total-training-steps', type=int, default=10000, metavar='N',
    #                     help='num of teacher training episodes (default: 1000)')
    parser.add_argument('--loss-metric', type=str, default='kl',
                        help='metric to build student objective')

    args = parser.parse_args()


    ########################
    # Distilling!
    ########################
    main(args)

# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

from policies import *
from torch.optim import Adam, SGD
import torch
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence

import random
import pickle
import gzip
from utils import AgentCollection, load_env_and_model, \
    get_wasserstein, get_kl
import numpy as np


class Student:
    def __init__(self, args):
        self.env, _ = load_env_and_model(args.env, args.algo, args.folder)

        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        self.learner_batch_size = args.learner_batch_size
        self.learner_eval_steps = args.learner_eval_steps
        self.loss_metric = args.loss_metric
        self.policy = Policy(num_inputs, num_actions, hidden_sizes=(args.hidden_size, ) * args.num_layers).to(
            args.device)
        self.agents = AgentCollection([self.env], [self.policy], render=args.render, num_agents=1)
        self.optimizer = Adam(self.policy.parameters(), lr=args.lr)

    def train(self, expert_data):
        batch = random.sample(expert_data, self.learner_batch_size)
        # print(batch[0])
        obses = torch.stack([x[0] for x in batch])
        teacher_mus = torch.stack([x[1] for x in batch])
        teacher_stds = torch.stack([x[2] for x in batch])

        # if action_dist:
        #     pass
        # else:
        #     fake_std = torch.from_numpy(np.array([1e-6] * len(means_teacher[0])))  # for deterministic
        #     stds_teacher = torch.stack([fake_std for _ in batch])

        # distilled actor
        student_mus = self.policy.mean_action(obses)
        student_stds = self.policy.get_std(obses)
        if self.loss_metric == 'kl':
            # TODO (cyzheng): better to use diagonal Gaussian here
            # loss = get_kl([teacher_mus, teacher_stds], [student_mus, student_stds])
            teacher_dist = Independent(Normal(loc=teacher_mus, scale=teacher_stds), 1)
            student_dist = Independent(Normal(loc=student_mus, scale=student_stds), 1)
            loss = torch.mean(kl_divergence(teacher_dist, student_dist))
        elif self.loss_metric == 'wasserstein':
            loss = get_wasserstein([teacher_mus, teacher_stds], [student_mus, student_stds])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def eval(self):
        _, logs = self.agents.collect_samples(self.learner_eval_steps, exercise=True)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward

    def save(self, ckp_name):
        with gzip.open(ckp_name, 'wb') as f:
            pickle.dump(self.policy, f)


class Teacher:
    def __init__(self, envs, policies, args):
        self.envs = envs
        self.policies = policies
        self.expert_batch_size = args.sample_batch_size
        self.agents = AgentCollection(self.envs, self.policies, render=args.render, num_agents=args.agent_count)

    def get_expert_sample(self):
        return self.agents.get_expert_sample(self.expert_batch_size)


class TrainedStudent:
    def __init__(self, args, optimizer=None):
        self.env, _ = load_env_and_model(args.env, args.algo, args.folder)
        self.testing_batch_size = args.testing_batch_size

        self.policy = self.load(args.path_to_student)
        self.agents = AgentCollection([self.env], [self.policy], render=args.render, num_agents=1)

    def test(self):
        memories, logs = self.agents.collect_samples(self.testing_batch_size, exercise=True)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward

    def load(self, ckp_name):
        with gzip.open(ckp_name, 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data

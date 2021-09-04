# reference : https://github.com/Mee321/policy-distillation & https://github.com/DLR-RM/rl-baselines3-zoo

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import os
import yaml
import glob
import importlib

import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

# from utils.replay_memory import Memory
# from utils.torch import *
from buffers import ReplayBuffer

from stable_baselines3.common.utils import set_random_seed

# import utils2.import_envs  # noqa: F401 pylint: disable=unused-import
# from utils2 import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams


################################################################
# to clear cv2 Import error
# ros_pack_path = '/opt/ros/%s/lib/python2.7/dist-packages' % os.getenv('ROS_DISTRO')
# if ros_pack_path in sys.path:
#     sys.path.remove(ros_pack_path)
################################################################
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from sb3_contrib import QRDQN, TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import \
    DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "her": HER,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "qrdqn": QRDQN,
    "tqc": TQC,
}


def get_kl(teacher_dist_info, student_dist_info):
    pi = Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])
    pi_new = Normal(student_dist_info[0], scale=student_dist_info[1])
    kl = torch.mean(kl_divergence(pi, pi_new))
    return kl


def get_wasserstein(teacher_dist_info, student_dist_info):
    means_t, stds_t = teacher_dist_info
    means_s, stds_s = student_dist_info
    return torch.sum((means_s - means_t) ** 2) + \
           torch.sum((torch.sqrt(stds_s) - torch.sqrt(stds_t)) ** 2)


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_id:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + f"/{env_id}_[0-9]*"):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(
        stats_path: str,
        norm_reward: bool = False,
        test_mode: bool = False,
):
    """
    :param stats_path:
    :param norm_reward:
    :param test_mode:
    :return:
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml"), "r") as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path


def get_wrapper_class(hyperparams):
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - utils.wrappers.PlotActionWrapper
        - utils.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env):
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def create_test_env(
        env_id,
        n_envs=1,
        stats_path=None,
        seed=0,
        log_dir=None,
        should_render=True,
        hyperparams=None,
        env_kwargs=None,
):
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :return:
    """
    # Avoid circular import
    # from utils2.exp_manager import ExperimentManager

    # Create the environment and wrap it if necessary
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs = {}
    vec_env_cls = DummyVecEnv
    # if n_envs > 1 or (ExperimentManager.is_bullet(env_id) and should_render):
    #     # HACK: force SubprocVecEnv for Bullet env
    #     # as Pybullet envs does not follow gym.render() interface
    #     vec_env_cls = SubprocVecEnv
    #     # start_method = 'spawn' for thread safe

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env


def load_env_and_model(env_id, algo, folder):
    # get experiment id
    exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    print(f"Loading latest experiment, id={exp_id}")

    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    # check & take get the model_path
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    # set random seed
    set_random_seed(0)

    # get stats_path & hyperparam_path
    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    # make gym environment
    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=0,
        log_dir=None,
        should_render=True,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    # Dummy buffer size as we don't need memory to enjoy the trained agent
    kwargs = dict(seed=0)
    kwargs.update(dict(buffer_size=1))

    # load pre-trained model
    model = ALGOS[algo].load(model_path, env=env, **kwargs)

    return env, model


def sample_generator(env, model, render=True, min_batch_size=10000, id_=0):
    log = dict()
    # memory = Memory()
    buffer = ReplayBuffer(
        obs_space=env.observation_space,
        action_space=env.action_space,
        capacity=min_batch_size,
        device=model.device,
        optimize_memory_usage=True,
    )

    num_steps = 0
    num_episodes = 0

    # main loop to enjoy for n_timesteps..
    episode_rewards, episode_lengths = [], []
    obs = env.reset()
    episode_reward = 0.0
    ep_len = 0
    while num_steps < min_batch_size:
        if render:
            env.render()

        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)

        episode_reward += reward[0]
        ep_len += 1

        not_done = 0 if done else 1
        buffer.add(obs, action, not_done, next_obs, reward, info)
        obs = next_obs

        if done:
            # NOTE: for env using VecNormalize, the mean reward
            # is a normalized reward when `--norm_reward` flag is passed
            print(f"Episode Reward: {episode_reward:.2f}")
            print("Episode Length", ep_len)
            episode_rewards.append(episode_reward)
            episode_lengths.append(ep_len)
            episode_reward = 0.0
            ep_len = 0

            num_episodes += 1

            obs = env.reset()

        num_steps += 1
    total_reward = sum(episode_rewards)
    min_reward = min(episode_rewards)
    max_reward = max(episode_rewards)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward

    return id_, buffer, log


class AgentCollection:
    def __init__(self, envs, policies, mean_action=False, render=False, num_agents=1):
        self.envs = envs
        self.policies = policies
        self.mean_action = mean_action
        self.render = render
        self.num_agents = num_agents
        self.num_teachers = len(policies)

    def collect_samples(self, min_batch_size, exercise=False):
        # print("collect_samples called!!")
        results = []
        for i in range(self.num_teachers):
            if not exercise:
                results.append(
                    sample_generator(self.envs[i], self.policies[i], self.render, min_batch_size, i)
                )
            else:
                results.append(
                    self.exercise(self.envs[i], self.policies[i], self.render, min_batch_size, i)
                )
        worker_logs = [None] * self.num_agents
        worker_memories = [None] * self.num_agents
        # print(len(result_ids))
        for result in results:
            pid, worker_memory, worker_log = result
            worker_memories[pid] = worker_memory
            worker_logs[pid] = worker_log

        # print("collect_samples done")
        return worker_memories, worker_logs

    def get_expert_sample(self, batch_size, deterministic=True):
        # print("get_expert_sample called!!")
        buffers, logs = self.collect_samples(batch_size)  # (cyzheng): only one teacher worker
        teacher_rewards = [log['avg_reward'] for log in logs if log is not None]
        teacher_average_reward = np.array(teacher_rewards).mean()
        # TODO better implementation of dataset and sampling
        # construct training dataset containing pairs {X:state, Y:output of teacher policy}
        dataset = []
        for buffer, policy in zip(buffers, self.policies):
            obses, _, _, _, _ = buffer.sample(buffer.capacity)
            obses = torch.squeeze(obses)
            # batched_state = np.array(batch.state).reshape(-1, policy.env.observation_space.shape[0])
            # states = torch.from_numpy(batched_state).to(torch.float).to('cpu')
            if isinstance(policy, TD3):
                mus = torch.Tensor(
                    policy.predict(obses.to('cpu').numpy(),
                                   deterministic=deterministic)[0]
                ).to(policy.device)
                stds = 1e-6 * torch.ones_like(mus, device=policy.device)
            else:
                mus = torch.Tensor(
                    policy.predict(obses.to('cpu').numpy(),
                                   deterministic=deterministic)[0]
                ).to(policy.device)
                stds = policy.get_std()

            dataset += [(obs, mu, std) for obs, mu, std in zip(obses, mus, stds)]
        return dataset, teacher_average_reward

    def exercise(self, env, policy, render=True, min_batch_size=10000, pid=0):
        torch.randn(pid)
        log = dict()
        # memory = Memory()
        buffer = ReplayBuffer(
            obs_space=env.observation_space,
            action_space=env.action_space,
            capacity=min_batch_size,
            device=policy.device,
            optimize_memory_usage=True,
        )
        num_steps = 0
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        num_episodes = 0

        while num_steps < min_batch_size:
            state = env.reset()
            reward_episode = 0

            for t in range(1000):
                state_var = torch.Tensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = policy.mean_action(state_var.to(torch.float))[0].numpy()
                next_state, reward, done, info = env.step(action)
                reward_episode += reward

                not_done = 0 if done else 1

                buffer.add(state, action, not_done, next_state, reward, info)

                if render:
                    env.render()
                if done:
                    break

                state = next_state

            # log states
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)
            print("reward_episode: %f"%reward_episode)
            print("num_steps: %d"%num_steps)


        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward

        return pid, buffer, log

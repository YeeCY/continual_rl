import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp
import glob


WINDOW_LENGTH = 10
SMOOTH_COEF = 0.20
CM = 1 / 2.54  # centimeters in inches


CURVE_FORMAT = {
    'mh_ppo_mlp_metaworld_single': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'PPO',
    },
    'mh_sac_mlp_metaworld_single': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'SAC',
    },
    'sac_rlkit_single': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'RLKIT_SAC',
    },
    'sac_garage_single': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'GARAGE_SAC',
    },
    'ppo_mlp_overparam_metaworld_single': {
        'color': [160, 32, 240],
        'style': '-',
        'label': 'PPO_overparameterized'
    },
    'sac_softlearning_single': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'SOFTLEARNING_SAC'
    },
    'ewc_lambda5000': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'EWC_lambda5000'
    },
    'ewc_lambda10': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'EWC_labmda10'
    },
    'si_c1': {
        'color': [204, 204, 0],
        'style': '-',
        'label': 'SI_c1'
    },
    'si_c10': {
        'color': [254, 153, 204],
        'style': '-',
        'label': 'SI_c10'
    },
    'si_c100': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'SI_c100'
    },
    'agem_ref_grad_batch_size256': {
        'color': [153, 76, 0],
        'style': '-',
        'label': 'AGEM_REF_GRAD_BATCH_SIZE256'
    },
    'agem_ref_grad_batch_size512': {
        'color': [64, 64, 64],
        'style': '-',
        'label': 'AGEM_REF_GRAD_BATCH_SIZE512'
    },
    'agem_ref_grad_batch_size1024': {
        'color': [0, 153, 153],
        'style': '-',
        'label': 'AGEM_REF_GRAD_BATCH_SIZE1024'
    },
    'agem_ref_grad_batch_size3072': {
        'color': [64, 64, 64],
        'style': '-',
        'label': 'AGEM_REF_GRAD_BATCH_SIZE3072'
    },
}


def window_smooth(y):
    window_size = int(WINDOW_LENGTH / 2)
    y_padding = np.concatenate([y[:1] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y_padding, y], axis=0)
    y_padding = np.concatenate([y[-1:] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y, y_padding], axis=0)

    coef = list()
    for i in range(WINDOW_LENGTH + 1):
        coef.append(np.exp(- SMOOTH_COEF * abs(i - window_size)))
    coef = np.array(coef)

    yw = list()
    for t in range(len(y)-WINDOW_LENGTH):
        yw.append(np.sum(y[t:t+WINDOW_LENGTH+1] * coef) / np.sum(coef))

    return np.array(yw).flatten()


def plot(ax, data, task_names, algos, curve_format=CURVE_FORMAT):
    algo_norm_data = {}
    for task_name in task_names:
        for algo in algos:
            algo_data = data[task_name][algo]
            if 'y' not in algo_data:
                continue

            # if len(algo_data['x']) != len(algo_data['y'][0]):
            #     min_len = len(algo_data['x'])
            #     for y in algo_data['y']:
            #         min_len = min(min_len, len(y))
            #
            #     algo_data['x'] = algo_data['x'][:min_len]
            #     for idx, y in enumerate(algo_data['y']):
            #         algo_data['y'][idx] = y[:min_len]
            y_len = 1E10
            for y in algo_data['y']:
                y_len = min(len(y), y_len)

            algo_data['x'] = algo_data['x'][:y_len]
            for idx in range(len(algo_data['y'])):
                algo_data['y'][idx] = algo_data['y'][idx][:y_len]

            # normalize via oracle returns
            y_mean = np.mean(np.array(algo_data['y']), axis=0)
            if algo not in algo_norm_data:
                algo_norm_data[algo] = {}
                algo_norm_data[algo]['x'] = algo_data['x']
                algo_norm_data[algo]['y'] = [y_mean]
            else:
                algo_norm_data[algo]['y'].append(y_mean)

    for algo in algos:
        if algo not in algo_norm_data:
            continue

        algo_data = algo_norm_data[algo]

        if 'y' not in algo_data:
            continue

        y_len = 1E10
        for y in algo_data['y']:
            y_len = min(len(y), y_len)

        x = algo_data['x'][:y_len]
        for idx in range(len(algo_data['y'])):
            algo_data['y'][idx] = algo_data['y'][idx][:y_len]

        y_mean = np.mean(np.array(algo_data['y']), axis=0)
        y_std = np.std(np.array(algo_data['y']), axis=0)
        y_mean = window_smooth(y_mean)
        y_std = window_smooth(y_std)

        color = np.array(curve_format[algo]['color']) / 255.
        style = curve_format[algo]['style']
        label = curve_format[algo]['label']
        ax.plot(x, y_mean, color=color, label=label, linestyle=style)
        ax.fill_between(x, y_mean - 0.5 * y_std, y_mean + 0.5 * y_std, facecolor=color, alpha=0.1)


def main(args):
    # exp_name = args.exp_name + '-setting-' + str(args.setting)
    exp_names = args.exp_names
    task_names = args.task_names
    data_dir = args.data_dir
    save_dir = args.save_dir
    stats = args.statistics
    seeds = args.seeds
    max_timesteps = args.max_timesteps
    # num_fig = len(stats)

    if not osp.exists(data_dir):
        print("The directory to load data doesn't exit")
    os.makedirs(save_dir, exist_ok=True)

    fig, _ = plt.subplots(1, len(stats))
    fig.set_size_inches(15, 15)

    for stat_idx, stat in enumerate(stats):
        ax = plt.subplot(1, len(stats), stat_idx + 1)
        # ax.set_title(task_name, fontsize=15)
        ax.set_xlabel('Total Timesteps', fontsize=15)
        ax.set_ylabel(stat, fontsize=15)
        data = {}

        for task_idx, task_name in enumerate(task_names):
            data[task_name] = {}
            for exp_name in exp_names:
                data[task_name][exp_name] = {}

                for seed in seeds:
                    if exp_name == 'sac_rlkit_single':
                        data_path = osp.join(data_dir, exp_name, 'SAC-' + task_name, 's-' + str(seed), 'progress.csv')
                    elif exp_name == 'sac_garage_single':
                        data_path = osp.join(data_dir, exp_name, 'sac-' + task_name, 's-' + str(seed), 'progress.csv')
                    elif exp_name == 'sac_softlearning_single':
                        data_path = osp.join(data_dir, exp_name, task_name.replace('-v2', ''),
                                             'v2', '*/*seed={}*'.format(str(seed)), 'progress.csv')
                        paths = glob.glob(r'{}'.format(data_path))
                        if len(paths) == 0:
                            print(f"Data path not found: {data_path}!")
                            continue
                        else:
                            data_path = paths[0]
                    else:
                        data_path = osp.join(data_dir, exp_name, 'sgd', task_name, str(seed), 'eval.csv')
                    data_path = os.path.abspath(data_path)

                    try:
                        df = pd.read_csv(data_path)
                    except:
                        print(f"Data path not found: {data_path}!")
                        continue

                    # file = file[file['step'] <= args.timesteps]
                    # x = file['exploration/all num steps total'].values

                    if exp_name == 'sac_rlkit_single':
                        task_df = df[df['exploration/num steps total'] <= max_timesteps]
                        data[task_name][exp_name].update(x=task_df['exploration/num steps total'].values)
                    elif exp_name == 'sac_garage_single':
                        task_df = df[df['TotalEnvSteps'] <= max_timesteps]
                        data[task_name][exp_name].update(x=task_df['TotalEnvSteps'].values)
                    elif exp_name == 'sac_softlearning_single':
                        task_df = df[df['total_timestep'] <= max_timesteps]
                        task_df = task_df.drop_duplicates(subset=['total_timestep'], keep='last')
                        data[task_name][exp_name].update(x=task_df['total_timestep'].values)
                    else:
                        task_df = df[df['task_name'] == task_name]
                        task_df = task_df[task_df['step'] <= max_timesteps]
                        data[task_name][exp_name].update(x=task_df['step'].values)

                    try:
                        if exp_name == 'sac_rlkit_single':
                            y = task_df['evaluation/Returns Mean'].values
                        elif exp_name == 'sac_garage_single':
                            y = task_df['Evaluation/AverageReturn'].values
                        elif exp_name == 'sac_softlearning_single':
                            y = task_df['evaluation/episode-reward-mean'].values
                        else:
                            y = task_df[stat].values

                        if 'y' not in data[task_name][exp_name]:
                            data[task_name][exp_name].update(y=[y])
                        else:
                            data[task_name][exp_name]['y'].append(y)
                    except:
                        raise RuntimeError(f"Statistics '{stat}' doesn't exist in '{data_path}'!")

        plot(ax, data, task_names, exp_names)

    fig_path = osp.abspath(osp.join(save_dir, args.file_name + '.png'))
    # plt.title(exp_name, fontsize=16)
    # fig.suptitle(exp_name + '/' + task_name, fontsize=20).set_y(0.9875)
    plt.tight_layout()
    plt.legend(framealpha=0., fontsize=20)
    plt.savefig(fname=fig_path)
    print(f"Save figure: {fig_path}")


if __name__ == '__main__':
    # custom argument type
    # def str_pair(s):
    #     splited_s = s.split(',')
    #     assert splited_s, 'invalid string pair'
    #     return (splited_s[0], splited_s[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', type=str, nargs='+', default=[
        'mh_ppo_mlp_metaworld_single',
        'mh_sac_mlp_metaworld_single'
    ])
    parser.add_argument('--data_dir', type=str, default='vec_logs')
    parser.add_argument('--save_dir', type=str, default='figures_single/compare')
    parser.add_argument('--task_names', type=str, nargs='+',
                        default=['reach-v2', 'window-close-v2', 'button-press-topdown-v2'])
    parser.add_argument('--file_name', type=str, default='compare_avg')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--max_timesteps', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('--statistics', type=str, nargs='+',
                        default=['episode_reward'])
    args = parser.parse_args()

    main(args)

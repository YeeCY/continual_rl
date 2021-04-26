import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp


WINDOW_LENGTH = 20
SMOOTH_COEF = 0.25


CURVE_FORMAT = {
    'task-0': ([139, 101, 8], '-'),
    'task-1': ([204, 153, 255], '-'),
    'task-2': ([0, 178, 238], '-'),
    'task-3': ([0, 100, 0], '-'),
    'task-4': ([160, 32, 240], '-'),
    'task-5': ([216, 30, 54], '-'),
    'task-6': ([55, 126, 184], '-'),
    'task-7': ([180, 180, 180], '-'),
    'task-8': ([0, 0, 0], '-'),
    'task-9': ([254, 151, 0], '-'),
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


def plot(datas, task_names, curve_format=CURVE_FORMAT):
    for task_name in task_names:
        data = datas[task_name]
        if len(data) == 1:
            continue

        x = data[0]
        y_len = 1E10

        for y in data:
            y_len = min(len(y), y_len)

        for y in range(len(data)):
            data[y] = data[y][:y_len]
        x = x[:y_len]

        y_mean = np.mean(np.array(data[1:]), axis=0)
        y_std = np.std(np.array(data[1:]), axis=0)

        y_mean = window_smooth(y_mean)

        x = np.array(x)

        key = 'task-' + str(task_names.index(task_name))
        color = np.array(curve_format[key][0]) / 255.
        style = curve_format[key][1]
        plt.plot(x, y_mean, color=color, label=task_name, linestyle=style)
        plt.fill_between(x, y_mean - 0.5 * y_std, y_mean + 0.5 * y_std, facecolor=color, alpha=0.1)


def main(args):
    # exp_name = args.exp_name + '-setting-' + str(args.setting)
    exp_name = args.exp_name
    task_names = args.task_names
    data_dir = args.data_dir
    save_dir = args.save_dir
    stats = args.statistics
    seeds = args.seeds
    max_timesteps = args.max_timesteps
    num_fig = len(stats)

    assert osp.exists(data_dir), print("The directory to load data doesn't exit")
    os.makedirs(save_dir, exist_ok=True)

    for s in range(num_fig):
        stat = stats[s]
        ax = plt.subplot(1, len(stats), s+1)
        plt.xlabel('Total Timesteps', fontsize=14)
        plt.ylabel(stat, fontsize=14)
        data = {}

        for d in range(len(seeds)):
            seed = seeds[d]
            data_path = osp.join(data_dir, exp_name, str(seed), 'eval.csv')
            data_path = os.path.abspath(data_path)

            try:
                df = pd.read_csv(data_path)
            except:
                print(f"Data path not found: {data_path}!")
                continue

            # file = file[file['step'] <= args.timesteps]
            # x = file['exploration/all num steps total'].values

            for task_name in task_names:
                task_df = df[df['task_name'] == task_name]
                task_df = task_df[task_df['step'] <= max_timesteps]
                x = task_df['step'].values

                if task_name not in data:
                    data[task_name] = [x]
                try:
                    # st = stat[0] + '/env-' + str(env) + '/' + stat[1]
                    # st = stat[0]
                    y = task_df[stat].values
                    data[task_name].append(y)
                except:
                    raise RuntimeError(f"Statistics '{stat}' doesn't exist in '{data_path}'!")

        plot(data, task_names)

    fig_path = osp.abspath(osp.join(save_dir, exp_name + '.png'))
    plt.title(exp_name, fontsize=16)
    plt.legend(framealpha=0.)
    plt.savefig(fname=fig_path)
    print(f"Save figure: {fig_path}")


if __name__ == '__main__':
    # custom argument type
    # def str_pair(s):
    #     splited_s = s.split(',')
    #     assert splited_s, 'invalid string pair'
    #     return (splited_s[0], splited_s[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='Skew-Fit-SawyerDoorHookEnv-setting-5-expl-3')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='figures')
    # parser.add_argument('--setting', type=int, default=1)
    parser.add_argument('--task_names', type=str, nargs='+',
                        default=['reach-v2', 'window-close-v2', 'button-press-topdown-v2'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--max_timesteps', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('--statistics', type=str, nargs='+',
                        default=['success_rate'])
    args = parser.parse_args()

    main(args)

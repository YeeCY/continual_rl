import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp


WINDOW_LENGTH = 20
SMOOTH_COEF = 0.25


CURVE_FORMAT = {
    'Env-0': ([139, 101, 8], '--'),
    'Env-1': ([204, 153, 255], '--'),
    'Env-2': ([0, 178, 238], '--'),
    'Env-3': ([0, 100, 0], '-'),
    'Env-4': ([160, 32, 240], '-'),
    'Env-5': ([216, 30, 54], '-'),
    'Env-6': ([55, 126, 184], '-'),
    'a': ([180, 180, 180], '-'),
    'b': ([0, 0, 0], '-'),
    'c': ([254, 151, 0], '-'),
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


def plot(datas, envs, curve_format=CURVE_FORMAT):
    for env in envs:
        data = datas[env]
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

        name = 'Env-' + str(env)
        c = np.array(curve_format[name][0]) / 255.
        s = curve_format[name][1]
        plt.plot(x, y_mean, color=c, label=name, linestyle=s)
        plt.fill_between(x, y_mean - 0.5 * y_std, y_mean + 0.5 * y_std, facecolor=c, alpha=0.1)


def main(args):
    # exp_name = args.exp_name + '-setting-' + str(args.setting)
    exp_name = args.exp_name
    save_dir = args.save_dir
    envs, stats, seeds = args.envs, args.statistics, args.seeds
    num_fig = len(stats)

    assert osp.exists(save_dir), "The directory to save figures doesn't exit"
    plt.figure(figsize=(num_fig * 8, 6))

    for s in range(num_fig):
        stat = stats[s]
        ax = plt.subplot(1, len(stats), s+1)
        plt.xlabel('Total Timesteps', fontsize=14)
        plt.ylabel(stat[1], fontsize=14)
        data = {}

        for d in range(len(seeds)):
            seed = seeds[d]
            file_dir = osp.join('data', exp_name, 's-' + str(seed), 'progress.csv')
            file_dir = os.path.abspath(file_dir)

            try:
                file = pd.read_csv(file_dir)
            except:
                print(file_dir + " not found!")
                continue

            file = file[file['exploration/all num steps total'] <= args.timesteps]
            x = file['exploration/all num steps total'].values

            for env in envs:
                if env not in data:
                    data[env] = [x]
                try:
                    st = stat[0] + '/env-' + str(env) + '/' + stat[1]
                    v = file[st].values
                    data[env].append(v)
                except:
                    print('env-' + env + ' not exist')
                    continue

        plot(data, envs)

    plt.title(exp_name, fontsize=16)
    plt.legend(framealpha=0.)
    plt.savefig(fname=osp.join(save_dir, exp_name + '.png'))


if __name__ == '__main__':
    # custom argument type
    def str_pair(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return (splited_s[0], splited_s[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='Skew-Fit-SawyerDoorHookEnv-setting-5-expl-3')
    parser.add_argument('--save_dir', type=str, default='figures')
    parser.add_argument('--setting', type=int, default=1)
    parser.add_argument('--envs', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--timesteps', type=int, default=int(5e5))
    parser.add_argument('--statistics', type=str_pair, nargs='+',
                        default=[('evaluation', 'Final angle_difference Mean')])
    args = parser.parse_args()

    main(args)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp


WINDOW_LENGTH = 10
SMOOTH_COEF = 0.20
CM = 1 / 2.54  # centimeters in inches


CURVE_FORMAT = {
    'sgd': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'SGD',
    },
    'ewc_lambda100': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'ewc_lambda100',
    },
    'ewc_lambda200': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'ewc_lambda200',
    },
    'ewc_lambda500': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'ewc_lambda500',
    },
    'ewc_lambda1000': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'ewc_lambda1000',
    },
    'ewc_lambda2000': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'ewc_lambda2000'
    },
    'ewc_lambda5000': {
        'color': [160, 32, 240],
        'style': '-',
        'label': 'ewc_lambda5000'
    },
    'ewc_lambda8000': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'EWC_lambda8000'
    },
    'ewc_lambda10000': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'ewc_lambda10000'
    },
    'ewc_lambda20000': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'ewc_lambda20000'
    },
    'si_c0.1': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'si_c0.1',
    },
    'si_c1': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'si_c1',
    },
    'si_c5': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'si_c5',
    },
    'si_c10': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'si_c10',
    },
    'si_c25': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'si_c25'
    },
    'si_c50': {
        'color': [160, 32, 240],
        'style': '-',
        'label': 'si_c50'
    },
    'si_c100': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'si_c100'
    },
    'si_c200': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'si_c200'
    },
    'si_c500': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'si_c500'
    },
    'agem_ref_grad_batch_size250': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'agem_ref_grad_batch_size250',
    },
    'agem_ref_grad_batch_size500': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'agem_ref_grad_batch_size500',
    },
    'agem_ref_grad_batch_size1000': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'agem_ref_grad_batch_size1000',
    },
    'agem_ref_grad_batch_size2500': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'agem_ref_grad_batch_size2500',
    },
    'agem_ref_grad_batch_size4500': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'agem_ref_grad_batch_size4500'
    },
    'agem_both_ref_grad_batch_size4500': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'agem_both_ref_grad_batch_size4500'
    },
    'agem_ppo_proj_loss_ref_grad_batch_size4500': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'agem_ppo_proj_loss_ref_grad_batch_size4500'
    },
    'original_agem_ref_grad_batch_size4500': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'original_agem_ref_grad_batch_size4500'
    },
    'oracle_agem_ref_grad_batch_size4500': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'oracle_agem_ref_grad_batch_size4500'
    },
    'oracle_critic_agem_ref_grad_batch_size4500': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'oracle_critic_agem_ref_grad_batch_size4500'
    },
    'oracle_grad_agem_ref_grad_batch_size4500': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'oracle_grad_agem_ref_grad_batch_size4500'
    },
    'oracle_actor_agem_ref_grad_batch_size4500': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'oracle_actor_agem_ref_grad_batch_size4500'
    },
    'oracle_actor_critic_agem_ref_grad_batch_size4500': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'oracle_actor_critic_agem_ref_grad_batch_size4500'
    },
    'agem_ref_grad_batch_size5000': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'agem_ref_grad_batch_size5000'
    },
    'agem_ref_grad_batch_size7500': {
        'color': [160, 32, 240],
        'style': '-',
        'label': 'agem_ref_grad_batch_size7500'
    },
    'agem_ref_grad_batch_size9216': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'agem_ref_grad_batch_size9216'
    },
    'agem_ref_grad_batch_size10000': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'agem_ref_grad_batch_size10000'
    },
    'agem_ref_grad_batch_size15000': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'agem_ref_grad_batch_size15000'
    },
    'agem_ref_grad_batch_size20000': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'agem_ref_grad_batch_size20000'
    },
    'cmaml_budget5000': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'cmaml_budget5000',
    },

    'fisher_brc_mh_bc_mlp_critic': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'fisher_brc_mh_bc_mlp_critic',
    },
    'fisher_brc_mh_bc_offset_critic': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'fisher_brc_mh_bc_offset_critic',
    },
    'fisher_brc_mt_bc_mlp_critic': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'fisher_brc_mt_bc_mlp_critic',
    },
    'fisher_brc_mt_bc_offset_critic': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'fisher_brc_mt_bc_offset_critic',
    },
    'agem_ref_grad_batch_size4500_hybrid': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'agem_ref_grad_batch_size4500_hybrid',
    },
    'agem_ref_grad_batch_size4500_replay_buffer': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'agem_ref_grad_batch_size4500_replay_buffer',
    },
    'agem_ref_grad_batch_size4500_rollout': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'agem_ref_grad_batch_size4500_rollout',
    },
    'ewc_lambda5000_hybrid': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'ewc_lambda5000_hybrid',
    },
    'ewc_lambda5000_replay_buffer': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'ewc_lambda5000_replay_buffer'
    },
    'ewc_lambda5000_rollout': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'ewc_lambda5000_rollout'
    },

    'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_1.0': {
        'color': [0, 0, 0],
        'style': '-',
        'label': 'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_1.0'
    },
    'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_10.0': {
        'color': [255, 0, 0],
        'style': '-',
        'label': 'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_10.0'
    },
    'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_100.0': {
        'color': [255, 0, 255],
        'style': '-',
        'label': 'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_100.0'
    },
    'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_0.1': {
        'color': [255, 0, 255],
        'style': '-',
        'label': 'ewc_lambda5000_hybrid_100_fisher_iters_grad_norm_coeff_0.1'
    },

    'agem_continual_actor_critic_grad_norm_reg_critic_prioritized_memory_ref_grad_bz5000_large_mem_hybrid': {
        'color': [0, 255, 255],
        'style': '-',
        'label': 'agem_continual_actor_critic_grad_norm_reg_critic_prioritized_memory_ref_grad_bz5000_large_mem_hybrid'
    },
    'agem_continual_actor_critic_grad_norm_reg_critic_ref_grad_bz5000_large_mem_hybrid': {
        'color': [220, 220, 0],
        'style': '-',
        'label': 'agem_continual_actor_critic_grad_norm_reg_critic_ref_grad_bz5000_large_mem_hybrid'
    },
    'agem_continual_actor_critic_ref_grad_bz5000_large_mem_hybrid': {
        'color': [0, 255, 0],
        'style': '-',
        'label': 'agem_continual_actor_critic_ref_grad_bz5000_large_mem_hybrid'
    },

    'distilled_actor_no_forgetting_reg': {
        'color': [255, 0, 0],
        'style': '-',
        'label': 'distilled_actor_no_forgetting_reg'
    },
    'distillation_multitask_uniform': {
        'color': [0, 255, 0],
        'style': '-',
        'label': 'distillation_multitask_uniform'
    },
    'distillation_ewc_lambda5000': {
        'color': [255, 0, 0],
        'style': '-',
        'label': 'distillation_ewc_lambda5000'
    },
    'distillation_si_c1.0': {
        'color': [255, 0, 255],
        'style': '-',
        'label': 'distillation_si_c1.0'
    },
    'distillation_agem_memory_budget50000': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'distillation_agem_memory_budget50000',
    },

    'task_embedding_hypernet_actor_reg_coeff0.1': {
        'color': [0, 255, 255],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_reg_coeff0.1'
    },
    'task_embedding_hypernet_actor_reg_coeff0.01': {
        'color': [220, 220, 0],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_reg_coeff0.01'
    },
    'task_embedding_hypernet_actor_reg_coeff1.0': {
        'color': [0, 255, 0],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_reg_coeff1.0'
    },
    'task_embedding_hypernet_actor_reg_coeff0.1_on_the_fly': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_reg_coeff0.1_on_the_fly'
    },
    'task_embedding_hypernet_actor_256_reg_coeff0.1': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_256_reg_coeff0.1',
    },
    'task_embedding_hypernet_actor_256_reg_coeff0.01': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_256_reg_coeff0.01',
    },
    'task_embedding_hypernet_actor_256_reg_coeff1.0': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_256_reg_coeff1.0',
    },
    'task_embedding_hypernet_actor_256_reg_coeff0.1_online_uniform': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_256_reg_coeff0.1_online_uniform',
    },

    'task_embedding_hypernet_actor_256_ewc_actor_loss_lambda5000': {
        'color': [255, 0, 0],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_256_ewc_actor_loss_lambda5000'
    },
    'task_embedding_hypernet_actor_256_si_actor_loss_c1.0': {
        'color': [255, 0, 255],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_256_si_actor_loss_c1.0'
    },
    'task_embedding_hypernet_actor_256_agem_actor_loss_memory_budget50000': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'task_embedding_hypernet_actor_256_agem_actor_loss_memory_budget50000',
    },

    'awp_coeff0.01': {
        'color': [255, 0, 0],
        'style': '-',
        'label': 'awp_coeff0.01'
    },
    'awp_coeff0.05': {
        'color': [255, 0, 255],
        'style': '-',
        'label': 'awp_coeff0.05'
    },
    'awp_coeff0.1': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'awp_coeff0.1',
    },
    'awp_coeff0.0': {
        'color': [0, 255, 0],
        'style': '-',
        'label': 'awp_coeff0.0',
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


def plot(ax, data, algos, curve_format=CURVE_FORMAT):
    for algo in algos:
        algo_data = data[algo]
        # if len(data['y']) == 1:
        #     continue
        if 'y' not in algo_data:
            continue

        if len(algo_data['x']) != len(algo_data['y'][0]):
            min_len = len(algo_data['x'])
            for y in algo_data['y']:
                min_len = min(min_len, len(y))

            algo_data['x'] = algo_data['x'][:min_len]
            for idx, y in enumerate(algo_data['y']):
                algo_data['y'][idx] = y[:min_len]

        x = np.array(algo_data['x'])
        y_len = 1E10

        for y in algo_data['y']:
            y_len = min(len(y), y_len)

        for y in range(len(algo_data['y'])):
            algo_data['y'][y] = algo_data['y'][y][:y_len]
        x = x[:y_len]

        y_mean = np.mean(np.array(algo_data['y']), axis=0)
        y_std = np.std(np.array(algo_data['y']), axis=0)

        y_mean = window_smooth(y_mean)
        y_std = window_smooth(y_std)

        # x = np.array(x)

        # key = 'task-' + str(task_names.index(task_name))
        color = np.array(curve_format[algo]['color']) / 255.
        style = curve_format[algo]['style']
        label = curve_format[algo]['label']
        ax.plot(x, y_mean, color=color, label=label, linestyle=style)
        ax.fill_between(x, y_mean - 0.5 * y_std, y_mean + 0.5 * y_std, facecolor=color, alpha=0.1)


def main(args):
    # exp_name = args.exp_name + '-setting-' + str(args.setting)
    exp_name = args.exp_name
    task_names = args.task_names
    algos = args.algos
    data_dir = args.data_dir
    save_dir = args.save_dir
    stats = args.statistics
    seeds = args.seeds
    max_timesteps = args.max_timesteps
    # num_fig = len(stats)

    if not osp.exists(data_dir):
        print("The directory to load data doesn't exit")
    os.makedirs(save_dir, exist_ok=True)

    fig, _ = plt.subplots(len(task_names), len(stats))
    fig.set_size_inches(16 * len(stats), 8 * len(task_names))
    for task_idx, task_name in enumerate(task_names):
        for stat_idx, stat in enumerate(stats):
            ax = plt.subplot(len(task_names), len(stats), task_idx * len(stats) + stat_idx + 1)
            ax.set_title(task_name, fontsize=15)
            ax.set_xlabel('Total Timesteps', fontsize=15)
            ax.set_ylabel(stat, fontsize=15)
            data = {}

            for algo in algos:
                data[algo] = {}

                for seed in seeds:
                    if 'distilled' in algo:
                        data_path = osp.join(data_dir, exp_name, algo, str(seed), 'distill', 'eval.csv')
                    elif 'distillation' in algo:
                        data_path = osp.join(data_dir, exp_name, algo, str(seed), 'distillation', 'eval.csv')
                    else:
                        data_path = osp.join(data_dir, exp_name, algo, str(seed), 'eval.csv')
                    data_path = os.path.abspath(data_path)

                    try:
                        df = pd.read_csv(data_path)
                    except:
                        print(f"Data path not found: {data_path}!")
                        continue

                    # file = file[file['step'] <= args.timesteps]
                    # x = file['exploration/all num steps total'].values

                    task_df = df[df['task_name'] == task_name]
                    task_df = task_df[task_df['step'] <= max_timesteps]
                    data[algo].update(x=task_df['step'].values)

                    try:
                        y = task_df[stat].values
                        if 'y' not in data[algo]:
                            data[algo].update(y=[y])
                        else:
                            data[algo]['y'].append(y)
                    except:
                        raise RuntimeError(f"Statistics '{stat}' doesn't exist in '{data_path}'!")

            plot(ax, data, algos)
            ax.legend(framealpha=0.)

    fig_path = osp.abspath(osp.join(save_dir, exp_name + '.png'))
    # plt.title(exp_name, fontsize=16)
    # fig.suptitle(exp_name, fontsize=20).set_y(0.9875)
    plt.tight_layout()
    plt.savefig(fname=fig_path)
    print(f"Save figure: {fig_path}")


if __name__ == '__main__':
    # custom argument type
    # def str_pair(s):
    #     splited_s = s.split(',')
    #     assert splited_s, 'invalid string pair'
    #     return (splited_s[0], splited_s[1])
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='reach_window-close_button-press-topdown')
    parser.add_argument('--data_dir', type=str, default='vec_logs')
    parser.add_argument('--save_dir', type=str, default='figures_continual')
    # parser.add_argument('--setting', type=int, default=1)
    parser.add_argument('--task_names', type=str, nargs='+',
                        default=['reach-v2', 'window-close-v2', 'button-press-topdown-v2'])
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['sgd', 'ewc', 'si'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--max_timesteps', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('--statistics', type=str, nargs='+',
                        default=['episode_reward'])
    args = parser.parse_args()

    main(args)

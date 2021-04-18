from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import csv
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('task_name', 'task_name', 'str'),
            ('episode', 'episode', 'int'), ('step', 'step', 'int'),
            ('duration', 'duration', 'time'), ('episode_reward', 'return', 'float'),
            ('batch_reward', 'batch_reward', 'float'), ('actor_loss', 'actor_loss', 'float'),
            ('critic_loss', 'critic_loss', 'float'),
            ('ss_inv_loss', 'ss_inv_loss', 'float'),
            ('episode_success', 'success', 'float'),
            ('recent_episode_reward', 'recent_return', 'float'),
            ('recent_success_rate', 'recent_success_rate', 'float')
        ],
        'eval': [
            ('task_name', 'task_name', 'str'),
            ('step', 'step', 'int'), ('episode_reward', 'return', 'float'),
            ('episode_ss_pred_var', 'ss_pred_var', 'float'),
            ('success_rate', 'success_rate', 'float'),
        ]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = self._prepare_file(file_name, 'log')
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        elif ty == 'str':
            template += '%s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix, save=True, info=None):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            if isinstance(info, dict):
                for key, val in info.items():
                    assert key.startswith('train') or key.startswith('eval'), \
                        "Keys in 'info' must begin with 'train' or 'eval'!"
                    if key.startswith('train'):
                        key = key[len('train') + 1:]
                    else:
                        key = key[len('eval') + 1:]
                    key = key.replace('/', '')
                    data[key] = val
            self._dump_to_file(data)
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self,
                 log_dir,
                 log_frequency=10000,
                 action_repeat=1,
                 save_tb=True,
                 config='rl'):
        """
            (chongyi zheng): update Logger to DrQ version
        """
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        self._action_repeat = action_repeat
        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval'),
            formating=FORMAT_CONFIG[config]['eval']
        )

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _update_step(self, step):
        return step * self._action_repeat

    def _try_sw_log(self, key, value, step):
        # step = self._update_step(step)
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        # step = self._update_step(step)
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step):
        # step = self._update_step(step)
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        # step = self._update_step(step)
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1, log_frequency=1, sw_prefix=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        if isinstance(value, (float, int, np.ndarray)):
            if sw_prefix is not None:
                sw_key = sw_prefix + key
            else:
                sw_key = key
            self._try_sw_log(sw_key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_image(key, image, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None, info=None):
        # step = self._update_step(step)
        if ty is None:
            self._train_mg.dump(step, 'train', save, info)
            self._eval_mg.dump(step, 'eval', save, info)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save, info)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save, info)
        else:
            raise f'invalid log type: {ty}'

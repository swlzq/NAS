# @Author: LiuZhQ

import os
import shutil
import sys
import torch
import numpy as np
import torch.backends.cudnn as cudnn


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeRecorder(object):
    """
    Recode training time.
    """

    def __init__(self, start_epoch, epochs, logger):
        self.total_time = 0.
        self.remaining_time = 0.
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.logger = logger

    def update(self, time):
        self.total_time += time
        self.start_epoch += 1
        self.remaining_time = time * (self.epochs - self.start_epoch)

        self.logger.info('Cost time=>' + self.format_time(self.total_time))
        self.logger.info('Remaining time=>' + self.format_time(self.remaining_time))

    @staticmethod
    def format_time(time):
        h = time // 3600
        m = (time % 3600) // 60
        s = (time % 3600) % 60
        return '{}h{}m{}s'.format(h, m, s)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def set_randomness(seed):
    """
    Control experimental randomness.
    :param seed: Random seed
    :return:
    """
    if not torch.cuda.is_available():
        print('No gpu device available!')
        sys.exit(1)
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)

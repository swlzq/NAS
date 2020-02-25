# author: LiuZhQ


import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

import utils


class BaseTrainer(object):
    def __init__(self, setting, logger):
        """
        Base Trainer: train and evaluate model
        :param setting:
        :param logger:
        """
        self.setting = setting
        self.logger = logger
        self.writer = SummaryWriter(self.setting.output_path)

        self.criterion = nn.CrossEntropyLoss().cuda()

        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.scheduler = None
        # 获取训练集和验证集
        self.train_queue, self.valid_queue = None, None

    def _get_lr(self):
        """
        Get learning rate.
        """
        assert self.scheduler is not None, 'Scheduler should not be none!'
        return self.scheduler.get_lr()[0]

    def run(self):
        """
        Run search trainer
        :return:
        """
        pass

    def train(self, epoch):
        """
        Train model
        :param epoch:
        :return:
        """
        pass

    def infer(self, epoch, auxiliary=False):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        assert self.model is not None, 'Model should not be none!'
        assert self.valid_queue is not None, 'Validation dataset should not be none!'
        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.valid_queue):
                input = input.cuda()
                target = target.cuda()

                if auxiliary:
                    logits, _ = self.model(input)
                else:
                    logits = self.model(input)
                loss = self.criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
                if (i + 1) % self.setting.log_interval == 0:
                    self._log_data('Valid', epoch, i + 1, top1.val, top1.avg, objs.val, objs.avg)

        return top1.avg, objs.avg

    def _write_data(self, values, epoch):
        """
        Write data to tensorboardX
        """
        for k, v in values.items():
            self.writer.add_scalar(k, v, epoch)

    def _log_data(self, flag, epoch, i, acc_val, acc_avg, loss_val, loss_avg):
        """
        Print and log data
        """
        assert self.train_queue is not None, 'Training dataset should not be none!'
        self.logger.info('{} Epoch: [{}/{}]([{}/{}])\t'
                         'Loss: {:.4f}({:.4f})\t'
                         'Accuracy: {:.4f}({:.4f})\t'
                         'LR: {}\t'.format(flag, epoch + 1, self.setting.epochs,
                                           i, len(self.train_queue),
                                           loss_val, loss_avg,
                                           acc_val, acc_avg,
                                           self._get_lr()
                                           ))

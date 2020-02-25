# author: LiuZhQ

import os
import time
import logging
import torch.nn.functional as F

from .base_trainer import *
from dataset import get_dataset
from model import Network
from model.architect import Architect


class SearchTrainer(BaseTrainer):
    def __init__(self, setting, logger):
        super(SearchTrainer, self).__init__(setting, logger)
        """
        Search Trainer: train and evaluate architecture
        :param setting:
        :param logger:
        """
        # Network with mixed operations
        self.model = Network(setting.init_channels, setting.num_classes, setting.layers, self.criterion).cuda()

        # 获取alphas参数的内存地址，过滤掉得到模型权重
        arch_params = list(map(id, self.model.arch_parameters()))
        weight_params = filter(lambda p: id(p) not in arch_params, self.model.parameters())

        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

        # SGD和CosineAnnealingLR训练模型权重w
        self.optimizer = torch.optim.SGD(
            weight_params,
            setting.lr,
            momentum=setting.momentum,
            weight_decay=setting.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(setting.epochs), eta_min=setting.lr_min
        )

        # 获取训练集和验证集
        self.train_queue, self.valid_queue = get_dataset(setting)

        # 用于alpha的更新优化
        self.architect = Architect(self.model, self.criterion, setting)

    def run(self):
        """
        Run search trainer
        :return:
        """
        training_time = utils.TimeRecorder(0, self.setting.epochs, self.logger)
        for epoch in range(self.setting.epochs):
            start_time = time.time()

            genotype = self.model.genotype()
            logging.info('genotype={}'.format(genotype))
            print(F.softmax(self.model.alphas_normal, dim=-1))
            print(F.softmax(self.model.alphas_reduce, dim=-1))
            # training
            train_acc, train_obj = self.train(epoch)

            train_values = {'lr': self._get_lr(), 'train_acc': train_acc, 'train_loss': train_obj}
            self._write_data(train_values, epoch)
            # validation
            valid_acc, valid_obj = self.infer(epoch)
            valid_values = {'valid_acc': valid_acc, 'valid_loss': valid_obj}
            self._write_data(valid_values, epoch)

            utils.save(self.model, os.path.join(self.setting.output_path, 'weights.pt'))
            self.scheduler.step()
            training_time.update(time.time() - start_time)

    def train(self, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        self.model.train()

        for i, (inputs, targets) in enumerate(self.train_queue):
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

            try:
                inputs_search, targets_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(self.valid_queue)
                inputs_search, targets_search = next(valid_queue_iter)

            inputs_search = inputs_search.cuda()
            targets_search = targets_search.cuda(non_blocking=True)

            # 搜索过程的下一步
            self.architect.step(inputs, targets, inputs_search, targets_search, self._get_lr(),
                                self.optimizer, unrolled=self.setting.unrolled)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.setting.grad_clip)
            self.optimizer.step()

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            objs.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            if (i + 1) % self.setting.log_interval == 0:
                self._log_data('Train', epoch, i + 1, top1.val, top1.avg, objs.val, objs.avg)

        return top1.avg, objs.avg

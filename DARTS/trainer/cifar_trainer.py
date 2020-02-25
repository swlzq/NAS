# author: LiuZhQ

import os
import time

from .base_trainer import *
from dataset import get_evaluate_dataset
from model import NetworkCIFAR, genotypes


class CifarTrainer(BaseTrainer):
    def __init__(self, setting, logger):
        super(CifarTrainer, self).__init__(setting, logger)
        """
        Cifar Trainer: test performance on CIFAR
        :param setting:
        :param logger:
        """

        genotype = eval("genotypes.%s" % setting.arch)
        model = NetworkCIFAR(setting.init_channels, setting.num_classes, setting.layers, setting.auxiliary, genotype)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
        else:
            self.model = model.cuda()
        logger.info(model)
        logger.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

        # SGD和CosineAnnealingLR训练模型权重w
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            setting.lr,
            momentum=setting.momentum,
            weight_decay=setting.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(setting.epochs)
        )

        # 获取训练集和测试集
        self.train_queue, self.valid_queue = get_evaluate_dataset(setting)

    def run(self):
        """
        Run search trainer
        :return:
        """
        training_time = utils.TimeRecorder(0, self.setting.epochs, self.logger)
        for epoch in range(self.setting.epochs):
            start_time = time.time()

            self.model.drop_path_prob = self.setting.drop_path_prob * epoch / self.setting.epochs
            # training
            train_acc, train_obj = self.train(epoch)

            train_values = {'lr': self._get_lr(), 'train_acc': train_acc, 'train_loss': train_obj}
            self._write_data(train_values, epoch)
            # validation
            valid_acc, valid_obj = self.infer(epoch, auxiliary=self.setting.auxiliary)
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
            self.optimizer.zero_grad()
            logits, logits_aux = self.model(inputs)
            loss = self.criterion(logits, targets)
            if self.setting.auxiliary:
                loss_aux = self.criterion(logits_aux, targets)
                loss += self.setting.auxiliary_weight * loss_aux
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

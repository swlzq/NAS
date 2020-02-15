# @Author:LiuZhQ

import os
import sys
import time
import glob
import pyhocon
import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import utils
from args import arg_parse
from model_search import Network
from architect import Architect

args = arg_parse()
_date = time.strftime('%Y-%m-%d-%H%M%S')
args.save = os.path.join('experiments', 'search-{}-{}'.format(args.save, _date))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('No gpu device available!')
        sys.exit(1)

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss().cuda()
    # 获取混合操作的网络结构
    model = Network(args.init_channels, args.num_classes, args.layers, criterion)
    model = model.cuda()
    # model = nn.DataParallel(model).module

    for name, params in model.named_parameters():
        print(name, params.size())
    # 获取alphas参数的内存地址，过滤掉得到模型权重
    arch_params = list(map(id, model.arch_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_params, model.parameters())

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # 获取CIFAR10数据集，并按比例划分训练集和验证集
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    # SGD和CosineAnnealingLR训练模型权重w
    optimizer = torch.optim.SGD(
        weight_params,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )

    architect = Architect(model, criterion, args)

    total_time = 0.
    for epoch in range(args.epochs):
        start_time = time.time()
        lr = scheduler.get_lr()[0]
        logging.info('epoch:{}, lr:{}'.format(epoch, lr))
        genotype = model.genotype()
        logging.info('genotype={}'.format(genotype))
        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)

        # validation
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        scheduler.step()

        epoch_time = time.time() - start_time
        total_time += epoch_time
        remain_time = epoch_time * (args.epochs - epoch + 1)
        print('Already cost time:{}h{}m{}s'.format(total_time // 3600, (total_time % 3600) // 60,
                                                   (total_time % 3600) % 60))
        print('Estimate remaining time:{}h{}m{}s'.format(remain_time // 3600, (remain_time % 3600) // 60,
                                                         (remain_time % 3600) % 60))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()

    for step, (inputs, targets) in enumerate(train_queue):
        n = inputs.size(0)

        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)

        try:
            inputs_search, targets_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            inputs_search, targets_search = next(valid_queue_iter)

        inputs_search = inputs_search.cuda()
        targets_search = targets_search.cuda(non_blocking=True)

        # 搜索过程的下一步
        architect.step(inputs, targets, inputs_search, targets_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

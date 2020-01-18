# @Time  :2020/1/14
# @Author:LiuZhQ

import argparse
import time
import os
import torch
import random
import numpy as np


class Argument(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Optional parameters')
        # ==================== Initialize directory path ====================
        parser.add_argument('--root_path',
                            type=str, help='Project root path')
        parser.add_argument('--data_path', default='/home/dataset/cifar',
                            type=str, help='Dataset root directory path')
        parser.add_argument('--result_path', default='results',
                            type=str, help='Result root directory path')
        parser.add_argument('--pretrained_model_path', default='pretrained_models',
                            type=str, help='Path to store previous trained models')
        parser.add_argument('--resume_path', default='',
                            type=str, help='Checkpoint path to resume training')
        parser.add_argument('--test_path', default='',
                            type=str, help='Test models path')
        # ==================== Initialize directory path ====================

        # ==================== Initialize models setting ====================
        parser.add_argument('--model_name', default='resnet50',
                            type=str, help='Model name')
        parser.add_argument('--input_size', default=224,
                            type=int, help='Input size of models')
        parser.add_argument('--num_classes', default=8,
                            type=int, help='For classification task')
        parser.add_argument('--input_images', default=5,
                            type=int, help='Number of input images')
        parser.add_argument('--use_cpu', action='store_true',
                            help='If True use cpu else cuda')
        # ==================== Initialize models setting ====================

        # ==================== Initialize optimizer setting ====================
        parser.add_argument('--lr', default=0.001,
                            type=float, help='Number of learning rate')
        parser.add_argument('--step_size', default=20,
                            type=int, help='Number of lr change interval')
        parser.add_argument('--gamma', default=0.1,
                            type=float, help='Number of lr multiple coefficient')
        parser.add_argument('--momentum', default=0.9,
                            type=float, help='Number of momentum')
        parser.add_argument('--weight_decay', default=5e-4,
                            type=float, help='Number of half l2 regularization coefficient')
        parser.add_argument('--patient', default=10,
                            type=int, help='Number of lr change patient')
        parser.add_argument('--milestones', default=[10],
                            type=int, nargs='+', help='Epoch of adjusting lr')
        # ==================== Initialize optimizer setting ====================

        # ==================== Initialize dataset setting ====================
        parser.add_argument('--train_dataset', default='DRIVER',
                            type=str, help='Dataset name for training')
        parser.add_argument('--test_dataset', default='DRIVER',
                            type=str, help='Dataset name for testing')
        parser.add_argument('--batch_size', default=128,
                            type=int, help='Number of batch size')
        parser.add_argument('--up_scale_train', default=1,
                            type=int, help='Traing up sample scale')
        parser.add_argument('--up_scale_test', default=1,
                            type=int, help='Testing up sample scale')
        parser.add_argument('--tsn_images', default=4,
                            type=int, help='Number of tsn images')
        parser.add_argument('--num_workers', default=4,
                            type=int, help='Number of threading')
        parser.add_argument('--pin_memory', action='store_true',
                            help='If False not use pin memory')
        # ==================== Initialize dataset setting ====================

        # ==================== Initialize training setting ====================
        parser.add_argument('--seed', default=7777,
                            type=int, help='Ensure training can be reproduced')
        parser.add_argument('--begin_epoch', default=0,
                            type=int, help='Number of beginning training epoch')
        parser.add_argument('--epochs', default=20,
                            type=int, help='Number of total training epochs')
        parser.add_argument('--log_interval', default=20,
                            type=int, help='Number of logging training status interval')
        parser.add_argument('--checkpoint_interval', default=5,
                            type=int, help='Number of save checkpoints interval')
        parser.add_argument('--flag', default='EXP',
                            type=str, help='Experiment flag')
        # ==================== Initialize training setting ====================
        self.args = parser.parse_args()

        self._init_file_setting()
        self._set_seed()

    def _init_file_setting(self):
        # Get root path
        root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.args.root_path = root_path
        # Each experiment has different flag to identity
        _date = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
        flag = ''
        flag = '_' + self.args.flag + '_' + _date if self.args.flag else '_' + _date
        self.args.result_path = os.path.join(self.args.root_path, self.args.result_path, flag)
        assert not os.path.exists(self.args.result_path), '{} has existed.'.format(self.args.result_path)
        # Create pretrained models directory
        self.args.pretrained_model_path = os.path.join(self.args.root_path, self.args.pretrained_model_path)
        if not os.path.exists(self.args.pretrained_model_path):
            os.makedirs(self.args.pretrained_model_path)

    def _set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def get_args(self):
        return self.args


if __name__ == '__main__':
    pass
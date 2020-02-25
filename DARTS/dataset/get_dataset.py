# @Author: LiuZhQ

import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_dataset(setting):
    if setting.data_name == 'CIFAR10':
        # 获取CIFAR10数据集，并按比例划分训练集和验证集
        train_transform, valid_transform = data_transforms_cifar10(setting.cutout, setting.cutout_length)
        train_data = dset.CIFAR10(root=setting.data_path, train=True, download=False, transform=train_transform)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(setting.train_portion * num_train))
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=setting.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=setting.num_workers)
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=setting.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=setting.num_workers)
    else:
        assert False, '{} error!'.format(setting.data_name)
    return train_queue, valid_queue


def get_evaluate_dataset(setting):
    if setting.data_name == 'CIFAR10':
        train_transform, valid_transform = data_transforms_cifar10(setting.cutout, setting.cutout_length)
        train_data = dset.CIFAR10(root=setting.data_path, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR10(root=setting.data_path, train=True, download=False, transform=train_transform)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=setting.batch_size,
            pin_memory=True, num_workers=setting.num_workers)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=setting.batch_size,
            pin_memory=True, num_workers=setting.num_workers)
    return train_queue, valid_queue


def data_transforms_cifar10(cutout=False, cutout_length=16):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

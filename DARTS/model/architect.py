# @Author: LiuZhQ

import numpy as np
import torch
import torch.nn as nn


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model, criterion, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.alpha_weight_decay
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=args.alpha_lr,
                                          betas=(0.5, 0.999), weight_decay=args.alpha_weight_decay)

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            pass
            # self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    # ξ=0
    def _backward_step(self, input_valid, target_valid):
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        pass

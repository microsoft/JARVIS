#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    """Set the learning rate for each parameter group using a polynomial defined as:
    lr = base_lr * (1 - T_cur/T_max) ^ (power), where T_cur is the current epoch and T_max is the maximum number of
    epochs.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of epochs
        power (int): Degree of polynomial
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, T_max, power, last_epoch=-1):
        self.T_max = T_max
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # base_lr * (1 - iter/max_iter) ^ (power)
        return [base_lr * (1 - self.last_epoch / self.T_max) ** self.power
                for base_lr in self.base_lrs]


class MultiStepMultiGammaLR(_LRScheduler):
    """Similar to torch.otpim.MultiStepLR, but instead of a single gamma value, specify a gamma value per-milestone.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gammas (list): List of gamma values. Must have same length as milestones.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        if len(milestones) != len(gammas):
            raise ValueError('Milestones and Gammas lists should be of same length.')

        self.milestones = milestones
        self.multiplicative_gammas = [1]
        for idx, gamma in enumerate(gammas):
            self.multiplicative_gammas.append(gamma * self.multiplicative_gammas[idx])

        super(MultiStepMultiGammaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = bisect_right(self.milestones, self.last_epoch)
        return [base_lr * self.multiplicative_gammas[idx] for base_lr in self.base_lrs]

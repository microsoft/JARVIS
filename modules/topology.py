#
# Copyright (c) 2019 Intel Corporation
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
"""Modules related to a model's topology"""
import torch.nn as nn


class BranchPoint(nn.Module):
    """Add a branch to an existing model."""
    def __init__(self, branched_module, branch_net):
        """
        :param branched_module: the module in the original network to which we add a branch.
        :param branch_net: the new branch
        """
        super().__init__()
        self.branched_module = branched_module
        self.branch_net = branch_net
        self.output = None

    def forward(self, x):
        x1 = self.branched_module.forward(x)
        self.output = self.branch_net.forward(x1)
        return x1


# This class is "borrowed" from PyTorch 1.3 until we integrate it
class Flatten(nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


# A temporary trick to see if we need to add Flatten to the `torch.nn` namespace for convenience.
try:
    Flatten = nn.Flatten
except AttributeError:
    nn.Flatten = Flatten
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
import torch
import torch.nn as nn


class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise addition"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class EltwiseSub(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise subtraction"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res -= t
        else:
            for t in input[1:]:
                res = res - t
        return res


class EltwiseMult(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise multiplication"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res


class EltwiseDiv(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise division"""
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor, y):
        if self.inplace:
            return x.div_(y)
        return x.div(y)


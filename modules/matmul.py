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
import torch
import torch.nn as nn


class Matmul(nn.Module):
    """
    A wrapper module for matmul operation between 2 tensors.
    """
    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return a.matmul(b)


class BatchMatmul(nn.Module):
    """
    A wrapper module for torch.bmm operation between 2 tensors.
    """
    def __init__(self):
        super(BatchMatmul, self).__init__()

    def forward(self, a: torch.Tensor, b:torch.Tensor):
        return torch.bmm(a, b)
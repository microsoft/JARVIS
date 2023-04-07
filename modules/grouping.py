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


class Concat(nn.Module):
    def __init__(self, dim=0):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *seq):
        return torch.cat(seq, dim=self.dim)


class Chunk(nn.Module):
    def __init__(self, chunks, dim=0):
        super(Chunk, self).__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, tensor):
        return tensor.chunk(self.chunks, dim=self.dim)


class Split(nn.Module):
    def __init__(self, split_size_or_sections, dim=0):
        super(Split, self).__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, tensor):
        return torch.split(tensor, self.split_size_or_sections, dim=self.dim)


class Stack(nn.Module):
    def __init__(self, dim=0):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, seq):
        return torch.stack(seq, dim=self.dim)

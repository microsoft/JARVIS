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

from .eltwise import *
from .grouping import *
from .matmul import *
from .rnn import *
from .aggregate import *
from .topology import *

__all__ = ['EltwiseAdd', 'EltwiseSub', 'EltwiseMult', 'EltwiseDiv', 'Matmul', 'BatchMatmul',
           'Concat', 'Chunk', 'Split', 'Stack',
           'DistillerLSTMCell', 'DistillerLSTM', 'convert_model_to_distiller_lstm',
           'Norm', 'Mean', 'BranchPoint', 'Print']


class Print(nn.Module):
    """Utility module to temporarily replace modules to assess activation shape.

    This is useful, e.g., when creating a new model and you want to know the size
    of the input to nn.Linear and you want to avoid calculating the shape by hand.
    """
    def forward(self, x):
        print(x.size())
        return x
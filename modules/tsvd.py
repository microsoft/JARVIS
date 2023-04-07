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

"""Truncated-SVD module.

For an example of how truncated-SVD can be used, see this Jupyter notebook:
https://github.com/IntelLabs/distiller/blob/master/jupyter/truncated_svd.ipynb

"""

def truncated_svd(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer using truncated SVD.

    For the original implementation (MIT license), see Faster-RCNN:
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/compress_net.py
    We replaced numpy operations with pytorch operations (so that we can leverage the GPU).

    Arguments:
      W: N x M weights matrix
      l: number of singular values to retain
    Returns:
      Ul, L: matrices such that W \approx Ul*L
    """

    U, s, V = torch.svd(W, some=True)

    Ul = U[:, :l]
    sl = s[:l]
    V = V.t()
    Vl = V[:l, :]

    SV = torch.mm(torch.diag(sl), Vl)
    return Ul, SV


class TruncatedSVD(nn.Module):
    def __init__(self, replaced_gemm, gemm_weights, preserve_ratio):
        super().__init__()
        self.replaced_gemm = replaced_gemm
        print("W = {}".format(gemm_weights.shape))
        self.U, self.SV = truncated_svd(gemm_weights.data, int(preserve_ratio * gemm_weights.size(0)))
        print("U = {}".format(self.U.shape))

        self.fc_u = nn.Linear(self.U.size(1), self.U.size(0)).cuda()
        self.fc_u.weight.data = self.U

        print("SV = {}".format(self.SV.shape))
        self.fc_sv = nn.Linear(self.SV.size(1), self.SV.size(0)).cuda()
        self.fc_sv.weight.data = self.SV#.t()

    def forward(self, x):
        x = self.fc_sv.forward(x)
        x = self.fc_u.forward(x)
        return x

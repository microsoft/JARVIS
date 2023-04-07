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

"""
Norm functions.

Norms functions map a tensor to a single real-valued scalar that represents
the tensor's magnitude according to some definition.  p-norms (Lp norms)
are the most common magnitude functions.

Many times we want to divide a large 4D/3D/2D tensor into groups of
equal-sized sub-tensors, to compute the norm of each sub-tensor. The
most common use-case is ranking of sub-tensors according to some norm.


For an interesting comparison of the characteristics of L1-norm vs. L2-norm,
see: https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms)

"""
import torch
import numpy as np
from functools import partial
from random import uniform


__all__ = ["kernels_lp_norm", "channels_lp_norm", "filters_lp_norm",
           "kernels_norm", "channels_norm", "filters_norm", "sub_matrix_norm",
           "rows_lp_norm", "cols_lp_norm",
           "rows_norm", "cols_norm",
           "l1_norm", "l2_norm", "max_norm",
           "rank_channels", "rank_filters"]


class NamedFunction:
    def __init__(self, f, name):
        self.f = f
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.name


""" Norm (magnitude) functions.

These functions are named-functions because it's convenient
to refer to them when logging.
"""


def _max_norm(t, dim=1):
    """Maximum norm.

    if t is some vector such that t = (t1, t2, ...,tn), then
        max_norm = max(|t1|, |t2|, ...,|tn|)
    """
    maxv, _ = t.abs().max(dim=dim)
    return maxv


l1_norm = NamedFunction(partial(torch.norm, p=1, dim=1), "L1")
l2_norm = NamedFunction(partial(torch.norm, p=2, dim=1), "L2")
max_norm = NamedFunction(_max_norm, "Max")


def kernels_lp_norm(param, p=1, group_len=1, length_normalized=False):
    """L1/L2 norm of kernel sub-tensors in a 4D tensor.

    A kernel is an m x n matrix used for convolving a feature-map to extract features.

    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        p: the exponent value in the norm formulation
        group_len: the numbers of (adjacent) kernels in each group.  Norms are calculated
           on the entire group.
        length_normalized: if True then normalize the norm.  I.e.
           norm = group_norm / num_elements_in_group

    Returns:
        1D tensor with norms of the groups
    """
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return kernels_norm(param, norm_fn, group_len, length_normalized)


def kernels_norm(param, norm_fn, group_len=1, length_normalized=False):
    """Compute some norm of 2D kernels of 4D parameter tensors.

    Assumes 4D weights tensors.
    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        norm_fn: a callable that computes a normal
        group_len: the numbers of (adjacent) kernels in each group.  Norms are calculated
           on the entire group.
        length_normalized: if True then normalize the norm.  I.e.
           norm = group_norm / num_elements_in_group

    Returns:
        1D tensor with lp-norms of the groups
    """
    assert param.dim() == 4, "param has invalid dimensions"
    group_size = group_len * np.prod(param.shape[2:])
    return generic_norm(param.view(-1, group_size), norm_fn, group_size, length_normalized, dim=1)


def channels_lp_norm(param, p=1, group_len=1, length_normalized=False):
    """L1/L2 norm of channels sub-tensors in a 4D tensor

    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        p: the exponent value in the norm formulation
        group_len: the numbers of (adjacent) channels in each group.  Norms are calculated
           on the entire group.
        length_normalized: if True then normalize the norm.  I.e.
           norm = group_norm / num_elements_in_group

    Returns:
        1D tensor with norms of the groups
    """
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return channels_norm(param, norm_fn, group_len, length_normalized)


def channels_norm(param, norm_fn, group_len=1, length_normalized=False):
    """Compute some norm of parameter input-channels.

    Computing the norms of weight-matrices input channels is logically similar to computing
    the norms of 4D Conv weights input channels so we treat both cases.
    Assumes 2D or 4D weights tensors.

    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        norm_fn: a callable that computes a normal
        group_len: the numbers of (adjacent) channels in each group.  Norms are calculated
           on the entire group.
        length_normalized: if True then normalize the norm.  I.e.
           norm = group_norm / num_elements_in_group

    Returns:
        1D tensor with lp-norms of the groups
    """
    assert param.dim() in (2, 4), "param has invalid dimensions"
    if param.dim() == 2:
        # For GEMM operations, PyTorch stores the weights matrices in a transposed format.  I.e.
        # before performing GEMM, a matrix is transposed.  This is because the output is computed
        # as follows (see torch.nn.functional.linear):
        #   y = x(W^T) + b ; where W^T is the transpose of W
        #
        # Therefore, W is expected to have shape (output_channels, input_channels), and to compute
        # the norms of input_channels, we compute the norms of W's columns.
        return cols_norm(param, norm_fn, group_len, length_normalized)
    param = param.transpose(0, 1).contiguous()
    group_size = group_len * np.prod(param.shape[1:])
    return generic_norm(param.view(-1, group_size), norm_fn, group_size, length_normalized, dim=1)


def filters_lp_norm(param, p=1, group_len=1, length_normalized=False):
    """L1/L2 norm of filters sub-tensors in a 4D tensor

    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        p: the exponent value in the norm formulation
        group_len: the numbers of (adjacent) filters in each group.  Norms are calculated
           on the entire group.
        length_normalized: if True then normalize the norm.  I.e.
           norm = group_norm / num_elements_in_group

    Returns:
        1D tensor with norms of the groups
    """
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return filters_norm(param, norm_fn, group_len, length_normalized)


def filters_norm(param, norm_fn, group_len=1, length_normalized=False):
    """Compute some norm of 3D filters of 4D parameter tensors.

    Assumes 4D weights tensors.
    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        norm_fn: a callable that computes a normal
        group_len: the numbers of (adjacent) filters in each group.  Norms are calculated
           on the entire group.
        length_normalized: if True then normalize the norm.  I.e.
           norm = group_norm / num_elements_in_group

    Returns:
        1D tensor with lp-norms of the groups
    """
    assert param.dim() == 4, "param has invalid dimensions"
    group_size = group_len * np.prod(param.shape[1:])
    return generic_norm(param.view(-1, group_size), norm_fn, group_size, length_normalized, dim=1)


def sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim):
    """Compute some norm of rows/cols of 2D parameter tensors.

    Assumes 2D weights tensors.
    Args:
        param: shape (num_filters(0), nun_channels(1), kernel_height(2), kernel_width(3))
        norm_fn: a callable that computes a normal
        group_len: the numbers of (adjacent) filters in each group.  Norms are calculated
           on the entire group.
        length_normalized: if True then normalize the norm.  I.e.
           norm = group_norm / num_elements_in_group

    Returns:
        1D tensor with lp-norms of the groups
    """
    assert param.dim() == 2, "param has invalid dimensions"
    group_size = group_len * param.size(abs(dim - 1))
    return generic_norm(param, norm_fn, group_size, length_normalized, dim)


def rows_lp_norm(param, p=1, group_len=1, length_normalized=False):
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=1)


def rows_norm(param, norm_fn, group_len=1, length_normalized=False):
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=1)


def cols_lp_norm(param, p=1, group_len=1, length_normalized=False):
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=0)


def cols_norm(param, norm_fn, group_len=1, length_normalized=False):
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=0)


def generic_norm(param, norm_fn, group_size, length_normalized, dim):
    with torch.no_grad():
        if dim is not None:
            norm = norm_fn(param, dim=dim)
        else:
            # The norm may have been specified as part of the norm function
            norm = norm_fn(param)
        if length_normalized:
            norm = norm / group_size
        return norm


"""
Ranking functions
"""


def num_structs_to_prune(n_elems, group_len, fraction_to_prune, rounding_fn):
    n_structs_to_prune = rounding_fn(fraction_to_prune * n_elems)
    n_structs_to_prune = int(rounding_fn(n_structs_to_prune * 1. / group_len) * group_len)

    # We can't allow removing all of the structs in a layer! --
    # Except when the fraction_to_prune is explicitly instructing us to do so.
    # n_ch_to_prune is the number of channels to prune.
    if n_structs_to_prune == n_elems and fraction_to_prune != 1.0:
        n_structs_to_prune = n_elems - group_len
    return n_structs_to_prune


def e_greedy_normal_noise(mags, e):
    """Epsilon-greedy noise

    If e>0 then with probability(adding noise) = e, multiply mags by a normally-distributed
    noise.
    :param mags: input magnitude tensor
    :param e: epsilon (real scalar s.t. 0 <= e <=1)
    :return: noise-multiplier.
    """
    if e and uniform(0, 1) <= e:
        # msglogger.info("%sRankedStructureParameterPruner - param: %s - randomly choosing channels",
        #                threshold_type, param_name)
        return torch.randn_like(mags)
    return 1


def k_smallest_elems(mags, k, noise):
    """Partial sort of tensor `mags` returning a list of the k smallest elements in order.

    :param mags: tensor of magnitudes to partially sort
    :param k: partition point
    :param noise: probability
    :return:
    """
    mags *= e_greedy_normal_noise(mags, noise)
    k_smallest_elements, _ = torch.topk(mags, k, largest=False, sorted=True)
    return k_smallest_elements, mags


def rank_channels(param, group_len, magnitude_fn, fraction_to_partition, rounding_fn, noise):
    assert param.dim() in (2, 4), "This ranking is only supported for 2D and 4D tensors"
    n_channels = param.size(1)
    n_ch_to_prune = num_structs_to_prune(n_channels, group_len, fraction_to_partition, rounding_fn)
    if n_ch_to_prune == 0:
        return None, None
    mags = channels_norm(param, magnitude_fn, group_len, length_normalized=True)
    return k_smallest_elems(mags, n_ch_to_prune, noise)


def rank_filters(param, group_len, magnitude_fn, fraction_to_partition, rounding_fn, noise):
    assert param.dim() == 4, "This ranking is only supported for 4D tensors"
    n_filters = param.size(0)
    n_filters_to_prune = num_structs_to_prune(n_filters, group_len, fraction_to_partition, rounding_fn)
    if n_filters_to_prune == 0:
        return None, None
    mags = filters_norm(param, magnitude_fn, group_len, length_normalized=True)
    return k_smallest_elems(mags, n_filters_to_prune, noise)


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

"""Tensor thresholding.

The code below supports fine-grained tensor thresholding and group-wise thresholding.
"""
import torch
import numpy as np
from distiller.norms import *


__all__ = ["GroupThresholdMixin",
           "group_threshold_binary_map", "group_threshold_mask"]


class GroupThresholdMixin(object):
    """A mixin class to add group thresholding capabilities

    TODO: this does not need to be a mixin - it should be made a simple function.  We keep this until we refactor
    """
    def group_threshold_mask(self, param, group_type, threshold, threshold_criteria):
        ret = group_threshold_mask(param, group_type, threshold, threshold_criteria)
        if isinstance(ret, tuple):
            return ret[0]
        return ret


def group_threshold_binary_map(param, group_type, threshold, threshold_criteria):
    """Return a threshold binary map for the provided parameter and group type.

    This function thresholds a parameter tensor, using the provided threshold.
    Thresholding is performed by breaking the parameter tensor into groups as
    specified by group_type, computing the norm of each group instance using
    threshold_criteria, and then thresholding that norm.  The result is called
    binary_map and contains 1s where the group norm was larger than the threshold
    value, zero otherwise.

    Args:
        param: The parameter to mask
        group_type: The elements grouping type (structure).
          One of:2D, 3D, Channels, Row, Cols
        threshold: The threshold
        threshold_criteria: The thresholding criteria.
          ('Mean_Abs', 'Mean_L1', 'L1') - thresholds the entire element group using the mean of the
          absolute values of the tensor elements.
          ('Mean_L2', 'L2') -  - thresholds the entire element group using the L2 norm
          'Max' - thresholds the entire group using the magnitude of the largest
          element in the group.

    Returns:
        binary_map
    """
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.item()
    length_normalized = 'Mean' in threshold_criteria
    if threshold_criteria in ('Mean_Abs', 'Mean_L1', 'L1'):
        norm_fn = l1_norm
    elif threshold_criteria in ('Mean_L2', 'L2'):
        norm_fn = l2_norm
    elif threshold_criteria == 'Max':
        norm_fn = max_norm
    else:
        raise ValueError("Illegal threshold_criteria %s", threshold_criteria)

    if group_type == '2D':
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        thresholds = param.new_full((param.size(0) * param.size(1),), threshold)
        norms = kernels_norm(param, norm_fn, length_normalized=length_normalized)

    elif group_type == 'Rows':
        assert param.dim() == 2, "This regularization is only supported for 2D weights"
        thresholds = param.new_full((param.size(0),), threshold)
        norms = sub_matrix_norm(param, norm_fn, group_len=1, length_normalized=length_normalized, dim=1)

    elif group_type == 'Cols':
        assert param.dim() == 2, "This regularization is only supported for 2D weights"
        thresholds = param.new_full((param.size(1),), threshold)
        norms = sub_matrix_norm(param, norm_fn, group_len=1, length_normalized=length_normalized, dim=0)

    elif group_type == '3D' or group_type == 'Filters':
        assert param.dim() == 4 or param.dim() == 3, "This pruning is only supported for 3D and 4D weights"
        n_filters = param.size(0)
        thresholds = param.new_full((n_filters,), threshold)
        norms = filters_norm(param, norm_fn, length_normalized=length_normalized)

    elif group_type == 'Channels':
        assert param.dim() == 4, "This thresholding is only supported for 4D weights"
        n_channels = param.size(1)
        thresholds = param.new_full((n_channels,),  threshold)
        norms = channels_norm(param, norm_fn, length_normalized=length_normalized)

    binary_map = norms.gt(thresholds).type(param.type())
    return binary_map


def group_threshold_mask(param, group_type, threshold, threshold_criteria, binary_map=None):
    """Return a threshold mask for the provided parameter and group type.

    Args:
        param: The parameter to mask
        group_type: The elements grouping type (structure).
          One of:2D, 3D, Channels, Row, Cols
        threshold: The threshold
        threshold_criteria: The thresholding criteria.
          'Mean_Abs' thresholds the entire element group using the mean of the
          absolute values of the tensor elements.
          'Max' thresholds the entire group using the magnitude of the largest
          element in the group.
        binary_map:

    Returns:
        (mask, binary_map)
    """
    assert group_type in ('2D', 'Rows', 'Cols', '3D', 'Filters', 'Channels')
    if binary_map is None:
        binary_map = group_threshold_binary_map(param, group_type, threshold, threshold_criteria)

    # Now let's expand back up to a 4D mask
    return expand_binary_map(param, group_type, binary_map)


def expand_binary_map(param, group_type, binary_map):
    """Expands a binary_map to the shape of the provided parameter.

    Args:
        param: The parameter to mask
        group_type: The elements grouping type (structure).
          One of:2D, 3D, 4D, Channels, Row, Cols
        binary_map: the binary map that matches the specified `group_type`

    Returns:
        (mask, binary_map)
    """
    assert group_type in ('2D', 'Rows', 'Cols', '3D', 'Filters', '4D', 'Channels')
    assert binary_map is not None

    # Now let's expand back up to a 4D mask
    if group_type == 'Channels' and param.dim() == 2:
        group_type = 'Cols'

    if group_type == '2D':
        a = binary_map.expand(param.size(2) * param.size(3),
                              param.size(0) * param.size(1)).t()
        return a.view(*param.shape), binary_map
    elif group_type == 'Rows':
        return binary_map.expand(param.size(1), param.size(0)).t(), binary_map
    elif group_type == 'Cols':
        return binary_map.expand(*param.shape), binary_map
    elif group_type == '3D' or group_type == 'Filters':
        a = binary_map.expand(np.prod(param.shape[1:]), param.size(0)).t()
        return a.view(*param.shape), binary_map
    elif group_type == 'Channels':
        num_filters, num_channels = param.size(0), param.size(1)
        a = binary_map.expand(num_filters, num_channels)
        c = a.unsqueeze(-1)
        d = c.expand(num_filters, num_channels, param.size(2) * param.size(3)).contiguous()
        return d.view(*param.shape), binary_map


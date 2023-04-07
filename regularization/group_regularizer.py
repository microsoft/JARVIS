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

"""Group Lasso regularization.

Group sparsity regularization (also referred to as structured sparsity regularization) penalizes
entire element groups (structures) in order to create blocks of zero elements, so that CPUs
and other inference acceleration hardware can perform computations more efficiently.
This is in contrast to element-wise regularization which "pushes" individual elements towards zero.

Group norms are calculated as Group Lasso terms (see [1]]).  This is fine for non-overlapping
groups (e.g. 'Channels'), but not so good for overlapping groups (e.g. 'Rows+Cols').

You can set the weight-decay of your Pytorch optimizer to a non-zero value to achieve Sparse
Group Lasso.  As discussed in [3], "this penalty yields solutions that are sparse at both the
group and individual feature levels"


[1] Yuan, M.; Lin, Y. (2006). "Model selection and estimation in regression with grouped variables".
    J. R. Stat. Soc. B. 68 (1): 49–67. doi:10.1111/j.1467-9868.2005.00532.x

[2] Jenatton, Rodolphe; Audibert, Jean-Yves; Bach, Francis (2009). "Structured Variable Selection with
    Sparsity-Inducing Norms". Journal of Machine Learning Research. 12 (2011): 2777–2824. arXiv:0904.3523

[3] J. Friedman, T. Hastie, and R. Tibshirani, “A note on the group lasso and a sparse group lasso,”
    arXiv preprint arXiv:1001.0736, 2010
"""

from .regularizer import _Regularizer, EPSILON
import distiller


class GroupLassoRegularizer(distiller.GroupThresholdMixin, _Regularizer):
    def __init__(self, name, model, reg_regims, threshold_criteria=None):
        """
        Arguments:
            reg_regims: regularization regiment.  A dictionary of
                        reg_regims[<param-name>] = [ lambda, structure-type]
        """
        super(GroupLassoRegularizer, self).__init__(name, model, reg_regims, threshold_criteria)
        assert threshold_criteria in [None, "Max", "Mean_Abs"]

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        if param_name in self.reg_regims.keys():
            group_type = self.reg_regims[param_name][1]
            strength = self.reg_regims[param_name][0]
            if group_type == '2D':
                regularizer_loss += GroupLassoRegularizer.__2d_kernelwise_reg(param, strength)
            elif group_type == 'Rows':
                regularizer_loss += GroupLassoRegularizer.__2d_rowwise_reg(param, strength)
            elif group_type == 'Cols':
                regularizer_loss += GroupLassoRegularizer.__2d_colwise_reg(param, strength)
            elif group_type in ['Rows+Cols', 'Cols+Rows']:
                regularizer_loss += GroupLassoRegularizer.__2d_rowwise_reg(param, strength)
                regularizer_loss += GroupLassoRegularizer.__2d_colwise_reg(param, strength)
            elif group_type == 'Channels':
                # This is also known as "input channels"
                regularizer_loss += GroupLassoRegularizer._3d_channelwise_reg(param, strength)
            elif group_type == 'Filters' or group_type == '3D':
                # This is also known as "output channels"
                regularizer_loss += GroupLassoRegularizer.__3d_filterwise_reg(param, strength)
            elif group_type == '4D':
                regularizer_loss += GroupLassoRegularizer.__4d_layerwise_reg(param, strength)
            elif group_type in ['Channels+4D', '4D+Channels']:
                regularizer_loss += GroupLassoRegularizer.__3d_channelwise_reg(param, strength)
                regularizer_loss += GroupLassoRegularizer.__4d_layerwise_reg(param, strength)
            else:
                raise ValueError('Unknown parameter grouping: ' + group_type)

        return regularizer_loss

    def threshold(self, param, param_name, zeros_mask_dict):
        if self.threshold_criteria is None or param_name not in self.reg_regims:
            return

        group_type = self.reg_regims[param_name][1]
        strength = self.reg_regims[param_name][0]
        zeros_mask_dict[param_name].mask = self.group_threshold_mask(param, group_type, strength, self.threshold_criteria)
        zeros_mask_dict[param_name].is_regularization_mask = True

    @staticmethod
    def __grouplasso_reg(groups, strength, dim):
        if dim == -1:
            # We only have single group
            return groups.norm(2) * strength
        return groups.norm(2, dim=dim).sum().mul_(strength)

    @staticmethod
    def __4d_layerwise_reg(layer_weights, strength, dim=0):
        """Group Lasso with group = 4D weights layer
        """
        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"
        return GroupLassoRegularizer.__grouplasso_reg(layer_weights, strength, dim=-1)

    @staticmethod
    def __3d_filterwise_reg(layer_weights, strength):
        """Group Lasso with group = 3D weights filter
        """
        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"

        # create a filter structure
        filters_view = layer_weights.view(layer_weights.size(0), -1)
        return GroupLassoRegularizer.__grouplasso_reg(filters_view, strength, dim=1)

    @staticmethod
    def _3d_channelwise_reg(layer_weights, strength):
        """Group Lasso with group = 3D input channel
        """
        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"

        # Sum of all channel L2s * regulization_strength
        layer_channels_l2 = GroupLassoRegularizer._channels_l2(layer_weights).sum().mul_(strength)
        return layer_channels_l2

    @staticmethod
    def _channels_l2(layer_weights):
        """Compute the L2-norm of convolution input channels weights.

        A weights input channel is composed of all the kernels that are applied to the
        same activation input channel.  Each kernel belongs to a different weights filter. 
        """
        # Now, for each group, we want to select a specific channel from all of the filters
        num_filters = layer_weights.size(0)
        num_kernels_per_filter = layer_weights.size(1)

        # First, reshape the weights tensor such that each channel (kernel) in the original
        # tensor, is now a row in the 2D tensor.
        view_2d = layer_weights.view(-1, layer_weights.size(2) * layer_weights.size(3))
        # Next, compute the sum of the squares (of the elements in each row/kernel)
        k_sq_sums = view_2d.pow(2).sum(dim=1)
        # Now we have a long vector in which the first num_kernels_per_filter elements
        # hold the sum-of-squares of channels 1..num_kernels_per_filter of the 1st filter,
        # the second num_kernels_per_filter hold the sum-of-squares of channels
        # 1..num_kernels_per_filter of the 2nd filter, and so on.
        # Let's reshape this vector into a matrix, in which each row holds
        # the sum-of-squares of the channels of some filter
        k_sq_sums_mat = k_sq_sums.view(num_filters, num_kernels_per_filter).t()

        # Now it's easy, just do Group Lasso on groups=rows
        channels_l2 = k_sq_sums_mat.sum(dim=1).add(EPSILON).pow(1/2.)
        return channels_l2

    @staticmethod
    def __2d_rowwise_reg(layer_weights, strength):
        assert layer_weights.dim() == 2, "This regularization is only supported for 2D weights"
        return GroupLassoRegularizer.__grouplasso_reg(layer_weights, strength, dim=1)

    @staticmethod
    def __2d_colwise_reg(layer_weights, strength):
        assert layer_weights.dim() == 2, "This regularization is only supported for 2D weights"
        return GroupLassoRegularizer.__grouplasso_reg(layer_weights, strength, dim=0)

    @staticmethod
    def __2d_kernelwise_reg(layer_weights, strength):
        """Group Lasso with one of:
            - group = 2D weights kernel (convolution)
            - group = 2D columns (fully connected)
            - group = 2D rows (fully connected)
        """
        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"
        view_2d = layer_weights.view(-1, layer_weights.size(2) * layer_weights.size(3))
        return GroupLassoRegularizer.__grouplasso_reg(view_2d, strength, dim=1)


class GroupVarianceRegularizer(GroupLassoRegularizer):
    """Group variance regularization.

    As described in [4].

    [4] Amirsina Torfi, Rouzbeh A. Shirvani, Sobhan Soleymani, Nasser M. Nasrabadi,
        “Attention-Based Guided Structured Sparsity of Deep Neural Networks,”
        arXiv preprint arXiv:1802.09902, ICLR 2018
    """
    def __init__(self, name, model, reg_regims):
        super(GroupVarianceRegularizer, self).__init__(name, model, reg_regims)

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        if param_name in self.reg_regims.keys():
            group_type = self.reg_regims[param_name][1]
            strength = self.reg_regims[param_name][0]

            if group_type == 'Channels':
                channels_l2 = GroupLassoRegularizer._channels_l2(param)
                var = channels_l2.var()
                var_loss = 1 / var
                regularizer_loss += strength * var_loss
            else:
                raise ValueError('Unknown parameter grouping: ' + group_type)
        return regularizer_loss

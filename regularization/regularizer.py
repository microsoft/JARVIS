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
EPSILON = 1e-8


class _Regularizer(object):
    def __init__(self, name, model, reg_regims, threshold_criteria):
        """Regularization base class.

        Args:
            name (str): the name of the regularizer.
            model (nn.Module): the model on which to apply regularization.
            reg_regims (dict[str, float or tuple[float, Any]]): regularization regiment.  A dictionary of
                        reg_regims[<param-name>] = [ lambda[, additional_configuration]]
            threshold_criteria (str): the criterion for which to calculate the threshold.
        """
        self.name = name
        self.model = model
        self.reg_regims = reg_regims
        self.threshold_criteria = threshold_criteria

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        """
        Applies the regularization loss onto regularization loss.
        Args:
            param (nn.Parameter): the parameter on which to calculate the regularization
            param_name (str): the name of the parameter relative to top level module.
            regularizer_loss (torch.Tensor): the previous regularization loss calculated,
            zeros_mask_dict (dict): the masks configuration.
        Returns:
            torch.Tensor: regularization_loss after applying the additional loss from current parameter.
        """
        raise NotImplementedError

    def threshold(self, param, param_name, zeros_mask_dict):
        """
        Calculates the threshold for pruning.
        Args:
            param (nn.Parameter): the parameter on which to calculate the regularization
            param_name (str): the name of the parameter relative to top level module.
            zeros_mask_dict (dict): the masks configuration.
        """
        raise NotImplementedError

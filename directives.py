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

"""Scheduling directives

Scheduling directives are instructions (directives) that the scheduler can
execute as part of scheduling pruning activities.
"""
from __future__ import division
import torch
import numpy as np
from collections import defaultdict
import logging
msglogger = logging.getLogger()

from torchnet.meter import AverageValueMeter
from distiller.utils import sparsity, density


class FreezeTraining(object):
    def __init__(self, name):
        print("------FreezeTraining--------")
        self.name = name

def freeze_training(model, which_params, freeze):
    """This function will freeze/defrost training for certain layers.

    Sometimes, when we prune and retrain a certain layer type,
    we'd like to freeze the training of the other layers.
    """
    for param in model.parameters():
        pname = model_find_param_name(model, param.data)
        if pname is None:
            continue
        for ptype in which_params:
            if ptype in pname:
                # see: http://pytorch.org/docs/master/notes/autograd.html?highlight=grad_fn
                param.requires_grad = not freeze
                if freeze:
                    msglogger.info('Freezing: ' + pname)
                else:
                    msglogger.info('Defrosting: ' + pname)


def freeze_all(model, freeze):
    msglogger.info('{} all parameters'.format('Freezing' if freeze else 'Defrosting'))
    for param in model.parameters():
        param.requires_grad = not freeze


def adjust_dropout(module, new_probabilty):
    """Replace the dropout probability of dropout layers

    As explained in the paper "Learning both Weights and Connections for
    Efficient Neural Networks":
        Dropout is widely used to prevent over-fitting, and this also applies to retraining.
        During retraining, however, the dropout ratio must be adjusted to account for the
        change in model capacity. In dropout, each parameter is probabilistically dropped
        during training, but will come back during inference. In pruning, parameters are
        dropped forever after pruning and have no chance to come back during both training
        and inference. As the parameters get sparse, the classifier will select the most
        informative predictors and thus have much less prediction variance, which reduces
        over-fitting. As pruning already reduced model capacity, the retraining dropout ratio
        should be smaller.
    """
    if type(module) in [torch.nn.Dropout,
                        torch.nn.Dropout2d,
                        torch.nn.Dropout3d,
                        torch.nn.AlphaDropout]:
        msglogger.info("Adjusting dropout probability")# for {}".format(str(module)))
        module.p = new_probabilty
    else:
        for child in module.children():
            adjust_dropout(child, new_probabilty)

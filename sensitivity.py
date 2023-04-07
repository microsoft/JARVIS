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

"""Perform sensitivity tests on layers and whole networks.

Construct a schedule for experimenting with network and layer sensitivity
to pruning.

The idea is to set the pruning level (percentage) of specific layers (or the
entire network), and then to prune once, run an evaluation on the test dataset,
and exit.  This should teach us about the "sensitivity" of the network/layers
to pruning.

This concept is discussed in "Learning both Weights and Connections for
Efficient Neural Networks" - https://arxiv.org/pdf/1506.02626v3.pdf
"""

from copy import deepcopy
from collections import OrderedDict
import logging
import csv
import distiller
from .scheduler import CompressionScheduler

msglogger = logging.getLogger()


def perform_sensitivity_analysis(model, net_params, sparsities, test_func, group):
    """Perform a sensitivity test for a model's weights parameters.

    The model should be trained to maximum accuracy, because we aim to understand
    the behavior of the model's performance in relation to pruning of a specific
    weights tensor.

    By default this function will test all of the model's parameters.

    The return value is a complex sensitivities dictionary: the dictionary's
    key is the name (string) of the weights tensor.  The value is another dictionary,
    where the tested sparsity-level is the key, and a (top1, top5, loss) tuple
    is the value.
    Below is an example of such a dictionary:

    .. code-block:: python
    {'features.module.6.weight':    {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.492, 79.1,   1.9161),
                                     0.10: (56.212, 78.854, 1.9315),
                                     0.15: (35.424, 60.3,   3.0866)},
     'classifier.module.1.weight':  {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.514, 79.07,  1.9159),
                                     0.10: (56.434, 79.074, 1.9138),
                                     0.15: (54.454, 77.854, 2.3127)} }

    The test_func is expected to execute the model on a test/validation dataset,
    and return the results for top1 and top5 accuracies, and the loss value.
    """
    if group not in ['element', 'filter', 'channel']:
        raise ValueError("group parameter contains an illegal value: {}".format(group))
    sensitivities = OrderedDict()

    for param_name in net_params:
        if model.state_dict()[param_name].dim() not in [2,4]:
            continue

        # Make a copy of the model, because when we apply the zeros mask (i.e.
        # perform pruning), the model's weights are altered
        model_cpy = deepcopy(model)

        sensitivity = OrderedDict()
        for sparsity_level in sparsities:
            sparsity_level = float(sparsity_level)
            msglogger.info("Testing sensitivity of %s [%0.1f%% sparsity]" % (param_name, sparsity_level*100))
            # Create the pruner (a level pruner), the pruning policy and the
            # pruning schedule.
            if group == 'element':
                # Element-wise sparasity
                sparsity_levels = {param_name: sparsity_level}
                pruner = distiller.pruning.SparsityLevelParameterPruner(name="sensitivity", levels=sparsity_levels)
            elif group == 'filter':
                # Filter ranking
                if model.state_dict()[param_name].dim() != 4:
                    continue
                pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                            group_type="Filters",
                                                                            desired_sparsity=sparsity_level,
                                                                            weights=param_name)
            elif group == 'channel':
                # Filter ranking
                if model.state_dict()[param_name].dim() != 4:
                    continue
                pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                            group_type="Channels",
                                                                            desired_sparsity=sparsity_level,
                                                                            weights=param_name)

            policy = distiller.PruningPolicy(pruner, pruner_args=None)
            scheduler = CompressionScheduler(model_cpy)
            scheduler.add_policy(policy, epochs=[0])

            # Compute the pruning mask per the pruner and apply the mask on the weights
            scheduler.on_epoch_begin(0)
            scheduler.mask_all_weights()

            # Test and record the performance of the pruned model
            prec1, prec5, loss = test_func(model=model_cpy)
            sensitivity[sparsity_level] = (prec1, prec5, loss)
            sensitivities[param_name] = sensitivity
    return sensitivities


def sensitivities_to_png(sensitivities, fname):
    """Create a mulitplot of the sensitivities.

    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    try:
        # sudo apt-get install python3-tk
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Function plot_sensitivity requires package matplotlib which"
              "is not installed in your execution environment.\n"
              "Skipping the PNG file generation")
        return

    msglogger.info("Generating sensitivity graph")

    for param_name, sensitivity in sensitivities.items():
        sense = [values[1] for sparsity, values in sensitivity.items()]
        sparsities = [sparsity for sparsity, values in sensitivity.items()]
        plt.plot(sparsities, sense, label=param_name)

    plt.ylabel('top5')
    plt.xlabel('sparsity')
    plt.title('Pruning Sensitivity')
    plt.legend(loc='lower center',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(fname, format='png')


def sensitivities_to_csv(sensitivities, fname):
    """Create a CSV file listing from the sensitivities dictionary.

    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write the header
        writer.writerow(['parameter', 'sparsity', 'top1', 'top5', 'loss'])
        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values))

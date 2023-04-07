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

from collections import namedtuple, OrderedDict, defaultdict
import re
import copy
import logging
import torch
import torch.nn as nn
import distiller
import warnings
from typing import Callable, Optional
from copy import deepcopy

msglogger = logging.getLogger()

QBits = namedtuple('QBits', ['acts', 'wts', 'bias'])

FP_BKP_PREFIX = 'float_'


def has_bias(module):
    return hasattr(module, 'bias') and module.bias is not None


def hack_float_backup_parameter(module, name, num_bits):
    try:
        param = dict(module.named_parameters())[name]
        param_id = id(param)
    except KeyError:
        raise ValueError('Module has no Parameter named ' + name)
    module.register_parameter(FP_BKP_PREFIX + name, param)
    assert id(getattr(module, FP_BKP_PREFIX + name)) == param_id
    delattr(module, name)
    module.register_buffer(name, torch.zeros_like(param))

    first = False
    if not hasattr(module, 'repr_mod'):
        setattr(module, 'repr_mod', ', \nDistiller_QuantAwareTrain: ')
        first = True
        module.original_extra_repr = module.extra_repr
        module.extra_repr = lambda: module.original_extra_repr() + module.repr_mod

    if not first:
        module.repr_mod += ' ; '
    module.repr_mod += '{0} --> {1} bits'.format(name, num_bits)


class _ParamToQuant(object):
    def __init__(self, module, module_name, fp_attr_name, q_attr_name, num_bits):
        self.module = module
        self.module_name = module_name
        self.fp_attr_name = fp_attr_name
        self.q_attr_name = q_attr_name
        self.num_bits = num_bits

    def __repr__(self):
        return "ParamToQuant(module_name=%s,num_bits=%s)" % (self.module_name, self.num_bits)


class Quantizer(object):
    r"""
    Base class for quantizers.

    Args:
        model (torch.nn.Module): The model to be quantized
        optimizer (torch.optim.Optimizer): An optimizer instance, required in cases where the quantizer is going
            to perform changes to existing model parameters and/or add new ones.
            Specifically, when train_with_fp_copy is True, this cannot be None.
        bits_activations/weights/bias (int): Default number of bits to use when quantizing each tensor type.
            Value of None means do not quantize.
        overrides (OrderedDict): Dictionary mapping regular expressions of layer name patterns to dictionary with
            overrides of default values.
            The keys in the overrides dictionary should be parameter names that the Quantizer accepts default values
            for in its init function.
            The parameters 'bits_activations', 'bits_weights', and 'bits_bias' which are accepted by the base Quantizer
            are supported by default.
            Other than those, each sub-class of Quantizer defines the set of parameter for which it supports
            over-riding.
            OrderedDict is used to enable handling of overlapping name patterns. So, for example, one could define
            certain override parameters for a group of layers, e.g. 'conv*', but also define different parameters for
            specific layers in that group, e.g. 'conv1'.
            The patterns are evaluated eagerly - the first match wins. Therefore, the more specific patterns must
            come before the broad patterns.
        train_with_fp_copy (bool): If true, will modify layers with weights to keep both a quantized and
            floating-point copy, such that the following flow occurs in each training iteration:
            1. q_weights = quantize(fp_weights)
            2. Forward through network using q_weights
            3. In back-prop:
                3.1 Gradients calculated with respect to q_weights
                3.2 We also back-prop through the 'quantize' operation from step 1
            4. Update fp_weights with gradients calculated in step 3.2
        Note:
            The `overrides` dictionary assumes the keys are *not* the module names in the
            `nn.DataParallel` case - i.e. without the `module.` prefix. e.g.:
            module.conv1 -> OrderedDict([('conv1', OrderedDict(...))])
    """
    def __init__(self, model, optimizer=None,
                 bits_activations=None, bits_weights=None, bits_bias=None,
                 overrides=None, train_with_fp_copy=False):
        if overrides is None:
            overrides = OrderedDict()
        if not isinstance(overrides, OrderedDict):
            raise TypeError('overrides must be an instance of collections.OrderedDict or None')

        if train_with_fp_copy and optimizer is None:
            raise ValueError('optimizer cannot be None when train_with_fp_copy is True')

        self.adjacency_map = None  # To be populated during prepare_model()

        self.default_qbits = QBits(acts=bits_activations, wts=bits_weights, bias=bits_bias)
        self.overrides = overrides

        self.model = model
        self.optimizer = optimizer

        # Stash some quantizer data in the model so we can re-apply the quantizer on a resuming model
        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_weights': bits_weights,
                                                    'bits_bias': bits_bias,
                                                    'overrides': copy.deepcopy(overrides)}}

        for k, v in self.overrides.items():
            if any(old_bits_key in v.keys() for old_bits_key in ['acts', 'wts', 'bias']):
                raise ValueError("Using 'acts' / 'wts' / 'bias' to specify bit-width overrides is deprecated.\n"
                                 "Please use the full parameter names: "
                                 "'bits_activations' / 'bits_weights' / 'bits_bias'")
            qbits = QBits(acts=v.pop('bits_activations', self.default_qbits.acts),
                          wts=v.pop('bits_weights', self.default_qbits.wts),
                          bias=v.pop('bits_bias', self.default_qbits.bias))
            v['bits'] = qbits

        # Prepare explicit mapping from each layer to QBits based on default + overrides
        patterns = []
        regex_overrides = None
        if overrides:
            patterns = list(overrides.keys())
            regex_overrides_str = '|'.join(['(^{0}$)'.format(pattern) for pattern in patterns])
            regex_overrides = re.compile(regex_overrides_str)

        self.module_qbits_map = {}  # type: OrderedDict[str, QBits]
        self.module_overrides_map = {}
        for module_full_name, module in model.named_modules():
            # Need to account for scenario where model is parallelized with DataParallel, which wraps the original
            # module with a wrapper module called 'module' :)
            name_to_match = module_full_name.replace('module.', '', 1)
            qbits = self.default_qbits
            override_entry = self.overrides.get(name_to_match, OrderedDict())
            if regex_overrides:
                m_overrides = regex_overrides.match(name_to_match)
                if m_overrides:
                    group_idx = 0
                    groups = m_overrides.groups()
                    while groups[group_idx] is None:
                        group_idx += 1
                    override_entry = copy.deepcopy(override_entry or self.overrides[patterns[group_idx]])
                    qbits = override_entry.pop('bits', self.default_qbits)

            self._add_qbits_entry(module_full_name, type(module), qbits)
            self._add_override_entry(module_full_name, override_entry)

        # Mapping from module type to function generating a replacement module suited for quantization
        # To be populated by child classes
        # Unspecified layer types return None by default.
        self.replacement_factory = OrderedDict([(nn.Identity, None)])
        self.default_repalcement_fn = None
        self.replacement_blacklist = []
        # Pointer to parameters quantization function, triggered during training process
        # To be populated by child classes
        self.param_quantization_fn = None

        self.train_with_fp_copy = train_with_fp_copy
        self.params_to_quantize = []

        # A dictionary of replaced modules and their respective names.
        self.modules_processed = OrderedDict()
        self.modules_processed_args = OrderedDict()

        self.prepared = False

    def _add_qbits_entry(self, module_name, module_type, qbits):
        if module_type not in [nn.Conv2d, nn.Conv3d, nn.Linear, nn.Embedding]:
            # For now we support weights quantization only for Conv, FC and Embedding layers (so, for example, we don't
            # support quantization of batch norm scale parameters)
            qbits = QBits(acts=qbits.acts, wts=None, bias=None)
        self.module_qbits_map[module_name] = qbits

    def _add_override_entry(self, module_name, entry):
        self.module_overrides_map[module_name] = entry

    def prepare_model(self, dummy_input=None):
        """
        Traverses the model and replaces sub-modules with quantized counterparts according to the bit-width
        and overrides configuration provided to __init__(), and according to the replacement_factory as
        defined by the Quantizer sub-class being used.

        Note:
            If multiple sub-modules within the model actually reference the same module, then that module
            is replaced only once, according to the configuration (bit-width and/or overrides) of the
            first encountered reference.
            Toy Example - say a module is constructed using this bit of code:

                shared_relu = nn.ReLU
                self.relu1 = shared_relu
                self.relu2 = shared_relu

            When traversing the model, a replacement will be generated when 'self.relu1' is encountered.
            Let's call it `new_relu1'. When 'self.relu2' will be encountered, it'll simply be replaced
            with a reference to 'new_relu1'. Any override configuration made specifically for 'self.relu2'
            will be ignored. A warning message will be shown.
        """
        if self.prepared:
            raise RuntimeError('prepare_model can be called only once')

        msglogger.info('Preparing model for quantization using {0}'.format(self.__class__.__name__))

        self.model.quantizer_metadata["dummy_input"] = dummy_input
        if dummy_input is not None:
            summary_graph = distiller.SummaryGraph(self.model, dummy_input)
            self.adjacency_map = summary_graph.adjacency_map(dedicated_modules_only=False)
            del summary_graph

        model_device = distiller.model_device(self.model)

        self._pre_prepare_model(dummy_input)

        self._pre_process_container(self.model)
        for module_name, module in self.model.named_modules():
            qbits = self.module_qbits_map[module_name]
            curr_parameters = dict(module.named_parameters())
            for param_name, param in curr_parameters.items():
                n_bits = qbits.bias if param_name.endswith('bias') else qbits.wts
                if n_bits is None:
                    continue
                fp_attr_name = param_name
                if self.train_with_fp_copy:
                    hack_float_backup_parameter(module, param_name, n_bits)
                    fp_attr_name = FP_BKP_PREFIX + param_name
                self.params_to_quantize.append(_ParamToQuant(module, module_name, fp_attr_name, param_name, n_bits))

                param_full_name = '.'.join([module_name, param_name])
                msglogger.debug(
                    "Parameter '{0}' will be quantized to {1} bits".format(param_full_name, n_bits))

        # If an optimizer was passed, assume we need to update it
        if self.optimizer:
            for pg in self._get_new_optimizer_params_groups():
                self.optimizer.add_param_group(pg)

        self._post_prepare_model()

        # Re-transfer model to the device it was on, in case the quantizer created new parameters/buffers
        self.model.to(model_device)

        distiller.assign_layer_fq_names(self.model)

        self.prepared = True

        msglogger.debug('Quantized model:\n\n{0}\n'.format(self.model))

    def _pre_prepare_model(self, dummy_input):
        pass

    def _pre_process_container(self, container, prefix=''):
        def replace_msg(module_name, modules=None):
            msglogger.debug('Module ' + module_name)
            if modules:
                msglogger.debug('\tReplacing: {}.{}'.format(modules[0].__module__, modules[0].__class__.__name__))
                msglogger.debug('\tWith:      {}.{}'.format(modules[1].__module__, modules[1].__class__.__name__))
            else:
                msglogger.debug('\tSkipping')

        # Iterate through model, insert quantization functions as appropriate
        for name, module in container.named_children():
            full_name = prefix + name
            if isinstance(module, tuple(self.replacement_blacklist)):
                replace_msg(full_name)
                continue
            if module in self.modules_processed:
                previous_name, previous_wrapper = self.modules_processed[module]
                warnings.warn("Module '{0}' references to same module as '{1}'."
                              ' Replacing with reference the same wrapper.'.format(full_name, previous_name),
                              UserWarning)
                if previous_wrapper:
                    replace_msg(full_name, (module, previous_wrapper))
                    setattr(container, name, previous_wrapper)
                else:
                    replace_msg(full_name)
                continue
            current_qbits = self.module_qbits_map[full_name]
            # TODO - Review necessity of the block below
            if current_qbits.acts is None and current_qbits.wts is None and not self.module_overrides_map[full_name]:
                # We indicate this module wasn't replaced by a wrapper
                replace_msg(full_name)
                self.modules_processed[module] = full_name, None
            else:
                # We use a type hint comment to let IDEs know replace_fn is a function
                replace_fn = self.replacement_factory.get(type(module),
                                                          self.default_repalcement_fn)  # type: Optional[Callable]
                # If the replacement function wasn't specified - continue without replacing this module.
                if replace_fn is not None:
                    valid_kwargs, invalid_kwargs = distiller.filter_kwargs(self.module_overrides_map[full_name],
                                                                           replace_fn)
                    if invalid_kwargs:
                        raise TypeError("""Quantizer of type %s doesn't accept \"%s\" 
                                            as override arguments for %s. Allowed kwargs: %s"""
                                        % (type(self), list(invalid_kwargs), type(module), list(valid_kwargs)))
                    new_module = replace_fn(module, full_name, self.module_qbits_map, **valid_kwargs)
                    if new_module != module:
                        replace_msg(full_name, (module, new_module))
                        # Add to history of prepared submodules
                        self.modules_processed[module] = full_name, new_module
                        # To allow recreating this wrapper later on
                        valid_args = full_name, deepcopy(self.module_qbits_map)
                        self.modules_processed_args[full_name] = valid_args, valid_kwargs
                        setattr(container, name, new_module)

                        # If a "leaf" module was replaced by a container, add the new layers to the QBits mapping
                        if not distiller.has_children(module) and distiller.has_children(new_module):
                            for sub_module_name, sub_module in new_module.named_modules():
                                self._add_qbits_entry(full_name + '.' + sub_module_name, type(sub_module),
                                                      current_qbits)
                            self.module_qbits_map[full_name] = QBits(acts=current_qbits.acts, wts=None, bias=None)
                    else:
                        replace_msg(full_name)
                        self.modules_processed[module] = full_name, None

            if distiller.has_children(module):
                # For container we call recursively
                self._pre_process_container(module, full_name + '.')

    def _get_new_optimizer_params_groups(self):
        """
        If the quantizer adds new trainable parameters to the model, this function should return a list of one
        or more parameter groups pertaining. Each parameter group is expected to be a dict in the format
        expected by torch.optim.Optimizer.
        For details, See https://pytorch.org/docs/stable/optim.html#per-parameter-options

        Subclasses which add parameters to the model should override as needed.

        :return: List of parameter groups
        """
        return list()

    def _post_prepare_model(self):
        pass

    def quantize_params(self):
        """
        Quantize all parameters using self.param_quantization_fn (with the defined number of bits for each parameter)
        """
        for ptq in self.params_to_quantize:
            q_param = self.param_quantization_fn(getattr(ptq.module, ptq.fp_attr_name), ptq)
            if self.train_with_fp_copy:
                setattr(ptq.module, ptq.q_attr_name, q_param)
            else:
                getattr(ptq.module, ptq.q_attr_name).data = q_param.data

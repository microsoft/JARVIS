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

"""A collection of useful utility functions.

This module contains various tensor sparsity/density measurement functions, together
with some random helper functions.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
import logging
import operator
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import inspect
import distiller

msglogger = logging.getLogger()


def model_device(model):
    """Determine the device the model is allocated on."""
    # Source: https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    if isinstance(model, nn.DataParallel):
        return model.src_device_obj
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        # Model has no parameters
        pass
    return 'cpu'


def optimizer_device_name(opt):
    return str(list(list(opt.state)[0])[0].device)


def to_np(var):
    return var.data.cpu().numpy()


def size2str(torch_size):
    if isinstance(torch_size, torch.Size):
        return size_to_str(torch_size)
    if isinstance(torch_size, (torch.FloatTensor, torch.cuda.FloatTensor)):
        return size_to_str(torch_size.size())
    if isinstance(torch_size, torch.autograd.Variable):
        return size_to_str(torch_size.data.size())
    if isinstance(torch_size, tuple) or isinstance(torch_size, list):
        return size_to_str(torch_size)
    raise TypeError


def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return '('+(', ').join(['%d' % v for v in torch_size])+')'


def pretty_int(i):
    return "{:,}".format(i)


class MutableNamedTuple(dict):
    def __init__(self, init_dict):
        for k, v in init_dict.items():
            self[k] = v

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def assign_layer_fq_names(container, name=None):
    """Assign human-readable names to the modules (layers).

    Sometimes we need to access modules by their names, and we'd like to use
    fully-qualified names for convenience.
    """
    for name, module in container.named_modules():
        module.distiller_name = name


def find_module_by_fq_name(model, fq_mod_name):
    """Given a module's fully-qualified name, find the module in the provided model.

    A fully-qualified name is assigned to modules in function assign_layer_fq_names.

    Arguments:
        model: the model to search
        fq_mod_name: the module whose name we want to look up

    Returns:
        The module or None, if the module was not found.
    """
    for module in model.modules():
        if hasattr(module, 'distiller_name') and fq_mod_name == module.distiller_name:
            return module
    return None


def normalize_module_name(layer_name):
    """Normalize a module's name.

    PyTorch let's you parallelize the computation of a model, by wrapping a model with a
    DataParallel module.  Unfortunately, this changs the fully-qualified name of a module,
    even though the actual functionality of the module doesn't change.
    Many time, when we search for modules by name, we are indifferent to the DataParallel
    module and want to use the same module name whether the module is parallel or not.
    We call this module name normalization, and this is implemented here.
    """
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)


def denormalize_module_name(parallel_model, normalized_name):
    """Convert back from the normalized form of the layer name, to PyTorch's name
    which contains "artifacts" if DataParallel is used.
    """
    fully_qualified_name = [mod_name for mod_name, _ in parallel_model.named_modules() if
                            normalize_module_name(mod_name) == normalized_name]
    if len(fully_qualified_name) > 0:
        return fully_qualified_name[-1]
    else:
        return normalized_name   # Did not find a module with the name <normalized_name>


def volume(tensor):
    """return the volume of a pytorch tensor"""
    if isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.cuda.FloatTensor):
        return np.prod(tensor.shape)
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return np.prod(tensor)
    raise ValueError


def density(tensor):
    """Computes the density of a tensor.

    Density is the fraction of non-zero elements in a tensor.
    If a tensor has a density of 1.0, then it has no zero elements.

    Args:
        tensor: the tensor for which we compute the density.

    Returns:
        density (float)
    """
    # Using torch.nonzero(tensor) can lead to memory exhaustion on
    # very large tensors, so we count zeros "manually".
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)


def sparsity(tensor):
    """Computes the sparsity of a tensor.

    Sparsity is the fraction of zero elements in a tensor.
    If a tensor has a density of 0.0, then it has all zero elements.
    Sparsity and density are complementary.

    Args:
        tensor: the tensor for which we compute the density.

    Returns:
        sparsity (float)
    """
    return 1.0 - density(tensor)


def sparsity_3D(tensor):
    """Filter-wise sparsity for 4D tensors"""
    if tensor.dim() != 4:
        return 0
    l1_norms = distiller.norms.filters_lp_norm(tensor, p=1, length_normalized=False)
    num_nonzero_filters = len(torch.nonzero(l1_norms))
    num_filters = tensor.size(0)
    return 1 - num_nonzero_filters / num_filters


def density_3D(tensor):
    """Filter-wise density for 4D tensors"""
    return 1 - sparsity_3D(tensor)


def sparsity_2D(tensor):
    """Create a list of sparsity levels for each channel in the tensor 't'

    For 4D weight tensors (convolution weights), we flatten each kernel (channel)
    so it becomes a row in a 3D tensor in which each channel is a filter.
    So if the original 4D weights tensor is:
        #OFMs x #IFMs x K x K
    The flattened tensor is:
        #OFMS x #IFMs x K^2

    For 2D weight tensors (fully-connected weights), the tensors is shaped as
        #IFMs x #OFMs
    so we don't need to flatten anything.

    To measure 2D sparsity, we sum the absolute values of the elements in each row,
    and then count the number of rows having sum(abs(row values)) == 0.
    """
    if tensor.dim() == 4:
        # For 4D weights, 2D structures are channels (filter kernels)
        view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    elif tensor.dim() == 2:
        # For 2D weights, 2D structures are either columns or rows.
        # At the moment, we only support row structures
        view_2d = tensor
    else:
        return 0

    num_structs = view_2d.size()[0]
    nonzero_structs = len(torch.nonzero(view_2d.abs().sum(dim=1)))
    return 1 - nonzero_structs/num_structs


def density_2D(tensor):
    """Kernel-wise sparsity for 4D tensors"""
    return 1 - sparsity_2D(tensor)


def non_zero_channels(tensor):
    """Returns the indices of non-zero channels.

    Non-zero channels are channels that have at least one coefficient that
    is not zero.  Counting non-zero channels involves some tensor acrobatics.
    """
    if tensor.dim() != 4:
        raise ValueError("Expecting a 4D tensor")

    norms = distiller.norms.channels_lp_norm(tensor, p=1)
    nonzero_channels = torch.nonzero(norms)
    return nonzero_channels


def sparsity_ch(tensor):
    """Channel-wise sparsity for 4D tensors"""
    if tensor.dim() != 4:
        return 0
    nonzero_channels = len(non_zero_channels(tensor))
    n_channels = tensor.size(1)
    return 1 - nonzero_channels/n_channels


def density_ch(tensor):
    """Channel-wise density for 4D tensors"""
    return 1 - sparsity_ch(tensor)


def sparsity_blocks(tensor, block_shape):
    """Block-wise sparsity for 4D tensors

    Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1
    """
    if tensor.dim() != 4:
        raise ValueError("sparsity_blocks is only supported for 4-D tensors")

    if len(block_shape) != 4:
        raise ValueError("Block shape must be specified as a 4-element tuple")
    block_repetitions, block_depth, block_height, block_width = block_shape
    if not block_width == block_height == 1:
        raise ValueError("Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1")

    super_block_volume = volume(block_shape)
    num_super_blocks = volume(tensor) / super_block_volume

    num_filters, num_channels = tensor.size(0), tensor.size(1)
    kernel_size = tensor.size(2) * tensor.size(3)

    # Create a view where each block is a column
    if block_depth > 1:
        view_dims = (
            num_filters*num_channels//(block_repetitions*block_depth),
            block_repetitions*block_depth,
            kernel_size,
            )
    else:
        view_dims = (
            num_filters // block_repetitions,
            block_repetitions,
            -1,
            )
    view1 = tensor.view(*view_dims)

    # Next, compute the sums of each column (block)
    block_sums = view1.abs().sum(dim=1)
    nonzero_blocks = len(torch.nonzero(block_sums))
    return 1 - nonzero_blocks/num_super_blocks


def sparsity_matrix(tensor, dim):
    """Generic sparsity computation for 2D matrices"""
    if tensor.dim() != 2:
        return 0

    num_structs = tensor.size()[dim]
    nonzero_structs = len(torch.nonzero(tensor.abs().sum(dim=1-dim)))
    return 1 - nonzero_structs/num_structs


def sparsity_cols(tensor, transposed=True):
    """Column-wise sparsity for 2D tensors

    PyTorch GEMM matrices are transposed before they are used in the GEMM operation.
    In other words the matrices are stored in memory transposed.  So by default we compute
    the sparsity of the transposed dimension.
    """
    if transposed:
        return sparsity_matrix(tensor, 0)
    return sparsity_matrix(tensor, 1)


def density_cols(tensor, transposed=True):
    """Column-wise density for 2D tensors"""
    return 1 - sparsity_cols(tensor, transposed)


def sparsity_rows(tensor, transposed=True):
    """Row-wise sparsity for 2D matrices

    PyTorch GEMM matrices are transposed before they are used in the GEMM operation.
    In other words the matrices are stored in memory transposed.  So by default we compute
    the sparsity of the transposed dimension.
    """
    if transposed:
        return sparsity_matrix(tensor, 1)
    return sparsity_matrix(tensor, 0)


def density_rows(tensor, transposed=True):
    """Row-wise density for 2D tensors"""
    return 1 - sparsity_rows(tensor, transposed)


def model_sparsity(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    """Returns the model sparsity as a fraction in [0..1]"""
    sparsity, _, _ = model_params_stats(model, param_dims, param_types)
    return sparsity


def model_params_size(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    """Returns the size of the model parameters, w/o counting zero coefficients"""
    _, _, sparse_params_cnt = model_params_stats(model, param_dims, param_types)
    return sparse_params_cnt


def model_params_stats(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    """Returns the model sparsity, weights count, and the count of weights in the sparse model.

    Returns:
        model_sparsity - the model weights sparsity (in percent)
        params_cnt - the number of weights in the entire model (incl. zeros)
        params_nnz_cnt - the number of weights in the entire model, excluding zeros.
                         nnz stands for non-zeros.
    """
    params_cnt = 0
    params_nnz_cnt = 0
    for name, param in model.state_dict().items():
        if param.dim() in param_dims and any(type in name for type in param_types):
            _density = density(param)
            params_cnt += torch.numel(param)
            params_nnz_cnt += param.numel() * _density
    model_sparsity = (1 - params_nnz_cnt/params_cnt)*100
    return model_sparsity, params_cnt, params_nnz_cnt


def norm_filters(weights, p=1):
    return distiller.norms.filters_lp_norm(weights, p)


def model_numel(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    """Count the number elements in a model's parameter tensors"""
    total_numel = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in param_types):
            total_numel += torch.numel(param)
    return total_numel


def activation_channels_l1(activation):
    """Calculate the L1-norms of an activation's channels.

    The activation usually has the shape: (batch_size, num_channels, h, w).

    When the activations are computed on a distributed GPU system, different parts of the
    activation tensor might be computed by a differnt GPU. If this function is called from
    the forward-callback of some activation module in the graph, we will only witness part
    of the batch.  For example, if the batch_size is 256, and we are using 4 GPUS, instead
    of seeing activations with shape = (256, num_channels, h, w), we may see 4 calls with
    shape = (64, num_channels, h, w).

    Since we want to calculate the average of the L1-norm of each of the channels of the
    activation, we need to move the partial sums results to the CPU, where they will be
    added together.

    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the L! magnitude of each channel).
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_norms = view_2d.norm(p=1, dim=1)  # (batch*channels) x 1
        featuremap_norms_mat = featuremap_norms.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_norms_mat = activation.norm(p=1, dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_l1: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_norms_mat.mean(dim=0).cpu()


def activation_channels_means(activation):
    """Calculate the mean of each of an activation's channels.

    The activation usually has the shape: (batch_size, num_channels, h, w).

    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."

    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the L1 magnitude of each channel).
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_means = view_2d.mean(dim=1)  # (batch*channels) x 1
        featuremap_means_mat = featuremap_means.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_means_mat = activation.mean(dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_means: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_means_mat.mean(dim=0).cpu()


def activation_channels_apoz(activation):
    """Calculate the APoZ of each of an activation's channels.

    APoZ is the Average Percentage of Zeros (or simply: average sparsity) and is defined in:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures".

    The activation usually has the shape: (batch_size, num_channels, h, w).

    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."

    Returns - for each channel: the batch-mean of its sparsity.
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_apoz = view_2d.abs().gt(0).sum(dim=1).float() / (activation.size(2) * activation.size(3))  # (batch*channels) x 1
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0).sum(dim=1).float() / activation.size(1)  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100).cpu()


def log_training_progress(stats_dict, params_dict, epoch, steps_completed, total_steps, log_freq, loggers):
    """Log information about the training progress, and the distribution of the weight tensors.

    Args:
        stats_dict: A tuple of (group_name, dict(var_to_log)).  Grouping statistics variables is useful for logger
          backends such as TensorBoard.  The dictionary of var_to_log has string key, and float values.
          For example:
              stats = ('Peformance/Validation/',
                       OrderedDict([('Loss', vloss),
                                    ('Top1', top1),
                                    ('Top5', top5)]))
        params_dict: A parameter dictionary, such as the one returned by model.named_parameters()
        epoch: The current epoch
        steps_completed: The current step in the epoch
        total_steps: The total number of training steps taken so far
        log_freq: The number of steps between logging records
        loggers: A list of loggers to send the log info to
    """
    if loggers is None:
        return
    if not isinstance(loggers, list):
        loggers = [loggers]
    for logger in loggers:
        logger.log_training_progress(stats_dict, epoch,
                                     steps_completed,
                                     total_steps, freq=log_freq)
        logger.log_weights_distribution(params_dict, steps_completed)


def log_activation_statistics(epoch, phase, loggers, collector):
    """Log information about the sparsity of the activations"""
    if collector is None:
        return
    if loggers is None:
        return
    for logger in loggers:
        logger.log_activation_statistic(phase, collector.stat_name, collector.value(), epoch)


def log_weights_sparsity(model, epoch, loggers):
    """Log information about the weights sparsity"""
    for logger in loggers:
        logger.log_weights_sparsity(model, epoch)


def log_model_buffers(model, buffer_names, tag_prefix, epoch, steps_completed, total_steps, log_freq, loggers=()):
    """
    Log values of model buffers. 'buffer_names' is a list of buffers to be logged (which not necessarily exist
    in all layers in the model).

    USE WITH CARE:
        * This logger logs each value within the buffers. As such, while any buffer can be passed
          it is not really intended for big buffers such as model weights.
        * Special attention is needed when using this using this functionality in TensorBoardLogger, as it could
          significantly slow down the load time of TensorBard. Please see the documentation of 'log_model_buffers'
          in that class.

    Args:
        model: Model containing buffers to be logged
        buffer_names: Names of buffers to be logged. Expected to be
        tag_prefix: Prefix to be used before buffer name by logger
        epoch: The current epoch
        steps_completed: The current step in the epoch
        total_steps: The total number of training steps taken so far
        log_freq: The number of steps between logging records
        loggers: An iterable of loggers to send the log info to
    """
    for logger in loggers:
        logger.log_model_buffers(model, buffer_names, tag_prefix, epoch, steps_completed, total_steps, log_freq)


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


def _validate_input_shape(dataset, input_shape):
    if dataset:
        try:
            return tuple(distiller.apputils.classification_get_input_shape(dataset))
        except ValueError:
            raise ValueError("Can't infer input shape for dataset {}, please pass shape directly".format(dataset))
    else:
        if input_shape is None:
            raise ValueError('Must provide either dataset name or input shape')
        if not isinstance(input_shape, tuple):
            raise TypeError('Shape should be a tuple of integers, or a tuple of tuples of integers')

        def val_recurse(in_shape):
            if all(isinstance(x, int) for x in in_shape):
                if any(x < 0 for x in in_shape):
                    raise ValueError("Shape can't contain negative dimensions: {}".format(in_shape))
                return in_shape
            if all(isinstance(x, tuple) for x in in_shape):
                return tuple(val_recurse(x) for x in in_shape)
            raise TypeError('Shape should be a tuple of integers, or a tuple of tuples of integers')

        return val_recurse(input_shape)


def get_dummy_input(dataset=None, device=None, input_shape=None):
    """Generate a representative dummy (random) input.

    If a device is specified, then the dummy_input is moved to that device.

    Args:
        dataset (str): Name of dataset from which to infer the shape
        device (str or torch.device): Device on which to create the input
        input_shape (tuple): Tuple of integers representing the input shape. Can also be a tuple of tuples, allowing
          arbitrarily complex collections of tensors. Used only if 'dataset' is None
    """
    def create_single(shape):
        t = torch.randn(shape)
        if device:
            t = t.to(device)
        return t

    def create_recurse(shape):
        if all(isinstance(x, int) for x in shape):
            return create_single(shape)
        return tuple(create_recurse(s) for s in shape)

    input_shape = _validate_input_shape(dataset, input_shape)
    return create_recurse(input_shape)


def set_model_input_shape_attr(model, dataset=None, input_shape=None):
    """Sets an attribute named 'input_shape' within the model instance, specifying the expected input shape

    Args:
          model (nn.Module): Model instance
          dataset (str): Name of dataset from which to infer input shape
          input_shape (tuple): Tuple of integers representing the input shape. Can also be a tuple of tuples, allowing
            arbitrarily complex collections of tensors. Used only if 'dataset' is None
    """
    if not hasattr(model, 'input_shape'):
        model.input_shape = _validate_input_shape(dataset, input_shape)


def make_non_parallel_copy(model):
    """Make a non-data-parallel copy of the provided model.

    torch.nn.DataParallel instances are removed.
    """
    def replace_data_parallel(container):
        for name, module in container.named_children():
            if isinstance(module, nn.DataParallel):
                setattr(container, name, module.module)
            if has_children(module):
                replace_data_parallel(module)

    # Make a copy of the model, because we're going to change it
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        new_model = new_model.module
    replace_data_parallel(new_model)

    return new_model


@contextlib.contextmanager
def get_nonparallel_clone_model(model):
    clone_model = make_non_parallel_copy(model)
    try:
        yield clone_model
    finally:
        del clone_model


def set_seed(seed):
    """Seed the PRNG for the CPU, Cuda, numpy and Python"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_deterministic(seed=0):
    '''Try to configure the system for reproducible results.

    Experiment reproducibility is sometimes important.  Pete Warden expounded about this
    in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
    For Pytorch specifics see: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    '''
    msglogger.debug('set_deterministic was invoked')
    if seed is None:
        seed = 0
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Function to load YAML file using an OrderedDict

    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)


def yaml_ordered_save(fname, ordered_dict):
    def ordered_dict_representer(self, value):
        return self.represent_mapping('tag:yaml.org,2002:map', value.items())

    yaml.add_representer(OrderedDict, ordered_dict_representer)

    with open(fname, 'w') as f:
        yaml.dump(ordered_dict, f, default_flow_style=False)


def float_range_argparse_checker(min_val=0., max_val=1., exc_min=False, exc_max=False):
    def checker(val_str):
        val = float(val_str)
        min_op, min_op_str = (operator.gt, '>') if exc_min else (operator.ge, '>=')
        max_op, max_op_str = (operator.lt, '<') if exc_max else (operator.le, '<=')
        if min_op(val, min_val) and max_op(val, max_val):
            return val
        raise argparse.ArgumentTypeError(
            'Value must be {} {} and {} {} (received {})'.format(min_op_str, min_val, max_op_str, max_val, val))
    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    return checker


def filter_kwargs(dict_to_filter, function_to_call):
    """Utility to check which arguments in the passed dictionary exist in a function's signature

    The function returns two dicts, one with just the valid args from the input and one with the invalid args.
    The caller can then decide to ignore the existence of invalid args, depending on context.
    """

    sig = inspect.signature(function_to_call)
    filter_keys = [param.name for param in sig.parameters.values() if (param.kind == param.POSITIONAL_OR_KEYWORD)]
    valid_args = {}
    invalid_args = {}

    for key in dict_to_filter:
        if key in filter_keys:
            valid_args[key] = dict_to_filter[key]
        else:
            invalid_args[key] = dict_to_filter[key]
    return valid_args, invalid_args


def convert_tensors_recursively_to(val, *args, **kwargs):
    """ Applies `.to(*args, **kwargs)` to each tensor inside val tree. Other values remain the same."""
    if isinstance(val, torch.Tensor):
        return val.to(*args, **kwargs)

    if isinstance(val, (tuple, list)):
        return type(val)(convert_tensors_recursively_to(item, *args, **kwargs) for item in val)

    return val


def model_setattr(model, attr_name, val, register=False):
    """
    Sets attribute of a model, through the entire hierarchy.
    Args:
        model (nn.Module): the model.
        attr_name (str): the attribute name as shown by model.named_<parameters/modules/buffers>()
        val: the value of the attribute
        register (bool): if True - register_buffer(val) if val is a torch.Tensor and
          register_parameter(val) if it's an nn.Parameter.
    """
    def split_name(name):
        if '.' in name:
            return name.rsplit('.', 1)
        else:
            return '', name
    modules_dict = OrderedDict(model.named_modules())
    lowest_depth_container_name, lowest_depth_attr_name = split_name(attr_name)
    while lowest_depth_container_name and lowest_depth_container_name not in modules_dict:
        container_name, attr = split_name(lowest_depth_container_name)
        lowest_depth_container_name = container_name
        lowest_depth_attr_name = '%s%s' % (attr, lowest_depth_attr_name)
    lowest_depth_container = modules_dict[lowest_depth_container_name]  # type: nn.Module

    if register and torch.is_tensor(val):
        if isinstance(val, nn.Parameter):
            lowest_depth_container.register_parameter(lowest_depth_attr_name, val)
        else:
            lowest_depth_container.register_buffer(lowest_depth_attr_name, val)
    else:
        setattr(lowest_depth_container, lowest_depth_attr_name, val)


def param_name_2_module_name(param_name):
    return '.'.join(param_name.split('.')[:-1])


def is_scalar(val):
    result = isinstance(val, torch.Tensor) and val.dim() == 0
    result |= np.isscalar(val)
    return result

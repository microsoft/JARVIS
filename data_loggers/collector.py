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

import contextlib
from functools import partial, reduce
import operator
import xlsxwriter
import enum
import yaml
import os
from sys import float_info
from collections import OrderedDict
from contextlib import contextmanager
import torch
from torchnet.meter import AverageValueMeter
import logging
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import distiller
from distiller.quantization.range_linear import is_post_train_quant_wrapper
from distiller.quantization.pytorch_quant_conversion import QFunctionalWrapper
import numpy as np
import concurrent.futures

msglogger = logging.getLogger()

__all__ = ['SummaryActivationStatsCollector', 'RecordsActivationStatsCollector', 'QuantCalibrationStatsCollector',
           'ActivationHistogramsCollector', 'RawActivationsCollector', 'CollectorDirection',
           'collect_quant_stats', 'collect_histograms', 'collect_raw_outputs',
           'collector_context', 'collectors_context']


class CollectorDirection(enum.Enum):
    OUT = 0
    OFM = 0
    IN = 1
    IFM = 1
    IFMS = 1


class ActivationStatsCollector(object):
    """Collect model activation statistics information.

    ActivationStatsCollector is the base class for classes that collect activations statistics.
    You may collect statistics on different phases of the optimization process (training, validation, test).

    Statistics data are accessible via .value() or by accessing individual modules.

    The current implementation has a few caveats:
    * It is slow - therefore it is advisable to use this only when needed.
    * It can't access the activations of torch.Functions, only torch.Modules.

    ActivationStatsCollector uses the forward hook of modules in order to access the
    feature-maps.  This is both slow and limits us to seeing only the outputs of torch.Modules.
    We can remove some of the slowness, by choosing to log only specific layers or use it only
    during validation or test. This can be achieved using the `classes` argument.
    """
    def __init__(self, model, stat_name, classes):
        """
        Args:
            model - the model we are monitoring.
            stat_name - name for the statistics being collected.
                You can access a module's activation statistics by referring to module.<stat_name>
                For example:
                    print(module.sparsity)
            classes - a list of class types for which we collect activation statistics.
                Passing an empty list or None will collect statistics for all class types.
        """
        super(ActivationStatsCollector, self).__init__()
        self.model = model
        self.stat_name = stat_name
        self.classes = classes
        self.fwd_hook_handles = []

        # The layer names are mangled, because torch.Modules don't have names and we need to invent
        # a unique, human-readable name per layer.
        distiller.utils.assign_layer_fq_names(model)

        # Currently this is internal, and its only purpose is to enable skipping collection
        # for wrapped modules inside post-training quantization wrapper classes.
        # When doing PTQ, the outputs of these wrapped modules are actually intermediate results
        # which are not relevant for tracking.
        self._dont_collect_list = [module.wrapped_module.distiller_name for module in model.modules() if
                                   is_post_train_quant_wrapper(module)]

    def value(self):
        """Return a dictionary containing {layer_name: statistic}"""
        activation_stats = OrderedDict()
        self.model.apply(partial(self._collect_activations_stats, activation_stats=activation_stats))
        return activation_stats

    def start(self, modules_list=None):
        """Start collecting activation stats.

        This will iteratively register the modules' forward-hooks, so that the collector
        will be called from the forward traversal and get exposed to activation data.
        modules_list (iterable): track stats for modules in the list. If None/empty - will track for all modules.
        """
        assert len(self.fwd_hook_handles) == 0
        if not modules_list:
            self.model.apply(self.start_module)
            return
        modules_dict = dict(self.model.named_modules())
        for module_name in modules_list:
            modules_dict[module_name].apply(self.start_module)

    def start_module(self, module):
        """Iteratively register to the forward-pass callback of all eligible modules.

        Eligible modules are currently filtered by their class type.
        """
        if self._should_collect(module):
            self.fwd_hook_handles.append(module.register_forward_hook(self._activation_stats_cb))
            self._start_counter(module)

    def stop(self):
        """Stop collecting activation stats.

        This will iteratively unregister the modules' forward-hooks.
        """
        for handle in self.fwd_hook_handles:
            handle.remove()
        self.fwd_hook_handles = []

    def reset(self):
        """Reset the statsitics counters of this collector."""
        self.model.apply(self._reset_counter)
        return self

    def save(self, fname):
        raise NotImplementedError

    def _activation_stats_cb(self, module, inputs, output):
        """Handle new activations ('output' argument).

        This is invoked from the forward-pass callback of module 'module'.
        """
        raise NotImplementedError

    def _start_counter(self, module):
        """Start a specific statistic counter - this is subclass-specific code"""
        raise NotImplementedError

    def _reset_counter(self, module):
        """Reset a specific statistic counter - this is subclass-specific code"""
        raise NotImplementedError

    def _collect_activations_stats(self, module, activation_stats, name=''):
        """Handle new activations - this is subclass-specific code"""
        raise NotImplementedError

    def _should_collect(self, module):
        if module.distiller_name in self._dont_collect_list:
            return False
        # In general, we only collect stats for "leaf" modules.
        # We make an exception for models that were quantized with 'PostTrainLinearQuantizer'. In these
        # models, the quantized modules are actually wrappers of the original FP32 modules, so they are
        # NOT leaf modules - but we still want to track them.
        if distiller.has_children(module) and not (is_post_train_quant_wrapper(module) or
                                                   isinstance(module, QFunctionalWrapper)):
            return False
        if isinstance(module, torch.nn.Identity):
            return False

        register_all_class_types = not self.classes
        if register_all_class_types or isinstance(module, tuple(self.classes)):
            return True

        return False


class WeightedAverageValueMeter(AverageValueMeter):
    """
    A correction to torchnet's AverageValueMeter which doesn't implement
    std collection correctly by taking into account the batch size.
    """
    def add(self, value, n=1):
        self.sum += value*n
        if n <= 0:
            raise ValueError("Cannot use a non-positive weight for the running stat.")
        elif self.n == 0:
            self.mean = 0.0 + value  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + n * (value - self.mean_old) / float(self.n+n)
            self.m_s += n*(value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n + n - 1.0))
        self.var = self.std**2

        self.n += n


class SummaryActivationStatsCollector(ActivationStatsCollector):
    """This class collects activations statistical summaries.

    This Collector computes the mean of some statistic of the activation.  It is rather
    light-weight and quicker than collecting a record per activation.
    The statistic function is configured in the constructor.

    collector_direction - enum type: IN for IFMs, OUT for OFM
    inputs_consolidate_func is called on tuple of tensors, and returns a tensor.
    """
    def __init__(self, model, stat_name, summary_fn,
                 classes=(torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU),
                 collector_direction=CollectorDirection.OUT,
                 inputs_consolidate_func=torch.cat):
        super(SummaryActivationStatsCollector, self).__init__(model, stat_name, classes)
        self.summary_fn = summary_fn
        self.collector_direction = collector_direction
        self.inputs_func = inputs_consolidate_func

    def _activation_stats_cb(self, module, inputs, output):
        """Record the activation sparsity of 'module'

        This is a callback from the forward() of 'module'.
        """
        feature_map = output if self.collector_direction == CollectorDirection.OUT else self.inputs_func(inputs)
        try:
            getattr(module, self.stat_name).add(self.summary_fn(feature_map.data), feature_map.data.numel())
        except RuntimeError as e:
            if "The expanded size of the tensor" in e.args[0]:
                raise ValueError("ActivationStatsCollector: a module ({} - {}) was encountered twice during model.apply().\n"
                                 "This is an indication that your model is using the same module instance, "
                                 "in multiple nodes in the graph.  This usually occurs with ReLU modules: \n"
                                 "For example in TorchVision's ResNet model, self.relu = nn.ReLU(inplace=True) is "
                                 "instantiated once, but used multiple times.  This is not permissible when using "
                                 "instances of ActivationStatsCollector.".
                                 format(module.distiller_name, type(module)))
            else:
                msglogger.info("Exception in _activation_stats_cb: {} {}".format(module.distiller_name, type(module)))
                raise

    def _start_counter(self, module):
        if not hasattr(module, self.stat_name):
            setattr(module, self.stat_name, WeightedAverageValueMeter())
            # Assign a name to this summary
            if hasattr(module, 'distiller_name'):
                getattr(module, self.stat_name).name = module.distiller_name
            else:
                getattr(module, self.stat_name).name = '_'.join((
                    module.__class__.__name__, str(id(module))))

    def _reset_counter(self, module):
        if hasattr(module, self.stat_name):
            getattr(module, self.stat_name).reset()

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if hasattr(module, self.stat_name):
            mean = getattr(module, self.stat_name).mean
            if isinstance(mean, torch.Tensor):
                mean = mean.tolist()
            activation_stats[getattr(module, self.stat_name).name] = mean

    def save(self, fname):
        """Save the stats to an Excel workbook"""
        if not fname.endswith('.xlsx'):
            fname = '.'.join([fname, 'xlsx'])
        with contextlib.suppress(OSError):
            os.remove(fname)

        def _add_worksheet(workbook, tab_name, record):
            try:
                worksheet = workbook.add_worksheet(tab_name)
            except xlsxwriter.exceptions.InvalidWorksheetName:
                worksheet = workbook.add_worksheet()

            col_names = []
            for col, (module_name, module_summary_data) in enumerate(record.items()):
                if not isinstance(module_summary_data, list):
                    module_summary_data = [module_summary_data]
                worksheet.write_column(1, col, module_summary_data)
                col_names.append(module_name)
            worksheet.write_row(0, 0, col_names)

        with xlsxwriter.Workbook(fname) as workbook:
            _add_worksheet(workbook, self.stat_name, self.value())

        return fname


class RecordsActivationStatsCollector(ActivationStatsCollector):
    """This class collects activations statistical records.

    This Collector computes a hard-coded set of activations statistics and collects a
    record per activation.  The activation records of the entire model (only filtered modules),
    can be saved to an Excel workbook.

    For obvious reasons, this is slower than SummaryActivationStatsCollector.
    """
    def __init__(self, model, classes=(torch.nn.ReLU,
                                       torch.nn.ReLU6,
                                       torch.nn.LeakyReLU)):
        super(RecordsActivationStatsCollector, self).__init__(model, "statistics_records", classes)

    def _activation_stats_cb(self, module, inputs, output):
        """Record the activation sparsity of 'module'

        This is a callback from the forward() of 'module'.
        """
        def to_np(stats):
            if isinstance(stats, tuple):
                return stats[0].detach().cpu().numpy()
            else:
                return stats.detach().cpu().numpy()

        # We get a batch of activations, from which we collect statistics
        if not output.is_contiguous():
            output = output.contiguous()
        act = output.view(output.size(0), -1)
        batch_min_list = to_np(torch.min(act, dim=1)).tolist()
        batch_max_list = to_np(torch.max(act, dim=1)).tolist()
        batch_mean_list = to_np(torch.mean(act, dim=1)).tolist()
        # If activation contains only a single element, standard-deviation is meaningless (and std() returns NaN)
        # Return 0 instead
        if act.shape[0] == act.numel():
            batch_std_list = to_np(torch.zeros(act.shape[0])).tolist()
        else:
            batch_std_list = to_np(torch.std(act, dim=1)).tolist()
        batch_l2_list = to_np(torch.norm(act, p=2, dim=1)).tolist()

        module.statistics_records['min'].extend(batch_min_list)
        module.statistics_records['max'].extend(batch_max_list)
        module.statistics_records['mean'].extend(batch_mean_list)
        module.statistics_records['std'].extend(batch_std_list)
        module.statistics_records['l2'].extend(batch_l2_list)
        module.statistics_records['shape'] = distiller.size2str(output)

    @staticmethod
    def _create_records_dict():
        records = OrderedDict()
        for stat_name in ['min', 'max', 'mean', 'std', 'l2']:
            records[stat_name] = []
        records['shape'] = ''
        return records

    def save(self, fname):
        """Save the records to an Excel workbook, with one worksheet per layer.
        """
        fname = ".".join([fname, 'xlsx'])
        try:
            os.remove(fname)
        except OSError:
            pass

        records_dict = self.value()
        with xlsxwriter.Workbook(fname) as workbook:
            for module_name, module_act_records in records_dict.items():
                try:
                    worksheet = workbook.add_worksheet(module_name)
                except xlsxwriter.exceptions.InvalidWorksheetName:
                    worksheet = workbook.add_worksheet()

                col_names = []
                for col, (col_name, col_data) in enumerate(module_act_records.items()):
                    if col_name == 'shape':
                        continue
                    worksheet.write_column(1, col, col_data)
                    col_names.append(col_name)
                worksheet.write_row(0, 0, col_names)
                worksheet.write(0, len(col_names)+2, module_act_records['shape'])
        return fname

    def _start_counter(self, module):
        if not hasattr(module, "statistics_records"):
            module.statistics_records = self._create_records_dict()

    def _reset_counter(self, module):
        if hasattr(module, "statistics_records"):
            module.statistics_records = self._create_records_dict()

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if hasattr(module, "statistics_records"):
            activation_stats[module.distiller_name] = module.statistics_records


class _QuantStatsRecord(object):
    @staticmethod
    def create_records_dict():
        records = OrderedDict()
        records['min'] = float_info.max
        records['max'] = -float_info.max
        for stat_name in ['avg_min', 'avg_max', 'mean', 'std', 'b']:
            records[stat_name] = 0
        records['shape'] = ''
        records['total_numel'] = 0
        return records

    def __init__(self):
        # We don't know the number of inputs at this stage so we defer records creation to the actual callback
        self.inputs = []
        self.output = self.create_records_dict()


def _verify_no_dataparallel(model):
    if torch.nn.DataParallel in [type(m) for m in model.modules()]:
        raise ValueError('Model contains DataParallel modules, which can cause inaccurate stats collection. '
                         'Either create a model without DataParallel modules, or call '
                         'distiller.utils.make_non_parallel_copy on the model before invoking the collector')


class QuantCalibrationStatsCollector(ActivationStatsCollector):
    """
    This class tracks activations stats required for quantization, for each layer and for each input
    and output. The tracked stats are:
      * Absolute min / max
      * Average min / max (calculate min / max per sample and average those)
      * Overall mean
      * Overall standard-deviation

    The generated stats dict has the following structure per-layer:
    'layer_name':
        'inputs':
            0:
                'min': value
                'max': value
                ...
            ...
            n:
                'min': value
                'max': value
                ...
        'output':
            'min': value
            'max': value
            ...
    Where n is the number of inputs the layer has.
    The calculated stats can be saved to a YAML file.

    If a certain layer operates in-place, that layer's input stats will be overwritten by its output stats.
    The collector can, optionally, check for such cases at runtime. In addition, a simple mechanism to disable inplace
    operations in the model can be used. See arguments details below.

    Args:
        model (torch.nn.Module): The model we are monitoring
        classes (list): List of class types for which we collect activation statistics. Passing an empty list or
          None will collect statistics for all class types.
        inplace_runtime_check (bool): If True will raise an error if an in-place operation is detected
        disable_inplace_attrs (bool): If True, will search all modules within the model for attributes controlling
          in-place operations and disable them.
        inplace_attr_names (iterable): If disable_inplace_attrs is enabled, this is the list of attribute name
          that will be searched for.

    TODO: Consider merging with RecordsActivationStatsCollector
    Current differences between the classes:
      * Track single value per-input/output-per-module for the entire run. Specifically, for standard deviation this
        cannot be done by tracking per-activation std followed by some post-processing
      * Track inputs in addition to outputs
      * Different serialization (yaml vs xlsx)
    """
    def __init__(self, model, classes=None, inplace_runtime_check=False,
                 disable_inplace_attrs=False, inplace_attr_names=('inplace',)):
        super(QuantCalibrationStatsCollector, self).__init__(model, "quant_stats", classes)

        _verify_no_dataparallel(model)

        self.batch_idx = 0
        self.inplace_runtime_check = inplace_runtime_check
        self.collecting_second_pass = False

        if disable_inplace_attrs:
            if not inplace_attr_names:
                raise ValueError('inplace_attr_names cannot by empty or None')
            for m in model.modules():
                for n in inplace_attr_names:
                    if hasattr(m, n):
                        setattr(m, n, False)

    def _check_required_stats(self):
        """
        Check whether the required statistics were collected to allow collecting laplace distribution stats.
        """
        for name, module in self.model.named_modules():
            if not self._should_collect(module):
                continue
            if not hasattr(module, 'quant_stats'):
                raise RuntimeError('Collection of Laplace distribution statistics is '
                                   'only allowed after collection of stats has started.')
            for i, input_stats_record in enumerate(module.quant_stats.inputs):
                if 'mean' not in input_stats_record:
                    raise RuntimeError('The required stats for input[%d] in module "%s" were not collected. '
                                       'Please collect the required statistics using `collector.start()` and evaluating'
                                       ' the model for enough batches.' % (i, name))
            if 'mean' not in module.quant_stats.output:
                raise RuntimeError('The required stats for the output in module "%s" were not collected. '
                                   'Please collect the required statistics using `collector.start()` and evaluating'
                                   ' the model for enough batches.' % name)

    def start_second_pass(self):
        self._check_required_stats()
        self.collecting_second_pass = True
        # reset batch_idx for all tracked modules
        for module in self.model.modules():
            if not self._should_collect(module):
                continue
            module.batch_idx = 0
            for record in module.quant_stats.inputs:
                record['total_numel'] = 0
            module.quant_stats.output['total_numel'] = 0

    def stop_second_pass(self):
        self.collecting_second_pass = False

    def _activation_stats_cb(self, module, inputs, output):
        """
        A callback for updating the required statistics for quantization in a module.
        """
        def update_running_mean(values, prev_mean, total_values_so_far):
            """
            Updates a running mean of a tensor of values
            Args:
                values (torch.Tensor): the new tensor
                prev_mean (float): the previous running mean
                total_values_so_far (int): the number of the values so far
            """
            curr_numel = values.numel()
            prev_numel = total_values_so_far
            return (prev_numel * prev_mean + values.sum().item()) / (prev_numel + curr_numel)

        def update_std(values, prev_std, mean, total_values_so_far):
            """
            Updates std of the tensor
            """
            prev_variance = prev_std ** 2
            curr_sqr_dists = (values - mean) ** 2
            new_variance = update_running_mean(curr_sqr_dists, prev_variance, total_values_so_far)
            return sqrt(new_variance)

        def update_b(values, previous_b, mean, total_values_so_far):
            """
            Updates the 'b' parameter of Laplace Distribution.
            """
            curr_abs_dists = (values - mean).abs_()
            return update_running_mean(curr_abs_dists, previous_b, total_values_so_far)

        def update_record(record, tensor):
            if tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
                # Mean function only works for float tensors
                tensor = tensor.to(torch.float32)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            act = tensor.view(tensor.size(0), -1)
            numel = act.numel()
            if self.collecting_second_pass:
                record['b'] = update_b(act, record['b'], record['mean'], record['total_numel'])
                record['std'] = update_std(act, record['std'], record['mean'], record['total_numel'])
                record['total_numel'] += numel
                return

            # In the general case, the average min/max that we're collecting are averages over the per-sample
            # min/max values. That is - we first calculate the min/max for each sample in the batch, then average
            # over that.
            # But - If each sample contains just a single value, then such a per-sample calculation we'll result in
            # avg_min = avg_max. So in that case we "revert" to calculating "global" values, for the whole batch,
            # instead of per-sample values
            dim = 0 if numel == act.shape[0] else 1

            min_per_sample = act.min(dim=dim)[0]
            max_per_sample = act.max(dim=dim)[0]
            record['min'] = min(record['min'], min_per_sample.min().item())
            record['max'] = max(record['max'], max_per_sample.max().item())
            record['avg_min'] = update_running_mean(min_per_sample, record['avg_min'], record['total_numel'])
            record['avg_max'] = update_running_mean(max_per_sample, record['avg_max'], record['total_numel'])
            new_mean = update_running_mean(act, record['mean'], record['total_numel'])
            record['mean'] = new_mean
            record['total_numel'] += numel

            if not record['shape']:
                record['shape'] = distiller.size2str(tensor)

        if self.inplace_runtime_check and any([id(input) == id(output) for input in inputs]):
            if not isinstance(module, torch.nn.modules.dropout._DropoutNd):
                raise RuntimeError('Inplace operation detected, meaning inputs stats are overridden by output stats. '
                                   'You can either disable this check or make sure no in-place operations occur. '
                                   'See QuantCalibrationStatsCollector class documentation for more info.')

        module.batch_idx += 1

        if not module.quant_stats.inputs:
            # Delayed initialization of inputs records, because only now we know the # of inputs
            for i in range(len(inputs)):
                module.quant_stats.inputs.append(_QuantStatsRecord.create_records_dict())

        with torch.no_grad():
            for idx, input in enumerate(inputs):
                update_record(module.quant_stats.inputs[idx], input)
            update_record(module.quant_stats.output, output)

    def _start_counter(self, module):
        # We don't know the number of inputs at this stage so we defer records creation to the actual callback
        module.quant_stats = _QuantStatsRecord()
        module.batch_idx = 0

    def _reset_counter(self, module):
        # We don't know the number of inputs at this stage so we defer records creation to the actual callback
        if hasattr(module, 'quant_stats'):
            module.quant_stats = _QuantStatsRecord()
            module.batch_idx = 0

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if not hasattr(module, 'quant_stats'):
            return

        activation_stats[module.distiller_name] = OrderedDict()
        if module.quant_stats.inputs:
            activation_stats[module.distiller_name]['inputs'] = OrderedDict()
            for idx, sr in enumerate(module.quant_stats.inputs):
                activation_stats[module.distiller_name]['inputs'][idx] = sr
        activation_stats[module.distiller_name]['output'] = module.quant_stats.output

    def save(self, fname):
        if not fname.endswith('.yaml'):
            fname = ".".join([fname, 'yaml'])
        try:
            os.remove(fname)
        except OSError:
            pass

        records_dict = self.value()
        distiller.yaml_ordered_save(fname, records_dict)

        return fname


class ActivationHistogramsCollector(ActivationStatsCollector):
    """
    This class collects activation histograms, for each layer and for each input and output tensor.
    It requires pre-computed min/max stats per tensor. This is done in order to prevent the need to save
    all of the activation tensors throughout the run. The histogram is created once according to these
    min/max values, and updated after each iteration. Any value outside the pre-computed range is clamped.

    The generated stats dict has the following structure per-layer:
    'layer_name':
        'inputs':
            0:
                'hist': tensor             # Tensor with bin counts
                'bin_centroids': tensor    # Tensor with activation values corresponding to center of each bin
            ...
            n:
                'hist': tensor
                'bin_centroids': tensor
        'output':
            'hist': tensor
            'bin_centroids': tensor
    Where n is the number of inputs the layer has.
    The generated stats dictionary can be saved to a file.
    Optionally, histogram images for all tensor can be saved as well

    Args:
        model (torch.nn.Module): The model we are monitoring
        activation_stats (str / dict): Either a path to activation stats YAML file, or a dictionary containing
          the stats. The stats are expected to be in the same structure as generated by QuantCalibrationStatsCollector.
        classes (list): List of class types for which we collect activation statistics. Passing an empty list or
          None will collect statistics for all class types.
        nbins (int): Number of histogram bins
        save_hist_imgs (bool): If set, calling save() will dump images of the histogram plots in addition to saving the
          stats dictionary
        hist_imgs_ext (str): The file type to be used when saving histogram images
    """
    def __init__(self, model, activation_stats, classes=None, nbins=2048,
                 save_hist_imgs=False, hist_imgs_ext='.svg'):
        super(ActivationHistogramsCollector, self).__init__(model, 'histogram', classes)

        _verify_no_dataparallel(model)

        if isinstance(activation_stats, str):
            if not os.path.isfile(activation_stats):
                raise ValueError("Model activation stats file not found at: " + activation_stats)
            msglogger.info('Loading activation stats from: ' + activation_stats)
            with open(activation_stats, 'r') as stream:
                activation_stats = distiller.utils.yaml_ordered_load(stream)
        elif not isinstance(activation_stats, (dict, OrderedDict)):
            raise TypeError('model_activation_stats must either be a string, a dict / OrderedDict or None')

        self.act_stats = activation_stats
        self.nbins = nbins
        self.save_imgs = save_hist_imgs
        self.imgs_ext = hist_imgs_ext if hist_imgs_ext[0] == '.' else '.' + hist_imgs_ext

    def _get_min_max(self, *keys):
        stats_entry = reduce(operator.getitem, keys, self.act_stats)
        return stats_entry['min'], stats_entry['max']

    def _activation_stats_cb(self, module, inputs, output):
        def get_hist(t, stat_min, stat_max):
            # torch.histc doesn't work on integral data types, so convert if needed
            if t.dtype not in [torch.float, torch.double, torch.half]:
                t = t.float()
            t_clamped = t.clamp(stat_min, stat_max)
            hist = torch.histc(t_clamped.cpu(), bins=self.nbins, min=stat_min, max=stat_max)
            return hist

        with torch.no_grad():
            for idx, input in enumerate(inputs):
                stat_min, stat_max = self._get_min_max(module.distiller_name, 'inputs', idx)
                curr_hist = get_hist(input, stat_min, stat_max)
                module.input_hists[idx] += curr_hist

            stat_min, stat_max = self._get_min_max(module.distiller_name, 'output')
            curr_hist = get_hist(output, stat_min, stat_max)
            module.output_hist += curr_hist

    def _reset(self, module):
        num_inputs = len(self.act_stats[module.distiller_name]['inputs'])
        module.input_hists = module.input_hists = [torch.zeros(self.nbins) for _ in range(num_inputs)]
        module.output_hist = torch.zeros(self.nbins)

    def _start_counter(self, module):
        self._reset(module)

    def _reset_counter(self, module):
        if hasattr(module, 'output_hist'):
            self._reset(module)

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if not hasattr(module, 'output_hist'):
            return

        def get_hist_entry(min_val, max_val, hist):
            od = OrderedDict()
            od['hist'] = hist
            bin_width = (max_val - min_val) / self.nbins
            od['bin_centroids'] = torch.linspace(min_val + bin_width / 2, max_val - bin_width / 2, self.nbins)
            return od

        stats_od = OrderedDict()
        inputs_od = OrderedDict()
        for idx, hist in enumerate(module.input_hists):
            inputs_od[idx] = get_hist_entry(*self._get_min_max(module.distiller_name, 'inputs', idx),
                                            module.input_hists[idx])

        output_od = get_hist_entry(*self._get_min_max(module.distiller_name, 'output'), module.output_hist)

        stats_od['inputs'] = inputs_od
        stats_od['output'] = output_od
        activation_stats[module.distiller_name] = stats_od

    def save(self, fname):
        hist_dict = self.value()

        if not fname.endswith('.pt'):
            fname = ".".join([fname, 'pt'])
        try:
            os.remove(fname)
        except OSError:
            pass

        torch.save(hist_dict, fname)

        if self.save_imgs:
            msglogger.info('Saving histogram images...')
            save_dir = os.path.join(os.path.split(fname)[0], 'histogram_imgs')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            def save_hist(layer_name, tensor_name, idx, bin_counts, bin_centroids, normed=True):
                if normed:
                    bin_counts = bin_counts / bin_counts.sum()
                plt.figure(figsize=(12, 12))
                plt.suptitle('\n'.join((layer_name, tensor_name)), fontsize=18, fontweight='bold')
                for subplt_idx, yscale in enumerate(['linear', 'log']):
                    plt.subplot(2, 1, subplt_idx + 1)
                    plt.fill_between(bin_centroids, bin_counts, step='mid', antialiased=False)
                    if yscale == 'linear':
                        plt.ylim(bottom=0)
                    plt.title(yscale + ' scale')
                    plt.yscale(yscale)
                    plt.xlabel('Activation Value')
                    plt.ylabel('Normalized Count')
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                idx_str = '{:03d}'.format(idx)
                plt.savefig(os.path.join(save_dir, '-'.join((idx_str, layer_name, tensor_name)) + self.imgs_ext))
                plt.close()

            cnt = 0
            for layer_name, data in hist_dict.items():
                for idx, od in data['inputs'].items():
                    cnt += 1
                    save_hist(layer_name, 'input_{}'.format(idx), cnt, od['hist'], od['bin_centroids'], normed=True)
                od = data['output']
                cnt += 1
                save_hist(layer_name, 'output', cnt, od['hist'], od['bin_centroids'], normed=True)
            msglogger.info('Done')
        return fname


class RawActivationsCollector(ActivationStatsCollector):
    def __init__(self, model, classes=None):
        super(RawActivationsCollector, self).__init__(model, "raw_acts", classes)

        _verify_no_dataparallel(model)

    def _activation_stats_cb(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            if output.is_quantized:
                module.raw_outputs.append(output.dequantize())
            else:
                module.raw_outputs.append(output.cpu())

    def _start_counter(self, module):
        module.raw_outputs = []

    def _reset_counter(self, module):
        if hasattr(module, 'raw_outputs'):
            module.raw_outputs = []

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if not hasattr(module, 'raw_outputs'):
            return

        if isinstance(module.raw_outputs, list) and len(module.raw_outputs) > 0:
            module.raw_outputs = torch.stack(module.raw_outputs)
        activation_stats[module.distiller_name] = module.raw_outputs

    def save(self, dir_name):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for idx, (layer_name, raw_outputs) in enumerate(self.value().items()):
                idx_str = '{:03d}'.format(idx + 1)
                executor.submit(torch.save, raw_outputs, os.path.join(dir_name,
                                                                      '-'.join((idx_str, layer_name)) + '.pt'))

        return dir_name


def collect_quant_stats(model, test_fn, save_dir=None, classes=None, inplace_runtime_check=False,
                        disable_inplace_attrs=False, inplace_attr_names=('inplace',),
                        modules_to_collect=None):
    """
    Helper function for collecting quantization calibration statistics for a model using QuantCalibrationStatsCollector

    Args:
        model (nn.Module): The model for which to collect stats
        test_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        save_dir (str): Path to directory where stats YAML file will be saved. If None then YAML will not be saved
          to disk.
        classes (iterable): See QuantCalibrationStatsCollector
        inplace_runtime_check (bool): See QuantCalibrationStatsCollector
        disable_inplace_attrs (bool): See QuantCalibrationStatsCollector
        inplace_attr_names (iterable): See QuantCalibrationStatsCollector
        modules_to_collect (iterable): enable stats collection for a predefined modules (specified by names).
          if None - will track stats for all layers.

    Returns:
        Dictionary with quantization stats (see QuantCalibrationStatsCollector for a description of the dictionary
        contents)
    """
    msglogger.info('Collecting quantization calibration stats for model')
    quant_stats_collector = QuantCalibrationStatsCollector(model, classes=classes,
                                                           inplace_runtime_check=inplace_runtime_check,
                                                           disable_inplace_attrs=disable_inplace_attrs,
                                                           inplace_attr_names=inplace_attr_names)
    with collector_context(quant_stats_collector, modules_to_collect):
        msglogger.info('Pass 1: Collecting min, max, avg_min, avg_max, mean')
        test_fn(model=model)
        # Collect Laplace distribution stats:
        msglogger.info('Pass 2: Collecting b, std parameters')
        quant_stats_collector.start_second_pass()
        test_fn(model=model)
        quant_stats_collector.stop_second_pass()

    msglogger.info('Stats collection complete')
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'acts_quantization_stats.yaml')
        quant_stats_collector.save(save_path)
        msglogger.info('Stats saved to ' + save_path)

    return quant_stats_collector.value()


def collect_histograms(model, test_fn, save_dir=None, activation_stats=None,
                       classes=None, nbins=2048, save_hist_imgs=False, hist_imgs_ext='.svg'):
    """
    Helper function for collecting activation histograms for a model using ActivationsHistogramCollector.
    Will perform 2 passes - one to collect the required stats and another to collect the histograms. The first
    pass can be skipped by passing pre-calculated stats.

    Args:
        model (nn.Module): The model for which to collect histograms
        test_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        save_dir (str): Path to directory where histograms will be saved. If None then data will not be saved to disk.
        activation_stats (str / dict / None): Either a path to activation stats YAML file, or a dictionary containing
          the stats. The stats are expected to be in the same structure as generated by QuantCalibrationStatsCollector.
          If None, then a stats collection pass will be performed.
        classes: See ActivationsHistogramCollector
        nbins: See ActivationsHistogramCollector
        save_hist_imgs: See ActivationsHistogramCollector
        hist_imgs_ext: See ActivationsHistogramCollector

    Returns:
        Dictionary with histograms data (See ActivationsHistogramCollector for a description of the dictionary
        contents)
    """
    msglogger.info('Pass 1: Stats collection')
    if activation_stats is not None:
        msglogger.info('Pre-computed activation stats passed, skipping stats collection')
    else:
        activation_stats = collect_quant_stats(model, test_fn, save_dir=save_dir, classes=classes,
                                               inplace_runtime_check=True, disable_inplace_attrs=True)

    msglogger.info('Pass 2: Histograms generation')
    histogram_collector = ActivationHistogramsCollector(model, activation_stats, classes=classes, nbins=nbins,
                                                        save_hist_imgs=save_hist_imgs, hist_imgs_ext=hist_imgs_ext)
    with collector_context(histogram_collector):
        test_fn(model=model)
    msglogger.info('Histograms generation complete')
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'acts_histograms.pt')
        histogram_collector.save(save_path)
        msglogger.info("Histogram data saved to " + save_path)
        if save_hist_imgs:
            msglogger.info('Histogram images saved in ' + os.path.join(save_dir, 'histogram_imgs'))

    return histogram_collector.value()


def collect_raw_outputs(model, test_fn, save_dir=None, classes=None):
    msglogger.info('Collecting raw layer outputs for model')
    collector = RawActivationsCollector(model, classes=classes)
    with collector_context(collector):
        test_fn(model=model)
    msglogger.info('Outputs collection complete')
    if save_dir is not None:
        msglogger.info('Saving outputs to disk...')
        save_path = os.path.join(save_dir, 'raw_outputs')
        collector.save(save_path)
        msglogger.info('Outputs saved to ' + save_path)
    return collector.value()


@contextmanager
def collector_context(collector, modules_list=None):
    """A context manager for an activation collector"""
    if collector is not None:
        collector.reset().start(modules_list)
    yield collector
    if collector is not None:
        collector.stop()


@contextmanager
def collectors_context(collectors_dict):
    """A context manager for a dictionary of collectors"""
    if len(collectors_dict) == 0:
        yield collectors_dict
        return
    for collector in collectors_dict.values():
        collector.reset().start()
    yield collectors_dict
    for collector in collectors_dict.values():
        collector.stop()


class TrainingProgressCollector(object):
    def __init__(self, stats={}):
        super(TrainingProgressCollector, self).__init__()
        object.__setattr__(self, '_stats', stats)

    def __setattr__(self, name, value):
        stats = self.__dict__.get('_stats')
        stats[name] = value

    def __getattr__(self, name):
        if name in self.__dict__['_stats']:
            return self.__dict__['_stats'][name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def value(self):
        return self._stats

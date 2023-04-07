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

"""Loggers frontends and backends.

- DataLogger is the generic logger interface.
- PythonLogger logs using the Python logger.
- TensorBoardLogger logs to files that can be read by Google's TensorBoard.
- CsvLogger logs to CSV files.

Note that not all loggers implement all logging methods.
"""

import torch
import tabulate
import distiller
from distiller.utils import density, sparsity, sparsity_2D, size_to_str, to_np, norm_filters
# TensorBoard logger
from .tbbackend import TBBackend
import csv
import logging
from contextlib import ExitStack
import os
#msglogger = logging.getLogger()

__all__ = ['PythonLogger', 'TensorBoardLogger', 'CsvLogger', 'NullLogger']


class DataLogger(object):
    """This is an abstract interface for data loggers

    Data loggers log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """
    def __init__(self):
        pass

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        pass

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        pass

    def log_weights_sparsity(self, model, epoch):
        pass

    def log_weights_distribution(self, named_params, steps_completed):
        pass

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        pass

# Log to null-space
NullLogger = DataLogger


class PythonLogger(DataLogger):
    def __init__(self, logger):
        super(PythonLogger, self).__init__()
        self.pylogger = logger

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        stats_dict = stats_dict[1]
        if epoch > -1:
            log = 'Epoch: [{}][{:5d}/{:5d}]    '.format(epoch, completed, int(total))
        else:
            log = 'Test: [{:5d}/{:5d}]    '.format(completed, int(total))
        for name, val in stats_dict.items():
            if isinstance(val, int):
                log = log + '{name} {val}    '.format(name=name, val=distiller.pretty_int(val))
            else:
                log = log + '{name} {val:.6f}    '.format(name=name, val=val)
        self.pylogger.info(log)

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        data = []
        for layer, statistic in activation_stats.items():
            data.append([layer, statistic])
        t = tabulate.tabulate(data, headers=['Layer', stat_name], tablefmt='psql', floatfmt=".2f")
        self.pylogger.info('\n' + t)

    def log_weights_sparsity(self, model, epoch):
        t, total = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        self.pylogger.info("\nParameters:\n" + str(t))
        self.pylogger.info('Total sparsity: {:0.2f}\n'.format(total))

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.

        Notes:
            1. Each buffer provided in 'buffer_names' is displayed in a separate table.
            2. Within each table, each value is displayed in a separate column.
        """
        datas = {name: [] for name in buffer_names}
        maxlens = {name: 0 for name in buffer_names}
        for n, m in model.named_modules():
            for buffer_name in buffer_names:
                try:
                    p = getattr(m, buffer_name)
                except AttributeError:
                    continue
                data = datas[buffer_name]
                values = p if isinstance(p, (list, torch.nn.ParameterList)) else p.view(-1).tolist()
                data.append([distiller.normalize_module_name(n) + '.' + buffer_name, *values])
                maxlens[buffer_name] = max(maxlens[buffer_name], len(values))

        for name in buffer_names:
            if datas[name]:
                headers = ['Layer'] + ['Val_' + str(i) for i in range(maxlens[name])]
                t = tabulate.tabulate(datas[name], headers=headers, tablefmt='psql', floatfmt='.4f')
                self.pylogger.info('\n' + name.upper() + ': (Epoch {0}, Step {1})\n'.format(epoch, completed) + t)


class TensorBoardLogger(DataLogger):
    def __init__(self, logdir):
        super(TensorBoardLogger, self).__init__()
        # Set the tensorboard logger
        self.tblogger = TBBackend(logdir)
        print('\n--------------------------------------------------------')
        print('Logging to TensorBoard - remember to execute the server:')
        print('> tensorboard --logdir=\'./logs\'\n')

        # Hard-code these preferences for now
        self.log_gradients = False      # True
        self.logged_params = ['weight'] # ['weight', 'bias']

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        def total_steps(total, epoch, completed):
            return total*epoch + completed

        prefix = stats_dict[0]
        stats_dict = stats_dict[1]

        for tag, value in stats_dict.items():
            self.tblogger.scalar_summary(prefix+tag, value, total_steps(total, epoch, completed))
        self.tblogger.sync_to_file()

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        group = stat_name + '/activations/' + phase + "/"
        for tag, value in activation_stats.items():
            self.tblogger.scalar_summary(group+tag, value, epoch)
        self.tblogger.sync_to_file()

    def log_weights_sparsity(self, model, epoch):
        params_size = 0
        sparse_params_size = 0

        for name, param in model.state_dict().items():
            if param.dim() in [2, 4]:
                _density = density(param)
                params_size += torch.numel(param)
                sparse_params_size += param.numel() * _density
                self.tblogger.scalar_summary('sparsity/weights/' + name,
                                             sparsity(param)*100, epoch)
                self.tblogger.scalar_summary('sparsity-2D/weights/' + name,
                                             sparsity_2D(param)*100, epoch)

        self.tblogger.scalar_summary("sparsity/weights/total", 100*(1 - sparse_params_size/params_size), epoch)
        self.tblogger.sync_to_file()

    def log_weights_filter_magnitude(self, model, epoch, multi_graphs=False):
        """Log the L1-magnitude of the weights tensors.
        """
        for name, param in model.state_dict().items():
            if param.dim() in [4]:
                self.tblogger.list_summary('magnitude/filters/' + name,
                                           list(to_np(norm_filters(param))), epoch, multi_graphs)
        self.tblogger.sync_to_file()

    def log_weights_distribution(self, named_params, steps_completed):
        if named_params is None:
            return
        for tag, value in named_params:
            tag = tag.replace('.', '/')
            if any(substring in tag for substring in self.logged_params):
                self.tblogger.histogram_summary(tag, to_np(value), steps_completed)
            if self.log_gradients:
                self.tblogger.histogram_summary(tag+'/grad', to_np(value.grad), steps_completed)
        self.tblogger.sync_to_file()

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.

        Notes:
            1. Buffers are logged separately per-layer (i.e. module) within model
            2. All values in a single buffer are logged such that they will be displayed on the same graph in
               TensorBoard
            3. Similarly, if multiple buffers are provided in buffer_names, all are presented on the same graph.
               If this is un-desirable, call the function separately for each buffer
            4. USE WITH CAUTION: While sometimes desirable, displaying multiple distinct values in a single
               graph isn't well supported in TensorBoard. It is achieved using a work-around, which slows
               down TensorBoard loading time considerably as the number of distinct values increases.
               Therefore, while not limited, this function is only meant for use with a very limited number of
               buffers and/or values, e.g. 2-5.

        """
        for module_name, module in model.named_modules():
            if distiller.has_children(module):
                continue

            sd = module.state_dict()
            values = []
            for buf_name in buffer_names:
                try:
                    values += sd[buf_name].view(-1).tolist()
                except KeyError:
                    continue

            if values:
                tag = '/'.join([tag_prefix, module_name])
                self.tblogger.list_summary(tag, values, total * epoch + completed, len(values) > 1)
        self.tblogger.sync_to_file()


class CsvLogger(DataLogger):
    def __init__(self, fname_prefix='', logdir=''):
        super(CsvLogger, self).__init__()
        self.logdir = logdir
        self.fname_prefix = fname_prefix

    def get_fname(self, postfix):
        fname = postfix + '.csv'
        if self.fname_prefix:
            fname = self.fname_prefix + '_' + fname
        return os.path.join(self.logdir, fname)

    def log_weights_sparsity(self, model, epoch):
        fname = self.get_fname('weights_sparsity')
        with open(fname, 'w') as csv_file:
            params_size = 0
            sparse_params_size = 0

            writer = csv.writer(csv_file)
            # write the header
            writer.writerow(['parameter', 'shape', 'volume', 'sparse volume', 'sparsity level'])

            for name, param in model.state_dict().items():
                if param.dim() in [2, 4]:
                    _density = density(param)
                    params_size += torch.numel(param)
                    sparse_params_size += param.numel() * _density
                    writer.writerow([name, size_to_str(param.size()),
                                     torch.numel(param),
                                     int(_density * param.numel()),
                                     (1-_density)*100])

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.

        Notes:
            1. Each buffer provided is logged in a separate CSV file
            2. Each CSV file is continuously updated during the run.
            3. In each call, a line is appended for each layer (i.e. module) containing the named buffers.
        """
        with ExitStack() as stack:
            files = {}
            writers = {}
            for buf_name in buffer_names:
                fname = self.get_fname(buf_name)
                new = not os.path.isfile(fname)
                files[buf_name] = stack.enter_context(open(fname, 'a'))
                writer = csv.writer(files[buf_name])
                if new:
                    writer.writerow(['Layer', 'Epoch', 'Step', 'Total', 'Values'])
                writers[buf_name] = writer

            for n, m in model.named_modules():
                for buffer_name in buffer_names:
                    try:
                        p = getattr(m, buffer_name)
                    except AttributeError:
                        continue
                    writer = writers[buffer_name]
                    if isinstance(p, (list, torch.nn.ParameterList)):
                        values = []
                        for v in p:
                            values += v.view(-1).tolist()
                    else:
                        values = p.view(-1).tolist()
                    writer.writerow([distiller.normalize_module_name(n) + '.' + buffer_name,
                                     epoch, completed, int(total)] + values)

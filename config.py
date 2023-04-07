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

"""CompressionScheduler configuration parsing.

There are three ways to configure an instance of CompressionScheduler:

1. Directly create and program scheduling objects.  For example:
    # Element-wise sparasity
    sparsity_levels = {net_param: sparsity_level}
    pruner = distiller.pruning.SparsityLevelParameterPruner(name='sensitivity', levels=sparsity_levels)
    policy = distiller.PruningPolicy(pruner, pruner_args=None)
    scheduler = CompressionScheduler(model)
    scheduler.add_policy(policy, epochs=[0, 2, 4])
2. Create a dictionary containing the configuration.
3. Create a YAML file containing the configuration.

When a YAML file is loaded, its dictionary is extracted and passed to ```dictConfig```.<br>

"""

import logging
from collections import OrderedDict
import yaml
import json
from torch.optim.lr_scheduler import *
import distiller
from distiller.thinning import *
from distiller.pruning import *
from distiller.regularization import *
from distiller.learning_rate import *
from distiller.quantization import *
from distiller.utils import filter_kwargs

msglogger = logging.getLogger()
app_cfg_logger = logging.getLogger("app_cfg")


def dict_config(model, optimizer, sched_dict, scheduler=None, resumed_epoch=None):
    app_cfg_logger.debug('Schedule contents:\n' + json.dumps(sched_dict, indent=2))

    if scheduler is None:
        scheduler = distiller.CompressionScheduler(model)

    pruners = __factory('pruners', model, sched_dict)
    regularizers = __factory('regularizers', model, sched_dict)
    quantizers = __factory('quantizers', model, sched_dict, optimizer=optimizer)
    if len(quantizers) > 1:
        raise ValueError("\nError: Multiple Quantizers not supported")
    extensions = __factory('extensions', model, sched_dict)

    try:
        lr_policies = []
        for policy_def in sched_dict['policies']:
            policy = None
            if 'pruner' in policy_def:
                try:
                    instance_name, args = __policy_params(policy_def, 'pruner')
                except TypeError as e:
                    print('\n\nFatal Error: a policy is defined with a null pruner')
                    print('Here\'s the policy definition for your reference:\n{}'.format(json.dumps(policy_def, indent=1)))
                    raise
                assert instance_name in pruners, "Pruner {} was not defined in the list of pruners".format(instance_name)
                pruner = pruners[instance_name]
                policy = distiller.PruningPolicy(pruner, args)

            elif 'regularizer' in policy_def:
                instance_name, args = __policy_params(policy_def, 'regularizer')
                assert instance_name in regularizers, "Regularizer {} was not defined in the list of regularizers".format(instance_name)
                regularizer = regularizers[instance_name]
                if args is None:
                    policy = distiller.RegularizationPolicy(regularizer)
                else:
                    policy = distiller.RegularizationPolicy(regularizer, **args)

            elif 'quantizer' in policy_def:
                instance_name, args = __policy_params(policy_def, 'quantizer')
                assert instance_name in quantizers, "Quantizer {} was not defined in the list of quantizers".format(instance_name)
                quantizer = quantizers[instance_name]
                policy = distiller.QuantizationPolicy(quantizer)

            elif 'lr_scheduler' in policy_def:
                # LR schedulers take an optimizer in their constructor, so postpone handling until we're certain
                # a quantization policy was initialized (if exists)
                lr_policies.append(policy_def)
                continue

            elif 'extension' in policy_def:
                instance_name, args = __policy_params(policy_def, 'extension')
                assert instance_name in extensions, "Extension {} was not defined in the list of extensions".format(instance_name)
                extension = extensions[instance_name]
                policy = extension

            else:
                raise ValueError("\nFATAL Parsing error while parsing the pruning schedule - unknown policy [%s]".format(policy_def))

            add_policy_to_scheduler(policy, policy_def, scheduler)

        # Any changes to the optimizer caused by a quantizer have occurred by now, so safe to create LR schedulers
        lr_schedulers = __factory('lr_schedulers', model, sched_dict, optimizer=optimizer,
                                  last_epoch=(resumed_epoch if resumed_epoch is not None else -1))
        for policy_def in lr_policies:
            instance_name, args = __policy_params(policy_def, 'lr_scheduler')
            assert instance_name in lr_schedulers, "LR-scheduler {} was not defined in the list of lr-schedulers".format(
                instance_name)
            lr_scheduler = lr_schedulers[instance_name]
            policy = distiller.LRPolicy(lr_scheduler)
            add_policy_to_scheduler(policy, policy_def, scheduler)

    except AssertionError:
        # propagate the assertion information
        raise
    except Exception as exception:
        print("\nFATAL Parsing error!\n%s" % json.dumps(policy_def, indent=1))
        print("Exception: %s %s" % (type(exception), exception))
        raise
    return scheduler


def add_policy_to_scheduler(policy, policy_def, scheduler):
    if 'epochs' in policy_def:
        scheduler.add_policy(policy, epochs=policy_def['epochs'])
    else:
        scheduler.add_policy(policy, starting_epoch=policy_def['starting_epoch'],
                            ending_epoch=policy_def['ending_epoch'],
                            frequency=policy_def['frequency'])


def file_config(model, optimizer, filename, scheduler=None, resumed_epoch=None):
    """Read the schedule from file"""
    with open(filename, 'r') as stream:
        msglogger.info('Reading compression schedule from: %s', filename)
        try:
            sched_dict = distiller.utils.yaml_ordered_load(stream)
            return dict_config(model, optimizer, sched_dict, scheduler, resumed_epoch)
        except yaml.YAMLError as exc:
            print("\nFATAL parsing error while parsing the schedule configuration file %s" % filename)
            raise


def config_component_from_file_by_class(model, filename, class_name, **extra_args):
    with open(filename, 'r') as stream:
        msglogger.info('Reading configuration from: %s', filename)
        try:
            config_dict = distiller.utils.yaml_ordered_load(stream)
            config_dict.pop('policies', None)
            for section_name, components in config_dict.items():
                for component_name, user_args in components.items():
                    if user_args['class'] == class_name:
                        msglogger.info(
                            'Found component of class {0}: Name: {1} ; Section: {2}'.format(class_name, component_name,
                                                                                            section_name))
                        user_args.update(extra_args)
                        return build_component(model, component_name, user_args)
            raise ValueError(
                'Component of class {0} does not exist in configuration file {1}'.format(class_name, filename))
        except yaml.YAMLError:
            print("\nFATAL parsing error while parsing the configuration file %s" % filename)
            raise


def __factory(container_type, model, sched_dict, **extra_args):
    container = {}
    if container_type in sched_dict:
        for name, user_args in sched_dict[container_type].items():
            try:
                instance = build_component(model, name, user_args, **extra_args)
                container[name] = instance
            except Exception as exception:
                print("\nFatal error while parsing [section: %s] [item: %s]" % (container_type, name))
                print("Exception: %s %s" % (type(exception), exception))
                raise

    return container


def build_component(model, name, user_args, **extra_args):
    # Instantiate component using the 'class' argument
    class_name = user_args.pop('class')
    try:
        class_ = globals()[class_name]
    except KeyError as ex:
        raise ValueError("Class named '{0}' does not exist".format(class_name)) from ex

    # First we check that the user defined dict itself does not contain invalid args
    valid_args, invalid_args = filter_kwargs(user_args, class_.__init__)
    if invalid_args:
        raise ValueError(
            '{0} does not accept the following arguments: {1}'.format(class_name, list(invalid_args.keys())))

    # Now we add some "hard-coded" args, which some classes may accept and some may not
    # So then we filter again, this time ignoring any invalid args
    valid_args.update(extra_args)
    valid_args['model'] = model
    valid_args['name'] = name
    final_valid_args, _ = filter_kwargs(valid_args, class_.__init__)
    instance = class_(**final_valid_args)
    return instance


def __policy_params(policy_def, type):
    name = policy_def[type]['instance_name']
    args = policy_def[type].get('args', None)
    return name, args

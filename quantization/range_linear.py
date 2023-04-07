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

import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import argparse
from enum import Enum
from collections import OrderedDict, namedtuple
from functools import reduce, partial, update_wrapper
import logging
import os
from copy import deepcopy
import warnings

import distiller
import distiller.utils
from .quantizer import Quantizer, QBits
from .q_utils import *
from .sim_bn_fold import SimulatedFoldedBatchNorm
import distiller.modules
import distiller.model_transforms as mt
from . import pytorch_quant_conversion as pytqc

import torch.quantization
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq

msglogger = logging.getLogger()


def _quant_param_to_str(val):
    if isinstance(val, torch.Tensor):
        if val.numel() > 1:
            return 'PerCh'
        else:
            return '{:.6f}'.format(val.item())
    return '{:.6f}'.format(val)


def _enum_to_str(enum_val):
    # TODO: This can probably be removed
    if isinstance(enum_val, str): # temporary fix
        return enum_val
    return str(enum_val).split('.')[1]


class ModuleQuantMode(namedtuple('ModuleQuantMode', ['activations', 'weights'])):
    """
    Named tuple for configuring the LinearQuantMode of both weights and activations of a module
    """
    def __new__(cls, activations, weights):
        if not isinstance(activations, LinearQuantMode) or not isinstance(weights, LinearQuantMode):
            raise ValueError('ModuleQuantMode must receive LinearQuantMode values')
        return super(ModuleQuantMode, cls).__new__(cls, activations, weights)


class ClipMode(Enum):
    # No clipping - absolute min/max values will be used
    NONE = 0
    # Clip value calculated by averaging over the max absolute values of samples within a batch
    AVG = 1
    # Clip value calculated as mean of tensor + N standard deviations. N should be specified separately
    N_STD = 2
    # ACIQ Clipping Modes -
    GAUSS = 3
    LAPLACE = 4


def _verify_enum_value(val, enum_cls):
    cls_name = enum_cls.__name__
    if isinstance(val, str):
        try:
            return enum_cls[val]
        except KeyError:
            raise ValueError("Input string '{0}' doesn't match any of the values of {1}: {2}"
                             .format(val, cls_name, [e.name for e in enum_cls]))
    elif isinstance(val, enum_cls):
        return val
    else:
        raise TypeError("Argument can be either a string or member of {0} (got {1})".format(cls_name, val))


def verify_quant_mode(mode):
    if isinstance(mode, ModuleQuantMode):
        return mode

    if isinstance(mode, dict):
        acts = _verify_enum_value(mode['activations'], LinearQuantMode)
        wts = _verify_enum_value(mode['weights'], LinearQuantMode)
    else:
        acts = wts = _verify_enum_value(mode, LinearQuantMode)
    return ModuleQuantMode(acts, wts)


def verify_clip_mode(mode):
    return _verify_enum_value(mode, ClipMode)


def _get_saturation_fn(quant_mode, clip_mode, num_stds, num_bits=None):
    if is_linear_quant_mode_symmetric(quant_mode):
        fns = {ClipMode.NONE: get_tensor_max_abs,
               ClipMode.AVG: get_tensor_avg_max_abs,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_max_abs, n_stds=num_stds),
               ClipMode.GAUSS: AciqSymmetricClipper(num_bits, AciqClipper.AciqClippingType.Gauss),
               ClipMode.LAPLACE: AciqSymmetricClipper(num_bits, AciqClipper.AciqClippingType.Laplace)}
    else:  # Asymmetric mode
        fns = {ClipMode.NONE: get_tensor_min_max,
               ClipMode.AVG: get_tensor_avg_min_max,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_min_max, n_stds=num_stds),
               ClipMode.GAUSS: AciqAsymmetricClipper(num_bits, AciqClipper.AciqClippingType.Gauss),
               ClipMode.LAPLACE: AciqAsymmetricClipper(num_bits, AciqClipper.AciqClippingType.Laplace)}
    return fns[clip_mode]


# TODO: Move to q_utils, add tests
def _get_quant_params_from_tensor(tensor, num_bits, mode, clip=ClipMode.NONE, per_channel=False, num_stds=None,
                                  half_range=False, scale_approx_mult_bits=None):
    if per_channel and tensor.dim() not in [2, 4]:
        raise UnsatisfiedRequirements('Per channel quantization possible only with '
                                      '2D or 4D tensors (linear or conv layer weights)')

    if clip == ClipMode.N_STD:
        if per_channel:
            raise ValueError('N_STD clipping not supported with per-channel quantization')
        if num_stds is None:
            raise UnsatisfiedRequirements('Clip mode set top N_STD but \'num_stds\' parameter not provided')

    dim = 0 if clip == ClipMode.AVG or per_channel else None
    sat_fn = _get_saturation_fn(mode, clip, num_stds, num_bits)
    if is_linear_quant_mode_symmetric(mode):
        sat_val = sat_fn(tensor, dim)
        if isinstance(sat_val, tuple):
            assert len(sat_val) == 2
            sat_val = torch.max(*sat_val)
        scale, zp = symmetric_linear_quantization_params(num_bits, sat_val,
                                                         restrict_qrange=mode == LinearQuantMode.SYMMETRIC_RESTRICTED)
    else:   # Asymmetric mode
        sat_min, sat_max = sat_fn(tensor, dim) if clip not in [ClipMode.GAUSS, ClipMode.LAPLACE] \
            else sat_fn(tensor, dim, half_range=half_range)
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    if per_channel:
        # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
        dims = [scale.shape[0]] + [1] * (tensor.dim() - 1)
        scale = scale.view(dims)
        zp = zp.view(dims)

    if scale_approx_mult_bits is not None:
        scale = approx_scale_as_mult_and_shift(scale, scale_approx_mult_bits)

    return scale, zp


# TODO: Move to q_utils, add tests
def _get_quant_params_from_stats_dict(stats, num_bits, mode, clip=ClipMode.NONE, num_stds=None, half_range=False,
                                      scale_approx_mult_bits=None):
    if clip == ClipMode.N_STD:
        if num_stds is None:
            raise ValueError('Clip mode set to N_STD but \'num_stds\' parameter not provided')
        if num_stds <= 0:
            raise ValueError('n_stds must be > 0, got {}'.format(num_stds))

    prefix = 'avg_' if clip == ClipMode.AVG else ''
    sat_min = torch.tensor(float(stats[prefix + 'min']))
    sat_max = torch.tensor(float(stats[prefix + 'max']))
    if clip == ClipMode.N_STD:
        mean = torch.tensor(float(stats['mean']))
        std = torch.tensor(float(stats['std']))
        sat_min = torch.max(sat_min, mean - num_stds * std)
        sat_max = torch.min(sat_max, mean + num_stds * std)
    elif clip in (ClipMode.LAPLACE, ClipMode.GAUSS):
        clip = AciqClipper.AciqClippingType.Laplace if clip == ClipMode.LAPLACE else AciqClipper.AciqClippingType.Gauss
        if is_linear_quant_mode_symmetric(mode):
            sat_min, sat_max = AciqSymmetricClipper(num_bits, clip)(stats)
        else:
            sat_min, sat_max = AciqAsymmetricClipper(num_bits, clip)(stats, half_range=half_range)

    if is_linear_quant_mode_symmetric(mode):
        scale, zp = symmetric_linear_quantization_params(num_bits, torch.max(sat_min.abs_(), sat_max.abs_()),
                                                         restrict_qrange=mode == LinearQuantMode.SYMMETRIC_RESTRICTED)
    else:
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    if scale_approx_mult_bits is not None:
        scale = approx_scale_as_mult_and_shift(scale, scale_approx_mult_bits)

    return scale, zp


def _get_clipping_values(scale, zp, num_bits, mode):
    """
    Gets the saturation values induced by quantization values
    Args:
        scale, zp (torch.Tensor or float): quantization params
        num_bits (int): number of bits
        mode (LinearQuantMode): mode of quantization
    Returns:
        min, max : tuple[float, float]
    """
    device = scale.device if isinstance(scale, torch.Tensor) else 'cpu'
    if is_linear_quant_mode_asymmetric(mode):
        t = torch.tensor([0, 2**num_bits-1], device=device)
    else:
        t = torch.tensor([-2**(num_bits-1), 2**(num_bits-1)-1], device=device)
    sat_min, sat_max = linear_dequantize(t, scale, zp)  # type: torch.Tensor
    return sat_min, sat_max


###############################################################################
# Post Training
###############################################################################

class TensorQuantMetadata(namedtuple('TensorQuantMetadata', ['scale', 'zero_point', 'min_q_val', 'max_q_val'])):
    __slots__ = ()

    def __str__(self):
        return '(scale={} ; zero_point={})'.format(_quant_param_to_str(self.scale),
                                                   _quant_param_to_str(self.zero_point))


class QuantSettings(object):
    def __init__(self, num_bits, quant_mode, clip_mode, clip_n_stds, clip_half_range, per_channel):
        self.num_bits = num_bits
        self.quant_mode = quant_mode
        self.clip_mode = clip_mode
        self.clip_n_stds = clip_n_stds
        self.clip_half_range = clip_half_range
        self.per_channel = per_channel

    def __str__(self):
        return '(num_bits={} ; quant_mode={} ; clip_mode={} ; clip_n_stds={} ; clip_half_range={}' \
               ' ; per_channel={})'.format(self.num_bits, _enum_to_str(self.quant_mode),
                                           _enum_to_str(self.clip_mode), self.clip_n_stds, self.clip_half_range,
                                           self.per_channel)


def linear_quantize_clamp_with_metadata(t, inplace=False):
    assert hasattr(t, 'quant_metadata')
    qmd = t.quant_metadata
    t = linear_quantize_clamp(t, *qmd, inplace)
    if not inplace:
        t.quant_metadata = qmd
    return t


def linear_dequantize_with_metadata(t, inplace=False):
    assert hasattr(t, 'quant_metadata')
    qmd = t.quant_metadata
    t = linear_dequantize(t, qmd.scale, qmd.zero_point, inplace)
    if not inplace:
        t.quant_metadata = qmd
    return t


def add_post_train_quant_args(argparser, add_lapq_args=False):
    str_to_quant_mode_map = OrderedDict([
        ('sym', LinearQuantMode.SYMMETRIC),
        ('sym_restr', LinearQuantMode.SYMMETRIC_RESTRICTED),
        ('asym_s', LinearQuantMode.ASYMMETRIC_SIGNED),
        ('asym_u', LinearQuantMode.ASYMMETRIC_UNSIGNED)
    ])

    str_to_clip_mode_map = OrderedDict([
        ('none', ClipMode.NONE), ('avg', ClipMode.AVG), ('n_std', ClipMode.N_STD),
        ('gauss', ClipMode.GAUSS), ('laplace', ClipMode.LAPLACE)
    ])

    def from_dict(val_str, d, optional):
        if not val_str and optional:
            return None
        try:
            return d[val_str]
        except KeyError:
            raise argparse.ArgumentTypeError('Must be one of {0} (received {1})'.format(list(d.keys()), val_str))

    linear_quant_mode_str = partial(from_dict, d=str_to_quant_mode_map, optional=False)
    linear_quant_mode_str_optional = partial(from_dict, d=str_to_quant_mode_map, optional=True)
    clip_mode_str = partial(from_dict, d=str_to_clip_mode_map, optional=False)

    group = argparser.add_argument_group('Post-Training Quantization Arguments')
    group.add_argument('--quantize-eval', '--qe', action='store_true',
                       help='Apply linear quantization to model before evaluation. Applicable only if '
                            '--evaluate is also set')
    group.add_argument('--qe-mode', '--qem', type=linear_quant_mode_str, default='sym',
                       help='Default linear quantization mode (for weights and activations). '
                            'Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    group.add_argument('--qe-mode-acts', '--qema', type=linear_quant_mode_str_optional, default=None,
                       help='Linear quantization mode for activations. Overrides --qe-mode`. '
                            'Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    group.add_argument('--qe-mode-wts', '--qemw', type=linear_quant_mode_str_optional, default=None,
                       help='Linear quantization mode for Weights. Overrides --qe-mode`. '
                            'Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    group.add_argument('--qe-bits-acts', '--qeba', type=int, default=8, metavar='NUM_BITS',
                       help='Number of bits for quantization of activations. Use 0 to not quantize activations. '
                            'Default value is 8')
    group.add_argument('--qe-bits-wts', '--qebw', type=int, default=8, metavar='NUM_BITS',
                       help='Number of bits for quantization of weights. Use 0 to not quantize weights. '
                            'Default value is 8')
    group.add_argument('--qe-bits-accum', type=int, default=32, metavar='NUM_BITS',
                       help='Number of bits for quantization of the accumulator')
    group.add_argument('--qe-clip-acts', '--qeca', type=clip_mode_str, default='none',
                       help='Activations clipping mode. Choices: ' + ' | '.join(str_to_clip_mode_map.keys()))
    group.add_argument('--qe-clip-n-stds', type=float,
                       help='When qe-clip-acts is set to \'n_std\', this is the number of standard deviations to use')
    group.add_argument('--qe-no-clip-layers', '--qencl', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                       help='List of layer names for which not to clip activations. Applicable '
                            'only if --qe-clip-acts is not \'none\'')
    group.add_argument('--qe-no-quant-layers', '--qenql', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                        help='List of layer names for which to skip quantization.')
    group.add_argument('--qe-per-channel', '--qepc', action='store_true',
                       help='Enable per-channel quantization of weights (per output channel)')
    group.add_argument('--qe-scale-approx-bits', '--qesab', type=int, metavar='NUM_BITS',
                       help='Enables scale factor approximation using integer multiply + bit shift, using '
                            'this number of bits the integer multiplier')
    group.add_argument('--qe-save-fp-weights', action='store_true',
                       help='Allow weights requantization.')
    group.add_argument('--qe-convert-pytorch', '--qept', action='store_true',
                       help='Convert the model to PyTorch native post-train quantization modules')
    group.add_argument('--qe-pytorch-backend', default='fbgemm', choices=['fbgemm', 'qnnpack'],
                       help='When --qe-convert-pytorch is set, specifies the PyTorch quantization backend to use')

    stats_group = group.add_mutually_exclusive_group()
    stats_group.add_argument('--qe-stats-file', type=str, metavar='PATH',
                             help='Path to YAML file with pre-made calibration stats')
    stats_group.add_argument('--qe-dynamic', action='store_true', help='Apply dynamic quantization')
    stats_group.add_argument('--qe-calibration', type=distiller.utils.float_range_argparse_checker(exc_min=True),
                             metavar='PORTION_OF_TEST_SET', default=None,
                             help='Run the model in evaluation mode on the specified portion of the test dataset and '
                                  'collect statistics')
    stats_group.add_argument('--qe-config-file', type=str, metavar='PATH',
                             help='Path to YAML file containing configuration for PostTrainRLinearQuantizer '
                                  '(if present, all other --qe* arguments are ignored)')

    if add_lapq_args:
        from .ptq_coordinate_search import add_coordinate_search_args
        group.add_argument('--qe-lapq', '--qe-coordinate-search', action='store_true',
                           help='Optimize post-training quantization parameters using LAPQ method')
        add_coordinate_search_args(argparser)


class UnsatisfiedRequirements(Exception):
    pass


def _check_clipping_val(val, quant_mode, half_range):
    if isinstance(val, float):
        if is_linear_quant_mode_symmetric(quant_mode):
            return -val, val
        elif half_range:
            return 0, val
        raise ValueError('For asymmetric quantization, setting clipping values only allowed '
                         'using both min/max values.')
    if isinstance(val, (tuple, list, np.ndarray, torch.Tensor)):
        assert all(distiller.is_scalar(v) for v in val), 'Elements of the clipping value must be scalar-like.'
        assert len(val) == 2, 'Clipping value must have 2 elements.'
        return tuple(val)
    raise TypeError('Clipping value should be a scalar or an iterable of these')


class RangeLinearQuantWrapper(nn.Module):
    """
    Base class for module which wraps an existing module with linear range-base quantization functionality

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        mode (ModuleQuantMode / LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed / unsigned)
        clip_acts (ClipMode): Activations clipping mode to use
        activation_stats (dict): Dict containing activation stats, used for static calculation of quantization
            parameters. Dict should be in the format exported by distiller.data_loggers.QuantCalibrationStatsCollector.
            If None then parameters are calculated dynamically.
        clip_n_stds (float): When clip_acts == ClipMode.N_STD, this is the number of standard deviations to use
        clip_half_range (bool): use half range clipping.
            NOTE - this only works with ACIQ clip modes i.e. GAUSS and LAPLACE
        scale_approx_mult_bits (int): If not None, scale factors will be approximated using an integer multiplication
            followed by a bit-wise shift. This eliminates floating-point scale factors, replacing them with integer
            calculations.
            If None, scale factors will be kept in their original FP32 values.
    """

    def __init__(self, wrapped_module, num_bits_acts, num_bits_accum=32, mode=LinearQuantMode.SYMMETRIC,
                 clip_acts=ClipMode.NONE, activation_stats=None, clip_n_stds=None, clip_half_range=False,
                 scale_approx_mult_bits=None, input_overrides=None, inputs_quant_auto_fallback=False):
        super(RangeLinearQuantWrapper, self).__init__()

        input_overrides = input_overrides or OrderedDict()

        mode = verify_quant_mode(mode)
        self.mode = mode
        self.wrapped_module = wrapped_module
        self.clip_half_range = clip_half_range
        self.scale_approx_mult_bits = scale_approx_mult_bits
        self.inputs_quant_auto_fallback = inputs_quant_auto_fallback

        self.output_quant_settings = QuantSettings(num_bits_acts, mode.activations, clip_acts, clip_n_stds,
                                                   clip_half_range, False)
        self.accum_quant_settings = QuantSettings(num_bits_accum, LinearQuantMode.SYMMETRIC,
                                                  ClipMode.NONE, None, False, False)

        self.preset_act_stats = False
        self.register_buffer('num_forwards', torch.zeros(1, dtype=torch.long))
        self.register_buffer('force_readjust', torch.tensor(False))

        # The accumulator is always signed
        self.accum_min_q_val, self.accum_max_q_val = get_quantized_range(num_bits_accum, signed=True,
                                                                         signed_restrict_qrange=False)

        # Activations not quantized - stop here
        if num_bits_acts is None:
            return

        # Activations are quantized - setup quantization parameters

        # Set-up inputs quantization settings
        self.inputs_quant_settings_overrides = OrderedDict()
        for k, v in input_overrides.items():
            idx = int(k)
            if v.pop('from_output', None):
                quant_settings = deepcopy(self.output_quant_settings)
                quant_settings.clip_half_range = False
            else:
                quant_settings = QuantSettings(
                    v.pop('bits_activations', self.output_quant_settings.num_bits),
                    verify_quant_mode(v.pop('mode', self.output_quant_settings.quant_mode)),
                    verify_clip_mode(v.pop('clip_acts', self.output_quant_settings.clip_mode)),
                    v.pop('clip_n_stds', self.output_quant_settings.clip_n_stds),
                    False, False)
                if v:
                    # Poor man's input checking on input overrides dict
                    raise ValueError('Input overrides dict contains unsupported keys:', list(v.keys()))
            self.inputs_quant_settings_overrides[idx] = quant_settings

        # Controls whether output is de-quantized at end of forward op. Meant as a debug / test flag only
        # (note that if False, the quantized output will be returned, but without any quantization parameters,
        #  so other than inspecting the contents there's not much to do with it)
        self._dequant_out = True

        signed = mode.activations != LinearQuantMode.ASYMMETRIC_UNSIGNED
        restrict_qrange = mode.activations == LinearQuantMode.SYMMETRIC_RESTRICTED
        self.acts_min_q_val, self.acts_max_q_val = get_quantized_range(num_bits_acts, signed=signed,
                                                                       signed_restrict_qrange=restrict_qrange)

        if activation_stats:
            self.preset_act_stats = True

            # Calculate inputs quantization parameters
            self.inputs_quant_metadata_fallback = OrderedDict()
            for idx, stats in activation_stats['inputs'].items():
                settings = self.inputs_quant_settings_overrides.get(idx, self.output_quant_settings)
                scale, zp = _get_quant_params_from_stats_dict(
                    stats, settings.num_bits, settings.quant_mode, settings.clip_mode,
                    settings.clip_n_stds, settings.clip_half_range, self.scale_approx_mult_bits
                )
                min_q_val, max_q_val = get_quantized_range(
                    settings.num_bits, settings.quant_mode != LinearQuantMode.ASYMMETRIC_UNSIGNED,
                    settings.quant_mode == LinearQuantMode.SYMMETRIC_RESTRICTED
                )
                qmd = TensorQuantMetadata(scale, zp, min_q_val, max_q_val)
                self.inputs_quant_metadata_fallback[idx] = qmd

            # Calculate output quantization parameters
            scale, zp = _get_quant_params_from_stats_dict(activation_stats['output'], num_bits_acts, mode.activations,
                                                          clip_acts, clip_n_stds, clip_half_range,
                                                          scale_approx_mult_bits)
            if not isinstance(scale, torch.Tensor):
                scale, zp = torch.tensor(scale), torch.tensor(zp)
            self.register_buffer('output_scale', scale)
            self.register_buffer('output_zero_point', zp)
        else:
            self.preset_act_stats = False

    def named_linear_quant_params(self, filter=False):
        if self.output_quant_settings.num_bits is not None and self.preset_act_stats:
            # Output scale buffers are saved in the model only when stats are used
            yield 'output_scale', self.output_scale
            if not filter or (is_linear_quant_mode_asymmetric(self.mode.activations) and not self.clip_half_range):
                yield 'output_zero_point', self.output_zero_point

    def set_linear_quant_param(self, name, val):
        if name in dict(self.named_clipping()):
            setattr(self, name, val)
        elif name not in dict(self.named_linear_quant_params()):
            raise ValueError('%s is not a quantization parameter.' % name)
        else:
            getattr(self, name).data.fill_(val)
        self.force_readjust.fill_(True)

    def _check_requirements_output_clipping(self):
        if not self.output_quant_settings.num_bits:
            raise UnsatisfiedRequirements('Cannot retrieve clipping values because '
                                          'the activations aren\'t quantized.')
        if not self.preset_act_stats:
            raise UnsatisfiedRequirements('Cannot retrieve clipping values '
                                          'because the activations stats were not provided.')

    @property
    def output_clipping(self):
        self._check_requirements_output_clipping()
        bits = self.output_quant_settings.num_bits
        scale, zp = self.output_scale, self.output_zero_point
        return _get_clipping_values(scale, zp, bits, self.output_quant_settings.quant_mode)

    @output_clipping.setter
    def output_clipping(self, val):
        """
        Args:
            val (float or tuple[float, float] or tuple[torch.Tensor, torch.Tensor]): the value to set
        """
        self._check_requirements_output_clipping()
        qset = self.output_quant_settings
        val_min, val_max = _check_clipping_val(val, qset.quant_mode, self.clip_half_range)
        qset.clip_mode, qset.clip_half_range, qset.clip_n_stds = ClipMode.NONE, None, None
        scale, zp = _get_quant_params_from_stats_dict({'min': val_min, 'max': val_max}, qset.num_bits, qset.quant_mode,
                                                      scale_approx_mult_bits=self.scale_approx_mult_bits)
        self.set_linear_quant_param('output_scale', scale.item())
        self.set_linear_quant_param('output_zero_point', zp.item())

    def named_clipping(self, filter=False):
        val = self.output_clipping
        if filter and (is_linear_quant_mode_symmetric(self.mode.activations) or self.clip_half_range):
            val = val[1]
        yield 'output_clipping', val

    def update_linear_quant_params(self, new_config):
        """
        Updates all the quant params using a dictionary.
        Args:
             new_config (dict): the new configuration dict.
        """
        for name, val in new_config.items():
            self.set_linear_quant_param(name, val)

    def forward(self, *inputs):
        if self.training:
            raise RuntimeError(self.__class__.__name__ + " can only be used in eval mode")

        if self.output_quant_settings.num_bits is None:
            # Pass through
            out = self.wrapped_module(*inputs)
            if self.clip_half_range:
                out = f.relu(out)
            return out

        device = inputs[0].device
        for buffer_name, buffer in self._buffers.items():
            setattr(self, buffer_name, buffer.to(device))

        inputs_q = [self._prepare_input(idx, input) for idx, input in enumerate(inputs)]

        # Forward through wrapped module
        accum = self.quantized_forward(*inputs_q)

        if self.clip_half_range:
            accum = f.relu(accum)

        # Re-quantize accumulator to quantized output range
        out_scale, out_zero_point = self.get_output_quantization_params(accum)
        requant_scale, requant_zero_point = self.get_accum_to_output_re_quantization_params(out_scale, out_zero_point)
        out_q = linear_quantize_clamp(accum.data, requant_scale, requant_zero_point,
                                      self.acts_min_q_val, self.acts_max_q_val, inplace=True)

        if not self._dequant_out:
            return torch.autograd.Variable(out_q)

        # De-quantize back to FP32
        out_f = linear_dequantize(out_q, out_scale, out_zero_point, inplace=True)

        out_f.quant_metadata = TensorQuantMetadata(out_scale, out_zero_point, self.acts_min_q_val, self.acts_max_q_val)

        self.num_forwards += 1

        return out_f

    def _prepare_input(self, idx, input):
        # Default implementation - quantize the input tensor
        # This works for all but RangeLinearFakeQuantWrapper
        input.quant_metadata = self._get_input_quant_metadata(idx, input)
        return linear_quantize_clamp_with_metadata(input, inplace=False)

    def _get_input_quant_metadata(self, idx, input):
        if hasattr(input, 'quant_metadata'):
            if idx in self.inputs_quant_settings_overrides:
                raise RuntimeError('<{}> Input {}: CONFLICT - Tensor has embedded quantization metadata AND user '
                                   'defined input quantization settings'.format(self.distiller_name, idx))
            qmd = input.quant_metadata
        else:
            # Input doesn't have embedded quantization data propagated from a previous layer
            # Our options are:
            #  If user set explicit settings for this input, use those
            #  OR
            #  If auto fallback is set, use the output quantization settings
            if idx not in self.inputs_quant_settings_overrides and not self.inputs_quant_auto_fallback:
                raise RuntimeError('<{}> Input {}: Expected tensor with embedded quantization metadata. Either:\n'
                                   '1. Make sure the previous operation is quantized\n'
                                   '2. Provide explicit input quantization settings\n'
                                   '3. Set inputs_quant_auto_fallback'.format(self.distiller_name, idx))
            if self.preset_act_stats:
                qmd = self.inputs_quant_metadata_fallback[idx]
            else:
                if idx in self.inputs_quant_settings_overrides:
                    q_settings = self.inputs_quant_settings_overrides[idx]
                else:
                    # If we're here then inputs_quant_auto_fallback is set
                    q_settings = deepcopy(self.output_quant_settings)
                    q_settings.clip_half_range = False
                scale, zp = _get_quant_params_from_tensor(input, q_settings.num_bits, q_settings.quant_mode,
                                                          q_settings.clip_mode, q_settings.per_channel,
                                                          q_settings.clip_n_stds, q_settings.clip_half_range,
                                                          self.scale_approx_mult_bits)
                signed = q_settings.quant_mode != LinearQuantMode.ASYMMETRIC_UNSIGNED
                restrict_qrange = q_settings.quant_mode == LinearQuantMode.SYMMETRIC_RESTRICTED
                min_q_val, max_q_val = get_quantized_range(q_settings.num_bits, signed, restrict_qrange)
                qmd = TensorQuantMetadata(scale, zp, min_q_val, max_q_val)

        # Make sure scale and zp are on correct device
        qmd = TensorQuantMetadata(qmd.scale.to(input.device), qmd.zero_point.to(input.device),
                                  qmd.min_q_val, qmd.max_q_val)
        return qmd

    def quantized_forward(self, *inputs_q):
        """
        Perform forward pass with quantized inputs and return quantized outputs

        :param inputs_q: Tensor (or list of tensors) with quantized input values
        :return: Tensor with quantized output values
        """
        raise NotImplementedError

    def get_output_quantization_params(self, accumulator):
        """
        Calculate quantization parameters (scale and zero-point) for the output.
        This is used for:
          * Calculating the accumulator-to-output re-quantization parameters
            (see get_accum_to_output_re_quantization_params)
          * De-quantizing the output back to FP32

        Should be overridden by all subclasses

        :param accumulator: Tensor with accumulator values
        :return: Tuple of scale and zero-point
        """
        raise NotImplementedError

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        """
        Calculate quantization parameters (scale and zero-point) for re-quantization, that is:
        Converting the intermediate integer accumulator to the output range

        Should be overridden by all subclasses

        :param output_scale: Output scale factor
        :param output_zero_point: Output zero-point
        :return: Tuple of scale and zero-point
        """
        raise NotImplementedError

    def to_pytorch_quant(self, reduce_range):
        assert self.output_quant_settings.num_bits == 8, \
            'Conversion to PyTorch PTQ supported only for 8-bit quantization'
        assert self.preset_act_stats, 'Conversion to PyTorch PTQ supported only for PTQ wrappers with activation stats'
        return self._convert_to_pytorch_quant(reduce_range)

    def _convert_to_pytorch_quant(self, reduce_range):
        raise NotImplementedError

    def extra_repr(self):
        if self.output_quant_settings.num_bits is None:
            return 'output_quant_settings=Not_Quantized'

        tmpstr = 'output_quant_settings={0}'.format(self.output_quant_settings)
        tmpstr += '\naccum_quant_settings={0}'.format(self.accum_quant_settings)
        overrides = self.inputs_quant_settings_overrides
        tmpstr += '\n  inputs_quant_auto_fallback={}'.format(self.inputs_quant_auto_fallback)
        tmpstr += ', forced_quant_settings_for_inputs={}'.format(
            'None' if not overrides else list(overrides.keys()))
        for idx, qset in overrides.items():
            tmpstr += '\n    input_{}_settings={}'.format(idx, qset)
        tmpstr += '\nscale_approx_mult_bits={}'.format(self.scale_approx_mult_bits)
        tmpstr += '\npreset_activation_stats={0}'.format(self.preset_act_stats)
        if self.preset_act_stats:
            tmpstr += '\n  output_scale={0}, output_zero_point={1}'.format(_quant_param_to_str(
                self.output_scale), _quant_param_to_str(self.output_zero_point))
            for idx in self.inputs_quant_settings_overrides:
                qmd = self.inputs_quant_metadata_fallback[idx]
                tmpstr += '\n  input_#{0}_scale={1}, input_#{0}_zero_point={2}'.format(
                    idx, _quant_param_to_str(qmd.scale), _quant_param_to_str(qmd.zero_point))
        return tmpstr


class RangeLinearQuantParamLayerWrapper(RangeLinearQuantWrapper):
    """
    Linear range-based quantization wrappers for layers with weights and bias (namely torch.nn.ConvNd and
    torch.nn.Linear)

    Assume:

    x_q = round(scale_x * x_f) - zero_point_x

    Hence:

    x_f = 1/scale_x * x_q + zero_point_x

    (And the same for y_q, w_q and b_q)

    So, we get: (use "zp" as abbreviation for zero_point)

    y_f = x_f * w_f + b_f

    y_q = scale_y * y_f + zp_y =  scale_y * (x_f * w_f + b_f) + zp_y =

                scale_y                                         scale_x * scale_w
        = ------------------- * ((x_q + zp_x) * (w_q + zp_w) + ------------------- * (b_q + zp_b)) + zp_y
           scale_x * scale_w                                         scale_b

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_params (int): Number of bits used for parameters (weights and bias) quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        mode (ModuleQuantMode / LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed/unsigned)
        clip_acts (ClipNode): See RangeLinearQuantWrapper
        per_channel_wts (bool): Enable quantization of weights using separate quantization parameters per
            output channel
        activation_stats (dict): See RangeLinearQuantWrapper
        clip_n_stds (float): See RangeLinearQuantWrapper
        clip_half_range (bool) : See RangeLinearQuantWrapper
        scale_approx_mult_bits (int): See RangeLinearQuantWrapper
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32,
                 mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE, per_channel_wts=False, activation_stats=None,
                 clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None, input_overrides=None,
                 inputs_quant_auto_fallback=False, save_fp_weights=False, also_clip_weights=False):
        super(RangeLinearQuantParamLayerWrapper, self).__init__(wrapped_module, num_bits_acts, num_bits_accum, mode,
                                                                clip_acts, activation_stats, clip_n_stds, clip_half_range,
                                                                scale_approx_mult_bits,
                                                                input_overrides=input_overrides,
                                                                inputs_quant_auto_fallback=inputs_quant_auto_fallback)

        if not isinstance(wrapped_module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            raise ValueError(self.__class__.__name__ + ' can wrap only Conv2D, Conv3D and Linear modules')

        # If activations are not quantized, we do fake quantization of the parameters, that is - quant and de-quant
        self.fake_quant_params = self.output_quant_settings.num_bits is None

        clip_wts_mode, clip_wts_n_stds = ClipMode.NONE, None
        if also_clip_weights:
            clip_wts_mode = self.output_quant_settings.clip_mode
            clip_wts_n_stds = self.output_quant_settings.clip_n_stds
        self.wts_quant_settings = QuantSettings(num_bits_params, self.mode.weights, clip_wts_mode, clip_wts_n_stds,
                                                False, per_channel_wts)

        self.params_min_q_val, self.params_max_q_val = get_quantized_range(
            self.wts_quant_settings.num_bits,
            self.wts_quant_settings.quant_mode != LinearQuantMode.ASYMMETRIC_UNSIGNED,
            self.wts_quant_settings.quant_mode == LinearQuantMode.SYMMETRIC_RESTRICTED
        )
        self.save_fp_weights = save_fp_weights
        # save the float weight to allow re-quantizing
        if save_fp_weights:
            wrapped_module.register_buffer('float_weight', wrapped_module.weight.clone().detach())

        # Quantize weights - overwrite FP32 weights
        w_scale, w_zero_point = _get_quant_params_from_tensor(wrapped_module.weight,
                                                              self.wts_quant_settings.num_bits,
                                                              self.wts_quant_settings.quant_mode,
                                                              clip=self.wts_quant_settings.clip_mode,
                                                              per_channel=self.wts_quant_settings.per_channel,
                                                              num_stds=self.wts_quant_settings.clip_n_stds)
        w_scale = w_scale if isinstance(w_scale, torch.Tensor) else torch.tensor(w_scale)
        w_zero_point = w_zero_point if isinstance(w_zero_point, torch.Tensor) else torch.tensor(w_zero_point)

        self.register_buffer('w_scale', w_scale)
        self.register_buffer('w_zero_point', w_zero_point)
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.w_zero_point, self.params_min_q_val,
                              self.params_max_q_val, inplace=True)

        # Quantize bias
        self.has_bias = hasattr(wrapped_module, 'bias') and wrapped_module.bias is not None
        if self.has_bias and (self.fake_quant_params or not self.preset_act_stats):
            b_scale, b_zero_point = _get_quant_params_from_tensor(wrapped_module.bias,
                                                                  self.accum_quant_settings.num_bits,
                                                                  self.accum_quant_settings.quant_mode)
            self.register_buffer('b_scale', b_scale)
            self.register_buffer('b_zero_point', b_zero_point)
            base_b_q = linear_quantize_clamp(wrapped_module.bias.data, self.b_scale, self.b_zero_point,
                                             self.accum_min_q_val, self.accum_max_q_val)
            if not self.preset_act_stats:
                # Dynamic ranges - save in auxiliary buffer,
                # requantize each time based on dynamic input scale factor
                self.register_buffer('base_b_q', base_b_q)

        # allow requantizing the bias:
        if self.has_bias and self.preset_act_stats:
            self.register_buffer('fp_bias', self.wrapped_module.bias.data.clone().detach())

        # Activations not quantized - de-quant parameters and return
        if self.fake_quant_params:
            linear_dequantize(wrapped_module.weight.data, self.w_scale, self.w_zero_point, inplace=True)
            if self.has_bias:
                wrapped_module.bias = torch.nn.Parameter(linear_dequantize(base_b_q, self.b_scale, self.b_zero_point))
            return

        # Activations are quantized - setup accumulator quantization parameters
        device = self.w_scale.device
        if self.preset_act_stats:
            t = torch.empty_like(self.w_scale)
            if self.wts_quant_settings.per_channel:
                t = t.squeeze(dim=-1)
            self.register_buffer('accum_scale', t)
        else:
            self.accum_scale = torch.ones(1).to(device)

        # A flag indicating that the simulated quantized weights are pre-shifted. for faster performance.
        # In the first forward pass - `w_zero_point` is added into the weights, to allow faster inference,
        # and all subsequent calls are done with these shifted weights.
        # Upon calling `self.state_dict()` - we restore the actual quantized weights.
        # i.e. is_simulated_quant_weight_shifted = False
        self.register_buffer('is_simulated_quant_weight_shifted', torch.tensor(False, device=device))

    def named_linear_quant_params(self, filter=False):
        if self.save_fp_weights:
            yield 'w_scale', self.w_scale
            if not filter or is_linear_quant_mode_asymmetric(self.mode.weights):
                yield 'w_zero_point', self.w_zero_point
        yield from super(RangeLinearQuantParamLayerWrapper, self).named_linear_quant_params(filter=filter)

    def set_linear_quant_param(self, name, val):
        if name in ['w_scale', 'w_zero_point']:
            if self.save_fp_weights:
                super().set_linear_quant_param(name, val)
                self.wrapped_module.weight.data.copy_(self.wrapped_module.float_weight.data)
                linear_quantize_clamp(self.wrapped_module.weight.data, self.w_scale, self.w_zero_point,
                                      self.params_min_q_val,
                                      self.params_max_q_val, inplace=True)
                if self.fake_quant_params:
                    linear_dequantize(self.wrapped_module.weight.data, self.w_scale, self.w_zero_point, inplace=True)

            else:
                raise UnsatisfiedRequirements('Cannot re-quantize the weights. Please specify \'save_fp_weights\' in '
                                              'the %s constructor to enable re-quantizing the weights.' %
                                              self.__class__.__name__)
        else:
            super().set_linear_quant_param(name, val)

    def _check_requirements_weights_clipping(self, setter=False):
        if not self.wts_quant_settings.num_bits:
            raise UnsatisfiedRequirements('Cannot retrieve clipping values because the weights aren\'t quantized.')
        if setter and not self.save_fp_weights:
            warnings.warn('Without saving fp32 version of weights, re-quantization is disabled. To enable, '
                          'please set \'save_fp_weights\' while constructing the wrapper.')

    @property
    def weight_clipping(self):
        self._check_requirements_weights_clipping(setter=False)
        bits, mode = self.wts_quant_settings.num_bits, self.wts_quant_settings.quant_mode
        scale, zp = self.w_scale, self.w_zero_point
        return _get_clipping_values(scale, zp, bits, mode)

    @weight_clipping.setter
    def weight_clipping(self, val):
        self._check_requirements_weights_clipping(setter=True)
        bits = self.wts_quant_settings.num_bits
        val_min, val_max = _check_clipping_val(val, self.wts_quant_settings.quant_mode, False)
        if is_linear_quant_mode_symmetric(self.wts_quant_settings.quant_mode):
            # in symmetric quantization - we only need one value
            scale, zp = symmetric_linear_quantization_params(bits, abs(max(val_min, val_max)))
        else:
            signed = self.wts_quant_settings.quant_mode == LinearQuantMode.ASYMMETRIC_SIGNED
            scale, zp = asymmetric_linear_quantization_params(bits, val_min, val_max, signed=signed)
        self.set_linear_quant_param('w_scale', scale)
        self.set_linear_quant_param('w_zero_point', zp)

    def named_clipping(self, filter=False):
        try:
            yield from super().named_clipping(filter=filter)
        except UnsatisfiedRequirements as ex:
            warnings.warn(str(ex))   # probably the output isn't quantized
        val = self.weight_clipping
        if filter and is_linear_quant_mode_symmetric(self.mode.weights):
            val = val[1]
        yield 'weight_clipping', val

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if not self.fake_quant_params and self.is_simulated_quant_weight_shifted:
            # We want to return the weights to their integer representation:
            self.wrapped_module.weight.data -= self.w_zero_point
            self.is_simulated_quant_weight_shifted.fill_(False) # i.e. is_simulated_quant_weight_shifted = False
        return super(RangeLinearQuantParamLayerWrapper, self).state_dict(destination, prefix, keep_vars)

    def quantized_forward(self, input_q):
        # See class documentation for quantized calculation details.

        def get_accum_scale(input_q):
            accum_scale = input_q.quant_metadata.scale * self.w_scale
            if self.wts_quant_settings.per_channel:
                accum_scale = accum_scale.squeeze(dim=-1)
            if self.scale_approx_mult_bits:
                accum_scale = approx_scale_as_mult_and_shift(accum_scale, self.scale_approx_mult_bits)
            return accum_scale

        if self.preset_act_stats:
            if self.num_forwards == 0 or self.force_readjust:
                self.accum_scale.copy_(get_accum_scale(input_q))
                if self.has_bias:
                    # Requantize bias to accumulator scale "permanently"
                    self.wrapped_module.bias.data.copy_(
                        linear_quantize_clamp(self.fp_bias, self.accum_scale.squeeze(), 0,
                                              self.accum_min_q_val, self.accum_max_q_val)
                    )
                self.force_readjust.fill_(False)
        else:
            self.accum_scale = get_accum_scale(input_q)
            if self.has_bias:
                # Re-quantize bias to match x * w scale: b_q' = (in_scale * w_scale / b_scale) * (b_q + b_zero_point)
                bias_requant_scale = self.accum_scale.squeeze() / self.b_scale
                if self.scale_approx_mult_bits is not None:
                    bias_requant_scale = approx_scale_as_mult_and_shift(bias_requant_scale, self.scale_approx_mult_bits)
                self.wrapped_module.bias.data = linear_quantize_clamp(self.base_b_q + self.b_zero_point,
                                                                      bias_requant_scale, 0,
                                                                      self.accum_min_q_val, self.accum_max_q_val)

        # Note the main terms within the summation is:
        #   (x_q + zp_x) * (w_q + zp_w)
        # In a performance-optimized solution, we would expand the parentheses and perform the computation similar
        # to what is described here:
        #   https://github.com/google/gemmlowp/blob/master/doc/low-precision.md#efficient-handling-of-offsets
        # However, for now we're more concerned with simplicity rather than speed. So we'll just add the zero points
        # to the input and weights and pass those to the wrapped model. Functionally, since at this point we're
        # dealing solely with integer values, the results are the same either way.

        if is_linear_quant_mode_asymmetric(self.wts_quant_settings.quant_mode) and \
                not self.is_simulated_quant_weight_shifted:
            # We "store" the w_zero_point inside our wrapped module's weights to
            # improve performance on inference.
            self.wrapped_module.weight.data += self.w_zero_point
            self.is_simulated_quant_weight_shifted.fill_(True)  # i.e. is_simulated_quant_weight_shifted = True

        input_q += input_q.quant_metadata.zero_point
        accum = self.wrapped_module(input_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        return accum

    def get_output_quantization_params(self, accumulator):
        if self.preset_act_stats:
            return self.output_scale, self.output_zero_point

        y_f = accumulator / self.accum_scale
        q_set = self.output_quant_settings
        return _get_quant_params_from_tensor(y_f, q_set.num_bits, q_set.quant_mode,
                                             clip=q_set.clip_mode, num_stds=q_set.clip_n_stds,
                                             half_range=q_set.clip_half_range,
                                             scale_approx_mult_bits=self.scale_approx_mult_bits)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        requant_scale = output_scale / self.accum_scale
        if self.scale_approx_mult_bits is not None:
            requant_scale = approx_scale_as_mult_and_shift(requant_scale, self.scale_approx_mult_bits)
        return requant_scale, output_zero_point

    def _convert_to_pytorch_quant(self, reduce_range):
        wrapped = self.wrapped_module
        supported = (nn.Conv2d, nn.Linear)
        # Tuple of module type and flag for relu fusing
        mapping = {
            (nn.Linear, False): nnq.Linear,
            (nn.Linear, True): nniq.LinearReLU,
            (nn.Conv2d, False): nnq.Conv2d,
            (nn.Conv2d, True): nniq.ConvReLU2d
        }
        if nn.Conv3d in torch.quantization.DEFAULT_MODULE_MAPPING:
            # Conv3D supported only from PyTorch 1.4
            supported += nn.Conv3d,
            mapping.update({
                (nn.Conv3d, False): nnq.Conv3d,
                (nn.Conv3d, True): nniq.ConvReLU3d,
            })
        assert isinstance(wrapped, supported), \
            'Conversion to PyTorch PTQ supported only for {}'.format(','.join(supported))
        assert self.wts_quant_settings.num_bits == 8, 'Conversion to PyTorch PTQ supported only for 8-bit quantization'

        # Convert weights - required by PyTorch to be signed 8-bit (torch.qint8)
        q_weight = pytqc.distiller_quantized_tensor_to_pytorch(wrapped.weight.clone().detach(),
                                                               self.w_scale, self.w_zero_point,
                                                               self.wts_quant_settings.num_bits,
                                                               self.wts_quant_settings.quant_mode, torch.qint8,
                                                               self.wts_quant_settings.per_channel, 0)

        # PyTorch PTQ modules expect the bias in FP32, we need to dequantize if necessary
        # With Distiller PTQ the bias is only quantized on the first forward - we do a crude check if it has
        # been quantized or not
        fp_bias = wrapped.bias.clone().detach()
        if self.has_bias:
            bias_quantized = (fp_bias == fp_bias.int()).all()
            if bias_quantized:
                fp_bias = linear_dequantize(fp_bias, self.accum_scale.squeeze(), 0, True)

        pytorch_cls = mapping[(type(wrapped), self.clip_half_range)]
        if isinstance(wrapped, nn.Linear):
            pytorch_module = pytorch_cls(wrapped.in_features, wrapped.out_features, wrapped.bias is not None)
        else:
            pytorch_module = pytorch_cls(wrapped.in_channels, wrapped.out_channels, wrapped.kernel_size,
                                         wrapped.stride, wrapped.padding, wrapped.dilation, wrapped.groups,
                                         wrapped.bias is not None, wrapped.padding_mode)

        pytorch_module.set_weight_bias(q_weight, fp_bias)

        # Convert activations qparams - required by PyTorch to be unsigned 8-bit (torch.quint8)
        out_scale, out_zp = pytqc.distiller_qparams_to_pytorch(self.output_scale, self.output_zero_point,
                                                               self.output_quant_settings.num_bits,
                                                               self.output_quant_settings.quant_mode, torch.quint8,
                                                               reduce_range)
        pytorch_module.scale = float(out_scale)
        pytorch_module.zero_point = int(out_zp)

        return pytorch_module

    def extra_repr(self):
        tmpstr = 'weights_quant_settings={0}\n'.format(self.wts_quant_settings)
        tmpstr += super(RangeLinearQuantParamLayerWrapper, self).extra_repr()
        tmpstr += '\nweights_scale={0}, weights_zero_point={1}'.format(_quant_param_to_str(self.w_scale),
                                                                       _quant_param_to_str(self.w_zero_point))
        if not self.preset_act_stats and self.has_bias:
            tmpstr += '\nbase_bias_scale={0}, base_bias_zero_point={1}'.format(_quant_param_to_str(self.b_scale),
                                                                               _quant_param_to_str(self.b_zero_point))
        return tmpstr


class RangeLinearQuantMatmulWrapper(RangeLinearQuantWrapper):
    """
    Wrapper for quantizing the Matmul/BatchMatmul operation between 2 input tensors.
    output = input1 @ input2
    where:
        input1.shape = (input_batch, input_size)
        input2.shape = (input_size, output_size)
    The mathematical calculation is:
        y_f = i1_f * i2_f
        iN_f = iN_q / scale_iN + zp_iN =>
        y_q = scale_y * y_f + zp_y =  scale_y * (i1_f * i2_f) + zp_y =

                    scale_y
        y_q = ------------------- * ((i1_q + zp_i1) * (i2_q + zp_i2) + zp_y
               scale_i1 * scale_i2
    Args:
        wrapped_module (distiller.modules.Matmul or distiller.modules.BatchMatmul): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        mode (ModuleQuantMode / LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed/unsigned)
        clip_acts (ClipNode): See RangeLinearQuantWrapper
        activation_stats (dict): See RangeLinearQuantWrapper
        clip_n_stds (int): See RangeLinearQuantWrapper
        scale_approx_mult_bits (int): See RangeLinearQuantWrapper
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_accum=32,
                 mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE, activation_stats=None,
                 clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None,
                 input_overrides=None, inputs_quant_auto_fallback=False):
        super(RangeLinearQuantMatmulWrapper, self).__init__(wrapped_module, num_bits_acts, num_bits_accum, mode,
                                                            clip_acts, activation_stats, clip_n_stds, clip_half_range,
                                                            scale_approx_mult_bits,
                                                            input_overrides=input_overrides,
                                                            inputs_quant_auto_fallback=inputs_quant_auto_fallback)

        if not isinstance(wrapped_module, (distiller.modules.Matmul, distiller.modules.BatchMatmul)):
            raise ValueError(self.__class__.__name__ + ' can wrap only Matmul modules')
        self.accum_scale = 1

    def quantized_forward(self, input0_q, input1_q):
        self.accum_scale = input0_q.quant_metadata.scale * input1_q.quant_metadata.scale
        accum = self.wrapped_module(input0_q + input0_q.quant_metadata.zero_point,
                                    input1_q + input1_q.quant_metadata.zero_point)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)
        return accum

    def get_output_quantization_params(self, accumulator):
        if self.preset_act_stats:
            return self.output_scale, self.output_zero_point

        y_f = accumulator / self.accum_scale
        q_set = self.output_quant_settings
        return _get_quant_params_from_tensor(y_f, q_set.num_bits, q_set.quant_mode,
                                             clip=q_set.clip_mode, num_stds=q_set.clip_n_stds,
                                             half_range=q_set.clip_half_range,
                                             scale_approx_mult_bits=self.scale_approx_mult_bits)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        requant_scale = output_scale / self.accum_scale
        if self.scale_approx_mult_bits is not None:
            requant_scale = approx_scale_as_mult_and_shift(requant_scale, self.scale_approx_mult_bits)
        return requant_scale, output_zero_point

    def _convert_to_pytorch_quant(self, reduce_range):
        # Convert activations qparams - required by PyTorch to be unsigned 8-bit (torch.quint8)
        scale, zp = pytqc.distiller_qparams_to_pytorch(self.output_scale, self.output_zero_point,
                                                       self.output_quant_settings.num_bits,
                                                       self.output_quant_settings.quant_mode, torch.quint8,
                                                       reduce_range)
        modules = [self.wrapped_module, nnq.Quantize(float(scale), int(zp), torch.quint8)]
        if self.clip_half_range:
            # The scale factor calculated in Distiller already considers the ReLU, so it's OK to apply the
            # ReLU after quantization
            modules.append(nnq.ReLU())
        return modules


class NoStatsError(NotImplementedError):
    pass


class RangeLinearQuantConcatWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None,
                 input_overrides=None, inputs_quant_auto_fallback=False):
        if not isinstance(wrapped_module, distiller.modules.Concat):
            raise ValueError(self.__class__.__name__ + ' can only wrap distiller.modules.Concat modules')

        if not activation_stats:
            raise NoStatsError(self.__class__.__name__ +
                               ' must get activation stats, dynamic quantization not supported')

        super(RangeLinearQuantConcatWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                            clip_acts=clip_acts, activation_stats=activation_stats,
                                                            clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                            scale_approx_mult_bits=scale_approx_mult_bits,
                                                            input_overrides=input_overrides,
                                                            inputs_quant_auto_fallback=inputs_quant_auto_fallback)

    def quantized_forward(self, *inputs_q):
        # For concatenation to make sense input scales need to match, so we re-quantize all inputs
        # based on the output scale
        inputs_re_q = [linear_quantize_clamp(input_q + input_q.quant_metadata.zero_point,
                                             self.output_scale / input_q.quant_metadata.scale, 0.,
                                             self.accum_min_q_val, self.accum_max_q_val, inplace=False)
                       for input_q in inputs_q]
        return self.wrapped_module(*inputs_re_q)

    def get_output_quantization_params(self, accumulator):
        return self.output_scale, self.output_zero_point

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        # Nothing to do here, since we already re-quantized in quantized_forward prior to the actual concatenation
        return 1., self.output_zero_point

    def _convert_to_pytorch_quant(self, reduce_range):
        # Convert activations qparams - required by PyTorch to be unsigned 8-bit (torch.quint8)
        scale, zp = pytqc.distiller_qparams_to_pytorch(self.output_scale, self.output_zero_point,
                                                       self.output_quant_settings.num_bits,
                                                       self.output_quant_settings.quant_mode, torch.quint8,
                                                       reduce_range)
        m = pytqc.QFunctionalCat(self.wrapped_module.dim)
        m.qfunc.scale = float(scale)
        m.qfunc.zero_point = int(zp)
        if self.clip_half_range:
            # The scale factor calculated in Distiller already considers the ReLU, so it's OK to apply the
            # ReLU after quantization
            m = nn.Sequential(m, nnq.ReLU())
        return m


class RangeLinearQuantEltwiseAddWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None,
                 input_overrides=None, inputs_quant_auto_fallback=False):
        if not isinstance(wrapped_module, distiller.modules.EltwiseAdd):
            raise ValueError(self.__class__.__name__ + ' can only wrap distiller.modules.EltwiseAdd modules')

        if not activation_stats:
            raise NoStatsError(self.__class__.__name__ +
                               ' must get activation stats, dynamic quantization not supported')

        super(RangeLinearQuantEltwiseAddWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                                clip_acts=clip_acts, activation_stats=activation_stats,
                                                                clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                                scale_approx_mult_bits=scale_approx_mult_bits,
                                                                input_overrides=input_overrides,
                                                                inputs_quant_auto_fallback=inputs_quant_auto_fallback)

    def quantized_forward(self, *inputs_q):
        # Re-scale inputs to the accumulator scale
        inputs_re_q = [linear_quantize_clamp(input_q + input_q.quant_metadata.zero_point,
                                             self.output_scale / input_q.quant_metadata.scale, 0,
                                             self.accum_min_q_val, self.accum_max_q_val, inplace=False)
                       for input_q in inputs_q]
        accum = self.wrapped_module(*inputs_re_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        return accum

    def get_output_quantization_params(self, accumulator):
        return self.output_scale, self.output_zero_point

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        return 1., self.output_zero_point

    def _convert_to_pytorch_quant(self, reduce_range):
        # Convert activations qparams - required by PyTorch to be unsigned 8-bit (torch.quint8)
        scale, zp = pytqc.distiller_qparams_to_pytorch(self.output_scale, self.output_zero_point,
                                                       self.output_quant_settings.num_bits,
                                                       self.output_quant_settings.quant_mode, torch.quint8,
                                                       reduce_range)
        m = pytqc.QFunctionalAddRelu() if self.clip_half_range else pytqc.QFunctionalAdd()
        m.qfunc.scale = float(scale)
        m.qfunc.zero_point = int(zp)
        return m


class RangeLinearQuantEltwiseMultWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None,
                 input_overrides=None, inputs_quant_auto_fallback=False):
        if not isinstance(wrapped_module, distiller.modules.EltwiseMult):
            raise ValueError(self.__class__.__name__ + ' can only wrap distiller.modules.EltwiseMult modules')

        if not activation_stats:
            raise NoStatsError(self.__class__.__name__ +
                               ' must get activation stats, dynamic quantization not supported')

        super(RangeLinearQuantEltwiseMultWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                                 clip_acts=clip_acts, activation_stats=activation_stats,
                                                                 clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                                 scale_approx_mult_bits=scale_approx_mult_bits,
                                                                 input_overrides=input_overrides,
                                                                 inputs_quant_auto_fallback=inputs_quant_auto_fallback)
        self.accum_scale = 1

    def quantized_forward(self, *inputs_q):
        input_scales = [input_q.quant_metadata.scale for input_q in inputs_q]
        self.accum_scale = reduce(lambda x, y: x * y, input_scales)

        for input_q in inputs_q:
            input_q += input_q.quant_metadata.zero_point

        accum = self.wrapped_module(*inputs_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        return accum

    def get_output_quantization_params(self, accumulator):
        return self.output_scale, self.output_zero_point

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        requant_scale = output_scale / self.accum_scale
        if self.scale_approx_mult_bits is not None:
            requant_scale = approx_scale_as_mult_and_shift(requant_scale, self.scale_approx_mult_bits)
        return requant_scale, output_zero_point

    def _convert_to_pytorch_quant(self, reduce_range):
        # Convert activations qparams - requirec by PyTorch to be unsigned 8-bit (torch.quint8)
        scale, zp = pytqc.distiller_qparams_to_pytorch(self.output_scale, self.output_zero_point,
                                                       self.output_quant_settings.num_bits,
                                                       self.output_quant_settings.quant_mode, torch.quint8,
                                                       reduce_range)
        m = pytqc.QFunctionalMul()
        m.qfunc.scale = float(scale)
        m.qfunc.zero_point = int(zp)
        if self.clip_half_range:
            # The scale factor calculated in Distiller already considers the ReLU, so it's OK to apply the
            # ReLU after quantization
            m = nn.Sequential(m, nnq.ReLU())
        return m


class FPWrapper(nn.Module):
    """
    A wrapper that replaces a module with a half precision version.
    Args:
        module (nn.Module): The module to be replaced.
        precision (Union[str, int]): the floating point precision to use. Either 16/32/64.
        convert_input (bool): Specifies whether an input conversion
            to module precision is required for forward. Default: True.
        return_fp32 (bool): Specifies whether the output needs
            to be converted back to fp32. Default: True.
    """
    def __init__(self, module: nn.Module, precision, convert_input=True, return_fp32=True):
        super(FPWrapper, self).__init__()
        precision = str(precision)
        self.dtype = {'16': torch.float16, '32': torch.float32, '64': torch.float64}[precision]
        self.wrapped_module = module.to(self.dtype)
        self.return_fp32 = return_fp32
        self.convert_input = convert_input

    def forward(self, *input):
        if self.convert_input:
            input = distiller.convert_tensors_recursively_to(input, dtype=self.dtype)

        result = self.wrapped_module(*input)
        if self.return_fp32:
            return distiller.convert_tensors_recursively_to(result, dtype=torch.float32)

        return result

    def extra_repr(self):
        tmpstr = 'float_dtype={}, convert_input={}, return_fp32={}'.format(self.dtype, self.convert_input,
                                                                           self.return_fp32)
        return tmpstr


class FP16Wrapper(FPWrapper):
    def __init__(self, module, convert_input=True, return_fp32=True):
        super(FP16Wrapper, self).__init__(module, 16, convert_input, return_fp32)


class RangeLinearEmbeddingWrapper(nn.Module):
    def __init__(self, wrapped_module, num_bits, mode=LinearQuantMode.SYMMETRIC, stats=None, save_fp_weights=False,
                 clip_acts=ClipMode.NONE, clip_n_stds=None, clip_half_range=False):
        if not isinstance(wrapped_module, nn.Embedding):
            raise ValueError(self.__class__.__name__ + ' can only wrap torch.nn.Embedding modules')

        super(RangeLinearEmbeddingWrapper, self).__init__()

        mode = verify_quant_mode(mode)
        self.mode = mode

        self.wts_quant_settings = QuantSettings(num_bits, self.mode.weights, clip_acts, clip_n_stds, clip_half_range,
                                                False)

        self.params_min_q_val, self.params_max_q_val = get_quantized_range(
            self.wts_quant_settings.num_bits,
            self.wts_quant_settings.quant_mode != LinearQuantMode.ASYMMETRIC_UNSIGNED,
            self.wts_quant_settings.quant_mode == LinearQuantMode.SYMMETRIC_RESTRICTED
        )
        self.save_fp_weights = save_fp_weights
        if save_fp_weights:
            wrapped_module.register_buffer('float_weight', wrapped_module.weight.clone().detach())

        if stats is None:
            w_scale, w_zero_point = _get_quant_params_from_tensor(wrapped_module.weight, num_bits, mode.weights,
                                                                  clip=clip_acts, num_stds=clip_n_stds,
                                                                  half_range=clip_half_range)
        else:
            w_scale, w_zero_point = _get_quant_params_from_stats_dict(stats['output'], num_bits, mode.weights,
                                                                      clip=clip_acts, num_stds=clip_n_stds,
                                                                      half_range=clip_half_range)

        device = wrapped_module.weight.device
        self.register_buffer('w_scale', w_scale.to(device))
        self.register_buffer('w_zero_point', w_zero_point.to(device))
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.w_zero_point,
                              self.params_min_q_val, self.params_max_q_val, inplace=True)
        self.quant_metadata = TensorQuantMetadata(self.w_scale, self.w_zero_point,
                                                  self.params_min_q_val, self.params_max_q_val)
        self.wrapped_module = wrapped_module

    def named_linear_quant_params(self, filter=False):
        yield 'w_scale', self.w_scale
        if not filter or is_linear_quant_mode_asymmetric(self.mode.weights):
            yield 'w_zero_point', self.w_zero_point

    def set_linear_quant_param(self, name, val):
        if name in dict(self.named_clipping()):
            setattr(self, name, val)
        elif name in ['w_scale', 'w_zero_point']:
            if self.save_fp_weights:
                getattr(self, name).fill_(val)
                self.wrapped_module.weight.data.copy_(self.wrapped_module.float_weight.data)
                linear_quantize_clamp(self.wrapped_module.weight.data, self.w_scale, self.w_zero_point,
                                      self.params_min_q_val,
                                      self.params_max_q_val, inplace=True)
            else:
                raise UnsatisfiedRequirements('Cannot re-quantize the weights. Please specify \'save_fp_weights\' in '
                                              'the %s constructor to enable re-quantizing the weights.' %
                                              self.__class__.__name__)
        else:
            raise KeyError('No quantization parameter called \'%s\'.' % name)

    def update_linear_quant_params(self, new_config):
        """
        Updates all the quant params using a dictionary.
        Args:
             new_config (dict): the new configuration dict.
        """
        for name, val in new_config.items():
            self.set_linear_quant_param(name, val)

    def _check_requirements_weights_clipping(self, setter=False):
        if not self.wts_quant_settings.num_bits:
            raise UnsatisfiedRequirements('Cannot retrieve clipping values because the weights aren\'t quantized.')
        if setter and not self.save_fp_weights:
            warnings.warn('Without saving fp32 version of weights, re-quantization is disabled. To enable, '
                          'please set \'save_fp_weights\' while constructing the wrapper.')

    @property
    def weight_clipping(self):
        self._check_requirements_weights_clipping(setter=False)
        bits, mode = self.wts_quant_settings.num_bits, self.wts_quant_settings.quant_mode
        scale, zp = self.w_scale, self.w_zero_point
        return _get_clipping_values(scale, zp, bits, mode)

    @weight_clipping.setter
    def weight_clipping(self, val):
        self._check_requirements_weights_clipping(setter=True)
        bits = self.wts_quant_settings.num_bits
        val_min, val_max = _check_clipping_val(val, self.wts_quant_settings.quant_mode, False)
        if is_linear_quant_mode_symmetric(self.wts_quant_settings.quant_mode):
            # in symmetric quantization - we only need one value
            scale, zp = symmetric_linear_quantization_params(bits, val_max)
        else:
            signed = self.wts_quant_settings.quant_mode == LinearQuantMode.ASYMMETRIC_SIGNED
            scale, zp = asymmetric_linear_quantization_params(bits, val_min, val_max, signed=signed)
        self.set_linear_quant_param('w_scale', scale)
        self.set_linear_quant_param('w_zero_point', zp)

    def named_clipping(self, filter=False):
        val = self.weight_clipping
        if filter and is_linear_quant_mode_symmetric(self.mode.weights):
            val = val[1]
        yield 'weight_clipping', val

    def forward(self, input):
        out_q = self.wrapped_module(input)
        out_f = linear_dequantize(out_q, self.w_scale, self.w_zero_point, inplace=True)
        out_f.quant_metadata = self.quant_metadata
        return out_f

    def to_pytorch_quant(self, reduce_range):
        # No quantized embedding in PyTorch, so use FP32 embedding followed by quantize
        emb = deepcopy(self.wrapped_module)
        with torch.no_grad():
            if self.save_fp_weights:
                w_dq = nn.Parameter(self.float_weight, requires_grad=False)
            else:
                w_dq = nn.Parameter(linear_dequantize(emb.weight, self.w_scale, self.w_zero_point),
                                    requires_grad=False)
        emb.weight = w_dq

        scale, zp = pytqc.distiller_qparams_to_pytorch(self.w_scale, self.w_zero_point,
                                                       self.wts_quant_settings.num_bits,
                                                       self.wts_quant_settings.quant_mode, torch.quint8,
                                                       reduce_range)

        return nn.Sequential(emb, nnq.Quantize(scale, zp, torch.quint8))


class RangeLinearFakeQuantWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None,
                 fpq_module=None, input_overrides=None, inputs_quant_auto_fallback=False, quantize_inputs=False):
        if isinstance(wrapped_module, (nn.ReLU, nn.ReLU6)):
            # In case of ReLU + Gauss/Laplace clipping, need to clip according to stats before ReLU is applied
            clip_half_range = True
            if clip_acts in (ClipMode.GAUSS, ClipMode.LAPLACE):
                activation_stats['output']['mean'] = activation_stats['inputs'][0]['mean']
                activation_stats['output']['std'] = activation_stats['inputs'][0]['std']
                activation_stats['output']['b'] = activation_stats['inputs'][0]['b']
        super(RangeLinearFakeQuantWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                          clip_acts=clip_acts, activation_stats=activation_stats,
                                                          clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                          scale_approx_mult_bits=scale_approx_mult_bits,
                                                          input_overrides=input_overrides,
                                                          inputs_quant_auto_fallback=inputs_quant_auto_fallback)
        self.fpq_module = str(fpq_module) if fpq_module else None
        self.dtype = torch.float
        self.quantize_inputs = quantize_inputs
        if self.fpq_module:
            self.dtype = {'16': torch.half, '32': torch.float, '64': torch.double}[self.fpq_module]
            self.wrapped_module.to(self.dtype)

    def _prepare_input(self, idx, input):
        if not self.quantize_inputs:
            return input

        previously_quantized = hasattr(input, 'quant_metadata')
        input.quant_metadata = self._get_input_quant_metadata(idx, input)
        if previously_quantized:
            return input

        # "Fresh" tensor, so need to quantize and the de-quantize (because this is the fake-quant wrapper)
        input_q = linear_quantize_clamp_with_metadata(input, inplace=False)
        return linear_dequantize_with_metadata(input_q, inplace=True)

    def quantized_forward(self, *inputs_q):
        inputs_q = distiller.convert_tensors_recursively_to(inputs_q, dtype=self.dtype)
        outputs = self.wrapped_module(*inputs_q)
        return distiller.convert_tensors_recursively_to(outputs, dtype=self.dtype)

    def get_output_quantization_params(self, accumulator):
        if self.preset_act_stats:
            return self.output_scale, self.output_zero_point
        else:
            q_set = self.output_quant_settings
            return _get_quant_params_from_tensor(accumulator, q_set.num_bits, q_set.quant_mode, q_set.clip_mode,
                                                 q_set.per_channel, q_set.clip_n_stds,q_set.clip_half_range,
                                                 self.scale_approx_mult_bits)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        return output_scale, output_zero_point

    def _convert_to_pytorch_quant(self, reduce_range):
        # A few PyTorch modules support quantized inputs/outputs
        supported = {
            nn.ReLU: nnq.ReLU(),
            nn.ReLU6: nnq.ReLU6(),
            nn.AvgPool2d: self.wrapped_module,
            nn.AdaptiveAvgPool2d: self.wrapped_module,
            nn.MaxPool2d: self.wrapped_module
        }
        q_module = supported.get(type(self.wrapped_module), None)
        if q_module is None:
            # No PyTorch quantized module - so fake it
            # Convert activations qparams - required by PyTorch to be unsigned 8-bit (torch.quint8)
            scale, zp = pytqc.distiller_qparams_to_pytorch(self.output_scale, self.output_zero_point,
                                                           self.output_quant_settings.num_bits,
                                                           self.output_quant_settings.quant_mode, torch.quint8,
                                                           reduce_range)
            modules = [pytqc.ConditionalDeQuantizeWrapper(self.wrapped_module),
                       nnq.Quantize(float(scale), int(zp), torch.quint8)]
        else:
            modules = [self.wrapped_module]
        if self.clip_half_range:
            # The scale factor calculated in Distiller already considers the ReLU, so it's OK to apply the
            # ReLU after quantization
            modules.append(nnq.ReLU())

        return modules[0] if len(modules) == 1 else nn.Sequential(*modules)

    def extra_repr(self):
        tmpstr = super(RangeLinearFakeQuantWrapper, self).extra_repr()
        if self.dtype:
            tmpstr += '\nwrapped_module_float_dtype={}.'.format(self.dtype)
        return tmpstr


_ptq_wrappers_int_only = (RangeLinearQuantWrapper, RangeLinearEmbeddingWrapper)
_ptq_wrappers_all = _ptq_wrappers_int_only + (FPWrapper,)


def is_post_train_quant_wrapper(module, include_fpwrapper=True):
    types = _ptq_wrappers_all if include_fpwrapper else _ptq_wrappers_int_only
    return isinstance(module, types)


class PostTrainLinearQuantizer(Quantizer):
    """
    Applies range-based linear quantization to a model.
    This quantizer is expected to be executed at evaluation only, on a pre-trained model

    The following modules / operations have dedicated implementations which consider quantization:
      * torch.nn.Conv2d/Conv3d
      * torch.nn.Linear
      * torch.nn.Embedding
      * distiller.modules.Concat
      * distiller.modules.EltwiseAdd
      * distiller.modules.EltwiseMult
      * distiller.modules.Matmul
      * distiller.modules.BatchMatmul
    An existing module will likely need to be modified to use the 'distiller.modules.*' modules. This needs to
    be done BEFORE creating the quantizer. See the docs for more details:
    https://intellabs.github.io/distiller/prepare_model_quant.html

    Any leaf module not in the list above will be "fake-quantized". That is - the floating-point module will be
    executed (FP64/32/16 can be specified with the fpq_module argument), and its output will be quantized.

    To completely disable quantization for any module (inc. "fake-quantization" as described above),
    use the overrides mechanism to specify NONE for bits_activations and/or bits_weights as needed.

    Args:
        model (torch.nn.Module): Model to be quantized
        bits_activations/parameters/accum (int): Number of bits to be used when quantizing each tensor type
        overrides (:obj:`OrderedDict`, optional): Overrides the layers quantization settings.
        mode (ModuleQuantMode / LinearQuantMode): Quantization mode to use (symmetric / asymmetric-signed / unsigned)
        clip_acts (ClipMode): Activations clipping mode to use
        per_channel_wts (bool): Enable quantization of weights using separate quantization parameters per
            output channel
        model_activation_stats (str / dict / OrderedDict): Either a path to activation stats YAML file, or a dictionary
            containing the stats. The stats are used for static calculation of quantization parameters.
            The dict should be in the format exported by distiller.data_loggers.QuantCalibrationStatsCollector.
            If None then parameters are calculated dynamically.
        fp16 (bool): Set to True to convert modules to half precision.
            WARNING - this argument is deprecated, use instead the argument `fpq_module`
        clip_n_stds (float): When clip_acts == ClipMode.N_STD, this is the number of standard deviations to use
        clip_half_range (bool): When clip_acts is
        scale_approx_mult_bits (int): If not None, scale factors will be approximated using an integer multiplication
            followed by a bit-wise shift. This eliminates floating-point scale factors, replacing them with integer
            calculations.
            If None, scale factors will be kept in their original FP32 values.
        inputs_quant_auto_fallback (bool): Enabled by default.
            See <distiller_root>/examples/post_train_quant/resnet18_imagenet_post_train_input_overrides.yaml
            For details what this does and how to override it.
        fpq_module (Union[int, str]): use the modules in floating point mode and only quantize their outputs.
            takes the values (16, 32, 64) only, this will use RangeLinearFakeQuantWrapper.
        save_fp_weights (bool): Indicates whether or not to save a copy of the floating point weights.
          This allows re-quantization of weight after the initial quantization.
          Defaults to False for performance.
    Note:
        If fpq_module is set, all the layers (except those overridden in `overrides`) will be converted
        to the set floating point precision, regardless of bits_activations/parameters/accum.
    """
    def __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 per_channel_wts=False, model_activation_stats=None, fp16=False,
                 clip_n_stds=None, clip_half_range=False,
                 scale_approx_mult_bits=None, inputs_quant_auto_fallback=True,
                 fpq_module=None, save_fp_weights=False, also_clip_weights=False):
        overrides_bkp = deepcopy(overrides)
        super(PostTrainLinearQuantizer, self).__init__(model, bits_activations=bits_activations,
                                                       bits_weights=bits_parameters, bits_bias=bits_accum,
                                                       overrides=overrides, train_with_fp_copy=False)
        if fp16 and str(fpq_module) not in ('16', 'None'):
            raise ValueError('Conflict - fp16 set to true and fpq_module set to other than 16.')
        mode = verify_quant_mode(mode)
        clip_acts = verify_clip_mode(clip_acts)
        if clip_acts == ClipMode.N_STD and clip_n_stds is None:
            raise ValueError('clip_n_stds must not be None when clip_acts set to N_STD')

        if model_activation_stats is not None:
            if isinstance(model_activation_stats, str):
                if not os.path.isfile(model_activation_stats):
                    raise ValueError("Model activation stats file not found at: " + model_activation_stats)
                msglogger.info('Loading activation stats from: ' + model_activation_stats)
                with open(model_activation_stats, 'r') as stream:
                    model_activation_stats = distiller.utils.yaml_ordered_load(stream)
            elif not isinstance(model_activation_stats, (dict, OrderedDict)):
                raise TypeError('model_activation_stats must either be a string, a dict / OrderedDict or None')

        mode_dict = {'activations': _enum_to_str(mode.activations), 'weights': _enum_to_str(mode.weights)}
        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters,
                                                    'bits_accum': bits_accum,
                                                    'mode': mode_dict,
                                                    'clip_acts': _enum_to_str(clip_acts),
                                                    'clip_n_stds': clip_n_stds,
                                                    'clip_half_range': clip_half_range,
                                                    'per_channel_wts': per_channel_wts,
                                                    'scale_approx_mult_bits': scale_approx_mult_bits,
                                                    'inputs_quant_auto_fallback': inputs_quant_auto_fallback,
                                                    'fpq_module': fpq_module,
                                                    'model_activation_stats': model_activation_stats,
                                                    'overrides': overrides_bkp}}

        def _check_fp16_arg(fp16, fpq_module):
            if fp16:
                warnings.warn("Argument 'fp16' is deprecated. Please use 'fpq_module'(=16/32/64) argument.",
                              DeprecationWarning)
                fpq_module = fpq_module or 16
            return fpq_module

        def replace_param_layer(module, name, qbits_map, per_channel_wts=per_channel_wts,
                                mode=mode, fp16=fp16, scale_approx_mult_bits=scale_approx_mult_bits,
                                clip_acts=None, clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                input_overrides=None, fpq_module=fpq_module, fake=False):
            fpq_module = _check_fp16_arg(fp16, fpq_module)
            if fpq_module and not fake:
                return FPWrapper(module, fpq_module)

            norm_name = distiller.utils.normalize_module_name(name)
            activation_stats = self.model_activation_stats.get(norm_name, None)
            clip_acts = verify_clip_mode(clip_acts or self.clip_acts)
            qbits = qbits_map[name]
            if qbits.acts is not None and qbits.wts is None:
                # Quantizing only activations equals fake-quantization
                fake = True

            if fake:
                return RangeLinearFakeQuantWrapper(module, qbits.acts, mode=mode, clip_acts=clip_acts,
                                                   activation_stats=activation_stats,
                                                   clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                   scale_approx_mult_bits=scale_approx_mult_bits,
                                                   fpq_module=fpq_module, input_overrides=input_overrides,
                                                   inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                                   quantize_inputs=False)

            return RangeLinearQuantParamLayerWrapper(module, qbits.acts, qbits.wts,
                                                     num_bits_accum=self.bits_accum, mode=mode, clip_acts=clip_acts,
                                                     per_channel_wts=per_channel_wts,
                                                     activation_stats=activation_stats,
                                                     clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                     scale_approx_mult_bits=scale_approx_mult_bits,
                                                     input_overrides=input_overrides,
                                                     inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                                     save_fp_weights=self.save_fp_weights,
                                                     also_clip_weights=self.also_clip_weights)

        def replace_non_param_layer(wrapper_type, module, name, qbits_map, fp16=fp16,
                                    scale_approx_mult_bits=scale_approx_mult_bits,
                                    clip_acts=None, clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                    input_overrides=None, inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                    fpq_module=fpq_module, fake=False):
            fpq_module = _check_fp16_arg(fp16, fpq_module)
            if fpq_module and not fake:
                return FPWrapper(module, fpq_module)

            norm_name = distiller.utils.normalize_module_name(name)
            activation_stats = self.model_activation_stats.get(norm_name, None)
            clip_acts = verify_clip_mode(clip_acts or self.clip_acts)
            qbits = qbits_map[name]

            if fake:
                return RangeLinearFakeQuantWrapper(module, qbits.acts, mode=mode, clip_acts=clip_acts,
                                                   activation_stats=activation_stats,
                                                   clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                   scale_approx_mult_bits=scale_approx_mult_bits,
                                                   fpq_module=fpq_module, input_overrides=input_overrides,
                                                   inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                                   quantize_inputs=False)
            try:
                return wrapper_type(module, qbits.acts, mode=mode, clip_acts=clip_acts,
                                    activation_stats=activation_stats,
                                    clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                    scale_approx_mult_bits=scale_approx_mult_bits,
                                    input_overrides=input_overrides,
                                    inputs_quant_auto_fallback=inputs_quant_auto_fallback)
            except NoStatsError:
                warnings.warn('WARNING: {0} - quantization of {1} without stats not supported. '
                              'Keeping the original FP32 module'.format(name, module.__class__.__name__), UserWarning)
                return module

        def replace_embedding(module, name, qbits_map, fp16=fp16, fpq_module=fpq_module, clip_acts=clip_acts,
                              clip_n_stds=clip_n_stds, clip_half_range=clip_half_range):
            fpq_module = _check_fp16_arg(fp16, fpq_module)
            if fpq_module:
                return FPWrapper(module, fpq_module, convert_input=False)
            norm_name = distiller.utils.normalize_module_name(name)
            if not self.also_clip_weights:
                clip_acts, clip_n_stds, clip_half_range = ClipMode.NONE, None, False
            return RangeLinearEmbeddingWrapper(module, qbits_map[name].wts, mode=mode,
                                               stats=self.model_activation_stats.get(norm_name, None),
                                               save_fp_weights=self.save_fp_weights, clip_acts=clip_acts,
                                               clip_n_stds=clip_n_stds, clip_half_range=clip_half_range)

        def replace_fake_quant(module, name, qbits_map, fp16=fp16,
                               clip_acts=None, clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                               scale_approx_mult_bits=scale_approx_mult_bits, input_overrides=None,
                               inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                               fpq_module=fpq_module, fake=True, make_identity=False, quantize_inputs=True):
            if isinstance(module, (nn.ReLU, nn.ReLU6)) and make_identity:
                named_modules = OrderedDict(self.model.named_modules())
                pred = self.adjacency_map[name].predecessors[0].name
                if isinstance(named_modules[pred], RangeLinearQuantWrapper):
                    return nn.Identity()

            if distiller.has_children(module):
                return module

            fpq_module = _check_fp16_arg(fp16, fpq_module)
            if not fake:
                return FPWrapper(module, fpq_module)

            norm_name = distiller.utils.normalize_module_name(name)
            clip_acts = verify_clip_mode(clip_acts or self.clip_acts)
            return RangeLinearFakeQuantWrapper(module, qbits_map[name].acts, mode=mode, clip_acts=clip_acts,
                                               activation_stats=self.model_activation_stats.get(norm_name, None),
                                               clip_n_stds=clip_n_stds,  clip_half_range=clip_half_range,
                                               scale_approx_mult_bits=scale_approx_mult_bits,
                                               fpq_module=fpq_module, input_overrides=input_overrides,
                                               inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                               quantize_inputs=quantize_inputs)

        self.clip_acts = clip_acts
        self.clip_n_stds = clip_n_stds
        self.model_activation_stats = model_activation_stats or {}
        self.bits_accum = bits_accum
        self.mode = mode
        self.save_fp_weights = save_fp_weights
        self.also_clip_weights = also_clip_weights

        self.replacement_factory[nn.Conv2d] = replace_param_layer
        self.replacement_factory[nn.Conv3d] = replace_param_layer
        self.replacement_factory[nn.Linear] = replace_param_layer

        factory_concat = partial(
            replace_non_param_layer, RangeLinearQuantConcatWrapper)
        factory_eltwiseadd = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseAddWrapper)
        factory_eltwisemult = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseMultWrapper)
        factory_matmul = partial(
            replace_non_param_layer, RangeLinearQuantMatmulWrapper)

        update_wrapper(factory_concat, replace_non_param_layer)
        update_wrapper(factory_eltwiseadd, replace_non_param_layer)
        update_wrapper(factory_eltwisemult, replace_non_param_layer)
        update_wrapper(factory_matmul, replace_non_param_layer)

        self.replacement_factory[distiller.modules.Concat] = factory_concat
        self.replacement_factory[distiller.modules.EltwiseAdd] = factory_eltwiseadd
        self.replacement_factory[distiller.modules.EltwiseMult] = factory_eltwisemult
        self.replacement_factory[distiller.modules.Matmul] = factory_matmul
        self.replacement_factory[distiller.modules.BatchMatmul] = factory_matmul
        self.replacement_factory[nn.Embedding] = replace_embedding

        self.default_repalcement_fn = replace_fake_quant
        self.replacement_blacklist.append(nn.Dropout)

        # To be filled by .prepare_model()
        self.linear_quant_params = None

    def named_linear_quant_params(self, yield_clipping_params=False, filter=False):
        if yield_clipping_params:
            yield from self.named_clipping(filter=filter)
            return
        for module_name, module in self.model.named_modules():
            if is_post_train_quant_wrapper(module, include_fpwrapper=False):
                for buff_name, buff in module.named_linear_quant_params(filter=filter):
                    full_buff_name = "%s.%s" % (module_name, buff_name)
                    yield full_buff_name, buff

    def named_clipping(self, filter=False):
        """
        Gets all the clipping parameters of the model.
        yields tuple[str, tuple[torch.Tensor, torch.Tensor]]
        """
        for module_name, module in self.model.named_modules():
            if not is_post_train_quant_wrapper(module, include_fpwrapper=False):
                continue
            for clip_name, clip_val in module.named_clipping(filter=filter):  # type: str, tuple[torch.Tensor, torch.Tensor]
                yield '%s.%s' % (module_name, clip_name), clip_val

    def set_clipping(self, name, val):
        """
        Sets a clipping parameter by name.
        Args:
            name (str): the name of the clipping parameter.
            val (tuple[float or torch.Tensor, float or torch.Tensor]): the value of the clipping.
        """
        module_name = distiller.param_name_2_module_name(name)
        clip_name = name.split('.')[-1]
        module = dict(self.model.named_modules())[module_name]
        if not is_post_train_quant_wrapper(module, False):
            raise ValueError('\'%s\' isn\'t a wrapper and has no clipping parameters.' % module_name)
        if clip_name not in dict(module.named_clipping()):
            raise ValueError('\'%s\' is not a clipping parameter.' % clip_name)
        setattr(module, clip_name, val)

    def update_clipping_parameters(self, clipping_config):
        """
        Updates all clipping paramters according to a configuration dict.
        Args:
            clipping_config (dict[str, tuple[float or torch.Tensor, float or torch.Tensor]]):
              the clipping configuration.
        """
        for name, val in clipping_config.items():
            self.set_clipping(name, val)

    def _is_clipping_parameter(self, name):
        module_name = distiller.param_name_2_module_name(name)
        clip_name = name.split('.')[-1]
        module = dict(self.model.named_modules())[module_name]
        return is_post_train_quant_wrapper(module, False) and clip_name in dict(module.named_clipping())

    def force_readjust_wrappers(self):
        def _force_readjust(module):
            if isinstance(module, RangeLinearQuantWrapper):
                module.force_readjust.fill_(True)
        self.model.apply(_force_readjust)

    def set_linear_quant_param(self, name, val):
        """
        Sets the the quant parameter by module_name.quant_param_name.
        Can also set the clipping values.
        Args:
             name (str): the name of the quant param [module_name].[quant_param_name]
             val: the new value.
        """
        if self._is_clipping_parameter(name):
            self.set_clipping(name, val)
        else:
            self.linear_quant_params[name].data.fill_(val)
        self.force_readjust_wrappers()

    def update_linear_quant_params(self, new_config):
        """
        Updates all the quant params using a dictionary.
        Args:
             new_config (dict): the new configuration dict.
        """
        for k, v in new_config.items():
            self.set_linear_quant_param(k, v)

    @classmethod
    def from_args(cls, model, args):
        """
        Returns an instance of PostTrainLinearQuantizer based on the set command-line arguments that are
        given by add_post_train_quant_args()
        """
        if args.qe_config_file:
            return distiller.config_component_from_file_by_class(model, args.qe_config_file,
                                                                 'PostTrainLinearQuantizer')
        else:
            if args.qe_bits_acts == 0:
                args.qe_bits_acts = None
            if args.qe_bits_wts == 0:
                args.qe_bits_wts = None
            overrides = OrderedDict(
                [
                    (layer, OrderedDict([('bits_activations', None), ('bits_weights', None)]))
                    for layer in args.qe_no_quant_layers
                ]
            )
            overrides.update(OrderedDict(
                [(layer, OrderedDict([('clip_acts', 'NONE')]))
                 for layer in args.qe_no_clip_layers if layer not in args.qe_no_quant_layers]
            ))
            mode_acts = args.qe_mode_acts or args.qe_mode
            mode_wts = args.qe_mode_wts or args.qe_mode
            mode = ModuleQuantMode(mode_acts, mode_wts)
            return cls(model,
                       bits_activations=args.qe_bits_acts,
                       bits_parameters=args.qe_bits_wts,
                       bits_accum=args.qe_bits_accum,
                       mode=mode,
                       clip_acts=args.qe_clip_acts,
                       per_channel_wts=args.qe_per_channel,
                       model_activation_stats=(None if args.qe_dynamic else args.qe_stats_file),
                       clip_n_stds=args.qe_clip_n_stds,
                       scale_approx_mult_bits=args.qe_scale_approx_bits,
                       overrides=overrides,
                       inputs_quant_auto_fallback=True,
                       save_fp_weights=args.qe_save_fp_weights)

    def save_per_layer_parameters(self, save_dir=''):
        defaults = OrderedDict(self.model.quantizer_metadata['params'])
        defaults.pop('bits_activations')
        defaults.pop('bits_parameters')
        defaults.pop('bits_accum')
        out = OrderedDict()
        for n in self.module_overrides_map:
            modules_dict = dict(self.model.named_modules())
            m = modules_dict[n]
            if distiller.has_children(m) and not is_post_train_quant_wrapper(m, include_fpwrapper=False):
                continue
            qbits = self.module_qbits_map[n]
            d = OrderedDict()
            d['bits_activations'] = qbits.acts
            d['bits_weights'] = qbits.wts
            d['bits_bias'] = qbits.bias
            for k, v in defaults.items():
                actual_v = self.module_overrides_map[n].get(k, v)
                d[k] = actual_v
            out[n] = d
        if self.linear_quant_params:
            out['linear_quant_params'] = lqp_dict = OrderedDict()
            for k, v in self.linear_quant_params.items():  # type: str, torch.Tensor
                if v.numel() == 1:
                    lqp_dict[k] = v.item()

        save_path = os.path.join(save_dir, 'layer_quant_params.yaml')
        distiller.yaml_ordered_save(save_path, out)
        msglogger.info('Per-layer quantization parameters saved to ' + save_path)

    def prepare_model(self, dummy_input=None):
        if not self.model_activation_stats:
            msglogger.warning("\nWARNING:\nNo stats file passed - Dynamic quantization will be used\n"
                              "At the moment, this mode isn't as fully featured as stats-based quantization, and "
                              "the accuracy results obtained are likely not as representative of real-world results."
                              "\nSpecifically:\n"
                              "  * Not all modules types are supported in this mode. Unsupported modules will remain "
                              "in FP32.\n"
                              "  * Optimizations for quantization of layers followed by Relu/Tanh/Sigmoid are only "
                              "supported when statistics are used.\nEND WARNING\n")

        self.has_bidi_distiller_lstm = any(isinstance(m, distiller.modules.DistillerLSTM) and m.bidirectional for
                                           _, m in self.model.named_modules())
        if self.has_bidi_distiller_lstm:
            warnings.warn('Model contains a bidirectional DistillerLSTM module. '
                          'Automatic BN folding and statistics optimization based on tracing is not yet '
                          'supported for models containing such modules.\n'
                          'Will perform specific optimization for the DistillerLSTM modules, but any other potential '
                          'opportunities for optimization in the model will be ignored.', UserWarning)
            # Setting dummy_input to None to make sure SummaryGraph won't be called
            dummy_input = None
        elif dummy_input is None:
            raise UnsatisfiedRequirements('PostTrainLinearQuantizer requires dummy '
                                          'input in order to perform certain optimizations')
        super(PostTrainLinearQuantizer, self).prepare_model(dummy_input)

        save_dir = msglogger.logdir if hasattr(msglogger, 'logdir') else '.'
        self.save_per_layer_parameters(save_dir)

    def _pre_prepare_model(self, dummy_input):
        if not self.has_bidi_distiller_lstm:
            self._apply_bn_folding(dummy_input)
            self._apply_activation_stats_fusions()
            self._apply_fuse_relu()
        else:
            self._apply_bidi_distiller_lstm_stats_fusion()

        save_dir = msglogger.logdir if hasattr(msglogger, 'logdir') else '.'
        save_path = os.path.join(save_dir, 'quant_stats_after_prepare_model.yaml')
        distiller.yaml_ordered_save(save_path, self.model_activation_stats)
        msglogger.info('Updated stats saved to ' + save_path)
        # for module_name, override in self.module_overrides_map.items():
        #     # Hack to bypass Quantizer pre-override check -
        #     # Quantizer class checks `qbit.acts` and `qbit.wts` before applying overrides
        #     # but since fp16 doesn't act as an intN - we need to override these
        #     # tensors to bypass the check
        #     if (override.get('fp16', False) or override.get('fpq_module', False)) and \
        #          not override.get('fake', False):
        #         self.module_qbits_map[module_name] = QBits('fp', None, None)

    def _clip_stats(self, entry, min_val, max_val):
        if entry['max'] < min_val:
            entry['min'] = entry['avg_min'] = entry['max'] = entry['avg_max'] = min_val
        elif entry['min'] > max_val:
            entry['min'] = entry['avg_min'] = entry['max'] = entry['avg_max'] = max_val
        else:
            entry['min'] = max(min_val, entry['min'])
            entry['avg_min'] = max(min_val, entry['avg_min'])
            entry['max'] = min(max_val, entry['max'])
            entry['avg_max'] = min(max_val, entry['avg_max'])

    def _apply_bn_folding(self, dummy_input):
        msglogger.info('Applying batch-norm folding ahead of post-training quantization')
        mt.fold_batch_norms(self.model, adjacency_map=self.adjacency_map, inference=True)

        # After BN folding model need to re-generate the adjacency map
        summary_graph = distiller.SummaryGraph(self.model, dummy_input)
        self.adjacency_map = summary_graph.adjacency_map(dedicated_modules_only=False)
        del summary_graph

        if not self.model_activation_stats:
            return

        # Update the activation stats to reflect BN folding
        msglogger.info('Propagating output statistics from BN modules to folded modules')
        named_modules = OrderedDict(self.model.named_modules())
        model_stats = self.model_activation_stats
        for n, m in named_modules.items():
            # Look for the mark left by distiller.model_transforms.fold_batch_norms
            fused_modules = getattr(m, 'fused_modules', None)
            if fused_modules is None:
                continue
            folded_bn_module = distiller.normalize_module_name(fused_modules[0])

            # Propagate the output stats of the folded BN module to this module
            # If stats were collected after folding was applied, then stats for the BN module won't exist,
            # in which case we just move on
            folded_bn_stats = model_stats.pop(folded_bn_module, None)
            if folded_bn_stats is None:
                continue
            model_stats[distiller.normalize_module_name(n)]['output'] = folded_bn_stats['output']
            msglogger.debug('  {} --> {}'.format(folded_bn_module, n))

    def _apply_activation_stats_fusions(self):
        # Now we look for certain "fusions" of layers and activations
        # We modify stats to make sure we quantize only the ranges relevant to the activation function
        # By doing so we reduce quantization error while still keeping all
        if not self.model_activation_stats:
            msglogger.info("No activation stats - skipping optimizations for modules followed by Relu/Tanh/Sigmoid")
            return

        msglogger.info('Optimizing output statistics for modules followed by ReLU/Tanh/Sigmoid')

        named_modules = OrderedDict(self.model.named_modules())
        model_stats = self.model_activation_stats
        for n, m in named_modules.items():
            # Don't fuse if module outputs aren't quantized:
            qbits = self.module_qbits_map.get(n, QBits(None, None, None))
            if qbits.acts is None:
                continue
            if (distiller.has_children(m) and not isinstance(m, SimulatedFoldedBatchNorm))\
                    or n not in self.adjacency_map or len(self.adjacency_map[n].successors) != 1:
                continue
            successor = self.adjacency_map[n].successors[0]
            n = distiller.normalize_module_name(n)
            m_stats = model_stats[n]

            succ_type = successor.type
            succ_stats = model_stats.get(distiller.normalize_module_name(successor.name), None)
            if succ_type == 'Split':
                # Handling case where layer output is split, with each chunk going into an activation function
                # This pattern occurs in LSTM, for example. If all the activations are "similar", we can still
                # optimize the quantization ranges of the output of the layer prior to the split
                post_split_ops = self.adjacency_map[successor.name].successors
                if all(op.type == 'Relu' for op in post_split_ops):
                    succ_type = 'Relu'
                elif all(op.type == 'Tanh' for op in post_split_ops):
                    # Tanh non-saturated input range is smaller than sigmoid, so we try this first
                    succ_type = 'Tanh'
                elif all(op.type in ('Sigmoid', 'Tanh') for op in post_split_ops):
                    # If we have both sigmoid and tanh (as in LSTM), we can go with sigmoid
                    succ_type = 'Sigmoid'
                succ_stats = None

            # Set the clipping values
            if succ_type == 'Relu':
                # ReLU zeros out all negative values, so there's no need to quantize them
                min_val = 0.
                max_val = m_stats['output']['max']
            elif succ_type == 'Sigmoid' or succ_type == 'Tanh':
                # Tanh / Sigmoid saturate at ~4 / ~6 respectively. No need to quantize their inputs outside
                # of these ranges
                max_val = 4. if succ_type == 'Tanh' else 6.
                min_val = -max_val
            elif isinstance(named_modules.get(successor.name, None), nn.ReLU6):
                succ_type = 'ReLU6'
                # ReLU zeros out all negative values, so there's no need to quantize them
                min_val = 0.
                max_val = min(m_stats['output']['max'], 6)
            else:
                continue

            # Clip the stats
            msglogger.debug('  Module {} followed by {}, updating stats'.format(n, succ_type))
            self._clip_stats(m_stats['output'], min_val, max_val)
            if succ_stats is not None:
                succ_stats['inputs'][0] = deepcopy(m_stats['output'])

    def _apply_fuse_relu(self):
        """Fuses ReLU layers to the linear layers before them."""
        model_overrides = self.module_overrides_map
        named_modules = dict(self.model.named_modules())
        for n, m in named_modules.items():
            # Don't fuse if module outputs aren't quantized:
            qbits = self.module_qbits_map.get(n, QBits(None, None, None))
            if qbits.acts is None:
                continue
            if (distiller.has_children(m) and not isinstance(m, SimulatedFoldedBatchNorm))\
                    or n not in self.adjacency_map or len(self.adjacency_map[n].successors) != 1:
                continue
            successor = self.adjacency_map[n].successors[0]
            successor_module = named_modules.get(successor.name, None)
            # Add half range clipping to module overrides
            m_override = model_overrides.get(n, OrderedDict())
            model_overrides[n] = m_override
            if successor.name in named_modules and isinstance(successor_module, (nn.ReLU, nn.ReLU6)):
                m_override['clip_half_range'] = True
                m_override = model_overrides.get(successor.name, OrderedDict())
                m_override['make_identity'] = True
                model_overrides[successor.name] = m_override

    def _apply_bidi_distiller_lstm_stats_fusion(self):
        distiller_lstm_cells = [n for n, m in self.model.named_modules() if
                                isinstance(m, distiller.modules.DistillerLSTMCell)]

        for name in distiller_lstm_cells:
            name += '.eltwiseadd_gate'
            msglogger.debug('  Module {} followed by Sigmoid, updating stats'.format(name))
            sat_val = 6.
            self._clip_stats(self.model_activation_stats[name]['output'], -sat_val, sat_val)

    def _post_prepare_model(self):
        m = self.model
        device = distiller.model_device(m)
        for param in m.parameters():
            param.data = param.data.to(device)
        for buffer in m.buffers():
            buffer.data = buffer.data.to(device)
        self.linear_quant_params = OrderedDict(self.named_linear_quant_params())

    def convert_to_pytorch(self, dummy_input, backend='fbgemm'):
        """
        Convert a model quantized using distiller.quantization.PostTrainLinearQuantizer to model comprised solely of
        native PyTorch static post-training quantization modules and operators.

        This is a convenience wrapper around distiller.quantization.convert_distiller_ptq_model_to_pytorch
        See that function's documentation for more details.
        """
        if not self.prepared:
            raise RuntimeError("Must call prepare_model before attempting to convert to PyTorch")

        return pytqc.convert_distiller_ptq_model_to_pytorch(self.model, dummy_input, backend=backend)


###############################################################################
# Quantization-aware training
###############################################################################


def update_ema(biased_ema, value, decay, step):
    biased_ema = biased_ema * decay + (1 - decay) * value
    unbiased_ema = biased_ema / (1 - decay ** step)  # Bias correction
    return biased_ema, unbiased_ema


def inputs_quantize_wrapped_forward(self, input):
    input = self.inputs_quant(input)
    return self.original_forward(input)


class FakeLinearQuantization(nn.Module):
    def __init__(self, num_bits=8, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, dequantize=True, inplace=False):
        super(FakeLinearQuantization, self).__init__()

        self.num_bits = num_bits
        self.mode = mode
        self.dequantize = dequantize
        self.inplace = inplace

        # We track activations ranges with exponential moving average, as proposed by Jacob et al., 2017
        # https://arxiv.org/abs/1712.05877
        # We perform bias correction on the EMA, so we keep both unbiased and biased values and the iterations count
        # For a simple discussion of this see here:
        # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max_biased', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def forward(self, input):
        # We update the tracked stats only in training
        #
        # Due to the way DataParallel works, we perform all updates in-place so the "main" device retains
        # its updates. (see https://pytorch.org/docs/stable/nn.html#dataparallel)
        # However, as it is now, the in-place update of iter_count causes an error when doing
        # back-prop with multiple GPUs, claiming a variable required for gradient calculation has been modified
        # in-place. Not clear why, since it's not used in any calculations that keep a gradient.
        # It works fine with a single GPU. TODO: Debug...
        if self.training:
            with torch.no_grad():
                current_min, current_max = get_tensor_min_max(input)
            self.iter_count += 1
            self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data,
                                                                             current_min, self.ema_decay,
                                                                             self.iter_count)
            self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data,
                                                                             current_max, self.ema_decay,
                                                                             self.iter_count)

        if is_linear_quant_mode_symmetric(self.mode):
            max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
            actual_min, actual_max = -max_abs, max_abs
            if self.training:
                self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(
                    self.num_bits, max_abs, restrict_qrange=self.mode == LinearQuantMode.SYMMETRIC_RESTRICTED)
        else:
            actual_min, actual_max = self.tracked_min, self.tracked_max
            signed = self.mode == LinearQuantMode.ASYMMETRIC_SIGNED
            if self.training:
                self.scale.data, self.zero_point.data = asymmetric_linear_quantization_params(self.num_bits,
                                                                                              self.tracked_min,
                                                                                              self.tracked_max,
                                                                                              signed=signed)

        input = clamp(input, actual_min.item(), actual_max.item(), False)
        input = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, False)

        return input

    def extra_repr(self):
        mode_str = str(self.mode).split('.')[1]
        return 'mode={0}, num_bits={1}, ema_decay={2:.4f})'.format(mode_str, self.num_bits, self.ema_decay)


class FakeQuantizationWrapper(nn.Module):
    def __init__(self, wrapped_module, num_bits, quant_mode, ema_decay):
        super(FakeQuantizationWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        self.fake_q = FakeLinearQuantization(num_bits, quant_mode, ema_decay, dequantize=True,
                                             inplace=getattr(wrapped_module, 'inplace', False))

    def forward(self, *input):
        res = self.wrapped_module(*input)
        res = self.fake_q(res)
        return res


class QuantAwareTrainRangeLinearQuantizer(Quantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_bias=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, per_channel_wts=False,
                 quantize_inputs=True, num_bits_inputs=None):
        super(QuantAwareTrainRangeLinearQuantizer, self).__init__(model, optimizer=optimizer,
                                                                  bits_activations=bits_activations,
                                                                  bits_weights=bits_weights,
                                                                  bits_bias=bits_bias,
                                                                  overrides=overrides,
                                                                  train_with_fp_copy=True)

        mode = verify_quant_mode(mode)

        mode_dict = {'activations': _enum_to_str(mode.activations), 'weights': _enum_to_str(mode.weights)}
        self.model.quantizer_metadata['params']['mode'] = mode_dict
        self.model.quantizer_metadata['params']['ema_decay'] = ema_decay
        self.model.quantizer_metadata['params']['per_channel_wts'] = per_channel_wts
        self.model.quantizer_metadata['params']['quantize_inputs'] = quantize_inputs

        # Keeping some parameters for input quantization
        self.quantize_inputs = quantize_inputs
        if num_bits_inputs is not None:
            self.num_bits_inputs = num_bits_inputs
        else:
            self.num_bits_inputs = bits_activations
        self.mode = mode
        self.decay = ema_decay
        self.per_channel_wts = per_channel_wts

        def linear_quantize_param(param_fp, param_meta):
            m = param_meta.module
            # We don't quantize the learned weights of embedding layers per-channel, because they're used
            # as inputs in subsequent layers and we don't support per-channel activations quantization yet
            perch = not isinstance(m, nn.Embedding) and per_channel_wts and param_fp.dim() in [2, 4]

            with torch.no_grad():
                scale, zero_point = _get_quant_params_from_tensor(param_fp, param_meta.num_bits, mode,
                                                                  per_channel=perch)
            setattr(m, param_meta.q_attr_name + '_scale', scale)
            setattr(m, param_meta.q_attr_name + '_zero_point', zero_point)
            out = LinearQuantizeSTE.apply(param_fp, scale, zero_point, True, False)
            return out

        def activation_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return FakeQuantizationWrapper(module, bits_acts, mode, ema_decay)

        self.param_quantization_fn = linear_quantize_param

        self.activation_replace_fn = activation_replace_fn
        self.replacement_factory[nn.ReLU] = self.activation_replace_fn

    def _post_prepare_model(self):
        if self.quantize_inputs:
            if isinstance(self.model, nn.DataParallel):
                m = self.model.module
            else:
                m = self.model

            m.inputs_quant = FakeLinearQuantization(self.num_bits_inputs, self.mode, self.decay,
                                                    dequantize=True, inplace=False)
            m.__class__.original_forward = m.__class__.forward
            m.__class__.forward = inputs_quantize_wrapped_forward

        # Prepare scale and zero point buffers in modules where parameters are being quantized
        # We're calculating "dummy" scale and zero point just to get their dimensions
        for ptq in self.params_to_quantize:
            m = ptq.module
            param_fp = getattr(m, ptq.fp_attr_name)
            perch = not isinstance(m, nn.Embedding) and self.per_channel_wts and param_fp.dim() in [2, 4]
            with torch.no_grad():
                scale, zero_point = _get_quant_params_from_tensor(param_fp, ptq.num_bits, self.mode,
                                                                  per_channel=perch)
            m.register_buffer(ptq.q_attr_name + '_scale', torch.ones_like(scale))
            m.register_buffer(ptq.q_attr_name + '_zero_point', torch.zeros_like(zero_point))


class NCFQuantAwareTrainQuantizer(QuantAwareTrainRangeLinearQuantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_bias=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, per_channel_wts=False):
        super(NCFQuantAwareTrainQuantizer, self).__init__(model, optimizer=optimizer,
                                                          bits_activations=bits_activations,
                                                          bits_weights=bits_weights,
                                                          bits_bias=bits_bias,
                                                          overrides=overrides,
                                                          mode=mode, ema_decay=ema_decay,
                                                          per_channel_wts=per_channel_wts,
                                                          quantize_inputs=False)

        # Remove 'quantize_inputs' from the metadata dict since this quantizer hard-codes it and doesn't
        # actually take it as an argument
        self.model.quantizer_metadata['params'].pop('quantize_inputs')

        self.replacement_factory[distiller.modules.EltwiseMult] = self.activation_replace_fn
        self.replacement_factory[distiller.modules.Concat] = self.activation_replace_fn
        self.replacement_factory[nn.Linear] = self.activation_replace_fn
        # self.replacement_factory[nn.Sigmoid] = self.activation_replace_fn

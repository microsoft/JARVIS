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

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import Quantizer
from .q_utils import *
import logging
msglogger = logging.getLogger()

###
# Clipping-based linear quantization (e.g. DoReFa, WRPN)
###


class ClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, clip_val, dequantize=True, inplace=False):
        super(ClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.scale, self.zero_point = asymmetric_linear_quantization_params(num_bits, 0, clip_val, signed=False)
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = clamp(input, 0, self.clip_val, self.inplace)
        input = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)


class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val, dequantize=True, inplace=False):
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        # Clip between 0 to the learned clip_val
        input = F.relu(input, self.inplace)
        # Using the 'where' operation as follows gives us the correct gradient with respect to clip_val
        input = torch.where(input < self.clip_val, input, self.clip_val)
        with torch.no_grad():
            scale, zero_point = asymmetric_linear_quantization_params(self.num_bits, 0, self.clip_val, signed=False)
        input = LinearQuantizeSTE.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val.item(),
                                                           inplace_str)


class WRPNQuantizer(Quantizer):
    """
    Quantizer using the WRPN quantization scheme, as defined in:
    Mishra et al., WRPN: Wide Reduced-Precision Networks (https://arxiv.org/abs/1709.01134)

    Notes:
        1. This class does not take care of layer widening as described in the paper
        2. The paper defines special handling for 1-bit weights which isn't supported here yet
    """
    def __init__(self, model, optimizer,
                 bits_activations=32, bits_weights=32, bits_bias=None,
                 overrides=None):
        super(WRPNQuantizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                            bits_weights=bits_weights, bits_bias=bits_bias,
                                            train_with_fp_copy=True, overrides=overrides)

        def wrpn_quantize_param(param_fp, param_meta):
            scale, zero_point = symmetric_linear_quantization_params(param_meta.num_bits, 1)
            out = param_fp.clamp(-1, 1)
            out = LinearQuantizeSTE.apply(out, scale, zero_point, True, False)
            return out

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

        self.param_quantization_fn = wrpn_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn


def dorefa_quantize_param(param_fp, param_meta):
    if param_meta.num_bits == 1:
        out = DorefaParamsBinarizationSTE.apply(param_fp)
    else:
        scale, zero_point = asymmetric_linear_quantization_params(param_meta.num_bits, 0, 1, signed=False)
        out = param_fp.tanh()
        out = out / (2 * out.abs().max()) + 0.5
        out = LinearQuantizeSTE.apply(out, scale, zero_point, True, False)
        out = 2 * out - 1
    return out


class DorefaParamsBinarizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace=False):
        if inplace:
            ctx.mark_dirty(input)
        E = input.abs().mean()
        output = torch.where(input == 0, torch.ones_like(input), torch.sign(input)) * E
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DorefaQuantizer(Quantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)

    Notes:
        1. Gradients quantization not supported yet
    """
    def __init__(self, model, optimizer,
                 bits_activations=32, bits_weights=32, bits_bias=None,
                 overrides=None):
        super(DorefaQuantizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                              bits_weights=bits_weights, bits_bias=bits_bias,
                                              train_with_fp_copy=True, overrides=overrides)

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

        self.param_quantization_fn = dorefa_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn


class PACTQuantizer(Quantizer):
    """
    Quantizer using the PACT quantization scheme, as defined in:
    Choi et al., PACT: Parameterized Clipping Activation for Quantized Neural Networks
    (https://arxiv.org/abs/1805.06085)

    Args:
        act_clip_init_val (float): Initial clipping value for activations, referred to as "alpha" in the paper
            (default: 8.0)
        act_clip_decay (float): L2 penalty applied to the clipping values, referred to as "lambda_alpha" in the paper.
            If None then the optimizer's default weight decay value is used (default: None)
    """
    def __init__(self, model, optimizer,
                 bits_activations=32, bits_weights=32, bits_bias=None,
                 overrides=None, act_clip_init_val=8.0, act_clip_decay=None):
        super(PACTQuantizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                            bits_weights=bits_weights, bits_bias=bits_bias,
                                            overrides=overrides, train_with_fp_copy=True)

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return LearnedClippedLinearQuantization(bits_acts, act_clip_init_val, dequantize=True,
                                                    inplace=module.inplace)

        self.param_quantization_fn = dorefa_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn

        self.act_clip_decay = act_clip_decay

    # In PACT, LearnedClippedLinearQuantization is used for activation, which contains a learnt 'clip_val' parameter
    # We optimize this value separately from the main model parameters
    def _get_new_optimizer_params_groups(self):
        clip_val_group = {'params': [param for name, param in self.model.named_parameters() if 'clip_val' in name]}
        if self.act_clip_decay is not None:
            clip_val_group['weight_decay'] = self.act_clip_decay
        return [clip_val_group]

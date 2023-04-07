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

from enum import Enum
import torch


class LinearQuantMode(Enum):
    SYMMETRIC = 1
    SYMMETRIC_RESTRICTED = 2
    ASYMMETRIC_UNSIGNED = 3
    ASYMMETRIC_SIGNED = 4


def is_linear_quant_mode_symmetric(quant_mode):
    return quant_mode in (LinearQuantMode.SYMMETRIC, LinearQuantMode.SYMMETRIC_RESTRICTED)


def is_linear_quant_mode_asymmetric(quant_mode):
    return not is_linear_quant_mode_symmetric(quant_mode)


def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out


def symmetric_linear_quantization_params(num_bits, saturation_val, restrict_qrange=False):
    """
    Calculate quantization parameters assuming float range of [-saturation_val, saturation_val].
    The returned zero-point is ALWAYS set to 0.
​
    Setting the 'restrict_qrange' parameter limits the quantized range to N-1 bins, where N = 2 ** num_bits -1.
    This matches the symmetric quantization mode in TensorFlow which uses signed integer and limits the quantized
    range to [-127, 127] (when using 8-bits), as opposed to the "vanilla" case of [-128, 127].
​
    See: https://arxiv.org/abs/1806.08342, section 2.2
    """
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)

    if any(sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    if restrict_qrange:
        n = 2 ** (num_bits - 1) - 1
    else:
        n = (2 ** num_bits - 1) / 2

    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = n / sat_val
    zero_point = torch.zeros_like(scale)

    if is_scalar:
        # If input was scalar, return scalars
        return scale.item(), zero_point.item()
    return scale, zero_point


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
    scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim >= t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]


def get_tensor_avg_min_max(t, across_dim=None):
    min_per_dim, max_per_dim = get_tensor_min_max(t, per_dim=across_dim)
    return min_per_dim.mean(), max_per_dim.mean()


def get_tensor_max_abs(t, per_dim=None):
    min_val, max_val = get_tensor_min_max(t, per_dim=per_dim)
    return torch.max(min_val.abs_(), max_val.abs_())


def get_tensor_avg_max_abs(t, across_dim=None):
    avg_min, avg_max = get_tensor_avg_min_max(t, across_dim=across_dim)
    return torch.max(avg_min.abs_(), avg_max.abs_())


def get_tensor_mean_n_stds_min_max(t, dim=None, n_stds=1):
    if dim is not None:
        raise NotImplementedError('Setting dim != None not supported yet')
    if n_stds <= 0:
        raise ValueError('n_stds must be > 0, got {}'.format(n_stds))
    mean = t.mean()
    std = t.std()
    min_val, max_val = get_tensor_min_max(t)
    min_val = torch.max(min_val, mean - n_stds * std)
    max_val = torch.min(max_val, mean + n_stds * std)
    return min_val, max_val


def get_tensor_mean_n_stds_max_abs(t, dim=None, n_stds=1):
    min_val, max_val = get_tensor_mean_n_stds_min_max(t, dim, n_stds)
    return torch.max(min_val.abs_(), max_val.abs_())


def get_scale_approximation_shift_bits(fp32_scale, mult_bits, limit=False):
    shift_bits = torch.log2((2 ** mult_bits - 1) / fp32_scale).floor()
    if limit:
        shift_bits = min(mult_bits, shift_bits)
    return shift_bits


def get_scale_approximation_mult(fp32_scale, shift_bits):
    return (fp32_scale * (2 ** shift_bits)).floor()


def get_scale_approximation_params(fp32_scale, mult_bits, limit=False):
    shift_bits = get_scale_approximation_shift_bits(fp32_scale, mult_bits, limit=limit)
    multiplier = get_scale_approximation_mult(fp32_scale, shift_bits)
    return multiplier, shift_bits


def approx_scale_as_mult_and_shift(fp32_scale, mult_bits, limit=False):
    multiplier, shift_bits = get_scale_approximation_params(fp32_scale, mult_bits, limit=limit)
    return multiplier / (2 ** shift_bits)


class AciqClipper(object):
    """
    Implemented according to https://arxiv.org/pdf/1810.05723.pdf
    """
    alpha_laplace = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
    alpha_laplace_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}
    alpha_gauss = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
    alpha_gauss_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}

    class AciqClippingType(Enum):
        Laplace = 1
        Gauss = 2

    @staticmethod
    def get_alpha_laplace(t, across_dim=None, num_bits=8, half_range=False):
        if isinstance(t, torch.Tensor):
            # Mean of means across dims is equivalent to global mean
            b = torch.mean(torch.abs(t - t.mean()))
        elif isinstance(t, dict):
            # t is Quant Calibration activation stats dict.
            b = t['b']
        else:
            raise TypeError("Only torch.Tensors or quantization calibration activation stats dicts are acceptable.")
        return b * (AciqClipper.alpha_laplace_positive[num_bits] if half_range
                    else AciqClipper.alpha_laplace[num_bits])

    @staticmethod
    def get_alpha_gauss(t, across_dim=None, num_bits=8, half_range=False):
        if isinstance(t, torch.Tensor):
            # Mean of means across dims is equivalent to global mean
            std = torch.std(t)
        elif isinstance(t, dict):
            # t is Quant Calibration activation stats dict.
            std = t['std']
        else:
            raise TypeError("Only torch.Tensors or quantization calibration activation stats dicts are acceptable.")
        return std * (AciqClipper.alpha_gauss_positive[num_bits] if half_range
                      else AciqClipper.alpha_gauss[num_bits])


class AciqSymmetricClipper(AciqClipper):
    def __init__(self, num_bits, clip_type=AciqClipper.AciqClippingType.Laplace):
        self.num_bits = num_bits
        self.clip_type = clip_type

    def __call__(self, t, across_dim=None):
        if self.clip_type == AciqClipper.AciqClippingType.Laplace:
            alpha = AciqClipper.get_alpha_laplace(t, across_dim, self.num_bits)
        else:
            alpha = AciqClipper.get_alpha_gauss(t, across_dim, self.num_bits)
        if isinstance(t, dict):
            mean = torch.tensor(t['mean'])
        else:
            mean = t.mean()

        clip_val = torch.abs(mean) + alpha
        return -clip_val, clip_val


class AciqAsymmetricClipper(AciqClipper):
    def __init__(self, num_bits, clip_type=AciqClipper.AciqClippingType.Laplace):
        self.num_bits = num_bits
        self.clip_type = clip_type

    def __call__(self, t, across_dim=None, half_range=False):
        if isinstance(t, dict):
            mean, min_val = torch.tensor(t['mean']), torch.tensor(t['avg_min'])
        else:
            mean = t.mean()
            min_val = get_tensor_min_max(t, across_dim)[0].mean()
        if self.clip_type == AciqClipper.AciqClippingType.Laplace:
            alpha = AciqClipper.get_alpha_laplace(t, across_dim, self.num_bits, half_range=half_range)
        else:
            alpha = AciqClipper.get_alpha_gauss(t, across_dim, self.num_bits, half_range=half_range)
        min_val = torch.max(min_val, mean - alpha)
        delta = alpha if half_range else 2 * alpha
        return min_val, min_val + delta


def get_quantized_range(num_bits, signed=True, signed_restrict_qrange=False):
    """
    Returns the min and max quantized values for a given number of bits and signed/unsigned mode.
​
    The 'signed_restrict_qrange' argument:
      If set, the quantized range is limited to N-1 "bins", where n = 2 ** num_bits - 1.
      This matches the symmetric quantization mode in TensorFlow which uses signed integer and limits the quantized
      range to [-127, 127] (when using 8-bits), as opposed to the "vanilla" case of [-128, 127].
​
      See: https://arxiv.org/abs/1806.08342, section 2.2
​
      This argument is ignored if the 'signed' argument is False. It should match the 'restrict_qrange' argument of
      'symmetric_linear_quantization_params'.
​
    An example for num_bits == 4:
    * signed == False:
        q_min = 0, q_max = 15
    * signed == True ; signed_restrict_qrange == False:
        q_min = -8, q_max = 7
    * signed == True ; signed_restrict_qrange == True:
        q_min = -7, q_max = 7
​
    """
    if signed:
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax if signed_restrict_qrange else -qmax - 1
        return qmin, qmax
    return 0, 2 ** num_bits - 1


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None

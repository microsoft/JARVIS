#
# Copyright (c) 2020 Intel Corporation
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

#
# Implementation of "Loss Aware Post-Training Quantization" (Nahshan et al., 2019)
#
# Paper: https://arxiv.org/abs/1911.07190
# Reference implementation: https://github.com/ynahshan/nn-quantization-pytorch/tree/master/lapq
#

import torch
import torch.nn as nn
from distiller.quantization.range_linear import PostTrainLinearQuantizer, ClipMode, \
    RangeLinearQuantWrapper, RangeLinearEmbeddingWrapper, RangeLinearQuantParamLayerWrapper, \
    is_post_train_quant_wrapper, LinearQuantMode
from distiller.quantization import is_linear_quant_mode_asymmetric, is_linear_quant_mode_symmetric
from functools import partial
from distiller.summary_graph import SummaryGraph
from distiller.model_transforms import fold_batch_norms
import distiller.modules
from distiller.data_loggers import collect_quant_stats
from collections import OrderedDict
from itertools import count
import logging
from copy import deepcopy
import scipy.optimize as opt
import numpy as np
import argparse


msglogger = logging.getLogger()


def _make_non_parallel_copy(model):
    if any([isinstance(m, nn.DataParallel) for m in model.modules()]):
        return distiller.make_non_parallel_copy(model)
    return model


def quant_params_dict2vec(p_dict, search_clipping=False):
    """
    Convert quantization params dictionary returned by post-train quantizer to a numpy array that can be used
    with scipy.opt.minimize
    """
    keys = []
    vals = []
    for k, v in p_dict.items():
        if search_clipping and isinstance(v, tuple):
            # When both min and amx values are optimized, we need to concatenate them in the array
            # We create dual matching keys so it's easy to convert back to a dict
            keys += [k + '_min', k + '_max']
            vals += [v[0].item(), v[1].item()]
        else:
            keys.append(k)
            vals.append(v.item())

    return keys, np.array(vals)


def quant_params_vec2dict(keys, vals, search_clipping=False):
    """
    Convert the vector(s) created by quant_params_dict2vec to a dictionary of quantization parameters that
    the post-training quantizer API can digest
    """
    res = OrderedDict()
    for idx, k in enumerate(keys):
        if search_clipping and k.endswith('_min'):
            res[k[:-4]] = sorted((vals[idx], vals[idx + 1]))
        elif search_clipping and k.endswith('_max'):
            continue
        else:
            res[k] = abs(vals[idx])
    return res


def lp_loss(x: torch.Tensor, y: torch.Tensor, p):
    tmp = (x - y).abs_().pow_(p)
    loss = (torch.sum(tmp) / x.numel()).item()
    return loss


def _check_qp_vec(keys, qp_vec, quant_mode=LinearQuantMode.SYMMETRIC, search_clipping=False):
    if is_linear_quant_mode_symmetric(quant_mode):
        return all(qp_vec > 0)
    if not search_clipping:
        idxs_scales = np.array(['scale' in key for key in keys])
        qp_vec_scales = qp_vec[idxs_scales]
        return all(qp_vec_scales > 0)


l1_loss = partial(lp_loss, p=1)
l2_loss = partial(lp_loss, p=2)
l3_loss = partial(lp_loss, p=3)


_INIT_MODES = {
    'NONE': ClipMode.NONE, 'AVG': ClipMode.AVG, 'LAPLACE': ClipMode.LAPLACE, 'GAUSS': ClipMode.GAUSS,
    'L1': l1_loss, 'L2': l2_loss, 'L3': l3_loss
}


def _init_mode_from_str(init_mode_str):
    init_mode_str = init_mode_str.upper()
    if init_mode_str not in _INIT_MODES:
        raise ValueError('Unsupported init mode \'%s\'. '
                         'The supported init modes are: %s.' % (init_mode_str, _INIT_MODES))
    return _INIT_MODES[init_mode_str]


def optimize_for_layer(layer, quantized_layer, loss_fn, input, method=None, search_clipping=False):
    """
    Searches for optimal linear quantization parameters (scale, zero_point) for a layer
    with respect to the loss function. Assumes loss_fn is of the signature `loss_fn(y, y_q)->float`

    We perform the initialization a bit differently compared to the paper/reference implementation:
    * In the reference:
      * Weights and activations are initialized based on quantization loss of their respective tensors.
      * Activations are initialized "online", meaning the input to the layer N being initialized is the
        output of the already quantized layer N-1.
    * In this implementation:
      * For a given layer, we initialize both activations and weights together (as applicable) based on the
        LP loss between the quantized layer output and the FP32 layer output.
      * But, we don't do "online" initialization. That is, each layer is initialized independently from the
        quantization parameters obtained for earlier layers.

    Args:
        layer (nn.Module): the original, pre-quantized, layer.
        quantized_layer (RangeLinearQuantWrapper or RangeLinearEmbeddingWrapper): the post-quantized layer.
        loss_fn (callable): the loss function to optimize with respect to it.
        method (str or callable): the method of optimization, as will be used by scipy.optimize.minimize.
        search_clipping (bool): if set, optimize clipping values, otherwise optimize scale factor
    Returns:
        quantized_layer after optimization
    """
    params_gen = quantized_layer.named_linear_quant_params(filter=True) if not search_clipping \
        else quantized_layer.named_clipping(filter=True)
    init_qp_dict = OrderedDict(params_gen)

    keys, init_qp_vec = quant_params_dict2vec(init_qp_dict, search_clipping)

    def feed_forward_fn(qp_vec):
        qp_dict = quant_params_vec2dict(keys, qp_vec, search_clipping)
        quantized_layer.update_linear_quant_params(qp_dict)
        # Using cloned input, required if the layer is inplace
        y = layer(input.clone().detach())
        if getattr(quantized_layer, 'clip_half_range', False):
            torch.relu_(y)
        q_y = quantized_layer(input.clone().detach())
        loss = loss_fn(y, q_y)
        return loss

    result = opt.minimize(feed_forward_fn, init_qp_vec, method=method)  # type: opt.OptimizeResult
    return quantized_layer


def get_input_for_layer(model, layer_name, eval_fn):
    layer = dict(model.named_modules())[layer_name]  # type: nn.Module
    layer_inputs = []

    def hook_layer_input(module, input):
        layer_inputs.append(input[0].clone().detach())

    handle = layer.register_forward_pre_hook(hook_layer_input)
    eval_fn(model)
    handle.remove()
    return torch.cat(layer_inputs)


def init_layer_linear_quant_params(quantizer, original_model, layer_name, init_mode=ClipMode.NONE,
                                   init_method='Powell', eval_fn=None, search_clipping=False,
                                   run_device='cpu'):
    """
    Initializes a layer's linear quant parameters.
    This is done to set the scipy.optimize.minimize initial guess.
    Args:
        quantizer (PostTrainLinearQuantizer): the quantizer, **after** calling prepare model.
        original_model (nn.Module): the original, pre-quantized, model.
        layer_name (str): the name of the layer.
        init_mode (ClipMode or callable or str): the initialization mode.
          If ClipMode, the initialization will be according to the respective ClipMode.
          If callable - init_mode will be treated as a loss function between the activations pre and post-quantization,
            and the initialization process will attempt to find the minimum of that loss function.
            E.g. if l1_loss has been passed, the initialization vector will be
              scale, zero_point = argmin_{s, zp} (l1_loss(layer(input), q_layer(input; s, zp)))
          If str - the mode will be chosen from a list of options. The options are:
            [NONE, AVG, LAPLACE, GAUSS, L1, L2 ,L3].
          Defaults to ClipMode.NONE
        init_method (str or callable): applicable only in the case of init_mode = 'L1/2/3' or callable.
          chooses the minimization method for finding the local argmin_{s, zp}.
          Defaults to 'Powell'
        eval_fn: evaluation function for the model. Assumed it has a signature of the form
          `eval_fn(model)->float`. this is the function to be minimized by the optimization algorithm.
          applicable only in the case of init_mode = 'L1/2/3' or callable.
        search_clipping (bool): if set, optimize clipping values, otherwise optimize scale factor
    """
    denorm_layer_name = distiller.denormalize_module_name(quantizer.model, layer_name)
    msglogger.info(denorm_layer_name)
    if isinstance(init_mode, str):
        init_mode = _init_mode_from_str(init_mode)
    layer = dict(original_model.named_modules())[layer_name]
    local_args, local_kwargs = quantizer.modules_processed_args[denorm_layer_name]
    if isinstance(init_mode, ClipMode):
        local_kwargs['clip_acts'] = init_mode
    replace_fn = quantizer.replacement_factory.get(type(layer), quantizer.default_repalcement_fn)
    quantized_layer = replace_fn(deepcopy(layer), *local_args, **local_kwargs).eval()
    if not is_post_train_quant_wrapper(quantized_layer, False):
        # the module wasn't quantized, nothing to do here
        return

    if callable(init_mode):
        input_for_layer = get_input_for_layer(original_model, layer_name, eval_fn)
        quantized_layer = optimize_for_layer(layer.to(device=run_device), quantized_layer.to(device=run_device),
                                             init_mode, input_for_layer, init_method, search_clipping=search_clipping)
        del input_for_layer

    distiller.model_setattr(quantizer.model, denorm_layer_name, quantized_layer)
    quantizer.model.eval()


def init_linear_quant_params(quantizer, original_model, eval_fn, dummy_input, init_mode,
                             init_method='Powell', search_clipping=False, run_device='cpu'):
    """
    Initializes all linear quantization parameters of the model.
    Args:
        quantizer (PostTrainLinearQuantizer): the quantizer, **after** calling prepare model.
        original_model (nn.Module): the original, pre-quantized, model.
        init_mode (ClipMode or callable or str or dict): See `init_layer_linear_qaunt_params`.
          if init_mode is dict - init_mode is configuration for the different layers,
          i.e. init_mode = Dict[layer_name:str, init_mode_layer: ClipMode or callable or str].
        eval_fn: evaluation function for the model. Assumed it has a signature of the form
          `eval_fn(model)->float`. this is the function to be minimized by the optimization algorithm.
          Note - unlike in `init_layer_linear_quant_params`, this argument is required here.
        dummy_input: dummy sample input to the model
        init_method: See `init_layer_linear_qaunt_params`.
        search_clipping (bool): if set, optimize clipping values, otherwise optimize scale factor
    """
    non_parallel_model = _make_non_parallel_copy(original_model).to(
        device=run_device if callable(init_mode) else 'cpu')
    layers_topological_order = SummaryGraph(non_parallel_model, dummy_input).layers_topological_order()
    q_named_modules = OrderedDict(quantizer.model.named_modules())
    for module_name in layers_topological_order:
        # check to see if it was quantized:
        q_module = q_named_modules[distiller.denormalize_module_name(quantizer.model, module_name)]
        if not is_post_train_quant_wrapper(q_module, False):
            continue
        module_init_mode = init_mode[module_name] if isinstance(init_mode, dict) else init_mode
        msglogger.debug('Initializing layer \'%s\' using %s mode' % (module_name, module_init_mode))
        init_layer_linear_quant_params(quantizer, non_parallel_model, module_name, module_init_mode,
                                       init_method=init_method,
                                       eval_fn=eval_fn,
                                       search_clipping=search_clipping, run_device=run_device)
    if non_parallel_model != original_model:
        del non_parallel_model

    quantizer._post_prepare_model()
    quantizer.model.eval()


def add_coordinate_search_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Post-Training Quantization Auto-Optimization (LAPQ) Arguments')
    group.add_argument('--lapq-maxiter', default=None, type=int,
                       help='Max iteration for minimization method.')
    group.add_argument('--lapq-maxfev', default=None, type=int,
                       help='Max iteration for minimization method.')
    group.add_argument('--lapq-method', default='Powell',
                       help='Minimization method used by scip.optimize.minimize.')
    group.add_argument('--lapq-basinhopping', '--lapq-bh', action='store_true', default=False,
                       help='Use scipy.optimize.basinhopping stochastic global minimum search.')
    group.add_argument('--lapq-basinhopping-niter', '--lapq-bh-niter', default=100,
                       help='Number of iterations for the basinhopping algorithm.')
    group.add_argument('--lapq-init-mode', default='NONE', type=_init_mode_from_str,
                       help='The mode of quant initalization. Choices: ' + '|'.join(list(_INIT_MODES)))
    group.add_argument('--lapq-init-method', default='Powell',
                       help='If --lapq-init-mode was specified as L1/L2/L3, this specifies the method of '
                            'minimization.')
    group.add_argument('--lapq-eval-size', type=float, default=1,
                       help='Portion of test dataset to use for evaluation function.')
    group.add_argument('--lapq-eval-memoize-dataloader', action='store_true', default=False,
                       help='Stores the input batch in memory to optimize performance.')
    group.add_argument('--lapq-search-clipping', action='store_true',
                       help='Search on clipping values instead of scale/zero_point.')


def cmdline_args_to_dict(args):
    """
    Convenience function converting command line arguments obtained from add_coordinate_search_args
    to a dictionary that can be passed as-is to ptq_coordinate_search.

    Example:
        # Assume pre-existing parser
        add_coordinate_search_args(parser)
        args = parser.parse_args()

        # Assume quantizer, dummy_input, eval_fn, and test_fn have been set up
        lapq_args_dict = cmdline_args_to_dict(args)
        ptq_coordinate_search(quantizer, dummy_input, eval_fn, test_fn=test_fn, **lapq_args_dict)
    """
    prefix = 'lapq_'
    len_prefix = len(prefix)
    lapq_args = {k[len_prefix:]: v for k, v in vars(args).items() if k.startswith(prefix)}
    lapq_args.pop('eval_size')
    lapq_args.pop('eval_memoize_dataloader')
    return lapq_args


def validate_quantization_settings(quantized_model, search_clipping):
    if search_clipping:
        return
    for n, m in quantized_model.named_modules():
        if not is_post_train_quant_wrapper(m, False):
            continue

        err_msg = 'Detected asymmetric quantization of {}. ' \
                  'Switch to symmetric quantization or enable search_clipping.'
        if not isinstance(m, RangeLinearEmbeddingWrapper):
            if m.output_quant_settings.num_bits and \
                    is_linear_quant_mode_asymmetric(m.mode.activations) and \
                    not m.clip_half_range:
                raise ValueError(err_msg.format('activations without fused ReLU'))
        if isinstance(m, (RangeLinearEmbeddingWrapper, RangeLinearQuantParamLayerWrapper)):
            if is_linear_quant_mode_asymmetric(m.mode.weights):
                raise ValueError(err_msg.format('weights'))


def ptq_coordinate_search(quantizer, dummy_input, eval_fn, test_fn=None, method='Powell',
                          maxiter=None, maxfev=None, basinhopping=False, basinhopping_niter=100,
                          init_mode=ClipMode.NONE, init_method=None, search_clipping=False,
                          minimizer_kwargs=None):
    """
    Searches for the optimal post-train quantization configuration (scale/zero_points)
    for a model using numerical methods, as described by scipy.optimize.minimize.
    Args:
        quantizer (distiller.quantization.PostTrainLinearQuantizer): A configured PostTrainLinearQuantizer object
          containing the model being quantized
        dummy_input: an sample expected input to the model
        eval_fn (callable): evaluation function for the model. Assumed it has a signature of the form
          `eval_fn(model)->float`. this is the function to be minimized by the optimization algorithm.
        test_fn (callable): a function to test the current performance of the model. Assumed it has a signature of
          the form `test_fn(model)->dict`, where the returned dict contains relevant results to be logged.
          For example: {'top-1': VAL, 'top-5': VAL, 'loss': VAL}
        method (str or callable): Minimization method as accepted by scipy.optimize.minimize.
        maxiter (int): Maximum number of iterations to perform during minimization
        maxfev (int): Maximum number of total function evaluations to perform during minimization
        basinhopping (bool): flag, indicates to use basinhopping as a global-minimization method,
          will pass the `method` argument to `scipy.optimize.basinhopping`.
        basinhopping_niter (int): Number of iterations to perform if basinhopping is set
        init_mode (ClipMode or callable or str or dict): See 'init_linear_quant_params'
        init_method (str or callable): See 'init_layer_linear_quant_params'
        search_clipping (bool): Search on clipping values instead of directly on scale/zero-point (scale and zero-
          point are inferred from the clipping values)
        minimizer_kwargs (dict): Optional additional arguments for scipy.optimize.minimize
    """
    if not isinstance(quantizer, PostTrainLinearQuantizer):
        raise ValueError('Only PostTrainLinearQuantizer supported, but got a {}'.format(quantizer.__class__.__name__))
    if quantizer.prepared:
        raise ValueError('Expecting a quantizer for which prepare_model has not been called')

    run_device = distiller.model_device(quantizer.model)

    original_model = deepcopy(quantizer.model).cpu()
    original_model = fold_batch_norms(original_model, dummy_input)

    if not quantizer.model_activation_stats:
        msglogger.info('Collecting stats for model...')
        model_temp = _make_non_parallel_copy(original_model).to(device=run_device)
        act_stats = collect_quant_stats(model_temp, eval_fn,
                                        inplace_runtime_check=True, disable_inplace_attrs=True,
                                        save_dir=getattr(msglogger, 'logdir', '.'))
        if model_temp != original_model:
            del model_temp
        quantizer.model_activation_stats = act_stats
        quantizer.model.quantizer_metadata['params']['model_activation_stats'] = act_stats

    # Preparing model and init conditions:
    msglogger.info("Initializing quantizer...")

    # Make sure weights are re-quantizable and clip-able
    quantizer.save_fp_weights = True
    quantizer.also_clip_weights = True

    # Disable any user set activations clipping - we'll be using init_args
    quantizer.clip_acts = ClipMode.NONE
    for overrides_dict in quantizer.module_overrides_map.values():
        overrides_dict.pop('clip_acts', None)

    quantizer.prepare_model(dummy_input)
    quantizer.model.eval()
    quantizer.model = quantizer.model.cpu()

    validate_quantization_settings(quantizer.model, search_clipping)

    msglogger.info("Initializing quantization parameters...")
    init_linear_quant_params(quantizer, original_model, eval_fn, dummy_input, init_mode, init_method,
                             search_clipping=search_clipping, run_device=run_device)

    msglogger.info("Evaluating initial quantization score...")
    best_data = {
        'score': eval_fn(quantizer.model),
        'qp_dict': deepcopy(quantizer.linear_quant_params)
    }
    msglogger.info("Evaluation set loss after initialization %.3f" % best_data['score'])
    if test_fn:
        msglogger.info('Evaluating on full test set...')
        results = test_fn(quantizer.model)
        s = ', '.join(['{} = {:.3f}'.format(k, v) for k, v in results.items()])
        msglogger.info('Test: ' + s)

    init_qp_dict = OrderedDict(quantizer.named_linear_quant_params(search_clipping, filter=True))
    keys, init_qp_vec = quant_params_dict2vec(init_qp_dict, search_clipping)

    iter_counter = count(1)
    eval_counter = count(1)

    def feed_forward_fn(qp_vec):
        # if not _check_qp_vec(keys, qp_vec, quant_mode, args.search_clipping):
        #     return 1e6
        qp_dict = quant_params_vec2dict(keys, qp_vec, search_clipping)
        quantizer.update_linear_quant_params(qp_dict)
        loss = eval_fn(quantizer.model)

        i = next(eval_counter)
        if i % 20 == 0:
            msglogger.info('%d evaluations: loss=%.3f' % (i, loss))

        return loss

    def callback(qp_vec):
        score = feed_forward_fn(qp_vec)
        i = next(iter_counter)
        msglogger.info("Iteration %d: \t Score=%.3f" % (i, score))
        if score < best_data['score']:
            best_data['score'] = score
            best_data['qp_dict'] = quant_params_vec2dict(keys, qp_vec, search_clipping)
            msglogger.info("Saving current best quantization parameters.")
        if test_fn:
            msglogger.info('Evaluating on full test set...')
            results = test_fn(quantizer.model)
            s = ', '.join(['{} = {:.3f}'.format(k, v) for k, v in results.items()])
            msglogger.info('Test: ' + s)

    options = OrderedDict()
    options['maxiter'] = maxiter
    options['maxfev'] = maxfev

    minimizer_kwargs = minimizer_kwargs or OrderedDict()
    minimizer_kwargs.update({
        'method': method, 'options': options
    })
    if basinhopping:
        msglogger.info('Using basinhopping global minimum search with "%s" local minimization method' % method)
        res = opt.basinhopping(feed_forward_fn, init_qp_vec, basinhopping_niter, callback=callback,
                               minimizer_kwargs=minimizer_kwargs)
    else:
        msglogger.info('Using "%s" minimization algorithm.' % method)
        res = opt.minimize(feed_forward_fn, init_qp_vec, callback=callback, **minimizer_kwargs)

    msglogger.info('Optimization done')
    msglogger.info('Best score: {}'.format(best_data['score']))
    msglogger.info('Best Configuration: {}'.format(best_data['qp_dict']))
    return quantizer.model, best_data['qp_dict']

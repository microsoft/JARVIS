#
# Copyright (c) 2019 Intel Corporation
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
r"""
Here we implement the greedy search algorithm for automatic quantization.
"""
import torch
import torch.nn as nn
from distiller.quantization import LinearQuantMode
from distiller.quantization.range_linear import PostTrainLinearQuantizer, ClipMode
from distiller.summary_graph import SummaryGraph
from distiller.model_transforms import fold_batch_norms
import distiller.modules
from distiller.data_loggers import collect_quant_stats
from distiller.models import create_model
from collections import OrderedDict, defaultdict
import logging
from copy import deepcopy
import distiller.apputils.image_classifier as classifier
import os
import distiller.apputils as apputils
import re
import argparse

__all__ = ['ptq_greedy_search']

msglogger = None

QUANTIZED_MODULES = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv3d,
    distiller.modules.Concat,
    distiller.modules.EltwiseAdd,
    distiller.modules.EltwiseMult,
    distiller.modules.Matmul,
    distiller.modules.BatchMatmul
)

FP16_LAYERS = (
    nn.Tanh,
    nn.Sigmoid
)

PARAM_MODULES = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv3d
)

UNQUANTIZED_MODULES = (
    nn.Softmax,
)

SKIP_MODULES = (
    nn.Identity,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.ReLU,
    nn.ReLU6
)

CLIP_MODES = ['NONE',
              'AVG',
              'GAUSS',
              'LAPLACE'
              ]


def get_default_args():
    parser = classifier.init_classifier_compression_arg_parser()
    parser.add_argument('--qe-no-quant-layers', '--qenql', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                       help='List of layer names for which to skip quantization.')
    parser.add_argument('--qe-calib-portion', type=float, default=1.0,
                        help='The portion of the dataset to use for calibration stats collection.')
    parser.add_argument('--qe-calib-batchsize', type=int, default=256,
                        help='The portion of the dataset to use for calibration stats collection.')
    parser.add_argument('--base-score', type=float, default=None)
    parser.add_argument('--quantize-inputs', type=str, nargs='+', metavar='LAYER_NAME#INPUT_IDX', default=[],
                        help='The inputs of layers to quantize')
    parser.add_argument('--resume-search-from', type=str, help='Search checkpoint file to resume.',
                        default=None)
    args = parser.parse_args()
    return args


def override_odict(**kwargs):
    return OrderedDict(kwargs)


def get_inputs_to_quantize(sg, args, recurrent=False):
    """
    Finds the modules in the graph that take the user input directly
    Args:
        sg (SummaryGraph): the summary graph of the model
        recurrent: see SummaryGraph.layers_topological_order
    TODO - implement properly
    """
    # input_modules = set()
    # layers = set(sg.layers_topological_order(recurrent))
    # for op in sg.top_level_ops():
    #     input_modules.update(set(sg.successors(op, 1)) & layers)
    # return list(input_modules)
    result = defaultdict(lambda: [])
    for input_str in args.quantize_inputs:
        module_name, input_idx_str = input_str.split('#')
        input_idx = int(input_idx_str)
        result[module_name].append(input_idx)
    return result


def input_override_generator(module, module_name, sg, overrides_dict, **kwargs):
    """
    Generator for overrides on inputs of the input layers.
    Args:
        module (nn.Module): the module
        module_name (str): module name as it appears in the summary graph
        sg (SummaryGraph): a summary graph of the model
        overrides_dict (OrderedDict): the fixed overrides already applied
        kwargs: additional arguments, if needed
    """
    bits_acts = kwargs.get('bits_activations', 8)
    bits_wts = kwargs.get('bits_weights', 8)
    input_nodes = sg.predecessors(module_name, 1)
    input_idx = kwargs.get('input_idx', 0)
    assert input_idx < len(input_nodes)
    for clip_mode in CLIP_MODES:
        input_idx_override = override_odict(bits_activations=bits_acts,
                                            clip_acts=clip_mode)
        input_overrides = OrderedDict([(input_idx, input_idx_override)])
        current_module_override = override_odict(input_overrides=input_overrides)
        # add basic quantization so the quantizer doesn't reject this override
        current_module_override['bits_activations'] = bits_acts
        if isinstance(module, PARAM_MODULES):
            current_module_override['bits_weights'] = bits_wts
        yield current_module_override


def module_override_generator(module, module_name, sg, overrides_dict, **kwargs):
    """
    Standard generator of overrides for the greedy search algorithm.
    Args:
        module (nn.Module): the module
        module_name (str): module name as it appears in the summary graph
        sg (SummaryGraph): a summary graph of the model
        overrides_dict (OrderedDict): the fixed overrides already applied
        kwargs: additional arguments, if needed
    """
    bits_acts = kwargs.get('bits_activations', 8)
    bits_wts = kwargs.get('bits_weights', 8)
    if isinstance(module, nn.ReLU):
        yield override_odict(make_identity=True,
                             bits_weights=bits_wts,
                             bits_activations=bits_acts)
        return
    adj_map = sg.adjacency_map()
    modules_dict = dict(sg._src_model.named_modules())
    successors_names = {op.name for op in adj_map[module_name].successors if op.name in modules_dict}
    use_half_range = all([isinstance(modules_dict[succ], nn.ReLU) for succ in successors_names])
    use_fake = False
    fpq_module = None
    if isinstance(module, FP16_LAYERS):
        fpq_module = 16
        use_fake = True
    if isinstance(module, UNQUANTIZED_MODULES) or not isinstance(module, QUANTIZED_MODULES):
        fpq_module = 32
        use_fake = True
    for clip_mode in CLIP_MODES:
        if isinstance(module, PARAM_MODULES):
            current_module_override = override_odict(clip_acts=clip_mode,
                                                     bits_weights=bits_wts,
                                                     bits_activations=bits_acts,
                                                     bits_bias=32)
        else:
            current_module_override = override_odict(clip_acts=clip_mode,
                                                     fpq_module=fpq_module,
                                                     fake=use_fake,
                                                     bits_weights=bits_wts,
                                                     bits_activations=bits_acts)
        current_module_override['clip_half_range'] = use_half_range and clip_mode in ['GAUSS', 'LAPLACE']

        yield current_module_override


def search_best_local_settings(module, module_name, sg, act_stats, eval_fn, best_overrides_dict, override_gen_fn,
                               **kwargs):
    msglogger.info('Searching optimal quantization in \'%s\'(%s):' % (module_name, module.__class__.__name__))
    overrides_dict = deepcopy(best_overrides_dict)
    best_performance, best_local_override = float("-inf"), OrderedDict()
    normalized_module_name = module_name
    if isinstance(model, nn.DataParallel):
        normalized_module_name = re.sub(r'module\.', '', normalized_module_name)
    for local_override in override_gen_fn(module, module_name, sg, best_overrides_dict, **kwargs):
        if not overrides_dict.get(normalized_module_name, None):
            overrides_dict[normalized_module_name] = OrderedDict()
        overrides_dict[normalized_module_name].update(local_override)
        temp_act_stats = deepcopy(act_stats)
        quantizer = PostTrainLinearQuantizer(deepcopy(model),
                                             bits_activations=None,
                                             bits_parameters=None,
                                             bits_accum=32,
                                             mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                             clip_acts=ClipMode.NONE,
                                             overrides=deepcopy(overrides_dict),
                                             model_activation_stats=deepcopy(temp_act_stats),
                                             inputs_quant_auto_fallback=False,
                                             per_channel_wts=kwargs.get('per_channel', False))
        quantizer.prepare_model(dummy_input)

        current_performance = eval_fn(quantizer.model)
        if not isinstance(module, UNQUANTIZED_MODULES):
            clip_mode = local_override.get('clip_acts', None)
            msglogger.info('\t%s\t score = %.3f\tLayer overrides: %s' %
                           (clip_mode or '', current_performance, local_override))
        else:
            msglogger.info('\t Module is not quantized to int8. Not clipping activations.')
            msglogger.info('\t score = %.3f\tLayer overrides: %s' %
                           (current_performance, local_override))
        if current_performance > best_performance:
            best_performance = current_performance
            best_local_override = local_override

    msglogger.info('\t Choosing overrides: %s' % best_local_override)
    return best_local_override


def ptq_greedy_search(model, dummy_input, eval_fn, calib_eval_fn=None,
                      recurrent=False, act_stats=None,
                      args=None,
                      module_override_gen_fn=None, input_override_gen_fn=None,
                      fold_sequences=True):
    """
    Perform greedy search on Post Train Quantization configuration for the model.
    Args:
        model (nn.Module): the model to quantize
        dummy_input (torch.Tensor): a dummy input to be passed to the model
        eval_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        calib_eval_fn (function): An 'evaluation' function to use for forward passing
          through the model to collection quantization calibration statistics.
          if None provided - will use `eval_fn` as a default.
        recurrent (bool): a flag to indicate whether the model has recurrent connections.
        act_stats (OrderedDict): quant calibration activation stats.
          if None provided - will be calculated on runtime.
        args (dict or argparse.Namespace): command line arguments. alternatively - a dict.
        module_override_gen_fn: A function to generate module overrides.
          assumes signature
          `def module_override_gen_fn(module: nn.Module,
                                      module_name: str,
                                      sg: distiller.SummaryGraph,
                                      overrides_dict: OrderedDict,
                                      **kwargs)-> Generator[OrderedDict, None, None]`
        input_override_gen_fn: Same as module_override_gen_fn, only quantized inputs to the top level layers.
        fold_sequences (bool): fold batch norms before quantizing
    Returns:
        (quantized_model, best_overrides_dict)
    Note:
        It is assumed that `eval_fn` returns a satisfying metric of performance (e.g. accuracy)
        and the greedy search aims to maximize this metric.
    """
    if args is None:
        args = get_default_args()
    elif isinstance(args, dict):
        updated_args = get_default_args()
        updated_args.__dict__.update(args)
        args = updated_args

    if fold_sequences:
        model = fold_batch_norms(model, dummy_input)
    best_overrides_dict = OrderedDict()
    if args.resume_search_from:
        with open(args.resume_search_from, 'r') as f:
            best_overrides_dict = distiller.yaml_ordered_load(f)
        msglogger.info('Loaded search checkpoint from %s' % args.resume_search_from)
    overrides_dict = OrderedDict()
    sg = SummaryGraph(model, dummy_input)
    modules_to_quantize = sg.layers_topological_order(recurrent)
    adjacency_map = sg.adjacency_map()
    modules_dict = OrderedDict(model.named_modules())  # type: OrderedDict[str, nn.Module]
    modules_to_quantize = [m for m in modules_to_quantize
                           if m not in args.qe_no_quant_layers]

    module_override_gen_fn = module_override_gen_fn or module_override_generator
    input_override_gen_fn = input_override_gen_fn or input_override_generator

    calib_eval_fn = calib_eval_fn or eval_fn
    if not act_stats:
        msglogger.info('Collecting stats for model...')
        model_temp = distiller.utils.make_non_parallel_copy(model)
        act_stats = collect_quant_stats(model_temp, calib_eval_fn)
        del model_temp
        if args:
            act_stats_path = '%s_act_stats.yaml' % args.arch
            msglogger.info('Done. Saving act stats into %s' % act_stats_path)
            distiller.yaml_ordered_save(act_stats_path, act_stats)
    msglogger.info('Evaluating baseline score for model...')
    base_score = args.base_score or eval_fn(model)
    msglogger.info("Base score: %.3f" % base_score)

    def recalibrate_stats(module_name, act_stats):
        """
        Re-collects quant-calibration stats for successor modules of the current module.
        """
        msglogger.info('Recalibrating stats...')
        modules_to_recalibrate = {op.name for op in adjacency_map[module_name].successors} & set(act_stats)
        if not modules_to_recalibrate:
            # either there aren't any successors or
            # the successors aren't in the stats file - skip
            return act_stats
        q = PostTrainLinearQuantizer(distiller.utils.make_non_parallel_copy(model),
                                     bits_activations=None,
                                     bits_parameters=None,
                                     bits_accum=32,
                                     mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                     clip_acts=ClipMode.NONE,
                                     overrides=deepcopy(best_overrides_dict),
                                     model_activation_stats=deepcopy(act_stats),
                                     inputs_quant_auto_fallback=False,
                                     per_channel_wts=args.qe_per_channel)
        q.prepare_model(dummy_input)
        # recalibrate on the current best quantized version of the model.
        recalib_act_stats = collect_quant_stats(q.model, calib_eval_fn, modules_to_collect=modules_to_recalibrate)
        msglogger.info('Done.')
        act_stats.update(recalib_act_stats)
        return act_stats

    loaded_from_checkpoint = []
    # Quantize inputs:
    input_modules = get_inputs_to_quantize(sg, args, recurrent)  # top level modules
    for module_name, input_idxs in input_modules.items():
        denormalized_module_name = distiller.denormalize_module_name(model, module_name)
        module = modules_dict[denormalized_module_name]
        if isinstance(module, SKIP_MODULES):
            msglogger.info('Skipping module \'%s\' of type %s.' % (module_name, type(module)))
            continue
        msglogger.info('Quantizing top level inputs for %s' % module_name)

        normalized_module_name = module_name
        if isinstance(model, nn.DataParallel):
            normalized_module_name = re.sub(r'module\.', '', normalized_module_name)
        if normalized_module_name in best_overrides_dict and \
                best_overrides_dict[normalized_module_name].get('input_overrides', None):
            # This means the loaded dict already has the module
            msglogger.info("  Quantizing '%s' based on loaded checkpoint: %s" %
                           (module_name, best_overrides_dict[normalized_module_name]))
            if best_overrides_dict[normalized_module_name].get('bits_activations'):
                loaded_from_checkpoint.append(normalized_module_name)
            continue
        if not best_overrides_dict.get(normalized_module_name, None):
            best_overrides_dict[normalized_module_name] = OrderedDict()
        for input_idx in input_idxs:
            best_module_override = search_best_local_settings(module, module_name, sg, act_stats, eval_fn,
                                                              best_overrides_dict,
                                                              input_override_gen_fn, input_idx=input_idx,
                                                              bits_activations=args.qe_bits_acts,
                                                              bits_weights=args.qe_bits_wts,
                                                              per_channel=args.qe_per_channel)
            best_overrides_dict[normalized_module_name].update(best_module_override)
        # Leave only the input_overrides settings:
        current_input_overrides = best_overrides_dict[normalized_module_name]['input_overrides']
        best_overrides_dict[normalized_module_name] = override_odict(input_overrides=current_input_overrides)

    # Quantize layers as a whole:
    for module_name in modules_to_quantize:
        module = modules_dict[module_name]
        if isinstance(module, SKIP_MODULES):
            msglogger.info('Skipping module \'%s\' of type %s.' % (module_name, module.__class__.__name__))
            continue

        normalized_module_name = module_name
        if isinstance(model, nn.DataParallel):
            normalized_module_name = re.sub(r'module\.', '', normalized_module_name)

        if normalized_module_name in best_overrides_dict and \
                best_overrides_dict[normalized_module_name].get('bits_activations', None)\
                and normalized_module_name not in loaded_from_checkpoint:
            # This means the loaded dict already has the module
            msglogger.info("  Quantizing '%s'(%s) based on loaded checkpoint: %s" %
                           (module_name, module.__class__.__name__, best_overrides_dict[normalized_module_name]))
            loaded_from_checkpoint.append(normalized_module_name)
            continue
        if not best_overrides_dict.get(normalized_module_name, None):
            best_overrides_dict[normalized_module_name] = OrderedDict()
        # Hard coded workaround for avgpool->reshape->fc
        if normalized_module_name == 'fc':
            input_override = override_odict(bits_activations=8,
                                            clip_acts='NONE')
            best_overrides_dict['fc'].update(OrderedDict([
                ('input_overrides', OrderedDict([
                    (0, input_override)
                ]))
            ]))
        best_module_override = search_best_local_settings(module, module_name, sg, act_stats, eval_fn,
                                                          best_overrides_dict,
                                                          module_override_gen_fn,
                                                          bits_activations=args.qe_bits_acts,
                                                          bits_weights=args.qe_bits_wts,
                                                          per_channel=args.qe_per_channel)
        best_overrides_dict[normalized_module_name].update(best_module_override)
        distiller.yaml_ordered_save('%s.ptq_greedy_search.yaml' % args.arch, best_overrides_dict)
        # # end of search - we update the calibration of the next layers:
        # recalibrate_stats(module_name, act_stats)

    quantizer = PostTrainLinearQuantizer(model,
                                         bits_activations=None,
                                         bits_parameters=None,
                                         bits_accum=32,
                                         mode=LinearQuantMode.ASYMMETRIC_SIGNED,
                                         clip_acts=ClipMode.NONE,
                                         overrides=deepcopy(best_overrides_dict),
                                         model_activation_stats=act_stats,
                                         inputs_quant_auto_fallback=False,
                                         per_channel_wts=args.qe_per_channel)
    quantizer.prepare_model(dummy_input)
    msglogger.info('best_overrides_dict: %s' % best_overrides_dict)
    msglogger.info('Best score: %f'% eval_fn(quantizer.model))
    return model, best_overrides_dict


if __name__ == "__main__":
    args = get_default_args()
    args.epochs = float('inf')  # hack for args parsing so there's no error in epochs
    cc = classifier.ClassifierCompressor(args, script_dir=os.path.dirname(__file__))
    args = deepcopy(cc.args)  # Get back args after modifications in ClassifierCompressor.__init__
    eval_data_loader = classifier.load_data(args, load_train=False, load_val=False)

    # quant calibration dataloader:
    args.effective_test_size = args.qe_calib_portion
    args.batch_size = args.qe_calib_batchsize
    calib_data_loader = classifier.load_data(args, fixed_subset=True, load_train=False, load_val=False)
    # logging
    logging.getLogger().setLevel(logging.WARNING)
    msglogger = logging.getLogger(__name__)
    msglogger.setLevel(logging.INFO)

    def test_fn(model):
        top1, top5, losses = classifier.test(eval_data_loader, model, cc.criterion, [cc.tflogger, cc.pylogger], None,
                                             args)
        return top1

    def calib_eval_fn(model):
        classifier.test(calib_data_loader, model, cc.criterion, [], None,
                        args)

    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    args.device = next(model.parameters()).device
    if args.resumed_checkpoint_path:
        args.load_model_path = args.resumed_checkpoint_path
    if args.load_model_path:
        msglogger.info("Loading checkpoint from %s" % args.load_model_path)
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
    dummy_input = torch.rand(*model.input_shape, device=args.device)
    if args.qe_stats_file:
        msglogger.info("Loading stats from %s" % args.qe_stats_file)
        with open(args.qe_stats_file, 'r') as f:
            act_stats = distiller.yaml_ordered_load(f)
    else:
        act_stats = None
    model, overrides = ptq_greedy_search(model, dummy_input, test_fn,
                                         calib_eval_fn=calib_eval_fn, args=args,
                                         act_stats=act_stats)
    # Prepare a compression scheduler yaml config file:
    quantizer_dict = OrderedDict([
        ('class', 'PostTrainLinearQuantizer')
    ])
    quantizer_dict.update(deepcopy(model.quantizer_metadata['params']))
    quantizer_dict['overrides'] = overrides
    quantizer_dict['model_activation_stats'] = os.path.abspath('%s_act_stats.yaml' % args.arch)
    sched_dict = OrderedDict([
        ('quantizers', OrderedDict([
            ('post_train_quantizer', quantizer_dict)
        ]))
    ])
    distiller.yaml_ordered_save('%s.ptqgs_quantizer_sched_dict.yaml' % args.arch, sched_dict)

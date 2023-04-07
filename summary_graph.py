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

import distiller
import re
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.jit as jit
import logging
from collections import OrderedDict, defaultdict
from collections.abc import MutableSequence, Iterable
msglogger = logging.getLogger()


def onnx_name_2_pytorch_name(name):
    # Convert a layer's name from an ONNX name, to a PyTorch name
    # For example:
    #   ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu] ==> layer3.0.relu

    # Split by square brackets
    name_parts = re.findall('\[.*?\]', name)
    name_parts = [part[1:-1] for part in name_parts]

    return '.'.join(name_parts)


class SummaryGraph(object):
    """We use Pytorch's JIT tracer to run a forward pass and generate a trace graph, which
    is an internal representation of the model.  We then use ONNX to "clean" this
    representation.  After builiding a new representation of the graph, we can print
    it to a table, a PNG, add information to nodes, etc.

    The trace is a C++ component and the API is not documented, so we need to dig into some
    Pytorch internals code to understand how to get the info we need.
    https://github.com/pytorch/pytorch/blob/master/torch/onnx/__init__.py
    https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py

    We think that using the trace output to generate a representation of the graph, is
    the best method available in Pytorch, due to the module's dynamic nature.
    Pytorch's module API naturally ignores layers/operations which are implemented as
    torch.autograd.Function, without an nn.Module.  For example:
        out = F.relu(self.bn1(self.conv1(x)))

    Another case where traversing the nn.Module API is not sufficient to create a
    representation of the graph, is the same nn.Module is used several times in the
    graph.  For example:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)    <=== First use of self.relu

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)    <=== Second use of self.relu
    """
    Edge = collections.namedtuple('Edge', 'src dst')

    def __init__(self, model, dummy_input, apply_scope_name_workarounds=True):
        self._src_model = model
        self._named_modules = OrderedDict(model.named_modules())
        self._adj_map = None
        self._layers_topological_order = None
        self._top_level_ops = set()
        model_clone = distiller.make_non_parallel_copy(model)

        # Switch all instances of torch.nn.ModuleList in the model to our DistillerModuleList
        # See documentation of _DistillerModuleList class for details on why this is done
        model_clone, converted_module_names_map = _to_distiller_modulelist(model_clone)

        with torch.onnx.set_training(model_clone, False):
            
            device = distiller.model_device(model_clone)
            dummy_input = distiller.convert_tensors_recursively_to(dummy_input, device=device)
            self.dummy_input = dummy_input
            trace, _ = jit.get_trace_graph(model_clone, dummy_input, _force_outplace=True)

            # As of PyTorch 1.3.0, ONNX trace optimization has an issue that results in incorrect scope names
            # of nodes in the trace graph.
            # These can make it impossible, in some cases, to derive the connectivity of the model using the original
            # module names. So we try to detect these cases and apply workarounds

            # The issue is:
            #   Dropout ops are removed by ONNX trace optimization. However, the op BEFORE the original dropout op
            #   gets the scope name of the dropout op
            pre_dropout_nodes_scope_names = OrderedDict()

            prev_non_dropout_op = None
            for node in trace.graph().nodes():
                kind = node.kind()
                if 'aten' not in kind:
                    continue
                if kind == 'aten::dropout':
                    if prev_non_dropout_op:
                        pre_dropout_nodes_scope_names[node.scopeName()] = prev_non_dropout_op.scopeName()
                else:
                    prev_non_dropout_op = node

            # Let ONNX do the heavy lifting: fusing the convolution nodes; fusing the nodes
            # composing a GEMM operation; etc.
            torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

            graph = trace.graph()
            self.ops = OrderedDict()
            self.module_ops_map = defaultdict(list)
            self.params = OrderedDict()
            self.edges = []
            self.temp = OrderedDict()

            in_out = list(graph.inputs()) + list(graph.outputs())
            for param in in_out:
                self.__add_param(param)

            for node in graph.nodes():
                new_op = self.__create_op(node)

                if apply_scope_name_workarounds:
                    # Here we apply the workaround to the issue of dropout op scope name overriding previous op's
                    # scope name
                    if new_op['name'] in pre_dropout_nodes_scope_names:
                        new_op['orig-name'] = pre_dropout_nodes_scope_names[new_op['name']]
                        new_op['name'] = new_op['orig-name']

                # Convert the graph node's scope name to a PyTorch module name
                module_name = onnx_name_2_pytorch_name(new_op['orig-name'])

                # Get name from before conversion to DistillerModuleList
                module_name = converted_module_names_map[module_name]

                if len(module_name) == 0:
                    # Special case where the module name is an empty string - this happens
                    # when the op is called from the "top-level" of the model
                    new_op['name'] = 'top_level_op'
                else:
                    new_op['name'] = module_name

                # Save the calling module name in the op dict. Denormalize it so it can
                # be directly matched with the actual model
                module_name = distiller.denormalize_module_name(self._src_model, module_name)
                new_op['module-name'] = module_name

                # The node's scope name in the graph corresponds to the module from which the op was called.
                # This means that when ops are invoked from the same module via functional calls or direct
                # operations on tensors, these ops will have the SAME MODEL NAME associated with them.
                # For example:
                #   t = t1 + t2
                #   t = F.relu(t)
                # In this case the add operation and the ReLU operation will have the same name, which is
                # derived from the module they're contained in.
                #
                # Another case where different ops will have the same module name is when a module is reused:
                #   out = self.conv1(x)
                #   out = self.relu(out)    <=== First use of self.relu
                #   out = self.conv2(out)
                #   out = self.relu(out)    <=== Second use of self.relu
                # In this case the graph will have 2 distinct ReLU nodes, with the same scope name.
                #
                # Operators with the same name create very confusing graphs (in ResNet, for example),
                # so we "unroll" them.
                same_module_cnt = len(self.module_ops_map[module_name])
                if same_module_cnt:
                    # TODO: Was this meant to be applied only to 'top_level_ops'? Also, it's not
                    #       applied to the first module that had the same name
                    new_op['name'] += "_%s_%d" % (new_op['type'], same_module_cnt)
                self.module_ops_map[module_name].append(new_op['name'])

                # Finally we register the new op in the ops collection
                self.ops[new_op['name']] = new_op

                for input_ in node.inputs():
                    self.__add_input(new_op, input_)
                    self.edges.append(SummaryGraph.Edge(input_.debugName(), new_op['name']))

                for output in node.outputs():
                    self.__add_output(new_op, output)
                    self.edges.append(SummaryGraph.Edge(new_op['name'], output.debugName()))

                new_op['attrs'] = OrderedDict([(attr_name, node[attr_name]) for attr_name in node.attributeNames()])

        self.__merge_pad_avgpool()
        self.add_macs_attr()
        self.add_footprint_attr()
        self.add_arithmetic_intensity_attr()
        del trace
        del graph
        del model_clone

    def __merge_pad_avgpool(self):
        """ The ONNX trace optimization converts average pool ops to a sequence of 2 operations: pad + pool.
        This "quirk" makes makes it unnecessarily difficult to detect the connectivity between an average pool
        op and its predecessor, and it doesn't serve any purpose in the context of SummaryGraph usages.
        So we get rid of the pad op here.
        """
        pad_op_name = None
        for curr_op_name, curr_op in list(self.ops.items()):
            curr_op_type = curr_op['type']
            if curr_op_type == 'Pad':
                pad_op_name = curr_op_name
            else:
                if pad_op_name and curr_op_type == 'AveragePool':
                    pad_op = self.ops[pad_op_name]
                    if pad_op['module-name'] != curr_op['module-name']:
                        continue
                    merged_op = OrderedDict(curr_op)
                    merged_op['name'] = pad_op_name
                    merged_op['inputs'] = pad_op['inputs']
                    self.ops[pad_op_name] = merged_op
                    self.ops.pop(curr_op_name)
                    self.module_ops_map[merged_op['module-name']].remove(curr_op_name)

                    sequence_input_idx = pad_op['inputs'][0]
                    first_edge = SummaryGraph.Edge(sequence_input_idx, pad_op_name)
                    idx = self.edges.index(first_edge)
                    del self.edges[idx:idx + 4]
                    self.edges.insert(idx, SummaryGraph.Edge(sequence_input_idx, pad_op_name))
                    self.edges.insert(idx + 1, SummaryGraph.Edge(pad_op_name, merged_op['outputs'][0]))

                pad_op_name = None

    def __create_op(self, onnx_node):
        op = OrderedDict()
        op['name'] = onnx_node.scopeName()
        op['orig-name'] = onnx_node.scopeName()
        op['type'] = onnx_node.kind().lstrip('::onnx')
        op['inputs'] = []
        op['outputs'] = []
        op['params'] = []
        return op

    def __add_input(self, op, n):
        param = self.__add_param(n)
        if param is None:
            return
        if param['id'] not in op['inputs']:
            op['inputs'].append(param['id'])

    def __add_output(self, op, n):
        param = self.__add_param(n)
        if param is None:
            return
        if param['id'] not in op['outputs']:
            op['outputs'].append(param['id'])

    def __add_param(self, n):
        if n.debugName() not in self.params:
            param = self.__tensor_desc(n)
            self.params[n.debugName()] = param
        else:
            param = self.params[n.debugName()]
        return param

    def __tensor_desc(self, n):
        tensor = OrderedDict()
        tensor['id'] = n.debugName()
        try:
            # try parsing the FM tensor type.  For example: Float(1, 64, 8, 8)
            s = str(n.node())
            s = s[s.find('(')+1: s.find(')')]
            tensor['shape'] = tuple(map(lambda x: int(x), s.split(',')))
        except ValueError:
            # Size not specified in type
            tensor['shape'] = 0,
        return tensor

    def param_shape(self, param_id):
        return self.params[param_id]['shape']

    @staticmethod
    def volume(dims):
        return np.prod(dims)

    def param_volume(self, param_id):
        return SummaryGraph.volume(self.param_shape(param_id))

    def add_macs_attr(self):
        for op in self.ops.values():
            op['attrs']['MACs'] = 0
            if op['type'] == 'Conv':
                conv_out = op['outputs'][0]
                conv_in = op['inputs'][0]
                conv_w = op['attrs']['kernel_shape']
                groups = op['attrs']['group']
                ofm_vol = self.param_volume(conv_out)
                try:
                    # MACs = volume(OFM) * (#IFM * K^2) / #Groups
                    op['attrs']['MACs'] = int(
                        ofm_vol * SummaryGraph.volume(conv_w) * self.params[conv_in]['shape'][1] / groups)
                except IndexError:
                    # Todo: change the method for calculating MACs
                    msglogger.error("An input to a Convolutional layer is missing shape information "
                                    "(MAC values will be wrong)")
                    msglogger.error("For details see https://github.com/IntelLabs/distiller/issues/168")
                    op['attrs']['MACs'] = 0
            elif op['type'] == 'Gemm':
                conv_out = op['outputs'][0]
                conv_in = op['inputs'][0]
                try:
                    n_ifm = self.param_shape(conv_in)[1]
                    n_ofm = self.param_shape(conv_out)[1]
                except IndexError:
                    msglogger.error("An input to a Convolutional layer is missing shape information.")
                    msglogger.error("For details see https://github.com/IntelLabs/distiller/issues/360")
                    n_ifm = n_ofm = 0

                # MACs = #IFM * #OFM
                op['attrs']['MACs'] = n_ofm * n_ifm

    def add_footprint_attr(self):
        for op in self.ops.values():
            op['attrs']['footprint'] = 0
            if op['type'] in ['Conv', 'Gemm', 'MaxPool']:
                conv_out = op['outputs'][0]
                conv_in = op['inputs'][0]
                ofm_vol = self.param_volume(conv_out)
                ifm_vol = self.param_volume(conv_in)
                if op['type'] == 'Conv' or op['type'] == 'Gemm':
                    if op['type'] == 'Conv':
                        kernel_size =  self.volume(op['attrs']['kernel_shape'])
                        group = op['attrs']['group']
                    else:
                        kernel_size, group = 1, 1
                    n_ifm = self.param_shape(conv_in)[1]
                    n_ofm = self.param_shape(conv_out)[1] 
                    weights_vol = kernel_size * n_ifm * n_ofm / group
                    op['attrs']['n_ifm'] = n_ifm
                    op['attrs']['n_ofm'] = n_ofm
                    op['attrs']['footprint'] = ofm_vol + ifm_vol + weights_vol
                    op['attrs']['fm_vol'] = ofm_vol + ifm_vol
                    op['attrs']['weights_vol'] = weights_vol
                elif op['type'] == 'MaxPool':
                    op['attrs']['footprint'] = ofm_vol + ifm_vol

    def add_arithmetic_intensity_attr(self):
        for op in self.ops.values():
            if op['attrs']['footprint'] == 0:
                op['attrs']['ai'] = 0
            else:
                # integers are enough, and note that we also round up
                op['attrs']['ai'] = ((op['attrs']['MACs']+0.5*op['attrs']['footprint']) // op['attrs']['footprint'])

    def get_attr(self, attr, f=lambda op: True):
        return [op['attrs'][attr] for op in self.ops.values() if attr in op['attrs'] and f(op)]

    def get_ops(self, attr, f=lambda op: True):
        return [op for op in self.ops.values() if attr in op['attrs'] and f(op)]

    def find_op(self, lost_op_name):
        return self.ops.get(distiller.normalize_module_name(lost_op_name), None)

    def find_param(self, data_name):
        return self.params.get(data_name, None)

    def predecessors(self, node, depth, done_list=None, denorm_names=True):
        """Returns a list of <op>'s predecessors"""
        if done_list is None:
            done_list = []

        node_name = node['name'] if isinstance(node, dict) else node
        preds = [edge.src for edge in self.edges if (edge.dst == node_name and
                                                     edge.src not in done_list)]
        done_list += preds

        if depth == 1:
            ret = preds
        else:
            ret = []
            for predecessor in preds:
                ret += self.predecessors(predecessor, depth - 1, done_list, denorm_names)

        if denorm_names:
            ret = [distiller.denormalize_module_name(self._src_model, x) for x in ret]
        return ret

    def predecessors_f(self, node_name, predecessors_types, done_list=None, logging=None, denorm_names=True):
        """Returns a list of <op>'s predecessors, if they match the <predecessors_types> criteria.
        """
        node_name = distiller.normalize_module_name(node_name)
        node = self.find_op(node_name)
        node_is_an_op = True
        if node is None:
            node_is_an_op = False
            node = self.find_param(node_name)
            if node is None:
                msglogger.warning("predecessors_f: Could not find node {}".format(node_name))
                return []

        if done_list is None:
            done_list = []

        done_list.append(node_name)

        if not isinstance(predecessors_types, list):
            predecessors_types = [predecessors_types]

        if node_is_an_op:
            # We check if we found the type of node we're looking for,
            # and that this is not the first node in our search.
            if node['type'] in predecessors_types and len(done_list) > 1:
                return [distiller.denormalize_module_name(self._src_model, node_name) if denorm_names else node_name]

            # This is an operation node
            preds = [edge.src for edge in self.edges if (edge.dst == node_name and
                                                         edge.src not in done_list)]
        else:
            # This is a data node
            preds = [edge.src for edge in self.edges if (edge.dst == node_name and
                                                         edge.src not in done_list)]
        ret = []
        for predecessor in preds:
            ret += self.predecessors_f(predecessor, predecessors_types, done_list, logging, denorm_names)

        return ret

    def successors(self, node, depth, done_list=None, denorm_names=True):
        """Returns a list of <op>'s successors"""
        if done_list is None:
            done_list = []

        node_name = node['name'] if isinstance(node, dict) else node
        succs = [edge.dst for edge in self.edges if (edge.src == node_name and
                                                     edge.dst not in done_list)]
        done_list += succs

        if depth == 1:
            ret = succs
        else:
            ret = []
            for successor in succs:
                ret += self.successors(successor, depth - 1, done_list, denorm_names)

        if denorm_names:
            ret = [distiller.denormalize_module_name(self._src_model, x) for x in ret]
        return ret

    def successors_f(self, node_name, successors_types, done_list=None, logging=None, denorm_names=True):
        """Returns a list of <op>'s successors, if they match the <successors_types> criteria.

        Traverse the graph, starting at node <node_name>, and search for successor
        nodes, that have one of the node types listed in <successors_types>.
        If none is found, then return an empty list.

        <node_name> and the returned list of successors are strings, because
        """
        node_name = distiller.normalize_module_name(node_name)
        node = self.find_op(node_name)
        node_is_an_op = True
        if node is None:
            node_is_an_op = False
            node = self.find_param(node_name)
            if node is None:
                msglogger.warning("successors_f: Could not find node {}".format(node_name))
                return []

        if done_list is None:
            done_list = []

        done_list.append(node_name)

        if not isinstance(successors_types, list):
            successors_types = [successors_types]

        if node_is_an_op:
            # We check if we found the type of node we're looking for,
            # and that this is not the first node in our search.
            if node['type'] in successors_types and len(done_list) > 1:
                return [distiller.denormalize_module_name(self._src_model, node_name) if denorm_names else node_name]

            # This is an operation node
            succs = [edge.dst for edge in self.edges if (edge.src == node_name and
                                                         edge.dst not in done_list)]
        else:
            # This is a data node
            succs = [edge.dst for edge in self.edges if (edge.src == node_name and
                                                         edge.dst not in done_list)]
        ret = []
        for successor in succs:
            ret += self.successors_f(successor, successors_types, done_list, logging, denorm_names)

        return ret

    def named_params_layers(self):
        for param_name, param in self._src_model.named_parameters():
            # remove the extension of param_name, and then normalize it
            # to create a normalized layer name
            normalized_layer_name = distiller.normalize_module_name(
                '.'.join(param_name.split('.')[:-1]))
            sgraph_layer_name = distiller.denormalize_module_name(
                self._src_model, normalized_layer_name)
            yield sgraph_layer_name, param_name, param

    def _dedicated_module_check(self, n, dedicated_modules_only=False):
        if not dedicated_modules_only:
            return True
        module_name = self.ops[n]['module-name']
        module = self._named_modules[module_name]
        return len(self.module_ops_map[module_name]) == 1 and not distiller.has_children(module)

    def adjacency_map(self, dedicated_modules_only=False):
        """Returns a mapping from each op in the graph to its immediate predecessors and successors.

        The keys in the generated mapping are op names, and the values are instances of AdjacentsEntry.

        The op names are "de-normalized", meaning they can be used directly with the underlying model's
        named_modules(), for example.

        Args:
            dedicated_modules_only (bool): If set, the generated mapping will not include any ops that can't be
              associated with a dedicated module within the underlying model. Examples of this will be
              functional calls, such as "F.relu()", and tensor operations, such as "t3 = t1 + t2".
        """
        if self._adj_map and not dedicated_modules_only:
            return self._adj_map
        adj_map = OrderedDict()

        for op_name, op in self.ops.items():

            def op_meta(n):
                return OpSimpleMetadata(distiller.denormalize_module_name(self._src_model, n), self.ops[n]['type'])

            if not self._dedicated_module_check(op_name, dedicated_modules_only):
                continue

            entry = AdjacentsEntry(op_meta(op_name))
            # Find the immediate preceding and succeeding modules. Depth of 1 gets us the
            # input and output tensors, depth of 2 gets the actual modules
            entry.predecessors = [op_meta(n) for n in self.predecessors(op, 2, denorm_names=False)
                                  if self._dedicated_module_check(n, dedicated_modules_only)]
            entry.successors = [op_meta(n) for n in self.successors(op, 2, denorm_names=False)
                                if self._dedicated_module_check(n, dedicated_modules_only)]

            adj_map[entry.op_meta.name] = entry
        self._adj_map = adj_map
        return adj_map

    def layers_topological_order(self, recurrent=False):
        """
        Prepares an ordered list of layers to quantize sequentially. This list has all the layers ordered by their
        topological order in the graph.
        Args:
            recurrent (bool): indication on whether the model might have recurrent connections.
        """
        if self._layers_topological_order:
            return self._layers_topological_order
        adj_map = self.adjacency_map()
        ranked_ops = OrderedDict([(k, _OpRank(v, 0)) for k, v in adj_map.items()])

        def _recurrent_ancestor(ranked_ops_dict, dest_op_name, src_op_name):
            def _is_descendant(parent_op_name, dest_op_name):
                successors_names = [op.name for op in adj_map[parent_op_name].successors]
                if dest_op_name in successors_names:
                    return True
                for succ_name in successors_names:
                    if _is_descendant(succ_name, dest_op_name):
                        return True
                return False

            return _is_descendant(dest_op_name, src_op_name) and \
                   (0 < ranked_ops_dict[dest_op_name].rank < ranked_ops_dict[src_op_name].rank)

        def rank_op(ranked_ops_dict, op_name, rank):
            ranked_ops_dict[op_name].rank = rank
            for child_op in adj_map[op_name].successors:
                # In recurrent models: if a successor is also an ancestor - we don't increment its rank.
                if not recurrent or not _recurrent_ancestor(ranked_ops_dict, child_op.name, op_name):
                    rank_op(ranked_ops_dict, child_op.name, ranked_ops_dict[op_name].rank + 1)

        roots = [k for k, v in adj_map.items() if len(v.predecessors) == 0]
        for root_op_name in roots:
            rank_op(ranked_ops, root_op_name, 0)

        # Take only the modules from the original model
        module_dict = dict(self._src_model.named_modules())
        ret = sorted([k for k in ranked_ops.keys() if k in module_dict],
                     key=lambda k: ranked_ops[k].rank)
        # Check that only the actual roots have a rank of 0
        assert {k for k in ret if ranked_ops[k].rank == 0} <= set(roots)
        self._layers_topological_order = ret
        return ret

    def top_level_ops(self):
        if self._top_level_ops:
            return self._top_level_ops
        for op_name in self.ops:
            if not self.predecessors(op_name, 1):
                self._top_level_ops.add(op_name)
        return self._top_level_ops

    def missing_modules(self):
        """
        Returns a list of ops that aren't registered as modules.
        """
        return [op_name for op_name in self.adjacency_map()
                if not self._dedicated_module_check(op_name, True)]


class _OpRank:
    def __init__(self, adj_entry, rank=None):
        self.adj_entry = adj_entry
        self._rank = rank or 0

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, val):
        self._rank = max(val, self._rank)

    def __repr__(self):
        return '_OpRank(\'%s\' | %d)' % (self.adj_entry.op_meta.name, self.rank)


class OpSimpleMetadata(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __repr__(self):
        return "Op('{}' | {})".format(self.name, self.type)

    def __eq__(self, other):
        return self.name == other.name and self.type == other.type


class AdjacentsEntry(object):
    def __init__(self, op_meta):
        self.op_meta = op_meta
        self.predecessors = []
        self.successors = []

    def __repr__(self):
        return 'OP: {0} ; PREDECESSORS: {1} ; SUCCESSORS: {2}'.format(self.op_meta, self.predecessors, self.successors)

    def __eq__(self, other):
        return self.op_meta == other.op_meta and \
               self.predecessors == other.predecessors and \
               self.successors == other.successors


class _DistillerModuleList(object):
    r"""A almost-drop-in replacement for torch.nn.ModuleList that results in full and unique scope-names when traced

    So why do we need this?
      Some flows in Distiller, such as modules fusion and "net-aware" quantization in PostTrainLinearQuantizer, rely
      on the ability to infer the connectivity within the model, at the Python API level. This is done using
      SummaryGraph, which internally uses PyTorch's trace capabilities. When tracing, each operation
      executed creates a node in the trace, which has a "scope-name". Distiller then uses the "scope-name" to do a
      reverse mapping - map from the trace node back to the actual nn.Module defined in the model code.

      These "scope-names" are generated by tracking the ".forward()" calls of modules. However, The torch.nn.ModuleList
      class itself doesn't have its own forward method. That makes perfect sense - it is only intended to be used as a
      container of modules which the user accesses explicitly.
      Unfortunately, this means that if an operation is part of a ModuleList, the name of the ModuleList instance
      does not appear in the "scope-name". This makes it impossible for us to do the reverse mapping mentioned
      above.

    From here on, we refer to the module which contains the DistillerModuleList instance as the "parent module".

    Similarities to torch.nn.ModuleList:
      * A DistillerModuleList can be indexed like a regular Python list, but the modules it contains are properly
        registered and will be visible to all torch.nn.Module methods.
      * The DistllerModuleList instance is registered as an attribute of the "parent module"
      * This means that in terms of accessing the modules and invoking them, DistillerModuleList behaves exactly the
        same as torch.nn.ModuleList. See the example below.

    Differences vs. torch.nn.ModuleList:
      * DistillerModuleList is NOT a sub-class of torch.nn.Module
      * This means that the modules in the list are NOT sub-modules of the list itself. They are registered as
        sub-modules of the "parent module". That is - the contents of a DistillerModuleList are "flattened" within the
        "parent module".
      * In addition, we can't use the '.' character to denote the "nesting" of a module within the list. We use '_'.
      * All of this means that calls to functions like state_dict() / named_modules() / named_children() / etc. on the
        "parent_module" return different results when this class is used compared to torch.nn.ModuleList.

    At the moment we don't see a usage for this class "in the wild", outside of SummaryGraph generation.
    In the context of SummaryGraph, we're going to take a pre-created model and replace any torch.nn.ModuleList
    instances with DistillerModuleLists. Once that is done, during model execution we expect that lists are being
    used as read-only (no modules are added to/removed from the list). We're not supporting loading state_dict "across"
    converted models.
    This means that:
      * We implement only a subset of the standard API of a Python sequence (see collections.abc.MutableSequence):
        'append()', 'extend()', '__len__()' and '__getitem()_'
        These are the only ones required to perform the conversion for an already created model.
      * We're not implementing:
        'insert()', '__setitem__()' and '__delitem__()'.

    If we see in the future that our assumptions break, we'll add the necessary APIs.

    For all the reasons mentioned above, and to avoid unnecessary confusion for users, we're keeping this class
    internal to summary_graph for now.

    Args:
        name (string): The base name to be used when registering modules added to the list
        parent_module (torch.nn.Module): The module to which the modules added to the list will be registered.
          NOTE: This is expected to be the module containing the list, but we can't enforce this.
        modules (iterable, optional): An iterable of modules to initialize the list with
    """
    def __init__(self, name, parent_module, modules=None):
        self.name = name
        if not isinstance(parent_module, nn.Module):
            raise TypeError('parent_module must be an instance of torch.nn.Module')
        self.parent_module = parent_module
        self._modules = []
        if modules is not None:
            self.extend(modules)

    def _name_for_idx(self, idx):
        return self.name + '_' + str(idx)

    def _verify_on_insertion(self, module, idx):
        if isinstance(module, nn.ModuleList):
            module = _DistillerModuleList(self._name_for_idx(idx), self.parent_module, module)
        if isinstance(module, _DistillerModuleList):
            if module.parent_module != self.parent_module:
                raise ValueError("When nesting one DistillerModuleList within another, both must have the same "
                                 "'parent_module'")
        return module

    def __getitem__(self, idx):
        return self._modules[idx]

    def __len__(self):
        return len(self._modules)

    def append(self, module):
        module = self._verify_on_insertion(module, len(self))
        if not isinstance(module, _DistillerModuleList):
            self.parent_module.add_module(self._name_for_idx(len(self)), module)
        self._modules.append(module)

    def extend(self, modules):
        if not isinstance(modules, Iterable):
            raise TypeError('DistillerModuleList.extend must be called with an iterable, but got ' +
                            modules.__class__.__name__)
        for module in modules:
            self.append(module)

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            # yield prefix, self
            for idx, module in enumerate(self._modules):
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + str(idx)
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def __repr__(self):
        # A simplified version of torch.nn.Module.__repr__
        from torch.nn.modules.module import _addindent

        child_lines = []
        for idx, module in enumerate(self._modules):
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + str(idx) + '): ' + mod_str)

        main_str = self.__class__.__name__ + '('
        if child_lines:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'
        return main_str


def _named_children_with_duplicates(module):
    """Version of torch.nn.Module.named_children() that includes duplicate modules"""
    for name, module in module._modules.items():
        if module is not None:
            yield name, module


def _named_modules_with_duplicates(module, prefix=''):
    """Version of torch.nn.Module.named_modules() that includes duplicate modules"""
    yield prefix, module
    for name, submodule in module._modules.items():
        if submodule is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        for m in _named_modules_with_duplicates(submodule, submodule_prefix):
            yield m


def _to_distiller_modulelist(model):
    """Replaces all instances of torch.nn.ModuleList in a model with DistillerModuleList instances

    Args:
        model (torch.nn.Module): Model to convert
    """
    def convert_container(container):
        # To maintain a similar order of registered modules compared to the original container, we unregister
        # all modules and then register them again
        # We take care to include duplicated modules, which are not returned by the original named_moduels/children
        # implementation in torch.nn.Module
        named_children = OrderedDict(_named_children_with_duplicates(container))
        for n, _ in named_children.items():
            delattr(container, n)
        for name, child in named_children.items():
            if isinstance(child, nn.ModuleList):
                child = _DistillerModuleList(name, container, child)
                to_check = child.modules()
            else:
                to_check = [child]
            setattr(container, name, child)
            for m in to_check:
                if isinstance(m, _DistillerModuleList):
                    continue
                if distiller.has_children(m):
                    convert_container(m)
        return container

    named_modules_orig = OrderedDict([(n, m) for n, m in _named_modules_with_duplicates(model)
                                      if not isinstance(m, nn.ModuleList)])
    model = convert_container(model)
    named_modules_dmlist = OrderedDict(_named_modules_with_duplicates(model))
    converted_module_names_map = OrderedDict(zip(named_modules_dmlist.keys(), named_modules_orig.keys()))

    return model, converted_module_names_map

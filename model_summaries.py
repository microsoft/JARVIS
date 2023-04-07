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

"""Model statistics summaries.

    - weights sparsities
    - optimizer state
    - model details
"""
import os
import pydot
from functools import partial
import pandas as pd
from tabulate import tabulate
import logging
import torch
import torch.optim
import distiller
from .summary_graph import SummaryGraph
from .data_loggers import PythonLogger, CsvLogger

msglogger = logging.getLogger()

__all__ = ['model_summary',
           'weights_sparsity_summary', 'weights_sparsity_tbl_summary',
           'model_performance_summary', 'model_performance_tbl_summary', 'masks_sparsity_tbl_summary',
           'attributes_summary', 'attributes_summary_tbl', 'connectivity_summary',
           'connectivity_summary_verbose', 'connectivity_tbl_summary', 'create_png', 'create_pydot_graph',
           'draw_model_to_file', 'draw_img_classifier_to_file', 'export_img_classifier_to_onnx']


def model_summary(model, what, dataset=None, logdir=''):
    if what.startswith('png'):
        png_fname = os.path.join(logdir, 'model.png')
        draw_img_classifier_to_file(model, png_fname, dataset, what == 'png_w_params')
    elif what == 'sparsity':
        pylogger = PythonLogger(msglogger)
        csvlogger = CsvLogger(logdir=logdir)
        distiller.log_weights_sparsity(model, -1, loggers=[pylogger, csvlogger])
    elif what == 'compute':
        try:
            dummy_input = distiller.get_dummy_input(dataset, distiller.model_device(model))
        except ValueError as e:
            print(e)
            return
        df = model_performance_summary(model, dummy_input, 1)
        t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
        total_macs = df['MACs'].sum()
        print(t)
        print("Total MACs: " + "{:,}".format(total_macs))
    elif what == 'model':
        # print the simple form of the model
        print(model)
    elif what == 'modules':
        # Print the names of non-leaf modules
        # Remember that in PyTorch not every node is a module (e.g. F.relu).
        # Also remember that parameterless modules, like nn.MaxPool2d, can be used multiple
        # times in the same model, but they will only appear once in the modules list.
        nodes = []
        for name, module in model.named_modules():
            # Only print leaf modules
            if len(module._modules) == 0:
                nodes.append([name, module.__class__.__name__])
        print(tabulate(nodes, headers=['Name', 'Type']))
    else:
        raise ValueError("%s is not a supported summary type" % what)


def weights_sparsity_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)',
                               'Cols (%)', 'Rows (%)', 'Ch (%)', '2D (%)', '3D (%)',
                               'Fine (%)', 'Std', 'Mean', 'Abs-Mean'])
    pd.set_option('precision', 2)
    params_size = 0
    sparse_params_size = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            _density = distiller.density(param)
            params_size += torch.numel(param)
            sparse_params_size += param.numel() * _density
            df.loc[len(df.index)] = ([
                name,
                distiller.size_to_str(param.size()),
                torch.numel(param),
                int(_density * param.numel()),
                distiller.sparsity_cols(param)*100,
                distiller.sparsity_rows(param)*100,
                distiller.sparsity_ch(param)*100,
                distiller.sparsity_2D(param)*100,
                distiller.sparsity_3D(param)*100,
                (1-_density)*100,
                param.std().item(),
                param.mean().item(),
                param.abs().mean().item()
            ])

    total_sparsity = (1 - sparse_params_size/params_size)*100

    df.loc[len(df.index)] = ([
        'Total sparsity:',
        '-',
        params_size,
        int(sparse_params_size),
        0, 0, 0, 0, 0,
        total_sparsity,
        0, 0, 0])

    if return_total_sparsity:
        return df, total_sparsity
    return df


def weights_sparsity_tbl_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df, total_sparsity = weights_sparsity_summary(model, return_total_sparsity=True, param_dims=param_dims)
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
    if return_total_sparsity:
        return t, total_sparsity
    return t


def masks_sparsity_summary(model, scheduler, param_dims=[2, 4]):
    df = pd.DataFrame(columns=['Name', 'Fine (%)'])
    pd.set_option('precision', 2)
    params_size = 0
    sparse_params_size = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            mask = scheduler.zeros_mask_dict[name].mask
            if mask is None:
                _density = 1
            else:
                _density = distiller.density(mask)
            params_size += torch.numel(param)
            sparse_params_size += param.numel() * _density
            df.loc[len(df.index)] = ([name, (1-_density)*100])

    assert params_size != 0
    total_sparsity = (1 - sparse_params_size/params_size)*100
    df.loc[len(df.index)] = (['Total sparsity:', total_sparsity])
    return df


def masks_sparsity_tbl_summary(model, scheduler, param_dims=[2, 4]):
    df = masks_sparsity_summary(model, scheduler, param_dims=param_dims)
    return tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")


# Performance data collection  code follows from here down

def conv_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Conv2d)
    if self in memo:
        return
    weights_vol = distiller.volume(self.weight)

    # Multiply-accumulate operations: MACs = volume(OFM) * (#IFM * K^2) / #Groups
    # Bias is ignored
    macs = (distiller.volume(output) *
            (self.in_channels / self.groups * self.kernel_size[0] * self.kernel_size[1]))
    attrs = 'k=' + '('+(', ').join(['%d' % v for v in self.kernel_size])+')'
    module_visitor(self, input, output, df, model, weights_vol, macs, attrs)


def fc_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Linear)
    if self in memo:
        return

    # Multiply-accumulate operations: MACs = #IFM * #OFM
    # Bias is ignored
    weights_vol = macs = distiller.volume(self.weight)
    module_visitor(self, input, output, df, model, weights_vol, macs)


def module_visitor(self, input, output, df, model, weights_vol, macs, attrs=None):
    in_features_shape = input[0].size()
    out_features_shape = output.size()

    mod_name = distiller.model_find_module_name(model, self)
    df.loc[len(df.index)] = ([mod_name, self.__class__.__name__,
                              attrs if attrs is not None else '',
                              distiller.size_to_str(in_features_shape), distiller.volume(input[0]),
                              distiller.size_to_str(out_features_shape), distiller.volume(output),
                              int(weights_vol), int(macs)])


def model_performance_summary(model, dummy_input, batch_size=1):
    """Collect performance data"""
    def install_perf_collector(m):
        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(m.register_forward_hook(
                                    partial(conv_visitor, df=df, model=model, memo=memo)))
        elif isinstance(m, torch.nn.Linear):
            hook_handles.append(m.register_forward_hook(
                                    partial(fc_visitor, df=df, model=model, memo=memo)))

    df = pd.DataFrame(columns=['Name', 'Type', 'Attrs', 'IFM', 'IFM volume',
                               'OFM', 'OFM volume', 'Weights volume', 'MACs'])

    hook_handles = []
    memo = []

    model = distiller.make_non_parallel_copy(model)
    model.apply(install_perf_collector)
    # Now run the forward path and collect the data
    dummy_input = dummy_input.to(distiller.model_device(model))
    model(dummy_input)
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    return df


def model_performance_tbl_summary(model, dummy_input, batch_size):
    df = model_performance_summary(model, dummy_input, batch_size)
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
    return t


def attributes_summary(sgraph, ignore_attrs):
    """Generate a summary of a graph's attributes.

    Args:
        sgraph: a SummaryGraph instance
        ignore_attrs: a list of attributes to ignore in the output datafraem

    Output:
        A Pandas dataframe
    """
    def pretty_val(val):
        if type(val) == int:
            return format(val, ",d")
        return str(val)

    def pretty_attrs(attrs, ignore_attrs):
        ret = ''
        for key, val in attrs.items():
            if key in ignore_attrs:
                continue
            ret += key + ': ' + pretty_val(val) + '\n'
        return ret

    df = pd.DataFrame(columns=['Name', 'Type', 'Attributes'])
    pd.set_option('precision', 5)
    for i, op in enumerate(sgraph.ops.values()):
        df.loc[i] = [op['name'], op['type'], pretty_attrs(op['attrs'], ignore_attrs)]
    return df


def attributes_summary_tbl(sgraph, ignore_attrs):
    df = attributes_summary(sgraph, ignore_attrs)
    return tabulate(df, headers='keys', tablefmt='psql')


def connectivity_summary(sgraph):
    """Generate a summary of each node's connectivity.

    Args:
        sgraph: a SummaryGraph instance
    """
    df = pd.DataFrame(columns=['Name', 'Type', 'Inputs', 'Outputs'])
    pd.set_option('precision', 5)
    for i, op in enumerate(sgraph.ops.values()):
        df.loc[i] = [op['name'], op['type'], op['inputs'], op['outputs']]
    return df


def connectivity_summary_verbose(sgraph):
    """Generate a summary of each node's connectivity, with details
    about the parameters.

    Args:
        sgraph: a SummaryGraph instance
    """
    def format_list(l):
        ret = ''
        for i in l: ret += str(i) + '\n'
        return ret[:-1]

    df = pd.DataFrame(columns=['Name', 'Type', 'Inputs', 'Outputs'])
    pd.set_option('precision', 5)
    for i, op in enumerate(sgraph.ops.values()):
        outputs = []
        for blob in op['outputs']:
            if blob in sgraph.params:
                outputs.append(blob + ": " + str(sgraph.params[blob]['shape']))
        inputs = []
        for blob in op['inputs']:
            if blob in sgraph.params:
                inputs.append(blob + ": " + str(sgraph.params[blob]['shape']))
        inputs = format_list(inputs)
        outputs = format_list(outputs)
        df.loc[i] = [op['name'], op['type'], inputs, outputs]

    return df


def connectivity_tbl_summary(sgraph, verbose=False):
    if verbose:
        df = connectivity_summary_verbose(sgraph)
    else:
        df = connectivity_summary(sgraph)
    return tabulate(df, headers='keys', tablefmt='psql')


def create_pydot_graph(op_nodes_desc, data_nodes, param_nodes, edges, rankdir='TB', styles=None):
    """Low-level API to create a PyDot graph (dot formatted).
    """
    pydot_graph = pydot.Dot('Net', graph_type='digraph', rankdir=rankdir)

    op_node_style = {'shape': 'record',
                     'fillcolor': '#6495ED',
                     'style': 'rounded, filled'}

    for op_node in op_nodes_desc:
        style = op_node_style
        # Check if we should override the style of this node.
        if styles is not None and op_node[0] in styles:
            style = styles[op_node[0]]
        pydot_graph.add_node(pydot.Node(op_node[0], **style, label="\n".join(op_node)))

    for data_node in data_nodes:
        pydot_graph.add_node(pydot.Node(data_node[0], label="\n".join(data_node[1:])))

    node_style = {'shape': 'oval',
                  'fillcolor': 'gray',
                  'style': 'rounded, filled'}

    if param_nodes is not None:
        for param_node in param_nodes:
            pydot_graph.add_node(pydot.Node(param_node[0], **node_style, label="\n".join(param_node[1:])))

    for edge in edges:
        pydot_graph.add_edge(pydot.Edge(edge[0], edge[1]))

    return pydot_graph


def create_png(sgraph, display_param_nodes=False, rankdir='TB', styles=None):
    """Create a PNG object containing a graphiz-dot graph of the network,
    as represented by SummaryGraph 'sgraph'.

    Args:
        sgraph (SummaryGraph): the SummaryGraph instance to draw.
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
    """

    def annotate_op_node(op):
        if op['type'] == 'Conv':
            return ["sh={}".format(distiller.size2str(op['attrs']['kernel_shape'])),
                    "g={}".format(str(op['attrs']['group']))]
        return ''   

    op_nodes = [op['name'] for op in sgraph.ops.values()]
    data_nodes = []
    param_nodes = []
    for id, param in sgraph.params.items():
        n_data = (id, str(distiller.volume(param['shape'])), str(param['shape']))
        if data_node_has_parent(sgraph, id):
            data_nodes.append(n_data)
        else:
            param_nodes.append(n_data)
    edges = sgraph.edges

    if not display_param_nodes:
        # Use only the edges that don't have a parameter source
        non_param_ids = op_nodes + [dn[0] for dn in data_nodes]
        edges = [edge for edge in sgraph.edges if edge.src in non_param_ids]
        param_nodes = None

    op_nodes_desc = [(op['name'], op['type'], *annotate_op_node(op)) for op in sgraph.ops.values()]
    pydot_graph = create_pydot_graph(op_nodes_desc, data_nodes, param_nodes, edges, rankdir, styles)
    png = pydot_graph.create_png()
    return png


def draw_model_to_file(sgraph, png_fname, display_param_nodes=False, rankdir='TB', styles=None):
    """Create a PNG file, containing a graphiz-dot graph of the netowrk represented
    by SummaryGraph 'sgraph'

    Args:
        sgraph (SummaryGraph): the SummaryGraph instance to draw.
        png_fname (string): PNG file name
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
        """
    png = create_png(sgraph, display_param_nodes=display_param_nodes)
    with open(png_fname, 'wb') as fid:
        fid.write(png)


def draw_img_classifier_to_file(model, png_fname, dataset=None, display_param_nodes=False,
                                rankdir='TB', styles=None, input_shape=None):
    """Draw a PyTorch image classifier to a PNG file.  This a helper function that
    simplifies the interface of draw_model_to_file().

    Args:
        model: PyTorch model instance
        png_fname (string): PNG file name
        dataset (string): one of 'imagenet' or 'cifar10'.  This is required in order to
                          create a dummy input of the correct shape.
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
        input_shape (tuple): List of integers representing the input shape.
                             Used only if 'dataset' is None
    """
    dummy_input = distiller.get_dummy_input(dataset=dataset,
                                            device=distiller.model_device(model),
                                            input_shape=input_shape)
    try:
        non_para_model = distiller.make_non_parallel_copy(model)
        g = SummaryGraph(non_para_model, dummy_input)

        draw_model_to_file(g, png_fname, display_param_nodes, rankdir, styles)
        print("Network PNG image generation completed")
    except FileNotFoundError:
        print("An error has occured while generating the network PNG image.")
        print("Please check that you have graphviz installed.")
        print("\t$ sudo apt-get install graphviz")
    finally:
        del non_para_model


def export_img_classifier_to_onnx(model, onnx_fname, dataset, add_softmax=True, **kwargs):
    """Export a PyTorch image classifier to ONNX.

    Args:
        add_softmax: when True, adds softmax layer to the output model.
        kwargs: arguments to be passed to torch.onnx.export
    """
    dummy_input = distiller.get_dummy_input(dataset, distiller.model_device(model))
    # Pytorch doesn't support exporting modules wrapped in DataParallel
    non_para_model = distiller.make_non_parallel_copy(model)

    try:
        if add_softmax:
            # Explicitly add a softmax layer, because it is needed for the ONNX inference phase.
            # TorchVision models use nn.CrossEntropyLoss for computing the loss,
            # instead of adding a softmax layer
            non_para_model.original_forward = non_para_model.forward
            softmax = torch.nn.Softmax(dim=-1)
            non_para_model.forward = lambda input: softmax(non_para_model.original_forward(input))
        torch.onnx.export(non_para_model, dummy_input, onnx_fname, **kwargs)
        msglogger.info('Exported the model to ONNX format at %s' % os.path.realpath(onnx_fname))
    finally:
        del non_para_model


def data_node_has_parent(g, id):
    for edge in g.edges:
        if edge.dst == id:
            return True
    return False

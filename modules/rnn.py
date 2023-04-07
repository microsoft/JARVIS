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

import torch
import torch.nn as nn
import numpy as np
from .eltwise import EltwiseAdd, EltwiseMult
from itertools import product

__all__ = ['DistillerLSTMCell', 'DistillerLSTM', 'convert_model_to_distiller_lstm']


# There is prevalent use of looping that depends on tensor sizes done in this implementation.
# This does not play well with the PyTorch tracing mechanism, and emits several different warnings.
# For "simple" cases, such as SummaryGraph creating a single trace based on a single forward pass,
# this is not an actual problem.
# TODO: Check if/how it's possible to have a tracer-friendly implementation


class DistillerLSTMCell(nn.Module):
    """
    A single LSTM block.
    The calculation of the output takes into account the input and the previous output and cell state:
    https://pytorch.org/docs/stable/nn.html#lstmcell
    Args:
        input_size (int): the size of the input
        hidden_size (int): the size of the hidden state / output
        bias (bool): use bias. default: True

    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(DistillerLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Treat f,i,o,c_ex as one single object:
        self.fc_gate_x = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.fc_gate_h = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.eltwiseadd_gate = EltwiseAdd()
        # Apply activations separately:
        self.act_f = nn.Sigmoid()
        self.act_i = nn.Sigmoid()
        self.act_o = nn.Sigmoid()
        self.act_g = nn.Tanh()
        # Calculate cell:
        self.eltwisemult_cell_forget = EltwiseMult()
        self.eltwisemult_cell_input = EltwiseMult()
        self.eltwiseadd_cell = EltwiseAdd()
        # Calculate hidden:
        self.act_h = nn.Tanh()
        self.eltwisemult_hidden = EltwiseMult()
        self.init_weights()

    def forward(self, x, h=None):
        """
        Implemented as defined in https://pytorch.org/docs/stable/nn.html#lstmcell.
        """
        x_bsz, x_device = x.size(1), x.device
        if h is None:
            h = self.init_hidden(x_bsz, device=x_device)
        
        h_prev, c_prev = h
        fc_gate = self.eltwiseadd_gate(self.fc_gate_x(x), self.fc_gate_h(h_prev))
        i, f, g, o = torch.chunk(fc_gate, 4, dim=1)
        i, f, g, o = self.act_i(i), self.act_f(f), self.act_g(g), self.act_o(o)
        cf, ci = self.eltwisemult_cell_forget(f, c_prev), self.eltwisemult_cell_input(i, g)
        c = self.eltwiseadd_cell(cf, ci)
        h = self.eltwisemult_hidden(o, self.act_h(c))
        return h, c

    def init_hidden(self, batch_size, device='cuda:0'):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        return h_0, c_0

    def init_weights(self):
        initrange = 1 / np.sqrt(self.hidden_size)
        self.fc_gate_x.weight.data.uniform_(-initrange, initrange)
        self.fc_gate_h.weight.data.uniform_(-initrange, initrange)

    def to_pytorch_impl(self):
        module = nn.LSTMCell(self.input_size, self.hidden_size, self.bias)
        module.weight_hh, module.weight_ih = \
            nn.Parameter(self.fc_gate_h.weight.clone().detach()), \
            nn.Parameter(self.fc_gate_x.weight.clone().detach())
        if self.bias:
            module.bias_hh, module.bias_ih = \
                nn.Parameter(self.fc_gate_h.bias.clone().detach()), \
                nn.Parameter(self.fc_gate_x.bias.clone().detach())
        return module

    @staticmethod
    def from_pytorch_impl(lstmcell: nn.LSTMCell):
        module = DistillerLSTMCell(input_size=lstmcell.input_size, hidden_size=lstmcell.hidden_size, bias=lstmcell.bias)
        module.fc_gate_x.weight = nn.Parameter(lstmcell.weight_ih.clone().detach())
        module.fc_gate_h.weight = nn.Parameter(lstmcell.weight_hh.clone().detach())
        if lstmcell.bias:
            module.fc_gate_x.bias = nn.Parameter(lstmcell.bias_ih.clone().detach())
            module.fc_gate_h.bias = nn.Parameter(lstmcell.bias_hh.clone().detach())

        return module

    def __repr__(self):
        return "%s(%d, %d)" % (self.__class__.__name__, self.input_size, self.hidden_size)


def process_sequence_wise(cell, x, h=None):
    """
    Process the entire sequence through an LSTMCell.
    Args:
         cell (DistillerLSTMCell): the cell.
         x (torch.Tensor): the input
         h (tuple of torch.Tensor-s): the hidden states of the LSTMCell.
    Returns:
         y (torch.Tensor): the output
         h (tuple of torch.Tensor-s): the new hidden states of the LSTMCell.
    """
    results = []
    for step in x:
        y, h = cell(step, h)
        results.append(y)
        h = (y, h)
    return torch.stack(results), h


def _repackage_hidden_unidirectional(h):
    """
    Repackages the hidden state into nn.LSTM format. (unidirectional use)
    """
    h_all = [t[0] for t in h]
    c_all = [t[1] for t in h]
    return torch.stack(h_all, 0), torch.stack(c_all, 0)


def _repackage_hidden_bidirectional(h_result):
    """
    Repackages the hidden state into nn.LSTM format. (bidirectional use)
    """
    h_all = [t[0] for t in h_result]
    c_all = [t[1] for t in h_result]
    return torch.cat(h_all, dim=0), torch.cat(c_all, dim=0)


def _unpack_bidirectional_input_h(h):
    """
    Unpack the bidirectional hidden states into states of the 2 separate directions.
    """
    h_t, c_t = h
    h_front, h_back = h_t[::2], h_t[1::2]
    c_front, c_back = c_t[::2], c_t[1::2]
    h_front = (h_front, c_front)
    h_back = (h_back, c_back)
    return h_front, h_back


class DistillerLSTM(nn.Module):
    """
    A modular implementation of an LSTM module.
    Args:
        input_size (int): size of the input
        hidden_size (int): size of the hidden connections and output.
        num_layers (int): number of LSTMCells
        bias (bool): use bias
        batch_first (bool): the format of the sequence is (batch_size, seq_len, dim). default: False
        dropout : dropout factor
        bidirectional (bool): Whether or not the LSTM is bidirectional. default: False (unidirectional).
        bidirectional_type (int): 1 or 2, corresponds to type 1 and type 2 as per
            https://github.com/pytorch/pytorch/issues/4930. default: 2
    """
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False,
                 dropout=0.5, bidirectional=False, bidirectional_type=2):
        super(DistillerLSTM, self).__init__()
        if num_layers < 1:
            raise ValueError("Number of layers has to be at least 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional_type = bidirectional_type

        if bidirectional:
            # Following https://github.com/pytorch/pytorch/issues/4930 -
            if bidirectional_type == 1:
                raise NotImplementedError
                # # Process each timestep at the entire layers chain -
                # # each timestep is forwarded through `front` and `back` chains independently,
                # # similarily to a unidirectional LSTM.
                # self.cells = self._create_cells_list(1)
                # self.cells_reverse = self._create_cells_list(2)
                # self.forward_fn = self.process_layer_wise
                # self.layer_chain_fn = self._layer_chain_bidirectional_type1

            elif bidirectional_type == 2:
                # Process the entire sequence at each layer consecutively -
                # the output of one layer is the sequence processed through the `front` and `back` cells
                # and the input to the next layers are both `output_front` and `output_back`.
                self.cells = self._create_cells_list(2)
                self.cells_reverse = self._create_cells_list(2)
                self.forward_fn = self._bidirectional_type2_forward

            else:
                raise ValueError("The only allowed types are [1, 2].")
        else:
            self.cells = self._create_cells_list()
            self.forward_fn = self.process_layer_wise
            self.layer_chain_fn = self._layer_chain_unidirectional

        self.dropout = nn.Dropout(dropout)
        self.dropout_factor = dropout

    def _create_cells_list(self, hidden_size_scale=1):
        # We always have the first layer
        cells = nn.ModuleList([DistillerLSTMCell(self.input_size, self.hidden_size, self.bias)])
        for i in range(1, self.num_layers):
            cells.append(DistillerLSTMCell(hidden_size_scale * self.hidden_size, self.hidden_size, self.bias))
        return cells

    def forward(self, x, h=None):
        is_packed_seq = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed_seq:
            return self.packed_sequence_forward(x, h)

        if self.batch_first:
            # Transpose to sequence_first format
            x = x.transpose(0, 1)
        x_bsz = x.size(1)

        if h is None:
            h = self.init_hidden(x_bsz)

        y, h = self.forward_fn(x, h)

        if self.batch_first:
            # Transpose back to batch_first format
            y = y.transpose(0, 1)
        return y, h

    def packed_sequence_forward(self, x, h=None):
        # Packed sequence treatment -
        # the sequences are not of the same size, hence
        # we split the padded tensor into the sequences.
        # we take the sequence from each row in the batch.
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_bsz = x.size(0)
        if h is None:
            h = self.init_hidden(x_bsz)
        y_results = []
        h_results = []
        for i, (sequence, seq_len) in enumerate(zip(x, lengths)):
            # Take the previous state according to the current batch.
            # we unsqueeze to have a 3D tensor
            h_current = (h[0][:, i, :].unsqueeze(1), h[1][:, i, :].unsqueeze(1))
            # Take only the relevant timesteps according to seq_len
            sequence = sequence[:seq_len].unsqueeze(1)  # sequence.shape = (seq_len, batch_size=1, input_dim)
            # forward pass:
            y, h_current = self.forward_fn(sequence, h_current)
            # sequeeze back the batch into a single sequence
            y_results.append(y.squeeze(1))
            h_results.append(h_current)
        # our result is a packed sequence
        y = nn.utils.rnn.pack_sequence(y_results)
        # concat hidden states per batches
        h = torch.cat([t[0] for t in h_results], dim=1), torch.cat([t[1] for t in h_results], dim=1)
        return y, h

    def process_layer_wise(self, x, h):
        results = []
        for step in x:
            y, h = self.layer_chain_fn(step, h)
            results.append(y)
        return torch.stack(results), h

    def _bidirectional_type2_forward(self, x, h):
        """
        Processes the entire sequence through a layer and passes the output sequence to the next layer.
        """
        out = x
        h_h_result = []
        h_c_result = []
        (h_front_all, c_front_all), (h_back_all, c_back_all) = _unpack_bidirectional_input_h(h)
        for i, (cell_front, cell_back) in enumerate(zip(self.cells, self.cells_reverse)):
            h_front, h_back = (h_front_all[i], c_front_all[i]), (h_back_all[i], c_back_all[i])

            # Sequence treatment:
            out_front, h_front = process_sequence_wise(cell_front, out, h_front)
            out_back, h_back = process_sequence_wise(cell_back, out.flip([0]), h_back)
            out = torch.cat([out_front, out_back.flip([0])], dim=-1)

            h_h_result += [h_front[0], h_back[0]]
            h_c_result += [h_front[1], h_back[1]]
            if i < self.num_layers-1:
                out = self.dropout(out)
        h = torch.stack(h_h_result, dim=0), torch.stack(h_c_result, dim=0)
        return out, h

    def _layer_chain_bidirectional_type1(self, x, h):
        # """
        # Process a single timestep through the entire bidirectional layer chain.
        # """
        # (h_front_all, c_front_all), (h_back_all, c_back_all) = _repackage_bidirectional_input_h(h)
        # h_result = []
        # out_front, out_back = x, x.flip([0])
        # for i, (cell_front, cell_back) in enumerate(zip(self.cells, self.cells_reverse)):
        #     h_front, h_back = (h_front_all[i], c_front_all[i]), (h_back_all[i], c_back_all[i])
        #     h_front, c_front = cell_front(out_front, h_front)
        #     h_back, c_back = cell_back(out_back, h_back)
        #     out_front, out_back = h_front, h_back
        #     if i < self.num_layers-1:
        #         out_front, out_back = self.dropout(out_front), self.dropout(out_back)
        #     h_current = torch.stack([h_front, h_back]), torch.stack([c_front, c_back])
        #     h_result.append(h_current)
        # h_result = _repackage_hidden_bidirectional(h_result)
        # return torch.cat([out_front, out_back], dim=-1), h_result
        raise NotImplementedError

    def _layer_chain_unidirectional(self, step, h):
        """
        Process a single timestep through the entire unidirectional layer chain.
        """
        step_bsz = step.size(0)
        if h is None:
            h = self.init_hidden(step_bsz)
        h_all, c_all = h
        h_result = []
        out = step
        for i, cell in enumerate(self.cells):
            h = h_all[i], c_all[i]
            out, hid = cell(out, h)
            if i < self.num_layers-1:
                out = self.dropout(out)
            h_result.append((out, hid))
        h_result = _repackage_hidden_unidirectional(h_result)
        return out, h_result

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        n_dir = 2 if self.bidirectional else 1
        return (weight.new_zeros(self.num_layers * n_dir, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers * n_dir, batch_size, self.hidden_size))

    def init_weights(self):
        for cell in self.hidden_cells:
            cell.init_weights()

    def flatten_parameters(self):
        pass

    def to_pytorch_impl(self):
        if self.bidirectional and self.bidirectional_type == 1:
            raise TypeError("Pytorch implementation of bidirectional LSTM doesn't support type 1.")

        module = nn.LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         dropout=self.dropout_factor,
                         bias=self.bias,
                         batch_first=self.batch_first,
                         bidirectional=self.bidirectional)
        param_gates = ['i', 'h']

        param_types = ['weight']
        if self.bias:
            param_types.append('bias')

        suffixes = ['']
        if self.bidirectional:
            suffixes.append('_reverse')

        for i in range(self.num_layers):
            for ptype, pgate, psuffix in product(param_types, param_gates, suffixes):
                cell = self.cells[i] if psuffix == '' else self.cells_reverse[i]
                lstm_pth_param_name = "%s_%sh_l%d%s" % (ptype, pgate, i, psuffix)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param_tensor = getattr(gate, ptype).clone().detach()

                # same as `module.weight_ih_l0 = nn.Parameter(param_tensor)`:
                setattr(module, lstm_pth_param_name, nn.Parameter(param_tensor))

        module.flatten_parameters()
        return module

    @staticmethod
    def from_pytorch_impl(lstm: nn.LSTM):
        bidirectional = lstm.bidirectional

        module = DistillerLSTM(lstm.input_size, lstm.hidden_size, lstm.num_layers, bias=lstm.bias,
                               batch_first=lstm.batch_first,
                               dropout=lstm.dropout, bidirectional=bidirectional)
        param_gates = ['i', 'h']

        param_types = ['weight']
        if lstm.bias:
            param_types.append('bias')

        suffixes = ['']
        if bidirectional:
            suffixes.append('_reverse')

        for i in range(lstm.num_layers):
            for ptype, pgate, psuffix in product(param_types, param_gates, suffixes):
                cell = module.cells[i] if psuffix == '' else module.cells_reverse[i]
                lstm_pth_param_name = "%s_%sh_l%d%s" % (ptype, pgate, i, psuffix)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param_tensor = getattr(lstm, lstm_pth_param_name).clone().detach()  # e.g. `lstm.weight_ih_l0.detach()`
                setattr(gate, ptype, nn.Parameter(param_tensor))

        return module

    def __repr__(self):
        return "%s(%d, %d, num_layers=%d, dropout=%.2f, bidirectional=%s)" % \
               (self.__class__.__name__,
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.dropout_factor,
                self.bidirectional)


def convert_model_to_distiller_lstm(model: nn.Module):
    """
    Replaces all `nn.LSTM`s and `nn.LSTMCell`s in the model with distiller versions.
    Args:
        model (nn.Module): the model
    """
    if isinstance(model, nn.LSTMCell):
        return DistillerLSTMCell.from_pytorch_impl(model)
    if isinstance(model, nn.LSTM):
        return DistillerLSTM.from_pytorch_impl(model)
    for name, module in model.named_children():
        module = convert_model_to_distiller_lstm(module)
        setattr(model, name, module)

    return model

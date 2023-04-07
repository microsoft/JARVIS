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

"""Policies for scheduling by a CompressionScheduler instance.

- PruningPolicy: prunning policy
- RegularizationPolicy: regulization scheduling
- LRPolicy: learning-rate decay scheduling
- QuantizationPolicy: quantization scheduling
"""
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from collections import namedtuple, OrderedDict
import logging
import distiller


__all__ = ['PruningPolicy', 'RegularizationPolicy', 'QuantizationPolicy', 'LRPolicy', 'ScheduledTrainingPolicy',
           'PolicyLoss', 'LossComponent']

msglogger = logging.getLogger()
PolicyLoss = namedtuple('PolicyLoss', ['overall_loss', 'loss_components'])
LossComponent = namedtuple('LossComponent', ['name', 'value'])


class ScheduledTrainingPolicy(object):
    """ Base class for all scheduled training policies.

    The CompressionScheduler invokes these methods as the training progresses.
    """
    def __init__(self, classes=None, layers=None):
        self.classes = classes
        self.layers = layers

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        """A new epcoh is about to begin"""
        pass

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch,
                           zeros_mask_dict, meta, optimizer=None):
        """The forward-pass of a new mini-batch is about to begin"""
        pass

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss, zeros_mask_dict,
                             optimizer=None):
        """The mini-batch training pass has completed the forward-pass,
        and is about to begin the backward pass.

        This callback receives a 'loss' argument. The callback should not modify this argument, but it can
        optionally return an instance of 'PolicyLoss' which will be used in place of `loss'.

        Note: The 'loss_components' parameter within 'PolicyLoss' should contain any new, individual loss components
              the callback contributed to 'overall_loss'. It should not contain the incoming 'loss' argument.
        """
        pass

    def before_parameter_optimization(self, model, epoch, minibatch_id, minibatches_per_epoch,
                                      zeros_mask_dict, meta, optimizer):
        """The mini-batch training pass has completed the backward-pass,
        and the optimizer is about to update the weights."""
        pass

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        """The mini-batch training pass has ended"""
        pass

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        """The current epoch has ended"""
        pass


class PruningPolicy(ScheduledTrainingPolicy):
    """Base class for pruning policies.
    """
    def __init__(self, pruner, pruner_args, classes=None, layers=None):
        """
        Arguments:
            mask_on_forward_only: controls what we do after the weights are updated by the backward pass.
            In issue #53 (https://github.com/IntelLabs/distiller/issues/53) we explain why in some
            cases masked weights will be updated to a non-zero value, even if their gradients are masked
            (e.g. when using SGD with momentum). Therefore, to circumvent this weights-update performed by
            the backward pass, we usually mask the weights again - right after the backward pass.  To
            disable this masking set:
                pruner_args['mask_on_forward_only'] = False

            use_double_copies: when set to `True`, two sets of weights are used. In the forward-pass we use
            masked weights to compute the loss, but in the backward-pass we update the unmasked weights (using
            gradients computed from the masked-weights loss).

            mini_batch_pruning_frequency: this controls pruning scheduling at the mini-batch granularity.  Every
            mini_batch_pruning_frequency training steps (i.e. mini_batches) we perform pruning.  This provides more
            fine-grained control over pruning than that provided by CompressionScheduler (epoch granularity).
            When setting 'mini_batch_pruning_frequency' to a value other than zero, make sure to configure the policy's
            schedule to once-every-epoch.

            fold_batchnorm: when set to `True`, the weights of BatchNorm modules are folded into the the weights of
            Conv-2D modules (if Conv2D->BN edges exist in the model graph).  Each weights filter is attenuated using
            a different pair of (gamma, beta) coefficients, so `fold_batchnorm` is relevant for fine-grained and
            filter-ranking pruning methods.  We attenuate using the running values of the mean and variance, as is
            done in quantization.
            This control argument is only supported for Conv-2D modules (i.e. other convolution operation variants and
            Linear operations are not supported).
         """
        super(PruningPolicy, self).__init__(classes, layers)
        self.pruner = pruner
        # Copy external policy configuration, if available
        if pruner_args is None:
            pruner_args = {}
        self.levels = pruner_args.get('levels', None)
        self.keep_mask = pruner_args.get('keep_mask', False)
        self.mini_batch_pruning_frequency = pruner_args.get('mini_batch_pruning_frequency', 0)
        self.mask_on_forward_only = pruner_args.get('mask_on_forward_only', False)
        self.mask_gradients = pruner_args.get('mask_gradients', False)
        if self.mask_gradients and not self.mask_on_forward_only:
            raise ValueError("mask_gradients and (not mask_on_forward_only) are mutually exclusive")
        self.backward_hook_handle = None   # The backward-callback handle
        self.use_double_copies = pruner_args.get('use_double_copies', False)
        self.discard_masks_at_minibatch_end = pruner_args.get('discard_masks_at_minibatch_end', False)
        self.skip_first_minibatch = pruner_args.get('skip_first_minibatch', False)
        self.fold_bn = pruner_args.get('fold_batchnorm', False)
        # These are required for BN-folding.  We cache them to improve performance
        self.named_modules = None
        self.sg = None
        # Initialize state
        self.is_last_epoch = False
        self.is_initialized = False

    @staticmethod
    def _fold_batchnorm(model, param_name, param, named_modules, sg):
        def _get_all_parameters(param_module, bn_module):
            w, b, gamma, beta = param_module.weight, param_module.bias, bn_module.weight, bn_module.bias
            if not bn_module.affine:
                gamma = 1.
                beta = 0.
            return w, b, gamma, beta

        def get_bn_folded_weights(conv_module, bn_module):
            """Compute the weights of `conv_module` after folding successor BN layer.

            In inference, DL frameworks and graph-compilers fold the batch normalization into
            the weights as defined by equations 20 and 21 of https://arxiv.org/pdf/1806.08342.pdf

            :param conv_module: nn.Conv2d module
            :param bn_module: nn.BatchNorm2d module which succeeds `conv_module`
            :return: Folded weights
            """
            w, b, gamma, beta = _get_all_parameters(conv_module, bn_module)
            with torch.no_grad():
                sigma_running = torch.sqrt(bn_module.running_var + bn_module.eps)
                w_corrected = w * (gamma / sigma_running).view(-1, 1, 1, 1)
            return w_corrected

        layer_name = distiller.utils.param_name_2_module_name(param_name)
        if not isinstance(named_modules[layer_name], nn.Conv2d):
            return param

        bn_layers = sg.successors_f(layer_name, ['BatchNormalization'])
        if bn_layers:
            assert len(bn_layers) == 1
            bn_module = named_modules[bn_layers[0]]
            conv_module = named_modules[layer_name]
            param = get_bn_folded_weights(conv_module, bn_module)
        return param

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        msglogger.debug("Pruner {} is about to prune".format(self.pruner.name))
        self.is_last_epoch = meta['current_epoch'] == (meta['ending_epoch'] - 1)
        if self.levels is not None:
            self.pruner.levels = self.levels

        meta['model'] = model
        is_initialized = self.is_initialized

        if self.fold_bn:
            # Cache this information (required for BN-folding) to improve performance
            self.named_modules = OrderedDict(model.named_modules())
            dummy_input = torch.randn(model.input_shape)
            self.sg = distiller.SummaryGraph(model, dummy_input)

        for param_name, param in model.named_parameters():
            if self.fold_bn:
                param = self._fold_batchnorm(model, param_name, param, self.named_modules, self.sg)
            if not is_initialized:
                # Initialize the maskers
                masker = zeros_mask_dict[param_name]
                masker.use_double_copies = self.use_double_copies
                masker.mask_on_forward_only = self.mask_on_forward_only
                # register for the backward hook of the parameters
                if self.mask_gradients:
                    masker.backward_hook_handle = param.register_hook(masker.mask_gradient)

                self.is_initialized = True
                if not self.skip_first_minibatch:
                    self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)
            else:
                self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch,
                           zeros_mask_dict, meta, optimizer=None):
        set_masks = False
        global_mini_batch_id = epoch * minibatches_per_epoch + minibatch_id
        if ((minibatch_id > 0) and
            (self.mini_batch_pruning_frequency != 0) and
            (global_mini_batch_id % self.mini_batch_pruning_frequency == 0)):
            # This is _not_ the first mini-batch of a new epoch (performed in on_epoch_begin)
            # and a pruning step is scheduled
            set_masks = True

        if self.skip_first_minibatch and global_mini_batch_id == 1:
            # Because we skipped the first mini-batch of the first epoch (global_mini_batch_id == 0)
            set_masks = True

        for param_name, param in model.named_parameters():
            if set_masks:
                if self.fold_bn:
                    param = self._fold_batchnorm(model, param_name, param, self.named_modules, self.sg)
                self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)
            zeros_mask_dict[param_name].apply_mask(param)

    def before_parameter_optimization(self, model, epoch, minibatch_id, minibatches_per_epoch,
                                      zeros_mask_dict, meta, optimizer):
        for param_name, param in model.named_parameters():
            zeros_mask_dict[param_name].revert_weights(param)

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        if self.discard_masks_at_minibatch_end:
            for param_name, param in model.named_parameters():
                zeros_mask_dict[param_name].mask = None

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        """The current epoch has ended"""
        if self.is_last_epoch:
            for param_name, param in model.named_parameters():
                masker = zeros_mask_dict[param_name]
                if self.keep_mask:
                    masker.use_double_copies = False
                    masker.mask_on_forward_only = False
                    masker.mask_tensor(param)
                if masker.backward_hook_handle is not None:
                    masker.backward_hook_handle.remove()
                    masker.backward_hook_handle = None


class RegularizationPolicy(ScheduledTrainingPolicy):
    """Regularization policy.

    """
    def __init__(self, regularizer, keep_mask=False):
        super(RegularizationPolicy, self).__init__()
        self.regularizer = regularizer
        self.keep_mask = keep_mask
        self.is_last_epoch = False

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        self.is_last_epoch = meta['current_epoch'] == (meta['ending_epoch'] - 1)

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss,
                             zeros_mask_dict, optimizer=None):
        regularizer_loss = torch.tensor(0, dtype=torch.float, device=loss.device)

        for param_name, param in model.named_parameters():
            self.regularizer.loss(param, param_name, regularizer_loss, zeros_mask_dict)

        policy_loss = PolicyLoss(loss + regularizer_loss,
                                 [LossComponent(self.regularizer.__class__.__name__ + '_loss', regularizer_loss)])
        return policy_loss

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        if self.regularizer.threshold_criteria is None:
            return

        keep_mask = False
        if (minibatches_per_epoch-1 == minibatch_id) and self.is_last_epoch and self.keep_mask:
            # If this is the last mini_batch in the last epoch, and the scheduler wants to
            # keep the regularization mask, then now is the time ;-)
            msglogger.info("RegularizationPolicy is keeping the regularization mask")
            keep_mask = True

        for param_name, param in model.named_parameters():
            self.regularizer.threshold(param, param_name, zeros_mask_dict)
            if keep_mask:
                zeros_mask_dict[param_name].is_regularization_mask = False
            zeros_mask_dict[param_name].apply_mask(param)


class LRPolicy(ScheduledTrainingPolicy):
    """Learning-rate decay scheduling policy.

    """
    def __init__(self, lr_scheduler):
        super(LRPolicy, self).__init__()
        self.lr_scheduler = lr_scheduler

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Note: ReduceLROnPlateau doesn't inherit from _LRScheduler
            self.lr_scheduler.step(kwargs['metrics'][self.lr_scheduler.mode],
                                   epoch=meta['current_epoch'] + 1)
        else:
            self.lr_scheduler.step(epoch=meta['current_epoch'] + 1)


class QuantizationPolicy(ScheduledTrainingPolicy):
    def __init__(self, quantizer):
        super(QuantizationPolicy, self).__init__()
        self.quantizer = quantizer
        self.quantizer.prepare_model()
        self.quantizer.quantize_params()

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        # After parameters update, quantize the parameters again
        # (Doing this here ensures the model parameters are quantized at training completion (and at validation time)
        self.quantizer.quantize_params()

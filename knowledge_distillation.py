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

import torch
import torch.nn.functional as F
from collections import namedtuple

from .policy import ScheduledTrainingPolicy, PolicyLoss, LossComponent

DistillationLossWeights = namedtuple('DistillationLossWeights',
                                     ['distill', 'student', 'teacher'])


def add_distillation_args(argparser, arch_choices=None, enable_pretrained=False):
    """
    Helper function to make it easier to add command line arguments for knowledge distillation to any script

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
        arch_choices: Optional list of choices to be enforced by the parser for model selection
        enable_pretrained (bool): Flag to enable/disable argument for "pre-trained" models.
    """
    group = argparser.add_argument_group('Knowledge Distillation Training Arguments')
    group.add_argument('--kd-teacher', choices=arch_choices, metavar='ARCH',
                       help='Model architecture for teacher model')
    if enable_pretrained:
        group.add_argument('--kd-pretrained', action='store_true', help='Use pre-trained model for teacher')
    group.add_argument('--kd-resume', type=str, default='', metavar='PATH',
                       help='Path to checkpoint from which to load teacher weights')
    group.add_argument('--kd-temperature', '--kd-temp', dest='kd_temp', type=float, default=1.0, metavar='TEMP',
                       help='Knowledge distillation softmax temperature')
    group.add_argument('--kd-distill-wt', '--kd-dw', type=float, default=0.5, metavar='WEIGHT',
                       help='Weight for distillation loss (student vs. teacher soft targets)')
    group.add_argument('--kd-student-wt', '--kd-sw', type=float, default=0.5, metavar='WEIGHT',
                       help='Weight for student vs. labels loss')
    group.add_argument('--kd-teacher-wt', '--kd-tw', type=float, default=0.0, metavar='WEIGHT',
                       help='Weight for teacher vs. labels loss')
    group.add_argument('--kd-start-epoch', type=int, default=0, metavar='EPOCH_NUM',
                       help='Epoch from which to enable distillation')


class KnowledgeDistillationPolicy(ScheduledTrainingPolicy):
    """
    Policy which enables knowledge distillation from a teacher model to a student model, as presented in [1].

    Notes:
        1. In addition to the standard policy callbacks, this class also provides a 'forward' function that must
           be called instead of calling the student model directly as is usually done. This is needed to facilitate
           running the teacher model in addition to the student, and for caching the logits for loss calculation.
        2. [TO BE ENABLED IN THE NEAR FUTURE] Option to train the teacher model in parallel with the student model,
           described as "scheme A" in [2]. This can be achieved by passing teacher loss weight > 0.
        3. [1] proposes a weighted average between the different losses. We allow arbitrary weights to be assigned
           to each loss.

    Arguments:
        student_model (nn.Module): The student model, that is - the main model being trained. If only initialized with
            random weights, this matches "scheme B" in [2]. If it has been bootstrapped with trained FP32 weights,
            this matches "scheme C".
        teacher_model (nn.Module): The teacher model from which soft targets are generated for knowledge distillation.
            Usually this is a pre-trained model, however in the future it will be possible to train this model as well
            (see Note 1 above)
        temperature (float): Temperature value used when calculating soft targets and logits (see [1]).
        loss_weights (DistillationLossWeights): Named tuple with 3 loss weights
            (a) 'distill' for student predictions (default: 0.5) vs. teacher soft-targets
            (b) 'student' for student predictions vs. true labels (default: 0.5)
            (c) 'teacher' for teacher predictions vs. true labels (default: 0). Currently this is just a placeholder,
                and cannot be set to a non-zero value.

    [1] Hinton et al., Distilling the Knowledge in a Neural Network (https://arxiv.org/abs/1503.02531)
    [2] Mishra and Marr, Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy
        (https://arxiv.org/abs/1711.05852)

    """
    def __init__(self, student_model, teacher_model, temperature=1.0,
                 loss_weights=DistillationLossWeights(0.5, 0.5, 0)):
        super(KnowledgeDistillationPolicy, self).__init__()

        if loss_weights.teacher != 0:
            raise NotImplementedError('Using teacher vs. labels loss is not supported yet, '
                                      'for now teacher loss weight must be set to 0')

        self.active = False

        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.loss_wts = loss_weights

        self.last_students_logits = None
        self.last_teacher_logits = None

    def forward(self, *inputs):
        """
        Performs forward propagation through both student and teacher models and caches the logits.
        This function MUST be used instead of calling the student model directly.

        Returns:
            The student model's returned output, to be consistent with what a script using this would expect
        """
        if not self.active:
            return self.student(*inputs)

        if self.loss_wts.teacher == 0:
            with torch.no_grad():
                self.last_teacher_logits = self.teacher(*inputs)
        else:
            self.last_teacher_logits = self.teacher(*inputs)

        out = self.student(*inputs)
        self.last_students_logits = out.clone()

        return out

    # Since the "forward" function isn't a policy callback, we use the epoch callbacks to toggle the
    # activation of distillation according the schedule defined by the user
    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        self.active = True

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        self.active = False

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss, zeros_mask_dict,
                             optimizer=None):
        # TODO: Consider adding 'labels' as an argument to this callback, so we can support teacher vs. labels loss
        # (Otherwise we can't do it with a sub-class of ScheduledTrainingPolicy)

        if not self.active:
            return None

        if self.last_teacher_logits is None or self.last_students_logits is None:
            raise RuntimeError("KnowledgeDistillationPolicy: Student and or teacher logits were not cached. "
                               "Make sure to call KnowledgeDistillationPolicy.forward() in your script instead of "
                               "calling the model directly.")

        # Calculate distillation loss
        soft_log_probs = F.log_softmax(self.last_students_logits / self.temperature, dim=1)
        # soft_targets = F.softmax(self.cached_teacher_logits[minibatch_id] / self.temperature)
        soft_targets = F.softmax(self.last_teacher_logits / self.temperature, dim=1)

        distillation_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean')

        # According to [1]:
        # "Since the magnitudes of the gradients produced by the soft targets scale as 1/(T^2) it is important
        #  to multiply them by T^2 when using both hard and soft targets. This ensures that the relative contributions
        #  of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed
        #  while experimenting with meta-parameters."
        distillation_loss_scaled = distillation_loss * self.temperature ** 2

        # The loss passed to the callback is the student's loss vs. the true labels, so we can use it directly, no
        # need to calculate again
        overall_loss = self.loss_wts.student * loss + self.loss_wts.distill * distillation_loss_scaled

        # For logging purposes, we return the un-scaled distillation loss so it's comparable between runs with
        # different temperatures
        return PolicyLoss(overall_loss,
                          [LossComponent('Distill Loss', distillation_loss)])

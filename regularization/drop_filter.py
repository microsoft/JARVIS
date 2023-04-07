import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .regularizer import _Regularizer


class Conv2dWithMask(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

        super(Conv2dWithMask, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.test_mask = None
        self.p_mask = 1.0
        self.frequency = 16

    def forward(self, input):
        if self.training:
            self.frequency -= 1
            if self.frequency == 0:
                sample = np.random.binomial(n=1, p=self.p_mask, size=self.out_channels)
                param = self.weight
                l1norm = param.detach().view(param.size(0), -1).norm(p=1, dim=1)
                mask = torch.tensor(sample)
                mask = mask.expand(param.size(1) * param.size(2) * param.size(3), param.size(0)).t().contiguous()
                mask = mask.view(self.weight.shape).to(param.device)
                mask = mask.type(param.type())
                masked_weights = self.weight * mask
                masked_l1norm = masked_weights.detach().view(param.size(0), -1).norm(p=1, dim=1)
                pruning_factor = (masked_l1norm.sum() / l1norm.sum()).item()
                pruning_factor = max(0.2, pruning_factor)
                weight = masked_weights / pruning_factor
                self.frequency = 16
            else:
                weight = self.weight
        else:
            weight = self.weight
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# replaces all conv2d layers in target`s model with 'Conv2dWithMask'
def replace_conv2d(container):
    for name, module in container.named_children():
        if (isinstance(module, nn.Conv2d)):
            print("replacing: ", name)
            new_module = Conv2dWithMask(in_channels=module.in_channels,
                                        out_channels=module.out_channels,
                                        kernel_size=module.kernel_size, padding=module.padding,
                                        stride=module.stride, bias=module.bias)
            setattr(container, name, new_module)
        replace_conv2d(module)


class DropFilterRegularizer(_Regularizer):
    def __init__(self, name, model, reg_regims, threshold_criteria=None):
        super().__init__(name, model, reg_regims, threshold_criteria)
        replace_conv2d(model)

import torch
import torch.nn as nn


class Norm(nn.Module):
    """
    A module wrapper for vector/matrix norm
    """
    def __init__(self, p='fro', dim=None, keepdim=False):
        super(Norm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor):
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)


class Mean(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Mean, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        return torch.mean(x, *self.args, **self.kwargs)

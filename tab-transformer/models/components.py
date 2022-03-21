import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class GEGLU(nn.Module):
    """
    activation function which is a variant of GLU.
    reference: https://arxiv.org/pdf/2002.05202v1.pdf
    """
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)



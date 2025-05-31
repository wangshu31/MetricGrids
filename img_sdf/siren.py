import torch
from torch import nn
from math import sqrt


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)

        w_std = (1 / in_dim) if self.is_first else (sqrt(c / in_dim) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if not self.is_last:
            out = self.activation(out)
        return out

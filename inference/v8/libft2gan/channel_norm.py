import torch
import torch.nn as nn

class ChannelNorm(nn.Module):
    def __init__(self, channels, dtype=torch.float):
        super(ChannelNorm, self).__init__()

        self.norm = nn.LayerNorm(channels)
        self.imag_norm = nn.LayerNorm(channels) if dtype == torch.cfloat else None

    def forward(self, x):
        if self.imag_norm is not None:
            xr, xi = x.real, x.imag
            xr = self.norm(xr.transpose(1,3)).transpose(1,3)
            xi = self.imag_norm(xi.transpose(1,3)).transpose(1,3)
            return torch.complex(xr, xi)

        return self.norm(x.transpose(1,3)).transpose(1,3)
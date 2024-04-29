import torch
import torch.nn as nn

from v7.libft2gan.multichannel_layernorm import MultichannelLayerNorm

class Dropout2d(nn.Module):
    def __init__(self, p=0.5, inplace=False, dtype=torch.float):
        super().__init__()

        self.dropout = nn.Dropout2d(p=p, inplace=inplace)
        self.dtype= dtype

    def forward(self, x):
        if self.dtype == torch.float:
            return self.dropout(x)
        else:
            real = self.dropout(x.real)
            imag = self.dropout(x.imag)
            return torch.complex(real, imag)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, kernel_size=3, padding=1, downsample=False, stride=(2,1), dropout=0, dtype=torch.float):
        super(ResBlock, self).__init__()

        self.dropout = Dropout2d(dropout, dtype=dtype) if dropout > 0 else nn.Identity()
        self.activate = nn.GELU()
        self.norm = MultichannelLayerNorm(in_channels, features, dtype=dtype)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, dtype=dtype)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride if downsample else 1, bias=False, dtype=dtype)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride if downsample else 1, bias=False, dtype=dtype) if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + self.dropout(h)

        return x
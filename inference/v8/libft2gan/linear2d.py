import torch
import torch.nn as nn
import math

class Linear2d(nn.Module):
    def __init__(self, in_channels, in_features, out_features, bias=False):
        super(Linear2d, self).__init__()

        self.bias = None
        self.weight = nn.Parameter(torch.empty(in_channels, out_features, in_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels, out_features, 1))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def __call__(self, x):
        x = torch.matmul(x.transpose(2,3), self.weight.transpose(1,2)).transpose(2,3)

        if self.bias is not None:
            x = x + self.bias
        
        return x
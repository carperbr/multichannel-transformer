import torch
import torch.nn as nn
import math
    
class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, bias=False, kernel_size=3, padding=1, groups=1):
        super(FrameConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.bias_pw = None
        self.weight_pw = nn.Parameter(torch.empty(out_channels, out_features, in_features))

        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_pw, -bound, bound)
        
        if bias:
            self.bias_pw = nn.Parameter(torch.empty(out_channels, out_features, 1))
            nn.init.uniform_(self.bias_pw, -bound, bound)
            
    def __call__(self, x):
        x = torch.matmul(self.conv(x).transpose(2,3), self.weight_pw.transpose(1,2)).transpose(2,3)

        if self.bias_pw is not None:
            x = x + self.bias_pw
        
        return x
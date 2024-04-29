import torch
import torch.nn as nn

class SquaredReLU(nn.Module):
    def __init__(self, dtype=torch.float):
        super().__init__()

        self.dtype= dtype

    def forward(self, x):
        if self.dtype == torch.float:
            return torch.relu(x) ** 2
        else:
            real = torch.relu(x.real) ** 2
            imag = torch.relu(x.imag) ** 2
            return torch.complex(real, imag)
        
class Cardioid(nn.Module):
    def forward(self, x):
        phase = torch.angle(x)
        scale = 0.5 * (1 + torch.cos(phase))
        return torch.complex(x.real * scale, x.imag * scale)

class Sigmoid(nn.Module):
    def __init__(self, dtype=torch.float):
        super().__init__()

        self.dtype= dtype

    def forward(self, x):
        if self.dtype == torch.float:
            return torch.sigmoid(x)
        else:
            real = torch.sigmoid(x.real)
            imag = torch.sigmoid(x.imag)
            return torch.complex(real, imag)

class Upsample(nn.Module):
    def __init__(self, scale_factor=(2,1), size=None, mode='bilinear', align_corners=True, dtype=torch.float):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, size=size, mode=mode, align_corners=align_corners)
        self.dtype= dtype

    def forward(self, x):
        if self.dtype == torch.float:
            return self.upsample(x)
        else:
            real = self.upsample(x.real)
            imag = self.upsample(x.imag)
            return torch.complex(real, imag)

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False, dtype=torch.float):
        super().__init__()

        self.dropout = nn.Dropout(p=p, inplace=inplace)
        self.dtype= dtype

    def forward(self, x):
        if self.dtype == torch.float:
            return self.dropout(x)
        else:
            real = self.dropout(x.real)
            imag = self.dropout(x.imag)
            return torch.complex(real, imag)

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
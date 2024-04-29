import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from v8.libft2gan.rotary_embedding_torch import RotaryEmbedding
from v8.libft2gan.channel_norm import ChannelNorm

class ConvolutionalMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(channels // num_heads)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.o_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape

        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)).transpose(2,3)
        v = self.v_proj(x if mem is None else mem).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)
        qk = torch.matmul(q,k) / math.sqrt(c)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(1,2).reshape(b,h,w,c).permute(0,3,1,2)
        x = self.o_proj(a)

        return x, qk

class ComplexConvolutionalMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(channels // num_heads, dtype=torch.cfloat)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=torch.cfloat)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=torch.cfloat)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=torch.cfloat)
        self.o_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=torch.cfloat)
        
    def forward(self, x, mem=None, prev_qkm=None, prev_qkp=None):
        b,c,h,w = x.shape

        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)).transpose(2,3)
        v = self.v_proj(x if mem is None else mem).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)

        qm, qp = torch.abs(q), torch.angle(q)
        km, kp = torch.abs(k), torch.angle(k)
        vm, vp = torch.abs(v), torch.angle(v)

        qkm = torch.matmul(qm,km) / math.sqrt(h)
        qkp = torch.matmul(qp,kp) / math.sqrt(h)

        if prev_qkm is not None:
            qkm = qkm + prev_qkm
            qkp = qkp + prev_qkp

        qkt = qkm * (torch.cos(qkp) + 1.j * torch.sin(qkp))
        scores = F.softmax(qkt.abs(), dim=-1)
        out_mag = torch.matmul(scores, vm)
        out_phase = torch.matmul(scores, vp)

        a = out_mag * (torch.cos(out_phase) + 1.j * torch.sin(out_phase))

        a = a.type_as(x).transpose(1,2).reshape(b,h,w,c).permute(0,3,1,2)
        x = self.o_proj(a)

        return x, qkm, qkp
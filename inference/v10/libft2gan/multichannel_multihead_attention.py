import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from v10.libft2gan.rotary_embedding_torch import RotaryEmbedding
from v10.libft2gan.multichannel_linear import MultichannelLinear
from v10.libft2gan.multichannel_layernorm import MultichannelLayerNorm

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, attention_maps, num_heads, features, kernel_size=3, padding=1, expansion=1, mem_channels=None, mem_features=None, dtype=torch.float):
        super().__init__()

        self.attention_maps = attention_maps
        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads, dtype=dtype)

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, attention_maps, kernel_size=kernel_size, padding=padding, bias=False),
            MultichannelLinear(attention_maps, attention_maps, features, features * expansion, bias=False),
            nn.Conv2d(attention_maps, attention_maps, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size, padding=padding, bias=False),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features * expansion, bias=False),
            nn.Conv2d(attention_maps, attention_maps, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size, padding=padding, bias=False),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features * expansion, bias=False),
            nn.Conv2d(attention_maps, attention_maps, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))
        
        self.o_proj = MultichannelLinear(attention_maps, attention_maps, features * expansion, features, bias=False)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,self.attention_maps,w,-1).transpose(2,3)
        out = self.o_proj(a)

        return out, qk
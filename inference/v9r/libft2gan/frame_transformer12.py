import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from v9r.libft2gan.multichannel_linear import MultichannelLinear
from v9r.libft2gan.rotary_embedding_torch import RotaryEmbedding

# expects x in [B,C,H,W]
def to_chunks(x, chunk_size):
    return x.unfold(2, chunk_size, chunk_size).unfold(3, chunk_size, chunk_size).reshape(x.shape[0], x.shape[1], (x.shape[2] // chunk_size) * (x.shape[3] // chunk_size), chunk_size, chunk_size)

# expects chunks in [B,C,L,Ch,Cw], x in [B,C,H,W]
def from_chunks(chunks, x):
    return chunks.reshape(chunks.shape[0], chunks.shape[1], x.shape[2] // chunks.shape[-1], x.shape[3] // chunks.shape[-1], chunks.shape[-1], chunks.shape[-1]).permute(0,1,2,4,3,5).reshape(chunks.shape[0], chunks.shape[1], x.shape[2], x.shape[3])

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=8, dropout=0.1, n_fft=2048, num_heads=4, expansion=2, num_attention_maps=1, num_layers=8, chunk_size=32):
        super(FrameTransformer, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2
        self.out_channels = out_channels
        self.chunk_size = chunk_size
        self.autoregressive = True

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False)
        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin, downsample=True) # [1024,2048] -> [512,1024]
        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2, downsample=True) # [512,1024] -> [256,512]
        self.enc4 = FrameEncoder(channels * 4, channels * 8, self.max_bin // 4, downsample=True) # [256,512] -> [128,256]
        self.transformer = nn.Sequential(*[FrameTransformerEncoder(channels * 8, channels * 8, self.chunk_size, dropout=dropout, expansion=expansion, num_heads=num_heads, kernel_size=3, padding=1) for _ in range(num_layers)])
        self.dec3 = FrameDecoder(channels * 8, channels * 4, channels * 4, self.max_bin // 4)
        self.dec2 = FrameDecoder(channels * 4, channels * 2, channels * 2, self.max_bin // 2)
        self.dec1 = FrameDecoder(channels * 2, channels, channels, self.max_bin)
        self.out = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        chunks = to_chunks(e4, self.chunk_size)

        prev_qk = None
        for encoder in self.transformer:
            chunks, prev_qk = encoder(chunks, prev_qk=prev_qk)

        e4 = from_chunks(chunks, e4)
        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out(d1)
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, expansion=1, downsample=True, kernel_size=3, padding=1):
        super(FrameEncoder, self).__init__()

        self.activate = nn.LeakyReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=2 if downsample else 1)
        self.idt = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=2 if downsample else 1) if in_channels != out_channels or downsample else nn.Identity()
        
    def forward(self, x):
        return self.idt(x) + self.conv2(self.activate(self.conv1(self.norm1(x))))

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1, prev_attn=0, freeze_layers=2):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if out_channels > 1 else nn.Identity()

        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features, kernel_size=kernel_size, padding=padding)

        self.norm2 = nn.InstanceNorm2d(channels * 2, affine=True)
        self.conv1 = MultichannelLinear(channels * 2, channels * 2, features, expansion)
        self.conv2 = MultichannelLinear(channels * 2, channels, expansion, features)

    def forward(self, x, prev_attn=None, prev_qk=None):
        b,c,l,ch,cw = x.shape
        xh = x.transpose(1,2).reshape((b*l,c,ch,cw))
        a, prev_qk = self.attn(self.norm1(xh).reshape((b,l,c,ch,cw)).transpose(1,2), prev_qk=prev_qk)

        b,c,l,ch,cw = a.shape
        z = torch.cat((xh, a.transpose(1,2).reshape((b*l,c,ch,cw))), dim=1)

        z = self.conv2(self.activate(self.conv1(self.norm2(z)))).reshape((b,l,c,ch,cw)).transpose(1,2)
        h = x + self.dropout(z.transpose(1,2).reshape((b*l,c,ch,cw))).reshape((b,l,c,ch,cw)).transpose(1,2)

        return h, prev_qk
    
class FrameDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, features, expansion=1, kernel_size=3, padding=1):
        super(FrameDecoder, self).__init__()

        self.activate = nn.LeakyReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(in_channels + skip_channels, affine=True)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, in_channels + skip_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.idt = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x, skip):
        x = torch.cat((skip, F.interpolate(x, size=[*skip.shape[2:]], mode='bilinear', align_corners=True)), dim=1)
        return self.idt(x) + self.conv2(self.activate(self.conv1(self.norm1(x))))

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1, expansion=1, mem_channels=None, mem_features=None, dtype=torch.float):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads, dtype=dtype)

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            MultichannelLinear(channels, channels, features, features, bias=False))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            MultichannelLinear(channels, channels, features if mem_features is None else mem_features, features, bias=False))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            MultichannelLinear(channels, channels, features if mem_features is None else mem_features, features, bias=False))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features, bias=False)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,l,ch,cw = x.shape
        h = x.transpose(1,2).reshape((b*l,c,ch,cw))

        q = self.q_proj(h)

        if mem is not None:
            mem = mem.transpose(1,2).reshape((mem.shape[0]*mem.shape[2],mem.shape[1],mem.shape[3],mem.shape[4]))

        k = self.k_proj(h if mem is None else mem)
        v = self.v_proj(h if mem is None else mem)

        q = q.reshape((b,l,c,ch*cw)).permute(0,2,1,3).reshape(b,c,l,self.num_heads,-1).permute(0,1,3,2,4) # [b,c,num_heads,l,ch*cw]
        k = k.reshape((b,l,c,ch*cw)).permute(0,2,1,3).reshape(b,c,l,self.num_heads,-1).permute(0,1,3,2,4) # [b,c,num_heads,ch*cw,l]
        v = v.reshape((b,l,c,ch*cw)).permute(0,2,1,3).reshape(b,c,l,self.num_heads,-1).permute(0,1,3,2,4) # [b,c,num_heads,l,ch*cw]

        q = self.embedding.rotate_queries_or_keys(q)
        k = self.embedding.rotate_queries_or_keys(k).transpose(3,4)

        qk = torch.matmul(q,k) / math.sqrt(ch)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,l,ch,cw)
        
        out = self.o_proj(a.transpose(1,2).reshape((b*l,c,ch,cw))).reshape((b,l,c,ch,cw)).transpose(1,2)

        return out, qk
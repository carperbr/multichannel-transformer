import torch
from torch import nn

from v8.libft2gan.multichannel_multihead_attention import MultichannelMultiheadAttention
from v8.libft2gan.multichannel_layernorm import MultichannelLayerNorm
from v8.libft2gan.multichannel_linear import MultichannelLinear
from v8.libft2gan.res_block import ResBlock

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_attention_maps=1, expansions=[-1,2.1,3,3,4,4,5]):
        super(FrameTransformer, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.autoregressive = False

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansions[1], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)

        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansions[2], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansions[3], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)

        self.enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansions[4], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)

        self.enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansions[5], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)

        self.enc6 = FrameEncoder(channels * 8, channels * 10, self.max_bin // 16)
        self.enc6_transformer = FrameTransformerEncoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansions[6], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)

        self.dec5 = FrameDecoder(channels * 10, channels * 8, self.max_bin // 16)
        self.dec5_transformer = FrameTransformerDecoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansions[5], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)

        self.dec4 = FrameDecoder(channels * 8, channels * 6, self.max_bin // 8)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansions[4], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)
        
        self.dec3 = FrameDecoder(channels * 6, channels * 4, self.max_bin // 4)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansions[3], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)
        
        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansions[2], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)
        
        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin // 1)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansions[1], num_heads=num_heads, src_channels=in_channels, src_features=self.max_bin)
        
        self.out = nn.Conv2d(channels, out_channels, 1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e1, qk1, pqks = self.enc1_transformer(e1, x)

        e2 = self.enc2(e1)
        e2, qk2, pqks = self.enc2_transformer(e2, x, prev_qk=qk1, prev_qk_src=pqks)

        e3 = self.enc3(e2)
        e3, qk3, pqks = self.enc3_transformer(e3, x, prev_qk=qk2, prev_qk_src=pqks)

        e4 = self.enc4(e3)
        e4, qk4, pqks = self.enc4_transformer(e4, x, prev_qk=qk3, prev_qk_src=pqks)

        e5 = self.enc5(e4)
        e5, qk5, pqks = self.enc5_transformer(e5, x, prev_qk=qk4, prev_qk_src=pqks)

        e6 = self.enc6(e5)
        e6, qk6, pqks = self.enc6_transformer(e6, x, prev_qk=qk5, prev_qk_src=pqks)

        h = self.dec5(e6, e5)
        h, pqk, pqks = self.dec5_transformer(h, e5, x, prev_qk=qk6, prev_qk_src=pqks, prev_qk_skip=qk5)
    
        h = self.dec4(h, e4)
        h, pqk, pqks = self.dec4_transformer(h, e4, x, prev_qk=pqk, prev_qk_src=pqks, prev_qk_skip=qk4)
        
        h = self.dec3(h, e3)
        h, pqk, pqks = self.dec3_transformer(h, e3, x, prev_qk=pqk, prev_qk_src=pqks, prev_qk_skip=qk3)

        h = self.dec2(h, e2)
        h, pqk, pqks = self.dec2_transformer(h, e2, x, prev_qk=pqk, prev_qk_src=pqks, prev_qk_skip=qk2)

        h = self.dec1(h, e1)
        h, pqk, pqks = self.dec1_transformer(h, e1, x, prev_qk=pqk, prev_qk_src=pqks, prev_qk_skip=qk1)

        out = self.out(h)

        return out
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, src_channels=None, src_features=None):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm = MultichannelLayerNorm(channels, features)
        self.self_attn = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1)
        self.src_attn = MultichannelMultiheadAttention(channels + out_channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=src_channels, mem_features=src_features)

        self.conv1 = MultichannelLinear(channels + out_channels * 2, channels + out_channels * 2, features, int(features * expansion), depthwise=True, bias=False)
        self.conv2 = MultichannelLinear(channels + out_channels * 2, channels, int(features * expansion), features, bias=False)

    def forward(self, x, src, prev_qk=None, prev_qk_src=None):
        h = self.norm(x)
        self_attn, prev_qk = self.self_attn(h, prev_qk=prev_qk)
        src_attn, prev_qk_src = self.src_attn(torch.cat((h, self_attn), dim=1), prev_qk=prev_qk_src, mem=src)
        z = self.conv2(self.activate(self.conv1(torch.cat((h, self_attn, src_attn), dim=1))))
        h = x + self.dropout(z)

        return h, prev_qk, prev_qk_src
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, src_channels=None, src_features=None):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm = MultichannelLayerNorm(channels, features)
        self.self_attn = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1)
        self.skip_attn = MultichannelMultiheadAttention(channels + out_channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=channels)
        self.src_attn = MultichannelMultiheadAttention(channels + out_channels * 2, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=src_channels, mem_features=src_features)

        self.conv1 = MultichannelLinear(channels + out_channels * 3, channels + out_channels * 3, features, int(features * expansion), depthwise=True, bias=False)
        self.conv2 = MultichannelLinear(channels + out_channels * 3, channels, int(features * expansion), features, bias=False)
        
    def forward(self, x, skip, src, prev_qk=None, prev_qk_src=None, prev_qk_skip=None):
        h = self.norm(x)
        self_attn, prev_qk = self.self_attn(h, prev_qk=prev_qk)
        skip_attn, _ = self.skip_attn(torch.cat((h, self_attn), dim=1), prev_qk=prev_qk_skip, mem=skip)
        src_attn, prev_qk_src = self.src_attn(torch.cat((h, self_attn, skip_attn), dim=1), prev_qk=prev_qk_src, mem=src)
        z = self.conv2(self.activate(self.conv1(torch.cat((h, self_attn, skip_attn, src_attn), dim=1))))
        h = x + self.dropout(z)

        return h, prev_qk, prev_qk_src

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, stride=(2,1)):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, stride=stride)

    def forward(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, dropout=0):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = ResBlock(in_channels + out_channels, out_channels, features, dropout=dropout)

    def forward(self, x: torch.Tensor, skip):
        x = torch.cat((self.upsample(x.half()).bfloat16(), skip), dim=1)
        x = self.body(x)

        return x
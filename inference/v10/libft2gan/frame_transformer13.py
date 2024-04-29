import torch
import torch.nn.functional as F
import torch.nn as nn

from v10.libft2gan.multichannel_layernorm import MultichannelLayerNorm
from v10.libft2gan.multichannel_multihead_attention import MultichannelMultiheadAttention

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, embedding=2, dropout=0.1, n_fft=2048, num_heads=8, expansion=4, num_attention_maps=1):
        super(FrameTransformer, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2
        self.out_channels = out_channels
        self.autoregressive = False

        self.activate = nn.LeakyReLU(inplace=True)

        attn_maps = [num_attention_maps * 1, num_attention_maps * 2, num_attention_maps * 4, num_attention_maps * 8, num_attention_maps * 12, num_attention_maps * 14, num_attention_maps * 14]

        self.enc1 = FrameEncoder(in_channels, embedding, self.max_bin, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(embedding, attn_maps[0], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=0)

        self.enc2 = FrameEncoder(embedding + attn_maps[0], embedding * 2, self.max_bin) # 2048 -> 1024
        self.enc2_transformer = FrameTransformerEncoder(embedding * 2, attn_maps[1], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=0)

        self.enc3 = FrameEncoder(embedding * 2 + attn_maps[1], embedding * 4, self.max_bin) # 1024 -> 512
        self.enc3_transformer = FrameTransformerEncoder(embedding * 4, attn_maps[2], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=0)

        self.enc4 = FrameEncoder(embedding * 4 + attn_maps[2], embedding * 6, self.max_bin) # 512 -> 256
        self.enc4_transformer = FrameTransformerEncoder(embedding * 6, attn_maps[3], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=0)

        self.enc5 = FrameEncoder(embedding * 6 + attn_maps[3], embedding * 8, self.max_bin) # 256 -> 128
        self.enc5_transformer = FrameTransformerEncoder(embedding * 8, attn_maps[4], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=0)

        self.enc6 = FrameEncoder(embedding * 8 + attn_maps[4], embedding * 10, self.max_bin) # 128 -> 64
        self.enc6_transformer = FrameTransformerEncoder(embedding * 10, attn_maps[5], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=0)

        # self.enc7 = FrameEncoder(embedding * 10 + attn_maps[5], embedding * 12, self.max_bin) # 64 -> 32
        # self.enc7_transformer = FrameTransformerEncoder(embedding * 12, attn_maps[6], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=0)

        # self.dec6 = FrameDecoder(embedding * 12 + attn_maps[6], embedding * 10 + attn_maps[5], embedding * 10, self.max_bin) # 32 -> 64
        # self.dec6_transformer = FrameTransformerDecoder(embedding * 10, embedding * 10 + attn_maps[5], attn_maps[5], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=attn_maps[5])

        self.dec5 = FrameDecoder(embedding * 10 + attn_maps[5], embedding * 8 + attn_maps[4], embedding * 8, self.max_bin) # 64 -> 128
        self.dec5_transformer = FrameTransformerDecoder(embedding * 8, embedding * 8 + attn_maps[4], attn_maps[4], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=attn_maps[4])

        self.dec4 = FrameDecoder(embedding * 8 + attn_maps[4] * 3, embedding * 6 + attn_maps[3], embedding * 6, self.max_bin) # 128 -> 256
        self.dec4_transformer = FrameTransformerDecoder(embedding * 6, embedding * 6 + attn_maps[3], attn_maps[3], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=attn_maps[3])
        
        self.dec3 = FrameDecoder(embedding * 6 + attn_maps[3] * 3, embedding * 4 + attn_maps[2], embedding * 4, self.max_bin) # 256 -> 512
        self.dec3_transformer = FrameTransformerDecoder(embedding * 4, embedding * 4 + attn_maps[2], attn_maps[2], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=attn_maps[2])
        
        self.dec2 = FrameDecoder(embedding * 4 + attn_maps[2] * 3, embedding * 2 + attn_maps[1], embedding * 2, self.max_bin) # 512 -> 1024
        self.dec2_transformer = FrameTransformerDecoder(embedding * 2, embedding * 2 + attn_maps[1], attn_maps[1], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=attn_maps[1])
        
        self.dec1 = FrameDecoder(embedding * 2 + attn_maps[1] * 3, embedding * 1 + attn_maps[0], embedding * 1, self.max_bin) # 1024 -> 2048
        self.dec1_transformer = FrameTransformerDecoder(embedding * 1, embedding * 1 + attn_maps[0], attn_maps[0], self.output_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, prev_attn=attn_maps[0])

        self.out = nn.Conv2d(embedding + attn_maps[0] * 3, out_channels, kernel_size=1, padding=0, bias=False)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e1, pa1, pqk1 = self.enc1_transformer(e1)

        e2 = self.enc2(e1)
        e2, pa2, pqk2 = self.enc2_transformer(e2)

        e3 = self.enc3(e2)
        e3, pa3, pqk3 = self.enc3_transformer(e3)

        e4 = self.enc4(e3)
        e4, pa4, pqk4 = self.enc4_transformer(e4)

        e5 = self.enc5(e4)
        e5, pa5, pqk5 = self.enc5_transformer(e5)

        e6 = self.enc6(e5)
        e6, pa6, pqk6 = self.enc6_transformer(e6)

        # e7 = self.enc7(e6)
        # e7, _, _ = self.enc7_transformer(e7)

        # h = self.dec6(e7, e6)
        # h = self.dec6_transformer(h, e6, pa6, prev_qk=pqk6, skip_qk=pqk6)
    
        h = self.dec5(e6, e5)
        h = self.dec5_transformer(h, e5, pa5, prev_qk=pqk5, skip_qk=pqk5)
    
        h = self.dec4(h, e4)
        h = self.dec4_transformer(h, e4, pa4, prev_qk=pqk4, skip_qk=pqk4)
        
        h = self.dec3(h, e3)
        h = self.dec3_transformer(h, e3, pa3, prev_qk=pqk3, skip_qk=pqk3)

        h = self.dec2(h, e2)
        h = self.dec2_transformer(h, e2, pa2, prev_qk=pqk2, skip_qk=pqk2)

        h = self.dec1(h, e1)
        h = self.dec1_transformer(h, e1, pa1, prev_qk=pqk1, skip_qk=pqk1)

        return self.out(h)
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, kernel_size=3, padding=1):
        super(FrameEncoder, self).__init__()

        self.activate = nn.LeakyReLU(inplace=True)
        self.norm1 = MultichannelLayerNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=(1,2) if downsample else 1)
        self.idt = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=(1,2) if downsample else 1) if in_channels != out_channels or downsample else nn.Identity()
        
    def forward(self, x):
        return self.idt(x) + self.conv2(self.activate(self.conv1(self.norm1(x))))
    
class FrameDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, features, kernel_size=3, padding=1):
        super(FrameDecoder, self).__init__()

        self.activate = nn.LeakyReLU(inplace=True)
        self.norm1 = MultichannelLayerNorm(in_channels + skip_channels, features)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, in_channels + skip_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.idt = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x, skip):
        x = torch.cat((skip, F.interpolate(x, size=[*skip.shape[2:]], mode='bilinear', align_corners=True)), dim=1)
        return self.idt(x) + self.conv2(self.activate(self.conv1(self.norm1(x))))

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1, prev_attn=0):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if out_channels > 1 else nn.Identity()

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels + prev_attn, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding)

        self.norm2 = MultichannelLayerNorm(channels + out_channels, features)
        self.conv1 = nn.Conv2d(channels + out_channels, (channels + out_channels) * expansion, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d((channels + out_channels) * expansion, channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x, prev_attn=None, prev_qk=None):
        a, prev_qk = self.attn(self.norm1(torch.cat((x, prev_attn), dim=1) if prev_attn is not None else x), prev_qk=prev_qk)
        z = self.conv2(self.activate(self.conv1(self.norm2(torch.cat((x, a), dim=1)))))
        h = x + self.dropout(z)

        return torch.cat((h, a), dim=1), torch.cat((prev_attn, a), dim=1) if prev_attn is not None else a, prev_qk

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, mem_channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1, prev_attn=0):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.self_attn = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding)

        self.norm2 = MultichannelLayerNorm(channels + out_channels, features)
        self.skip_attn = MultichannelMultiheadAttention(channels + out_channels, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding, mem_channels=mem_channels + prev_attn)

        self.norm3 = MultichannelLayerNorm(channels + out_channels * 2 + prev_attn, features)
        self.conv1 = nn.Conv2d(channels + out_channels * 2 + prev_attn, (channels + out_channels * 2 + prev_attn) * expansion, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d((channels + out_channels * 2 + prev_attn) * expansion, channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x, mem, prev_attn=None, prev_qk=None, skip_qk=None):
        a, prev_qk = self.self_attn(self.norm1(x), prev_qk=prev_qk)
        a2, _ = self.skip_attn(self.norm2(torch.cat((x, a), dim=1)), prev_qk=skip_qk, mem=torch.cat((mem, prev_attn), dim=1))

        h = torch.cat((prev_attn, a, a2), dim=1)
        z = self.conv2(self.activate(self.conv1(self.norm3(torch.cat((x, h), dim=1)))))
        h = x + self.dropout(z)

        return torch.cat((h, prev_attn, a, a2), dim=1)
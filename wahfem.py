import math
import torch

import torch.nn as nn


class WaHFEM(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_heads=4):
        super(WaHFEM, self).__init__()
        
        self.depthwise_conv = DepthConv(in_channels, out_channels)

        self.local_branch = nn.Sequential(
            DilatedResBlock(out_channels),
            DilatedResBlock(out_channels),
            DilatedResBlock(out_channels)
        )
        self.global_branch = nn.Sequential(
            TransformerBranch(out_channels, num_heads),
            TransformerBranch(out_channels, num_heads)
        )

        self.channel_attention = ChannelAttention(out_channels * 2)

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.depthwise_conv(x)

        cnn_feat = self.local_branch(x)
        transformer_feat = self.global_branch(x)

        fused_feat = torch.cat((cnn_feat, transformer_feat), dim=1)
        attention_feat = self.channel_attention(fused_feat)

        out = self.output_conv(attention_feat)

        return out


class DilatedResBlock(nn.Module):
    def __init__(self, channels):
        super(DilatedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        x = nn.LeakyReLU()(self.conv1(x))
        x = nn.LeakyReLU()(self.conv2(x))
        x = self.conv3(x)

        return x + residual


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class TransformerBranch(nn.Module):
    def __init__(self, dim, num_heads):
        super(TransformerBranch, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = dim // num_heads

        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)
    
    def transpose_for_scores(self, x):
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        output = self.norm(ctx_layer)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        avg_out = self.global_avg_pool(x).view(B, C)
        attention = self.mlp(avg_out).view(B, C, 1, 1)

        return x * attention

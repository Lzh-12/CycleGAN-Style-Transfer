import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
       通道注意力模块
       输入: (B, C, H, W)
       输出: (B_1500, C, 1, 1) 的注意力权重
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分别通过平均池化和最大池化得到两个分支
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        # 相加后通过sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
       空间注意力模块
       输入: (B, C, H, W)
       输出: (B, 1, H, W) 的注意力权重
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # 在通道维度上做平均池化和最大池化
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上计算平均和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B_1500, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B_1500, 1, H, W)

        # 拼接后通过卷积
        concat = torch.cat([avg_out, max_out], dim=1)  # (B_1500, 2, H, W)
        out = self.conv(concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    CBAM完整模块：先通道注意力，后空间注意力
    参数:
        in_channels: 输入通道数
        reduction_ratio: 通道压缩比例
        kernel_size: 空间卷积核大小
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_att(x)
        # Apply spatial attention
        x = x * self.spatial_att(x)
        return x

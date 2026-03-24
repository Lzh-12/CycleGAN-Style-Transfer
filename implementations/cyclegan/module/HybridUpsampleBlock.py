import torch
import torch.nn as nn


class HybridUpsampleBlock(nn.Module):
    """
    混合上采样块 (HUB)
    一个轻量级但有效的上采样模块，融合了：
      - 保持平滑性的双线性插值路径
      - 增强细节的 PixelShuffle 路径
    专为无配对图像到图像转换任务设计（例如 CycleGAN）。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        scale_factor (int, optional): 上采样因子。默认：2.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

        # 路径 1: 双线性插值 + 卷积（保持平滑性）
        self.bilinear_path = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 路径 2: PixelShuffle 路径（增强细节）
        self.pixelshuffle_path = nn.Sequential(
            nn.Conv2d(in_channels, (out_channels // 2) * (scale_factor ** 2), kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 特征融合与通道匹配
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两个互补的上采样路径
        x_bilinear = self.bilinear_path(x)  # (N, C/2, H*r, W*r)
        x_shuffle = self.pixelshuffle_path(x)  # (N, C/2, H*r, W*r)

        # 拼接并融合
        x_fused = torch.cat([x_bilinear, x_shuffle], dim=1)  # (N, C, H*r, W*r)
        x_out = self.fuse(x_fused)
        return x_out

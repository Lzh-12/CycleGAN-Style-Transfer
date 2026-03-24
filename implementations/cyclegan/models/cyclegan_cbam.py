import torch.nn as nn
import torch

from ..module.CBAM import CBAM


def weights_init_normal(m):
    """
    使用正态分布初始化网络权重
    
    args:
        m: 网络模块(如Conv2d, BatchNorm2d等)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # 卷积层权重使用均值为0，标准差为0.02的正态分布初始化
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            # 偏置项初始化为0
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        # BatchNorm层权重初始化为均值1，标准差0.02的正态分布
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    """
    残差块实现，用于生成器中的特征提取
    
    args:
        in_features: 输入特征通道数
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # 反射填充，保持特征图尺寸不变
            nn.ReflectionPad2d(1),
            # 3x3卷积，保持通道数不变
            nn.Conv2d(in_features, in_features, 3),
            # Instance Normalization，保持每个样本的特征分布稳定
            nn.InstanceNorm2d(in_features),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 再次反射填充
            nn.ReflectionPad2d(1),
            # 第二个3x3卷积
            nn.Conv2d(in_features, in_features, 3),
            # 再次Instance Normalization
            nn.InstanceNorm2d(in_features),
        )

        self.cbam = CBAM(in_features)

    def forward(self, x):
        """
        前向传播，实现残差连接
        
        args:
            x: 输入特征图
            
        return:
            残差连接后的特征图 (输入 + 块输出)
        """
        residual = self.block(x)
        residual = self.cbam(residual)
        return x + residual


class GeneratorResNet(nn.Module):
    """
    基于ResNet的生成器, 用于CycleGAN中的图像转换
    
    args:
        input_shape: 输入图像的形状 (通道数, 高度, 宽度)
        num_residual_blocks: 残差块的数量
    """
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # 初始卷积块
        out_features = 64
        model = [
            # 反射填充，保持特征图尺寸不变
            nn.ReflectionPad2d(channels),
            # 7x7卷积，将输入通道数映射到64通道
            nn.Conv2d(channels, out_features, 7),
            # Instance Normalization
            nn.InstanceNorm2d(out_features),
            # ReLU激活函数
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # 下采样阶段（降低空间分辨率，增加通道数）
        for _ in range(2):
            out_features *= 2
            model += [
                # 3x3卷积，步长为2实现下采样
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # Instance Normalization
                nn.InstanceNorm2d(out_features),
                # ReLU激活函数
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 残差块序列，用于提取深层特征
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # 上采样阶段（恢复空间分辨率，减少通道数）
        for _ in range(2):
            out_features //= 2
            model += [
                # 上采样，尺寸放大2倍
                nn.Upsample(scale_factor=2),
                # 3x3卷积，调整通道数
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                # Instance Normalization
                nn.InstanceNorm2d(out_features),
                # ReLU激活函数
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 输出层，将特征映射回原始通道数
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            # Tanh激活函数，输出范围[-1, 1]
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        前向传播
        
        args:
            x: 输入图像批次
            
        return:
            生成的图像批次
        """
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    """
    判别器网络, 使用PatchGAN结构, 用于区分真实图像和生成图像
    
    args:
        input_shape: 输入图像的形状 (通道数, 高度, 宽度)
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # 计算PatchGAN的输出形状（每个输出像素对应输入图像的一个感受野区域）
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """
            创建判别器的下采样块
            
            avrgs:
                in_filters: 输入通道数
                out_filters: 输出通道数
                normalize: 是否使用Instance Normalization
                
            return:
                下采样层列表
            """
            layers = [
                # 4x4卷积，步长为2，实现下采样
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            ]
            if normalize:
                # 除了第一个块外，其他块都使用Instance Normalization
                layers.append(nn.InstanceNorm2d(out_filters))
            # 使用LeakyReLU激活函数，负斜率为0.2
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 构建判别器网络
        self.model = nn.Sequential(
            # 第一个块，不使用归一化
            *discriminator_block(channels, 64, normalize=False),
            # 第二个块
            *discriminator_block(64, 128),
            # 第三个块
            *discriminator_block(128, 256),
            # 第四个块
            *discriminator_block(256, 512),
            # 零填充，调整特征图尺寸
            nn.ZeroPad2d((1, 0, 1, 0)),
            # 最终输出层，输出每个感受野的真假判断
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        """
        前向传播
        
        args:
            img: 输入图像批次
            
        return:
            每个感受野区域的判别结果（PatchGAN的输出）
        """
        return self.model(img)

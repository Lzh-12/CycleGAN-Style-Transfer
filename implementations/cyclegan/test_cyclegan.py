#!/usr/bin/python3
import os
import sys

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from utils.utils import parse_arguments, load_config
from utils.metrics import *
from datasets import ImageDataset
from implementations.cyclegan.constants import DEFAULT_MODEL_TYPE
from implementations.cyclegan.cyclegan import get_model_classes


def test(opt):
    """
    CycleGAN 测试主函数

    args:
        opt: 测试配置参数（字典）
    """
    # ================== 设备设置 ==================
    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')

    if opt.cuda and not torch.cuda.is_available():
        print("WARNING: --cuda specified but CUDA is not available! Using CPU.")

    # 定义变量
    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # 初始化两个方向的生成器
    GeneratorResNet, _ = get_model_classes(opt.get('model_type', DEFAULT_MODEL_TYPE))
    netG_A2B = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
    netG_B2A = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)

    # 加载预训练权重（使用 map_location 确保 CPU/GPU 兼容）
    checkpoint_dir = f"{opt.checkpoints_dir}/{opt.get('model_type', DEFAULT_MODEL_TYPE)}/{opt.dataset_name}"
    checkpoint_epoch = opt.get('checkpoint_epoch', 200)

    generator_A2B_path = os.path.join(checkpoint_dir, f"G_AB_{checkpoint_epoch}.pth")
    generator_B2A_path = os.path.join(checkpoint_dir, f"G_BA_{checkpoint_epoch}.pth")
    
    netG_A2B.load_state_dict(torch.load(generator_A2B_path, map_location=device))
    netG_B2A.load_state_dict(torch.load(generator_B2A_path, map_location=device))

    # 设置为测试模式
    netG_A2B.eval()
    netG_B2A.eval()

    # 定义图像预处理变换
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # 创建数据加载器
    dataloader = DataLoader(
        ImageDataset(opt.dataroot, transforms_=transforms_, mode='test', preload=false),
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.n_cpu
    )
    ###################################

    ###### 测试设置 ######
    # 保存生成的图像和指标结果
    base_dir = f'output/{opt.get("model_type", "baseline")}/{opt.dataset_name}'
    output_dir_A = f'{base_dir}/A'
    output_dir_B = f'{base_dir}/B'
    metrics_dir = f'{base_dir}/saved_results'
    
    os.makedirs(output_dir_A, exist_ok=True)
    os.makedirs(output_dir_B, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ================== 初始化指标计算变量 ==================
    total_psnr_A = 0.0  # A 域 PSNR 累计值
    total_ssim_A = 0.0  # A 域 SSIM 累计值
    total_psnr_B = 0.0  # B 域 PSNR 累计值
    total_ssim_B = 0.0  # B 域 SSIM 累计值
    count = 0  # 批次计数器

    # FID 计算用的特征列表
    real_A_features = []  # A 域真实图像特征
    fake_A_features = []  # A 域生成图像特征
    real_B_features = []  # B 域真实图像特征
    fake_B_features = []  # B 域生成图像特征

    # LPIPS 指标
    total_lpips_A = 0.0
    total_lpips_B = 0.0

    # ================== 主测试循环 ==================
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)  # A 域真实图像
        real_B = batch['B'].to(device)  # B 域真实图像

        # 前向传播（不计算梯度
        with torch.no_grad():
            fake_B = 0.5 * (netG_A2B(real_A) + 1.0)  # [-1,1] -> [0,1]
            fake_A = 0.5 * (netG_B2A(real_B) + 1.0)

        # 保存生成的图像到磁盘（必须是 CPU 上的 Tensor）
        save_image(fake_A.cpu(), os.path.join(output_dir_A, f'{i+1:04d}.png'))
        save_image(fake_B.cpu(), os.path.join(output_dir_B, f'{i+1:04d}.png'))

        # 将真实图像转换到 [0, 1] 范围（在设备上）
        real_A_01 = ((batch['A'] + 1) / 2).to(device)
        real_B_01 = ((batch['B'] + 1) / 2).to(device)

        # ===== PSNR 计算（峰值信噪比）=====
        psnr_A = calculate_psnr(real_A_01, fake_A)
        psnr_B = calculate_psnr(real_B_01, fake_B)

        # SSIM
        ssim_A = calculate_ssim(fake_A, real_A_01)  # A 域结构相似性
        ssim_B = calculate_ssim(fake_B, real_B_01)  # B 域结构相似性

        # 累加指标
        total_psnr_A += psnr_A
        total_ssim_A += ssim_A
        total_psnr_B += psnr_B
        total_ssim_B += ssim_B
        count += 1

        # ===== LPIPS 计算（学习感知图像块相似度）=====
        lpips_a_val = calculate_lpips(
            normalize_to_neg1pos1(real_A_01),
            normalize_to_neg1pos1(fake_A)
        )
        lpips_b_val = calculate_lpips(
            normalize_to_neg1pos1(real_B_01),
            normalize_to_neg1pos1(fake_B)
        )

        if lpips_a_val is not None:
            total_lpips_A += lpips_a_val
        if lpips_b_val is not None:
            total_lpips_B += lpips_b_val

        # ===== FID 特征提取（Fréchet Inception Distance）=====
        real_a_feat = extract_inception_features(real_A_01)
        fake_a_feat = extract_inception_features(fake_A)
        real_b_feat = extract_inception_features(real_B_01)
        fake_b_feat = extract_inception_features(fake_B)

        if real_a_feat is not None:
            real_A_features.append(real_a_feat)
            fake_A_features.append(fake_a_feat)
            real_B_features.append(real_b_feat)
            fake_B_features.append(fake_b_feat)

        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

    sys.stdout.write('\n')

    # ================== 计算平均指标 ==================
    avg_psnr_A = total_psnr_A / count
    avg_ssim_A = total_ssim_A / count
    avg_psnr_B = total_psnr_B / count
    avg_ssim_B = total_ssim_B / count

    print(f"\nAverage PSNR (A domain): {avg_psnr_A:.4f} dB")
    print(f"Average SSIM (A domain): {avg_ssim_A:.4f}")
    print(f"Average PSNR (B domain): {avg_psnr_B:.4f} dB")
    print(f"Average SSIM (B domain): {avg_ssim_B:.4f}")

    # LPIPS 平均值
    avg_lpips_A = avg_lpips_B = None
    if total_lpips_A > 0:
        avg_lpips_A = total_lpips_A / count
        avg_lpips_B = total_lpips_B / count
        print(f"\nAverage LPIPS (A domain): {avg_lpips_A:.4f}")
        print(f"Average LPIPS (B domain): {avg_lpips_B:.4f}")

    # FID 和 KID 计算
    fid_A = fid_B = None
    kid_mean_A, kid_std_A = None, None
    kid_mean_B, kid_std_B = None, None

    if len(real_A_features) > 0:
        fid_A = calculate_fid(real_A_features, fake_A_features)
        fid_B = calculate_fid(real_B_features, fake_B_features)
        print(f"\nFID (A domain): {fid_A:.4f}")
        print(f"FID (B domain): {fid_B:.4f}")

        kid_mean_A, kid_std_A = calculate_kid(real_A_features, fake_A_features)
        kid_mean_B, kid_std_B = calculate_kid(real_B_features, fake_B_features)
        print(f"KID (A domain): {kid_mean_A:.6f} ± {kid_std_A:.6f}")
        print(f"KID (B domain): {kid_mean_B:.6f} ± {kid_std_B:.6f}")
    else:
        print("\nFID/KID calculation skipped (Inception V3 not available)")

    # ================== 保存结果 ==================
    with open(os.path.join(metrics_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Average PSNR (A domain): {avg_psnr_A:.4f} dB\n")
        f.write(f"Average SSIM (A domain): {avg_ssim_A:.4f}\n")
        f.write(f"Average PSNR (B domain): {avg_psnr_B:.4f} dB\n")
        f.write(f"Average SSIM (B domain): {avg_ssim_B:.4f}\n")
        if avg_lpips_A is not None:
            f.write(f"Average LPIPS (A domain): {avg_lpips_A:.4f}\n")
            f.write(f"Average LPIPS (B domain): {avg_lpips_B:.4f}\n")
        else:
            f.write("LPIPS calculation skipped\n")
        if fid_A is not None:
            f.write(f"FID (A domain): {fid_A:.4f}\n")
            f.write(f"FID (B domain): {fid_B:.4f}\n")
            f.write(f"KID (A domain): {kid_mean_A:.6f} ± {kid_std_A:.6f}\n")
            f.write(f"KID (B domain): {kid_mean_B:.6f} ± {kid_std_B:.6f}\n")
        else:
            f.write("FID/KID calculation skipped\n")

    print(f"\nMetrics saved to {os.path.join(metrics_dir, 'metrics.txt')}")


if __name__ == '__main__':
    # 命令行参数
    args = parse_arguments()
    # 加载配置
    config = load_config(args)
    # 测试
    test(config)
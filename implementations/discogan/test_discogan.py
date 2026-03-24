#!/usr/bin/python3

import argparse
import sys
import os
import numpy as np
import yaml

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

# 修改
from models import GeneratorUNet
from datasets import ImageDataset
from utils.utils import parse_arguments, load_config, print_config
from utils.metrics import (
    init_metrics_models,
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    extract_inception_features,
    calculate_fid,
    calculate_kid,
    normalize_to_01,
    normalize_to_neg1pos1
)


def test():
    # ================== 设备设置 ==================
    device = torch.device('cuda' if opt['cuda'] and torch.cuda.is_available() else 'cpu')
    
    # 检查是否使用了可用的GPU
    print(f"Using device: {device}")
    if opt['cuda'] and not torch.cuda.is_available():
        print("WARNING: --cuda specified but CUDA is not available! Using CPU.")

    ###### Definition of variables ######
    # Networks
    input_shape = (opt['channels'], opt['img_height'], opt['img_width'])
    netG_A2B = GeneratorUNet(input_shape).to(device)
    netG_B2A = GeneratorUNet(input_shape).to(device)

    # Load state dicts with map_location for safety
    netG_A2B.load_state_dict(torch.load(opt['generator_A2B'], map_location=device))
    netG_B2A.load_state_dict(torch.load(opt['generator_B2A'], map_location=device))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Dataset loader
    transforms_ = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    # 拼接完整的数据路径
    data_path = os.path.join(opt['dataroot'], opt['dataset_name'])
    
    dataloader = DataLoader(
        ImageDataset(data_path, transforms_=transforms_, mode=''),
        batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=opt['n_cpu']
    )
    ###################################

    ###### Testing ######
    base_dir = f'{opt.get("output_dir", "output")}/{opt["model_name"]}/{opt["dataset_name"]}'
    output_dir_A = f'{base_dir}/A'
    output_dir_B = f'{base_dir}/B'
    metrics_dir = f'{base_dir}/saved_results'
    os.makedirs(output_dir_A, exist_ok=True)
    os.makedirs(output_dir_B, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ================== 初始化评估模型 ==================
    print("\n📊 初始化评估指标模型...")
    metrics_models = init_metrics_models(device=str(device))
    
    # ================== 初始化指标计算变量 ==================
    total_psnr_A = 0.0
    total_ssim_A = 0.0
    total_psnr_B = 0.0
    total_ssim_B = 0.0
    total_lpips_A = 0.0
    total_lpips_B = 0.0
    count = 0

    real_A_features = []
    fake_A_features = []
    real_B_features = []
    fake_B_features = []

    # ================== 主测试循环 ==================
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        with torch.no_grad():
            fake_B = 0.5 * (netG_A2B(real_A) + 1.0)  # [-1,1] -> [0,1]
            fake_A = 0.5 * (netG_B2A(real_B) + 1.0)

        # Save images (must be CPU tensors)
        save_image(fake_A.cpu(), os.path.join(output_dir_A, f'{i+1:04d}.png'))
        save_image(fake_B.cpu(), os.path.join(output_dir_B, f'{i+1:04d}.png'))

        # Convert real images to [0,1] on device
        real_A_01 = normalize_to_01(batch['A']).to(device)
        real_B_01 = normalize_to_01(batch['B']).to(device)

        # PSNR
        psnr_A = calculate_psnr(real_A_01, fake_A)
        psnr_B = calculate_psnr(real_B_01, fake_B)

        # SSIM
        ssim_A = calculate_ssim(real_A_01, fake_A)
        ssim_B = calculate_ssim(real_B_01, fake_B)

        total_psnr_A += psnr_A
        total_ssim_A += ssim_A
        total_psnr_B += psnr_B
        total_ssim_B += ssim_B
        count += 1

        # LPIPS
        # LPIPS需要[-1,1]范围
        real_A_neg1pos1 = normalize_to_neg1pos1(real_A_01)
        fake_A_neg1pos1 = normalize_to_neg1pos1(fake_A)
        lpips_A_val = calculate_lpips(real_A_neg1pos1, fake_A_neg1pos1)
        if lpips_A_val is not None:
            total_lpips_A += lpips_A_val

        real_B_neg1pos1 = normalize_to_neg1pos1(real_B_01)
        fake_B_neg1pos1 = normalize_to_neg1pos1(fake_B)
        lpips_B_val = calculate_lpips(real_B_neg1pos1, fake_B_neg1pos1)
        if lpips_B_val is not None:
            total_lpips_B += lpips_B_val

        # FID features
        real_A_feat = extract_inception_features(real_A_01.cpu())
        if real_A_feat is not None:
            real_A_features.append(real_A_feat)
            
        fake_A_feat = extract_inception_features(fake_A.cpu())
        if fake_A_feat is not None:
            fake_A_features.append(fake_A_feat)
            
        real_B_feat = extract_inception_features(real_B_01.cpu())
        if real_B_feat is not None:
            real_B_features.append(real_B_feat)
            
        fake_B_feat = extract_inception_features(fake_B.cpu())
        if fake_B_feat is not None:
            fake_B_features.append(fake_B_feat)

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

    avg_lpips_A = avg_lpips_B = None
    if total_lpips_A > 0:
        avg_lpips_A = total_lpips_A / count
        avg_lpips_B = total_lpips_B / count
        print(f"\nAverage LPIPS (A domain): {avg_lpips_A:.4f}")
        print(f"Average LPIPS (B domain): {avg_lpips_B:.4f}")

    fid_A = fid_B = None
    if real_A_features and fake_A_features:
        fid_A = calculate_fid(real_A_features, fake_A_features)
        fid_B = calculate_fid(real_B_features, fake_B_features)
        print(f"\nFID (A domain): {fid_A:.4f}")
        print(f"FID (B domain): {fid_B:.4f}")

        # === 计算 KID ===
        kid_mean_A, kid_std_A = calculate_kid(real_A_features, fake_A_features)
        kid_mean_B, kid_std_B = calculate_kid(real_B_features, fake_B_features)

        if kid_mean_A is not None:
            print(f"KID (A domain): {kid_mean_A:.6f} ± {kid_std_A:.6f}")
            print(f"KID (B domain): {kid_mean_B:.6f} ± {kid_std_B:.6f}")
    else:
        print("\nFID calculation skipped (Inception features not available)")

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
            # KID 保存
            f.write(f"KID (A domain): {kid_mean_A:.6f} ± {kid_std_A:.6f}\n")
            f.write(f"KID (B domain): {kid_mean_B:.6f} ± {kid_std_B:.6f}\n")
        else:
            f.write("FID calculation skipped\n")

    print(f"\nMetrics saved to {os.path.join(metrics_dir, 'metrics.txt')}")


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    # 加载配置文件
    opt = load_config(args)
    # 打印配置信息
    print_config(opt)
    # 测试模型    
    test()

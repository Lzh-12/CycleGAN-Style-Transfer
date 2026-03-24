"""
图像质量评估指标模块

提供常用的图像生成质量评估指标：
- PSNR: 峰值信噪比
- SSIM: 结构相似性
- LPIPS: 学习感知图像块相似度
- FID: Fréchet Inception Distance
- KID: Kernel Inception Distance
"""

import torch
import numpy as np
from scipy.linalg import sqrtm
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM


# ================== 全局工具变量 ==================
_lpips_model = None
_inception = None
_inception_transform = None
_ssim_metric = None


def init_metrics_models(device='cuda'):
    """
    初始化所有评估模型（LPIPS、Inception V3、SSIM）

    参数:
        device: 计算设备 (cpu/cuda)

    返回:
        dict: 包含所有初始化好的模型
    """
    global _lpips_model, _inception, _inception_transform, _ssim_metric

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 初始化 LPIPS 模型
    try:
        _lpips_model = lpips.LPIPS(net='alex').eval().to(device)
        print(f"LPIPS model loaded on {device}")
    except Exception as e:
        print(f"Warning: LPIPS model not available. Error: {e}")
        _lpips_model = None

    # 初始化 Inception V3
    try:
        from torchvision import models, transforms as T
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()
        inception.eval().to(device)

        _inception = inception
        _inception_transform = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Inception V3 model loaded for FID/KID calculation")
    except Exception as e:
        print(f"Warning: Inception V3 not available. Error: {e}")
        _inception = None
        _inception_transform = None

    # 初始化 SSIM
    _ssim_metric = SSIM(data_range=1.0).to(device)

    return {
        'lpips_model': _lpips_model,
        'inception': _inception,
        'inception_transform': _inception_transform,
        'ssim_metric': _ssim_metric
    }


def calculate_psnr(real, fake):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)

    参数:
        real: 真实图像 Tensor，范围 [0, 1]
        fake: 生成图像 Tensor，范围 [0, 1]

    返回:
        float: PSNR 值 (dB)
    """
    mse = torch.mean((real - fake) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(real, fake):
    """
    计算结构相似性 (Structural Similarity Index)

    参数:
        real: 真实图像 Tensor，范围 [0, 1]
        fake: 生成图像 Tensor，范围 [0, 1]

    返回:
        float: SSIM 值 (0~1)
    """
    global _ssim_metric
    if _ssim_metric is None:
        raise RuntimeError("SSIM metric not initialized. Call init_metrics_models() first.")

    ssim_val = _ssim_metric(fake, real)
    return ssim_val.item()


def calculate_lpips(real, fake):
    """
    计算学习感知图像块相似度 (Learned Perceptual Image Patch Similarity)

    参数:
        real: 真实图像 Tensor，范围 [-1, 1]
        fake: 生成图像 Tensor，范围 [-1, 1]

    返回:
        float: LPIPS 值 (越小越好)，如果不可用则返回 None
    """
    global _lpips_model
    if _lpips_model is None:
        return None

    with torch.no_grad():
        lpips_val = _lpips_model(real, fake)
    return lpips_val.mean().item()


def extract_inception_features(images):
    """
    从 Inception V3 提取特征用于 FID/KID 计算

    参数:
        images: 图像 Tensor，范围 [0, 1]，形状 (N, C, H, W)

    返回:
        Tensor: 特征向量 (N, 2048)，如果不可用则返回 None
    """
    global _inception, _inception_transform
    if _inception is None or _inception_transform is None:
        return None

    images = _inception_transform(images).to(_inception.device)

    with torch.no_grad():
        features = _inception(images)

    return features.cpu().detach()


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1.0):
    """
    计算多项式核函数

    公式：K(x, y) = (gamma * <x, y> + coef0) ^ degree

    参数:
        X: 输入特征矩阵 (N, D)
        Y: 目标特征矩阵 (M, D)，如果为 None 则 Y=X
        degree: 多项式次数，默认 3
        gamma: 缩放系数，默认 1/D
        coef0: 偏置项，默认 1.0

    返回:
        核矩阵 K (N, M)
    """
    if gamma is None:
        gamma = 1.0 / X.size(1)
    if Y is None:
        Y = X

    K = torch.mm(X, Y.t())
    K *= gamma
    K += coef0
    K.clamp_min_(0)
    K.pow_(degree)
    return K


def calculate_fid(real_features, fake_features):
    """
    计算 Fréchet Inception Distance

    公式：FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_rΣ_f))

    参数:
        real_features: 真实图像特征列表或 Tensor [(N, D), ...]
        fake_features: 生成图像特征列表或 Tensor [(M, D), ...]

    返回:
        float: FID 分数，如果不可用则返回 None
    """
    if len(real_features) == 0 or len(fake_features) == 0:
        return None

    # 拼接所有批次的特征
    if isinstance(real_features[0], torch.Tensor):
        real_features = torch.cat(real_features, dim=0)
        fake_features = torch.cat(fake_features, dim=0)

    # 计算均值向量
    mu_real = torch.mean(real_features, dim=0)
    mu_fake = torch.mean(fake_features, dim=0)

    # 计算协方差矩阵
    sigma_real = torch.cov(real_features.T)
    sigma_fake = torch.cov(fake_features.T)

    # 均值差的平方和
    diff = mu_real - mu_fake
    diff_norm = torch.sum(diff * diff)

    # 计算协方差乘积的平方根迹
    cov_product = sigma_real @ sigma_fake
    cov_product_np = cov_product.cpu().numpy()
    cov_mean_np = sqrtm(cov_product_np)

    # 处理复数情况
    if np.iscomplexobj(cov_mean_np):
        cov_mean_np = np.real(cov_mean_np)

    cov_mean = torch.from_numpy(cov_mean_np).float().to(real_features.device)

    # FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_rΣ_f))
    fid = diff_norm + torch.trace(sigma_real + sigma_fake - 2 * cov_mean)

    return fid.item()


def calculate_kid(real_features, fake_features, num_subsets=100, subset_size=1000):
    """
    计算 Kernel Inception Distance (KID)
    使用无偏估计器

    参考论文：https://arxiv.org/abs/1801.01401

    参数:
        real_features: 真实图像特征列表或 Tensor [(N, D), ...]
        fake_features: 生成图像特征列表或 Tensor [(M, D), ...]
        num_subsets: 子集数量，默认 100
        subset_size: 每个子集的大小，默认 1000

    返回:
        tuple: (mean, std) KID 的平均值和标准差，如果不可用则返回 (None, None)
    """
    if len(real_features) == 0 or len(fake_features) == 0:
        return None, None

    # 拼接所有批次的特征
    if isinstance(real_features[0], torch.Tensor):
        real_features = torch.cat(real_features, dim=0)
        fake_features = torch.cat(fake_features, dim=0)

    n_real = real_features.size(0)
    n_fake = fake_features.size(0)

    # 如果样本不足，调整子集大小
    if subset_size > min(n_real, n_fake):
        subset_size = min(n_real, n_fake)
        print(f"Warning: subset_size reduced to {subset_size} due to limited data.")

    kid_scores = []
    for _ in range(num_subsets):
        # 随机采样子集（不放回）
        real_idx = torch.randperm(n_real)[:subset_size]
        fake_idx = torch.randperm(n_fake)[:subset_size]

        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]

        # 计算核矩阵
        k_rr = polynomial_kernel(real_subset, degree=3,
                                gamma=1.0 / real_subset.size(1), coef0=1.0)
        k_ff = polynomial_kernel(fake_subset, degree=3,
                                gamma=1.0 / fake_subset.size(1), coef0=1.0)
        k_rf = polynomial_kernel(real_subset, fake_subset, degree=3,
                                gamma=1.0 / real_subset.size(1), coef0=1.0)

        # MMD^2 的无偏估计
        m = subset_size
        kid_score = (
            (k_rr.sum() - torch.trace(k_rr)) / (m * (m - 1)) +
            (k_ff.sum() - torch.trace(k_ff)) / (m * (m - 1)) -
            2 * k_rf.sum() / (m * m)
        )
        kid_scores.append(kid_score.item())

    return np.mean(kid_scores), np.std(kid_scores)


def normalize_to_01(tensor):
    """
    将 [-1, 1] 范围的 Tensor 转换到 [0, 1]

    参数:
        tensor: 输入 Tensor，范围 [-1, 1]

    返回:
        Tensor: 范围 [0, 1]
    """
    return (tensor + 1) / 2


def normalize_to_neg1pos1(tensor):
    """
    将 [0, 1] 范围的 Tensor 转换到 [-1, 1]

    参数:
        tensor: 输入 Tensor，范围 [0, 1]

    返回:
        Tensor: 范围 [-1, 1]
    """
    return tensor * 2 - 1

import datetime
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from datasets import ImageDataset
from utils.utils import print_config
from implementations.cyclegan.utils.utils import parse_arguments, load_config, LambdaLR, ReplayBuffer
from constants import DEFAULT_MODEL_TYPE


def initialize_models(opt, input_shape, cuda):
    """
    初始化和加载模型

    参数:
        opt: 配置参数
        input_shape: 输入图像形状
        cuda: 是否使用 GPU

    返回:
        tuple: (G_AB, G_BA, D_A, D_B)
    """
    # 获取模型类
    model_type = opt.get('model_type', DEFAULT_MODEL_TYPE)
    GeneratorResNet, Discriminator, weights_init_normal = get_model_classes(model_type)
    
    # 初始化生成器和判别器
    G_AB = GeneratorResNet(input_shape, opt['n_residual_blocks'])
    G_BA = GeneratorResNet(input_shape, opt['n_residual_blocks'])
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    # 移动到 GPU
    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # 加载预训练模型或初始化权重
    if opt['epoch'] != 0:
        G_AB.load_state_dict(torch.load(f"checkpoints/{opt['model_type']}/{opt['dataset_name']}/G_AB_{opt['epoch']}.pth"))
        G_BA.load_state_dict(torch.load(f"checkpoints/{opt['model_type']}/{opt['dataset_name']}/G_BA_{opt['epoch']}.pth"))
        D_A.load_state_dict(torch.load(f"checkpoints/{opt['model_type']}/{opt['dataset_name']}/D_A_{opt['epoch']}.pth"))
        D_B.load_state_dict(torch.load(f"checkpoints/{opt['model_type']}/{opt['dataset_name']}/D_B_{opt['epoch']}.pth"))
        print(f"✅ 成功加载 epoch {opt['epoch']} 的模型")
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
        print("✅ 模型已初始化")

    return G_AB, G_BA, D_A, D_B


def initialize_optimizers_and_schedulers(opt, models):
    """
    初始化优化器和学习率调度器

    参数:
        opt: 配置参数
        models: 模型元组 (G_AB, G_BA, D_A, D_B)

    返回:
        tuple: (optimizer_G, optimizer_D_A, optimizer_D_B,
                lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B)
    """
    G_AB, G_BA, D_A, D_B = models

    # 优化器
    optimizer_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=opt['lr'],
        betas=(opt['b1'], opt['b2'])
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))

    # 学习率调度器
    lr_lambda = LambdaLR(opt['n_epochs'], opt['epoch'], opt['decay_epoch']).step
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

    return optimizer_G, optimizer_D_A, optimizer_D_B, lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B


def prepare_dataloader(opt):
    """
    准备数据加载器

    参数:
        opt: 配置参数

    返回:
        DataLoader: 训练数据加载器
    """
    import torchvision.transforms as T

    # 数据增强
    transforms_ = [
        T.Resize(int(opt['img_height'] * 1.12), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop((opt['img_height'], opt['img_width'])),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # 数据加载器
    dataloader = DataLoader(
        ImageDataset("./data/%s" % opt['dataset_name'], transforms_=transforms_),
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt['n_cpu'],
    )

    return dataloader


def train_generator(G_AB, G_BA, D_A, D_B, real_A, real_B, valid, fake,
                   criterion_GAN, criterion_cycle, criterion_identity,
                   optimizer_G, lambda_cyc, lambda_id):
    """
    训练生成器

    返回:
        dict: 包含各种损失值
    """
    G_AB.train()
    G_BA.train()
    optimizer_G.zero_grad()

    # Identity loss
    loss_id_A = criterion_identity(G_BA(real_A), real_A)
    loss_id_B = criterion_identity(G_AB(real_B), real_B)
    loss_identity = (loss_id_A + loss_id_B) / 2

    # GAN loss
    fake_B = G_AB(real_A)
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
    fake_A = G_BA(real_B)
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

    # Cycle consistency loss
    recov_A = G_BA(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)
    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

    # Total loss
    loss_G_A = loss_GAN_AB + lambda_cyc * loss_cycle_A + lambda_id * loss_id_A
    loss_G_B = loss_GAN_BA + lambda_cyc * loss_cycle_B + lambda_id * loss_id_B
    loss_G = (loss_G_A + loss_G_B) / 2

    loss_G.backward()
    optimizer_G.step()

    return {
        'G': loss_G.item(),
        'G_A': loss_G_A.item(),
        'G_B': loss_G_B.item(),
        'GAN': loss_GAN.item(),
        'GAN_AB': loss_GAN_AB.item(),
        'GAN_BA': loss_GAN_BA.item(),
        'cycle': loss_cycle.item(),
        'cycle_A': loss_cycle_A.item(),
        'cycle_B': loss_cycle_B.item(),
        'identity': loss_identity.item(),
        'identity_A': loss_id_A.item(),
        'identity_B': loss_id_B.item()
    }


def train_discriminator(discriminator, real_data, fake_data, valid, fake,
                       criterion_GAN, optimizer_D, fake_buffer):
    """
    训练判别器

    返回:
        float: 判别器损失
    """
    discriminator.train()
    optimizer_D.zero_grad()

    # Real loss
    loss_real = criterion_GAN(discriminator(real_data), valid)

    # Fake loss (使用缓冲区)
    fake_from_buffer = fake_buffer.push_and_pop(fake_data)
    loss_fake = criterion_GAN(discriminator(fake_from_buffer.detach()), fake)

    # Total loss
    loss_D = (loss_real + loss_fake) / 2

    loss_D.backward()
    optimizer_D.step()

    return loss_D.item()


@torch.no_grad()
def sample_images(batches_done, real_A, real_B, G_AB, G_BA, dataset_name):
    """
    采样并保存图像

    参数:
        batches_done: 已完成的 batch 数
        real_A: A 域真实图像
        real_B: B 域真实图像
        G_AB: A→B 生成器
        G_BA: B→A 生成器
        dataset_name: 数据集名称

    返回:
        Tensor: 合并后的图像
    """
    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

    # 合并图像进行显示（按行排列：A→fake_B→B→fake_A）
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), 0)

    # 保存图像
    save_path = f"images/{dataset_name}/{batches_done}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    from torchvision.utils import save_image
    save_image(img_sample, save_path, range(-1, 1), nrow=4)

    G_AB.train()
    G_BA.train()

    return img_sample


def save_checkpoint(epoch, models, opt):
    """
    保存检查点

    参数:
        epoch: 当前 epoch
        models: 模型元组
        opt: 配置参数
    """
    G_AB, G_BA, D_A, D_B = models

    checkpoint_dir = f"checkpoints/{opt['model_type']}/{opt['dataset_name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(G_AB.state_dict(), f"{checkpoint_dir}/G_AB_{epoch}.pth")
    torch.save(G_BA.state_dict(), f"{checkpoint_dir}/G_BA_{epoch}.pth")
    torch.save(D_A.state_dict(), f"{checkpoint_dir}/D_A_{epoch}.pth")
    torch.save(D_B.state_dict(), f"{checkpoint_dir}/D_B_{epoch}.pth")


def log_epoch_losses(writer, epoch, losses, num_batches):
    """
    记录每个 epoch 的平均损失到 TensorBoard

    参数:
        writer: TensorBoard SummaryWriter 对象
        epoch: 当前 epoch 数
        losses: 包含所有损失累加值的字典
        num_batches: batch 数量
    """
    # 计算平均损失
    avg_losses = {}
    for key, value in losses.items():
        avg_losses[key] = value / num_batches

    # ===== 记录到 TensorBoard（每个 epoch 记录一次）=====

    # 1. 判别器损失（A 和 B 域在同一图中）
    writer.add_scalar('Loss/Discriminator/D_A', avg_losses['D_A'], epoch)
    writer.add_scalar('Loss/Discriminator/D_B', avg_losses['D_B'], epoch)

    # 2. 生成器总损失
    writer.add_scalar('Loss/Generator/G_A_total', avg_losses['G_A'], epoch)
    writer.add_scalar('Loss/Generator/G_B_total', avg_losses['G_B'], epoch)

    # 3. GAN 对抗损失（A→B 和 B→A 在同一图中）
    writer.add_scalar('Loss/GAN_Loss/G_AB', avg_losses['GAN_AB'], epoch)
    writer.add_scalar('Loss/GAN_Loss/G_BA', avg_losses['GAN_BA'], epoch)

    # 4. 循环一致性损失（A 和 B 域在同一图中）
    writer.add_scalar('Loss/Cycle_Loss/Cycle_A', avg_losses['Cycle_A'], epoch)
    writer.add_scalar('Loss/Cycle_Loss/Cycle_B', avg_losses['Cycle_B'], epoch)

    # 5. 恒等损失（A 和 B 域在同一图中）
    writer.add_scalar('Loss/Identity_Loss/Identity_A', avg_losses['Identity_A'], epoch)
    writer.add_scalar('Loss/Identity_Loss/Identity_B', avg_losses['Identity_B'], epoch)


def log_learning_rates(writer, epoch, optimizers):
    """
    记录学习率到 TensorBoard

    参数:
        writer: TensorBoard SummaryWriter 对象
        epoch: 当前 epoch 数
        optimizers: 优化器字典 (optimizer_G, optimizer_D_A, optimizer_D_B)
    """
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers

    current_lr_G = optimizer_G.param_groups[0]['lr']
    current_lr_D_A = optimizer_D_A.param_groups[0]['lr']
    current_lr_D_B = optimizer_D_B.param_groups[0]['lr']

    writer.add_scalar('LearningRate/G_optimizer', current_lr_G, epoch)
    writer.add_scalar('LearningRate/D_A_optimizer', current_lr_D_A, epoch)
    writer.add_scalar('LearningRate/D_B_optimizer', current_lr_D_B, epoch)


def train(opt):
    """
    CycleGAN 主训练函数

    参数:
        opt: 训练配置参数
    """
    print_config(opt)

    # 创建目录
    os.makedirs(f"output/{opt['model_type']}/{opt['dataset_name']}", exist_ok=True)
    os.makedirs(f"checkpoints/{opt['model_type']}/{opt['dataset_name']}", exist_ok=True)

    # 初始化 TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{opt['model_type']}/{opt['dataset_name']}/{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard 日志目录：{log_dir}")
    print(f"💡 运行 'tensorboard --logdir logs/{opt['model_type']}/{opt['dataset_name']}' 查看训练曲线\n")

    # 定义损失函数
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # 设备检查
    cuda = torch.cuda.is_available()
    if not cuda and opt.get('cuda', False):
        print("⚠️ CUDA 不可用，将使用 CPU 训练")

    # 设备
    device = torch.device("cuda" if cuda else "cpu")

    input_shape = (opt['channels'], opt['img_height'], opt['img_width'])

    # 初始化模型
    print("\n📦 正在初始化模型...")
    G_AB, G_BA, D_A, D_B = initialize_models(opt, input_shape, cuda)

    # 移动损失函数到 GPU
    if cuda:
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    # 初始化优化器和调度器
    print("📦 正在初始化优化器...")
    optimizers_schedulers = initialize_optimizers_and_schedulers(opt, (G_AB, G_BA, D_A, D_B))
    optimizer_G, optimizer_D_A, optimizer_D_B, lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = optimizers_schedulers

    # 图像缓冲区
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 数据加载器
    print("📦 正在加载数据集...")
    dataloader = prepare_dataloader(opt)
    print(f"✅ 数据集加载完成，共 {len(dataloader)} 个批次\n")

    # 开始训练
    print("🚀 开始训练...\n")
    prev_time = time.time()

    for epoch in range(opt['epoch'], opt['n_epochs']):
        epoch_start_time = time.time()

        # 记录 epoch 平均损失（分 A 和 B 域）
        epoch_losses = {
            'D_A': 0.0,
            'D_B': 0.0,
            'G': 0.0,
            'GAN_AB': 0.0,
            'GAN_BA': 0.0,
            'Cycle_A': 0.0,
            'Cycle_B': 0.0,
            'Identity_A': 0.0,
            'Identity_B': 0.0
        }

        # 使用 tqdm 显示进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{opt['n_epochs']}", leave=True)

        for i, batch in enumerate(pbar):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *D_A.output_shape), device=device, requires_grad=False)
            fake = torch.zeros((real_A.size(0), *D_A.output_shape), device=device, requires_grad=False)

            # ------ 训练生成器 ------
            losses_G = train_generator(
                G_AB, G_BA, D_A, D_B, real_A, real_B, valid, fake,
                criterion_GAN, criterion_cycle, criterion_identity,
                optimizer_G, opt['lambda_cyc'], opt['lambda_id']
            )

            # ------ 训练判别器 A ------
            loss_D_A = train_discriminator(
                D_A, real_A, G_BA(real_B), valid, fake,
                criterion_GAN, optimizer_D_A, fake_A_buffer
            )

            # ------ 训练判别器 B ------
            loss_D_B = train_discriminator(
                D_B, real_B, G_AB(real_A), valid, fake,
                criterion_GAN, optimizer_D_B, fake_B_buffer
            )

            loss_D = (loss_D_A + loss_D_B) / 2

            # 累加 epoch 损失（分 A 和 B 域）
            epoch_losses['D_A'] += loss_D_A
            epoch_losses['D_B'] += loss_D_B
            epoch_losses['G'] += losses_G['G']
            epoch_losses['GAN_AB'] += losses_G['GAN_AB']
            epoch_losses['GAN_BA'] += losses_G['GAN_BA']
            epoch_losses['Cycle_A'] += losses_G['cycle_A']
            epoch_losses['Cycle_B'] += losses_G['cycle_B']
            epoch_losses['Identity_A'] += losses_G['identity_A']
            epoch_losses['Identity_B'] += losses_G['identity_B']

            # 更新进度条
            batches_done = epoch * len(dataloader) + i
            batches_left = opt['n_epochs'] * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            pbar.set_postfix({
                'D_loss': f'{loss_D:.4f}',
                'G_loss': f'{losses_G["G"]:.4f}',
                'GAN': f'{losses_G["GAN"]:.4f}',
                'cyc': f'{losses_G["cycle"]:.4f}',
                'id': f'{losses_G["identity"]:.4f}',
                'ETA': str(time_left)
            })

            # 采样图像
            if batches_done % opt['sample_interval'] == 0:
                img_sample = sample_images(batches_done, real_A, real_B, G_AB, G_BA, opt['dataset_name'])
                # 将图像记录到 TensorBoard
                writer.add_image('Sample_Images', img_sample, batches_done, dataformats='CHW')

        pbar.close()

        # 记录损失到 TensorBoard
        log_epoch_losses(writer, epoch, epoch_losses, len(dataloader))

        # 记录学习率
        log_learning_rates(writer, epoch, (optimizer_G, optimizer_D_A, optimizer_D_B))

        # 更新学习率
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # 打印 epoch 总结
        epoch_time = time.time() - epoch_start_time
        avg_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
        print(f"\n✅ Epoch {epoch + 1}/{opt['n_epochs']} 完成 - 耗时：{datetime.timedelta(seconds=epoch_time)}")
        print(f"   D_A: {avg_losses['D_A']:.4f} | D_B: {avg_losses['D_B']:.4f} | "
              f"G: {avg_losses['G']:.4f} | GAN_AB: {avg_losses['GAN_AB']:.4f} | GAN_BA: {avg_losses['GAN_BA']:.4f} | "
              f"Cycle_A: {avg_losses['Cycle_A']:.4f} | Cycle_B: {avg_losses['Cycle_B']:.4f} | "
              f"Id_A: {avg_losses['Identity_A']:.4f} | Id_B: {avg_losses['Identity_B']:.4f}\n")

        # 保存检查点
        if opt['checkpoint_interval'] != -1 and (epoch + 1) % opt['checkpoint_interval'] == 0:
            save_checkpoint(epoch + 1, (G_AB, G_BA, D_A, D_B), opt)
            print(f"💾 已保存 epoch {epoch + 1} 的检查点\n")

    print("\n🎉 训练完成！")
    writer.close()


if __name__ == "__main__":
    # 命令行参数
    args = parse_arguments()
    # 加载配置
    config = load_config(args)
    # 训练
    train(config)

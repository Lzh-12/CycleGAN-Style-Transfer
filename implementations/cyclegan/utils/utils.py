import argparse
import os
import random
import torch
import yaml

from torch.autograd import Variable
from torchvision.utils import save_image


def parse_arguments():
    """
    解析命令行参数

    返回:
        argparse.Namespace: 命令行参数对象
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml", help="YAML 配置文件路径")
    parser.add_argument("--epoch", type=int, default=None, help="开始训练的起始 epoch（覆盖配置文件）")
    parser.add_argument("--dataset_name", type=str, default=None, help="数据集名称（覆盖配置文件）")
    parser.add_argument("--model_type", type=str, default="cyclegan", 
                        choices=["cyclegan", "cbam", "spectral_norm", "hybrid_upsample"], 
                        help="模型类型选择（cyclegan/cbam/spectral_norm/hybrid_upsample）")
    return parser.parse_args()


def load_config(args):
    """
    从 YAML 文件加载配置，并允许命令行参数覆盖

    参数:
        args: 命令行参数

    返回:
        dict: 配置字典
    """
    config_path = args.config

    # 加载主配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 如果有_base_字段，先加载基础配置
    if '_base_' in config:
        base_config_path = os.path.join(os.path.dirname(config_path), config['_base_'])
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)

            # 合并配置（主配置覆盖基础配置）
            merged_config = {**base_config, **config}
            config = merged_config

    # 遍历命令行参数，如果提供了则覆盖 YAML 配置
    if args.epoch is not None:
        config['epoch'] = args.epoch
    if args.dataset_name is not None:
        config['dataset_name'] = args.dataset_name
    if args.model_type is not None:
        config['model_type'] = args.model_type

    return config


def sample_images(opt, batches_done, real_A, real_B, G_AB, G_BA):
    """
    使用当前训练批次保存生成的样本图像
    展示：真实图像 A → 生成图像 B → 重建图像 A
         真实图像 B → 生成图像 A → 重建图像 B

    args:
        opt: 配置选项 (字典)
        batches_done: 已完成的批次数量
        real_A: 域 A 的真实图像
        real_B: 域 B 的真实图像
        G_AB: A→B 的生成器
        G_BA: B→A 的生成器
    """
    with torch.no_grad():
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)

        row1 = torch.cat((real_A, fake_B, recov_A), -1)
        row2 = torch.cat((real_B, fake_A, recov_B), -1)
        img_sample = torch.cat((row1, row2), -2)

        save_image(img_sample, "images/%s/%s.png" % (opt['dataset_name'], batches_done), nrow=1, normalize=True)


def get_model_classes(model_type):
    """根据模型类型动态导入模型类"""
    module_name = MODEL_TYPES.get(model_type, MODEL_TYPES[DEFAULT_MODEL_TYPE])
    module = importlib.import_module(module_name)
    return module.GeneratorResNet, module.Discriminator, module.weights_init_normal

    
class ReplayBuffer:
    """
    经验回放缓冲区（Experience Replay Buffer）

    用于存储之前生成的假图像，目的是打破样本之间的相关性，
    提高判别器的训练稳定性，防止模式崩溃。

    工作原理:
        1. 当缓冲区未满时，直接存储新生成的样本
        2. 当缓冲区已满时，以 50% 的概率用新样本替换旧样本
        3. 以 50% 的概率直接返回缓冲区中的历史样本

    属性:
        max_size (int): 缓冲区最大容量，默认 50
        data (list): 存储图像的列表
    """
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                # 缓冲区未满，直接添加
                self.data.append(element)
                to_return.append(element)
            else:
                # 缓冲区已满，按概率决定操作
                if random.uniform(0, 1) > 0.5:
                    # 50% 概率：返回旧样本，用新样本替换
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    # 50% 概率：直接返回新样本
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    """
    学习率衰减调度器（Lambda Learning Rate Scheduler）

    实现线性学习率衰减策略：
    - 在 decay_start_epoch 之前保持初始学习率不变
    - 从 decay_start_epoch 开始，学习率线性下降到 0

    学习率计算公式:
        lr = initial_lr * (1.0 - max(0, epoch + offset - decay_start_epoch) / (n_epochs - decay_start_epoch))

    属性:
        n_epochs (int): 总训练 epoch 数
        offset (int): 起始 epoch 偏移量（用于断点续训）
        decay_start_epoch (int): 开始衰减的 epoch
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        """
        初始化学习率调度器

        参数:
            n_epochs (int): 总训练 epoch 数
            offset (int): 已完成的 epoch 数（断点续训时使用）
            decay_start_epoch (int): 开始学习率衰减的 epoch

        异常:
           AssertionError: 如果衰减开始时间在训练结束时间之后
        """
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

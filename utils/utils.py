"""
通用工具函数模块
"""

import argparse
import os
import yaml


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
    parser.add_argument("--dataroot", type=str, default=None, help="数据集根目录（覆盖配置文件）")
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
    
    # 如果配置文件路径不完整，添加默认路径
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", config_path)
    
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
    if args.dataroot is not None:
        config['dataroot'] = args.dataroot
    
    return config


def print_config(opt):
    """
    打印配置信息
    
    参数:
        opt: 配置字典
    """
    print("*" * 50)
    print("配置信息:")
    print("=" * 50)
    for key, value in opt.items():
        print(f"{key}: {value}")
    print("=" * 50)

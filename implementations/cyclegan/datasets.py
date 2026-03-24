"""
CycleGAN 数据集加载模块
用于加载和处理两个域（A域和B域）的图像数据
"""
import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    图像数据集类，用于加载两个域的图像对
    
    参数:
        root: 数据集根目录
        transforms_: 图像变换列表
        mode: 数据集模式（'train'、'test'等）
        preload: 是否预加载数据到内存（默认False）
    """
    def __init__(self, root, transforms_=None, mode="train", preload=False):
        # 组合图像变换
        self.transform = transforms.Compose(transforms_)
        self.preload = preload

        # 构建路径：root + mode + /A 和 /B
        path_A = os.path.join(root, mode, "A")
        path_B = os.path.join(root, mode, "B")

        # 获取所有图像文件（支持 jpg/png 等格式）
        self.files_A = sorted(glob.glob(os.path.join(path_A, "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(path_B, "*.*")))

        # 打印数据集信息
        print(f"[Dataset] Found {len(self.files_A)} images in {mode}/A")
        print(f"[Dataset] Found {len(self.files_B)} images in {mode}/B")

        # 检查数据集是否为空
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise RuntimeError(f"Empty dataset! Check paths: {path_A}, {path_B}")
        
        # 预加载数据到内存
        if preload:
            print(f"[Dataset] Preloading images to memory...")
            self.images_A = []
            self.images_B = []
            
            # 加载A域图像
            for file_path in self.files_A:
                img = Image.open(file_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                self.images_A.append(img)
            
            # 加载B域图像
            for file_path in self.files_B:
                img = Image.open(file_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                self.images_B.append(img)
            
            print(f"[Dataset] Preloading completed. {len(self.images_A)} images loaded.")
        else:
            self.images_A = None
            self.images_B = None

    def __getitem__(self, index):
        """
        获取单个数据样本
        
        参数:
            index: 数据索引
            
        返回:
            包含 A 域和 B 域图像的字典
        """
        # 从 A 域取图（使用模运算循环索引）
        if self.preload:
            img_A = self.images_A[index % len(self.images_A)]
        else:
            img_A = Image.open(self.files_A[index % len(self.files_A)])
            if img_A.mode != "RGB":
                img_A = img_A.convert("RGB")
        
        # 从 B 域随机取图（避免固定配对，增加多样性）
        b_index = random.randint(0, len(self.files_B) - 1)
        if self.preload:
            img_B = self.images_B[b_index]
        else:
            img_B = Image.open(self.files_B[b_index])
            if img_B.mode != "RGB":
                img_B = img_B.convert("RGB")

        # 随机水平翻转（数据增强，提高模型泛化能力）
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)

        # 应用图像变换（如 resize、归一化等）
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        """
        返回数据集长度
        
        返回:
            两个域中图像数量的较大值
        """
        if self.preload:
            return max(len(self.images_A), len(self.images_B))
        else:
            return max(len(self.files_A), len(self.files_B))

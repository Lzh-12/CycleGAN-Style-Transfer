# import glob
# import random
# import os
# import numpy as np
#
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms
#
#
# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.transform = transforms.Compose(transforms_)
#
#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#
#     def __getitem__(self, index):
#
#         img = Image.open(self.files[index % len(self.files)])
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))
#
#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
#
#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)
#
#         return {"A": img_A, "B": img_B}
#
#     def __len__(self):
#         return len(self.files)


import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        # 构建路径：root + mode + /A 和 /B
        path_A = os.path.join(root, mode, "A")
        path_B = os.path.join(root, mode, "B")

        # 获取所有图像文件（支持 jpg/png 等）
        self.files_A = sorted(glob.glob(os.path.join(path_A, "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(path_B, "*.*")))

        # 调试信息
        print(f"[Dataset] Found {len(self.files_A)} images in {mode}/A")
        print(f"[Dataset] Found {len(self.files_B)} images in {mode}/B")

        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise RuntimeError(f"Empty dataset! Check paths: {path_A}, {path_B}")

    def __getitem__(self, index):
        # 从 A 域取图（循环索引）
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        # 从 B 域随机取图（避免固定配对）
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])

        # 强制转为 RGB（防止灰度图出错）
        if img_A.mode != "RGB":
            img_A = img_A.convert("RGB")
        if img_B.mode != "RGB":
            img_B = img_B.convert("RGB")

        # 随机水平翻转（数据增强）
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)

        # 应用变换
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

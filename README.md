# 基于深度学习的图片风格转化

## 项目简介

本项目实现了基于深度学习的图像风格转换系统，主要基于CycleGAN架构，并进行了多项改进和优化。通过生成对抗网络（GAN）技术，实现了不同领域之间的图像风格迁移，无需成对训练数据即可完成高质量的风格转换。

## 技术栈

- **编程语言**: Python 3.9+
- **深度学习框架**: PyTorch 2.5.+
- **图像处理**: OpenCV, PIL
- **可视化**: TensorBoard, Matplotlib
- **其他工具**: NumPy, YAML, tqdm

## 项目特点

1. **改进的CycleGAN架构**：在基础CycleGAN模型上进行了多项改进
2. **注意力机制集成**：集成CBAM（Convolutional Block Attention Module）注意力机制
3. **谱归一化优化**：引入谱归一化（Spectral Normalization）提升模型稳定性
4. **混合上采样策略**：采用改进的上采样模块提升生成图像质量
5. **多种风格转换**：支持多种风格域之间的相互转换
6. **完整的实验体系**：包含对比实验和多个消融实验

## 项目结构

```
CycleGAN-Style-Transfer/
|—— checkpoints/            # 模型检查点目录
│   ├── cyclegan/            # CycleGAN模型检查点
│   ├── cyclegan_cbam/       # CycleGAN+CBAM模型检查点
├── data/                    # 数据集目录
│   ├── download_cyclegan_dataset.sh  # CycleGAN数据集下载脚本
│   └── download_pix2pix_dataset.sh   # Pix2Pix数据集下载脚本
├── implementations/         # 代码实现
│   ├── cyclegan/            # CycleGAN实现
│   │   ├── config/          # 配置文件
│   │   │   ├── base.yaml    # 基础配置
│   │   │   ├── test.yaml    # 测试配置
│   │   │   └── train.yaml   # 训练配置
│   │   ├── models/          # 模型定义
│   │   │   ├── ahs_models.py    # 改进的模型定义
│   │   │   ├── cbam_models.py   # CBAM注意力模型
│   │   │   ├── hybrid_upsample_models.py  # 混合上采样模型
│   │   │   ├── models.py        # 基础模型定义
│   │   │   └── sn_models.py     # 谱归一化模型
│   │   ├── module/          # 模块实现
│   │   │   ├── CBAM.py          # CBAM注意力模块
│   │   │   └── HybridUpsampleBlock.py  # 混合上采样模块
│   │   ├── utils/           # 工具函数
│   │   │   ├── __init__.py
│   │   │   └── utils.py     # 通用工具
│   │   ├── __pycache__/     # Python缓存目录
│   │   ├── __init__.py
│   │   ├── constants.py     # 常量定义
│   │   ├── cyclegan.py      # 训练主脚本
│   │   ├── datasets.py      # 数据集加载
│   │   └── test_cyclegan.py # 测试脚本
│   ├── discogan/            # DiscoGAN实现
│   └── dualgan/             # DualGAN实现
├── output/                  # 输出目录（生成图像）
│   ├── cbam/                # CBAM模型生成结果
│   │   └── monet2photo/     # Monet到照片转换结果
│   └── spectral_normalization/  # 谱归一化模型生成结果
│       └── vangogh2photo/   # Van Gogh到照片转换结果
├── utils/                   # 项目通用工具
│   ├── metrics.py           # 评估指标
│   └── utils.py             # 通用工具函数
├── .gitignore               # Git忽略文件
└── README.md                # 项目说明文档
```

## 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/Lzh-12/CycleGAN-Style-Transfer.git
   cd CycleGAN-Style-Transfer
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install torch torchvision
   pip install opencv-python pillow numpy pyyaml tqdm matplotlib
   ```

## 使用方法

### 1. 下载数据集

```bash
cd data
bash download_cyclegan_dataset.sh horse2zebra
# 支持的数据集：horse2zebra, monet2photo, apple2orange, ukiyoe2photo, vangogh2photo, maps, facades
```

### 2. 训练模型

```bash
# 训练基线模型
cd implementations/cyclegan
python cyclegan.py --config config/train.yaml --dataset_name horse2zebra

# 训练CBAM改进版
python cyclegan.py --config config/train_cbam.yaml --dataset_name horse2zebra
```

### 3. 测试模型

```bash
# 测试基线模型
python test_cyclegan.py --config config/test.yaml --dataset_name horse2zebra --epoch 200

# 测试CBAM改进版
python test_cyclegan.py --config config/test_cbam.yaml --dataset_name horse2zebra --epoch 200
```

### 4. 查看训练过程

```bash
tensorboard --logdir experiments/horse2zebra/baseline/logs
```


## 核心改进点

1. **CBAM注意力机制**：在生成器中集成通道注意力和空间注意力模块，提升特征表达能力
2. **谱归一化优化**：在判别器中应用谱归一化，提高GAN训练的稳定性和收敛速度
3. **混合上采样策略**：结合亚像素重排（Pixel Shuffle）和双线性插值法的优点，减少棋盘格效应，提升图像质量
4. **改进的损失函数**：优化循环一致性损失和身份映射损失的权重配比，平衡生成质量和风格转换效果

## 配置说明

主要配置参数位于 `implementations/cyclegan/config/` 目录：

- `base.yaml`: 基础配置
- `train.yaml`: 训练配置
- `test.yaml`: 测试配置

## 致谢

- 感谢CycleGAN原论文作者的贡献
- 感谢PyTorch团队提供的优秀深度学习框架
- 感谢所有开源社区的贡献者

## 参考文献

1. Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).

2. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19).

## 联系方式

- 作者: Lzh
- 邮箱: li20922024@163.com

---

*本项目为毕业设计作品，仅供学习和研究使用。*
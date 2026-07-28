# TinyVision\-AI Challenge


# 面向无人机边缘设备的实时语义分割算法系统

---

# 一、项目定位

## 项目类型

模拟真实 AI 算法工程项目：

> 在有限计算资源下，实现一个高精度、高效率的无人机视觉理解系统。
> 
> 

---

## 核心任务

输入：

无人机 RGB 图像

```Plain Text
Image:
H × W × 3
```

输出：

像素级语义类别

```Plain Text
Mask:
H × W
```

目标类别：

---

# 二、项目核心问题

无人机视觉相比普通图像任务具有特殊困难：

## 1\. 小目标问题

例如：

车辆、行人只占几十个pixel。

经过网络：

```Plain Text
512×512

↓

256×256

↓

128×128

↓

64×64
```

大量细节丢失。

---

## 2\. 多尺度问题

同一张图：

建筑：

几百像素

车辆：

几十像素

模型需要同时理解：

大结构

- 

小细节。

---

## 3\. 精度与速度平衡

无人机无法部署巨大模型。

需要考虑：

Accuracy

↓

FPS

↓

Memory

三者平衡。

---

# 三、整体技术路线

```Plain Text
Level 0
项目工程初始化

↓

Level 1
数据处理与任务理解

↓

Level 2
Baseline模型开发

↓

Level 3
实验体系建立

↓

Level 4
经典分割模型进阶

↓

Level 5
无人机场景优化

↓

Level 6
模型轻量化与部署

↓

Level 7
完整系统交付
```

预计周期：

8\~10周。

---

# Level 0 项目工程初始化

## Theme

建立标准深度学习项目开发环境。

---

# Task 0\.1 项目结构搭建

## 目标

从第一天开始按照真实算法项目组织代码。

最终结构：

```Plain Text
TinyVision-AI

├── configs
│
├── datasets
│
├── models
│
├── losses
│
├── utils
│
├── tools
│
├── train.py
│
├── test.py
│
├── inference.py
│
└── README.md
```

---

## 实现内容

完成：

- PyTorch环境

- GPU训练

- 参数配置

- checkpoint保存

- logging

- tensorboard

---

## 学习重点

理解：

一个模型训练项目不仅是：

model\.py

而是：

```Plain Text
Dataset

↓

Dataloader

↓

Model

↓

Loss

↓

Optimizer

↓

Training Loop

↓

Evaluation
```

---

## 完成标准

能够：

只修改config：

改变：

- 模型

- batch size

- learning rate

- 数据路径

不用改训练代码。

---

# Task 0\.2 阅读成熟代码框架

## 目标

提升阅读工业代码能力。

阅读：

- torchvision segmentation

- mmsegmentation结构

重点理解：

```Plain Text
Dataset

↓

Backbone

↓

Neck

↓

Head

↓

Loss
```

---

## 完成标准

能够解释：

一个语义分割模型代码从输入到输出的数据流。

---

# Level 1 数据处理与任务理解

## Theme

理解真实视觉任务，而不是直接训练模型。

---

# Task 1\.1 UAV数据集构建

数据：

UAVid

完成：

- 数据下载

- 图片读取

- mask解析

- 类别映射

- train/val划分

---

## 学习内容

理解：

语义分割数据：

Image:

```Plain Text
RGB
```

Mask:

```Plain Text
pixel label
```

---

## 完成标准

实现：

```Python
dataset[i]

return:

image, mask
```

---

# Task 1\.2 数据分析与可视化

完成工具：

显示：

```Plain Text
Original Image

+

Ground Truth Mask
```

分析：

- 类别比例

- 目标尺寸

- 场景复杂度

---

## 目的

回答：

这个任务为什么困难？

---

# Task 1\.3 数据增强Pipeline

实现：

基础：

- Resize

- Crop

- Flip

- Rotation

- ColorJitter

进阶：

- Mixup

- CutMix

---

## 理解

augmentation本质：

改变训练数据分布，提高泛化能力。

---

# Level 2 Baseline模型开发

## Theme

建立可靠性能基线。

---

# Task 2\.1 手写UNet

## 目标

实现经典Encoder\-Decoder结构。

---

理解：

Encoder：

```Plain Text
空间信息降低

语义信息增强
```

Decoder：

```Plain Text
恢复空间结构
```

Skip Connection：

```Plain Text
低层细节

+

高层语义
```

---

实现模块：

- ConvBlock

- DownSample

- UpSample

- Skip Connection

---

完成标准：

模型可以稳定训练并产生合理mask。

---

# Task 2\.2 训练策略优化

实验：

## Optimizer

比较：

- Adam

- SGD

---

## Learning Rate

比较：

- Step decay

- Cosine decay

---

## Loss

实现：

- CrossEntropy

- Dice Loss

- Focal Loss

---

目标：

建立稳定训练方案。

---

# Level 3 实验体系建立

## Theme

培养算法工程实验能力。

---

# Task 3\.1 实验管理

建立：

统一实验记录。

记录：

- 参数

- 模型

- 数据

- 指标

---

例如：

---

# Task 3\.2 错误分析工具

实现：

分析：

模型错误区域。

包括：

- False Positive

- False Negative

- Boundary Error

---

目标：

以后优化模型必须来自错误分析。

---

# Level 4 经典模型进阶

## Theme

理解不同Architecture如何改变Representation。

只研究三个模型：

---

# Task 4\.1 UNet → DeepLabV3\+

## 动机

UNet：

局部卷积。

问题：

大范围目标理解不足。

---

学习：

ASPP:

不同尺度感受野。

---

比较：

```Plain Text
UNet

vs

DeepLabV3+
```

理解：

CNN如何扩大信息交互范围。

---

# Task 4\.2 DeepLabV3\+ → SegFormer

## 动机

CNN：

局部建模。

Transformer：

全局关系。

---

学习：

- Patch Embedding

- Self Attention

- Hierarchical Feature

---

比较：

```Plain Text
CNN segmentation

vs

Transformer segmentation
```

---

# Task 4\.3 模型综合比较

统一测试：

- mIoU

- Params

- FPS

- Memory

形成：

模型选择依据。

---

# Level 5 无人机场景优化

## Theme

针对真实问题设计改进。

这是整个项目核心。

---

# Task 5\.1 小目标增强 ⭐核心任务

## 问题

无人机：

车辆、人：

尺寸小。

Baseline：

容易漏检。

---

## 分析

原因：

CNN下采样导致：

高频细节丢失。

---

## 优化方向

设计：

Multi\-scale Feature Fusion

例如：

增加：

High Resolution Branch

融合：

低层空间信息

- 

高层语义信息

---

## 验证

比较：

Baseline

vs

改进模型

观察：

small object IoU。

---

# Task 5\.2 模型结构优化

根据实验结果：

选择：

- Feature Fusion

- Attention Module

- Decoder优化

之一。

---

要求：

不是增加复杂度。

而是解决具体问题。

---

# Level 6 模型轻量化与部署

## Theme

从实验模型到实际应用。

---

# Task 6\.1 模型效率分析

测试：

不同模型：

---

理解：

模型性能不是只有accuracy。

---

# Task 6\.2 Knowledge Distillation

设计：

Teacher:

大模型

Student:

轻量模型

目标：

保持精度降低计算。

---

# Task 6\.3 ONNX部署

完成：

PyTorch

↓

ONNX

测试：

真实推理速度。

---

# Level 7 完整系统交付

## Theme

形成最终算法项目。

---

# Task 7\.1 推理Demo

实现：

输入：

无人机图片

输出：

```Plain Text
Original Image

↓

Segmentation Mask

↓

Visualization
```

---

# Task 7\.2 Github整理

最终：

```Plain Text
TinyVision-AI

├── configs

├── datasets

├── models

├── train.py

├── inference.py

├── demo

└── README
```

---

README：

包含：

- 项目介绍

- 环境配置

- 使用方法

- 模型结果

---



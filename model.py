from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18_model(num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    构建基于 ResNet18 的分类模型，并返回模型与输入图像的期望尺寸。

    这里采用 torchvision 提供的预训练 ResNet18 作为特征提取 backbone，
    冻结其大部分参数，仅微调最后的全连接层（迁移学习）。
    """
    # 根据 torchvision 新版 API，weights 参数用于加载预训练权重
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # 冻结除最后一层外的所有参数，降低过拟合风险并加快训练
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 替换最后一层全连接，以适配 4 类 PD 缺陷分类任务
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 新的全连接层参数需要参与训练
    for param in model.fc.parameters():
        param.requires_grad = True

    # ResNet18 默认输入图像尺寸为 224x224
    image_size = 224

    return model, image_size


if __name__ == "__main__":
    """
    简单自测：构建模型并打印结构与参数量。
    """
    model, image_size = build_resnet18_model(num_classes=4, pretrained=True)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params}, 可训练参数量: {trainable_params}")
    print(f"期望输入图像尺寸: {image_size}x{image_size}")


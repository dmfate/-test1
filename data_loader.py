import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224):
    """
    定义训练/验证/测试时统一使用的图像预处理与增强流程。

    注意：由于我们使用的是基于 ImageNet 预训练的 ResNet18，
    因此这里采用与 ImageNet 一致的归一化参数。
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    从 preprocess.py 生成的 PD_Dataset 目录中读取图像，
    并划分为训练/验证/测试集，返回 DataLoader。

    目录结构示例：
    PD_Dataset/
      ├── Void/
      ├── Surface/
      ├── Corona/
      └── Floating/
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"数据集目录 {data_root} 不存在，请先运行 preprocess.py 生成图像。")

    train_tf, eval_tf = get_transforms()

    # 使用 ImageFolder 自动读取类别子目录
    full_dataset = datasets.ImageFolder(root=data_root, transform=train_tf)
    class_names = full_dataset.classes

    total_size = len(full_dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 验证集与测试集使用 eval_transform（不做随机增强）
    val_dataset.dataset.transform = eval_tf
    test_dataset.dataset.transform = eval_tf

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    """
    简单自测：仅在命令行运行本脚本时，打印数据集大小信息。

    示例：
    python data_loader.py --data_root ./PD_Dataset
    """
    import argparse

    parser = argparse.ArgumentParser(description="测试数据加载模块")
    parser.add_argument(
        "--data_root", type=str, default="PD_Dataset", help="PD 图像数据集根目录"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )

    print(f"类别列表: {class_names}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")


import os
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from data_loader import create_dataloaders
from model import build_resnet18_model


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    单次训练 epoch，返回本 epoch 的平均 Loss 与 Accuracy。
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return {"loss": epoch_loss, "acc": epoch_acc}


def eval_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    """
    在验证或测试集上评估，返回平均 Loss 与 Accuracy。
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return {"loss": epoch_loss, "acc": epoch_acc}


def main():
    """
    训练入口脚本。

    典型使用方式：
    python train.py ^
        --data_root ./PD_Dataset ^
        --epochs 30 ^
        --batch_size 32 ^
        --output_dir ./checkpoints
    """
    parser = argparse.ArgumentParser(description="训练基于 ResNet18 的 PD 缺陷分类模型")
    parser.add_argument(
        "--data_root",
        type=str,
        default="PD_Dataset",
        help="PD 图像数据集根目录（由 preprocess.py 生成）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="训练批大小",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="保存模型与训练曲线的目录",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="强制使用 CPU 训练（一般建议使用 GPU）",
    )

    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"使用设备: {device}")

    # 构建数据加载器
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)
    print(f"检测到类别数: {num_classes}, 类别名称: {class_names}")

    # 构建模型
    model, image_size = build_resnet18_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # 仅优化可训练参数（最后一层）
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(args.epochs):
        print(f"\n===== Epoch [{epoch + 1}/{args.epochs}] =====")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = eval_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['acc']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )

        # 保存当前最优模型
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": image_size,
                },
                best_model_path,
            )
            print(f"验证集准确率提升到 {best_val_acc:.4f}，已保存最优模型到 {best_model_path}")

    # 保存训练过程曲线数据，供 evaluate.py 绘图
    history_path = os.path.join(args.output_dir, "training_history.pth")
    torch.save(history, history_path)
    print(f"训练曲线数据已保存到 {history_path}")

    # 训练结束后，在测试集上评估一次最终模型表现
    test_metrics = eval_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(
        f"\n===== 使用最后一轮模型在测试集上的表现 =====\n"
        f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['acc']:.4f}"
    )


if __name__ == "__main__":
    main()


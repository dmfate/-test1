import os
import argparse
from typing import List

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from model import build_resnet18_model


def evaluate_and_plot_confusion_matrix(
    model: nn.Module,
    test_loader,
    device: torch.device,
    class_names: List[str],
    output_dir: str,
) -> None:
    """
    在测试集上评估模型并生成混淆矩阵图。
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"混淆矩阵已保存到 {cm_path}")


def plot_training_curves(history_path: str, output_dir: str) -> None:
    """
    从 train.py 保存的 training_history.pth 中读取 loss/acc 曲线，
    并绘制训练与验证的 Loss/Accuracy 变化图。
    """
    if not os.path.isfile(history_path):
        print(f"未在 {history_path} 找到训练历史文件，跳过曲线绘制。")
        return

    history = torch.load(history_path)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    epochs = range(1, len(train_loss) + 1)
    os.makedirs(output_dir, exist_ok=True)

    # Loss 曲线
    plt.figure()
    plt.plot(epochs, train_loss, "r-o", label="Train Loss")
    plt.plot(epochs, val_loss, "b-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Loss 曲线已保存到 {loss_path}")

    # Accuracy 曲线
    plt.figure()
    plt.plot(epochs, train_acc, "r-o", label="Train Acc")
    plt.plot(epochs, val_acc, "b-o", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Accuracy 曲线已保存到 {acc_path}")


def main():
    """
    模型评估与可视化入口脚本。

    典型使用方式：
    python evaluate.py ^
        --data_root ./PD_Dataset ^
        --checkpoint_dir ./checkpoints
    """
    parser = argparse.ArgumentParser(description="评估 PD 缺陷分类模型并生成可视化结果")
    parser.add_argument(
        "--data_root",
        type=str,
        default="PD_Dataset",
        help="PD 图像数据集根目录（由 preprocess.py 生成）",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="训练脚本保存模型与曲线的目录",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="强制使用 CPU 进行推理",
    )

    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"使用设备: {device}")

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    history_path = os.path.join(args.checkpoint_dir, "training_history.pth")

    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(
            f"未在 {best_model_path} 找到最优模型文件，请先运行 train.py 完成训练。"
        )

    # 读取模型信息
    checkpoint = torch.load(best_model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model, _ = build_resnet18_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 重新构建数据加载器，仅需 test_loader
    _, _, test_loader, class_names_from_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=32,
    )
    assert (
        class_names == class_names_from_loader
    ), "模型保存时的类别顺序与当前数据集类别顺序不一致，请检查。"

    # 绘制 Loss / Accuracy 曲线
    plot_training_curves(history_path=history_path, output_dir=args.checkpoint_dir)

    # 在测试集上生成混淆矩阵
    evaluate_and_plot_confusion_matrix(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()


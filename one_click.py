"""一键启动脚本。

用法示例：
  python one_click.py
  python one_click.py --mode full
  python one_click.py --mode full --no-gui
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "PD_Dataset"
MAT_ROOT = ROOT / "mat_files"
CKPT_DIR = ROOT / "checkpoints"
BEST_MODEL = CKPT_DIR / "best_model.pth"
HISTORY = CKPT_DIR / "training_history.pth"


def run_step(cmd: list[str], name: str) -> None:
    print(f"\n[STEP] {name}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def dataset_ready(data_root: Path) -> bool:
    if not data_root.exists() or not data_root.is_dir():
        return False
    classes = [p for p in data_root.iterdir() if p.is_dir()]
    return len(classes) >= 2


def checkpoints_ready() -> bool:
    return BEST_MODEL.is_file() and HISTORY.is_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PD 项目一键启动")
    parser.add_argument(
        "--mode",
        choices=["gui", "full"],
        default="gui",
        help="gui: 直接启动界面；full: 自动补齐预处理/训练/评估后再启动界面",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="full 模式下训练轮数（仅在需要训练时生效）",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="训练 batch size")
    parser.add_argument("--use-cpu", action="store_true", help="训练/评估阶段强制使用 CPU")
    parser.add_argument("--no-gui", action="store_true", help="full 模式下执行完评估后不启动 GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "full":
        if not dataset_ready(DATA_ROOT):
            if not MAT_ROOT.is_dir():
                raise FileNotFoundError(
                    f"未找到数据集目录 {DATA_ROOT}，且 mat 原始数据目录 {MAT_ROOT} 也不存在。"
                )
            run_step(
                [
                    sys.executable,
                    "preprocess.py",
                    "--mat_root",
                    str(MAT_ROOT),
                    "--output_root",
                    str(DATA_ROOT),
                ],
                "生成 PRPS 图像数据集",
            )
        else:
            print("[SKIP] 已检测到可用数据集，跳过 preprocess。")

        if not checkpoints_ready():
            train_cmd = [
                sys.executable,
                "train.py",
                "--data_root",
                str(DATA_ROOT),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--output_dir",
                str(CKPT_DIR),
            ]
            if args.use_cpu:
                train_cmd.append("--use_cpu")
            run_step(train_cmd, "训练模型")
        else:
            print("[SKIP] 已检测到 best_model.pth + training_history.pth，跳过训练。")

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--data_root",
            str(DATA_ROOT),
            "--checkpoint_dir",
            str(CKPT_DIR),
        ]
        if args.use_cpu:
            eval_cmd.append("--use_cpu")
        run_step(eval_cmd, "评估并生成可视化结果")

        if not args.no_gui:
            run_step([sys.executable, "main.py"], "启动 GUI")
        return

    # mode == "gui"
    run_step([sys.executable, "main.py"], "启动 GUI")


if __name__ == "__main__":
    main()

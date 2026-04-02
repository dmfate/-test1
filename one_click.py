"""一键启动脚本（含自动环境自举）。

用法示例：
  python one_click.py
  python one_click.py --mode full
  python one_click.py --mode full --no-gui
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "PD_Dataset"
MAT_ROOT = ROOT / "mat_files"
CKPT_DIR = ROOT / "checkpoints"
BEST_MODEL = CKPT_DIR / "best_model.pth"
HISTORY = CKPT_DIR / "training_history.pth"
VENV_DIR = ROOT / ".venv"


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def valid_venv(venv_dir: Path) -> bool:
    """
    判断虚拟环境是否完整可用。
    仅 python 可执行文件存在但 pyvenv.cfg 缺失时，视为损坏环境。
    """
    return (venv_dir / "pyvenv.cfg").is_file() and venv_python(venv_dir).is_file()


def ensure_venv_and_reexec(argv: list[str], already_bootstrapped: bool) -> None:
    """
    保证使用仓库内 .venv 运行当前脚本。
    - 若当前不在 .venv 中：自动创建 .venv、安装依赖并重启到 .venv。
    - 若已在 .venv 或已经完成过自举：直接返回。
    """
    current_prefix = Path(getattr(sys, "prefix", ""))
    target_python = venv_python(VENV_DIR)

    in_target_venv = (
        valid_venv(VENV_DIR)
        and current_prefix == VENV_DIR
        and Path(sys.executable).resolve() == target_python.resolve()
    )

    if in_target_venv or already_bootstrapped:
        return

    if VENV_DIR.exists() and not valid_venv(VENV_DIR):
        print(f"[BOOTSTRAP] 检测到损坏的虚拟环境，正在重建: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    if not valid_venv(VENV_DIR):
        print(f"[BOOTSTRAP] 未检测到虚拟环境，正在创建: {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
        target_python = venv_python(VENV_DIR)

    print("[BOOTSTRAP] 安装/更新依赖 requirements.txt ...")
    try:
        subprocess.run(
            [str(target_python), "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=ROOT,
            check=True,
        )
    except subprocess.CalledProcessError:
        # 兼容 Windows 上偶发 “No pyvenv.cfg file” 的损坏环境场景：重建后重试一次。
        print("[BOOTSTRAP] 依赖安装失败，尝试重建 .venv 后重试一次。")
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
        venv.create(VENV_DIR, with_pip=True)
        target_python = venv_python(VENV_DIR)
        subprocess.run(
            [str(target_python), "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=ROOT,
            check=True,
        )

    print("[BOOTSTRAP] 依赖就绪，切换到 .venv 继续执行。")
    new_argv = [str(target_python), __file__, *argv, "--_bootstrapped"]
    subprocess.run(new_argv, cwd=ROOT, check=True)
    raise SystemExit(0)


def run_step(cmd: list[str], name: str, dry_run: bool = False) -> None:
    print(f"\n[STEP] {name}")
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def dataset_ready(data_root: Path) -> bool:
    if not data_root.exists() or not data_root.is_dir():
        return False
    classes = [p for p in data_root.iterdir() if p.is_dir()]
    return len(classes) >= 2


def checkpoints_ready() -> bool:
    return BEST_MODEL.is_file() and HISTORY.is_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PD 项目一键启动（自动环境）")
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
    parser.add_argument("--dry-run", action="store_true", help="仅打印将执行的步骤，不实际运行")
    parser.add_argument("--_bootstrapped", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_venv_and_reexec(sys.argv[1:], already_bootstrapped=args._bootstrapped)

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
                dry_run=args.dry_run,
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
            run_step(train_cmd, "训练模型", dry_run=args.dry_run)
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
        run_step(eval_cmd, "评估并生成可视化结果", dry_run=args.dry_run)

        if not args.no_gui:
            run_step([sys.executable, "main.py"], "启动 GUI", dry_run=args.dry_run)
        return

    # mode == "gui"
    run_step([sys.executable, "main.py"], "启动 GUI", dry_run=args.dry_run)


if __name__ == "__main__":
    main()

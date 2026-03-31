import os
import argparse
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_mat_file(mat_path: str) -> Dict[str, np.ndarray]:
    """
    读取单个 .mat 文件，并返回其中的关键信息数组。

    期望 .mat 文件中包含以下变量：
    - phase: 放电相位 (0~360 度)
    - pks_realistic: 放电幅值 (单位：V)
    - cycle_num: 工频周期序号 (1~N)
    - t_peaks: 放电发生的绝对时间点（本脚本中暂不使用）
    """
    data = sio.loadmat(mat_path)

    def _get(key: str) -> np.ndarray:
        if key not in data:
            raise KeyError(f"在文件 {mat_path} 中未找到变量 '{key}'")
        arr = np.asarray(data[key]).squeeze()
        return arr.astype(float)

    phase = _get("phase")
    pks_realistic = _get("pks_realistic")
    cycle_num = _get("cycle_num")
    t_peaks = _get("t_peaks")

    if not (len(phase) == len(pks_realistic) == len(cycle_num) == len(t_peaks)):
        raise ValueError(f"{mat_path} 中的四个数组长度不一致，请检查数据。")

    return {
        "phase": phase,
        "pks_realistic": pks_realistic,
        "cycle_num": cycle_num,
        "t_peaks": t_peaks,
    }


def generate_prps_histogram(
    phase: np.ndarray,
    cycle_num: np.ndarray,
    pks_realistic: np.ndarray,
    num_bins: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于相位 (phase) 与工频周期序号 (cycle_num) 生成 PRPS 直方图矩阵。

    这里采用 numpy.histogram2d 来模拟 MATLAB 中的 3D 直方图统计：
    - x 轴：相位 (0~360 度)
    - y 轴：序列（工频周期号）
    - 权重：放电幅值绝对值，反映放电能量大小

    返回：
    - H: 形状为 [num_bins, num_bins] 的二维矩阵
    - phase_edges: 相位方向的 bin 边界
    - cycle_edges: 序列方向的 bin 边界
    """
    # 相位范围固定到 [0, 360]
    phase_min, phase_max = 0.0, 360.0

    # 周期范围根据数据自动确定
    cycle_min, cycle_max = float(np.min(cycle_num)), float(np.max(cycle_num))

    H, phase_edges, cycle_edges = np.histogram2d(
        phase,
        cycle_num,
        bins=[num_bins, num_bins],
        range=[[phase_min, phase_max], [cycle_min, cycle_max]],
        weights=np.abs(pks_realistic),
    )

    # 为了可视化效果，对直方图做对数或归一化处理
    H = H.astype(float)
    if np.max(H) > 0:
        H = H / np.max(H)

    return H, phase_edges, cycle_edges


def apply_random_perturbation(
    phase: np.ndarray,
    cycle_num: np.ndarray,
    pks_realistic: np.ndarray,
    phase_jitter_deg: float = 5.0,
    amp_jitter_ratio: float = 0.05,
    subsample_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对原始散点数据进行随机扰动，用于数据增强：
    - 相位加上小范围高斯噪声
    - 幅值乘以 (1 + 小高斯噪声)
    - 随机子采样一部分点
    """
    n = len(phase)
    idx = np.random.choice(n, size=int(n * subsample_ratio), replace=False)

    phase_aug = phase[idx] + np.random.randn(len(idx)) * phase_jitter_deg
    # 相位保持在 [0, 360]
    phase_aug = np.mod(phase_aug, 360.0)

    amp_aug = pks_realistic[idx] * (1.0 + np.random.randn(len(idx)) * amp_jitter_ratio)
    cycle_aug = cycle_num[idx]

    return phase_aug, cycle_aug, amp_aug


def save_hist_as_png(hist: np.ndarray, out_path: str, cmap: str = "jet") -> None:
    """
    将 2D 直方图矩阵保存为 PNG 彩色图像。

    为了保证尺寸为 224x224，这里直接以 224x224 bin 生成矩阵，
    图像保存时使用 imshow，关闭坐标轴与边距。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.imshow(hist.T, origin="lower", cmap=cmap, aspect="auto")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=224 / 2.24, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_single_mat(
    mat_path: str,
    class_name: str,
    output_root: str,
    num_images: int = 200,
) -> None:
    """
    针对单个 .mat 文件，批量生成带随机扰动的 PRPS 图像。
    """
    data = load_mat_file(mat_path)
    phase = data["phase"]
    pks_realistic = data["pks_realistic"]
    cycle_num = data["cycle_num"]

    class_dir = os.path.join(output_root, class_name)
    os.makedirs(class_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(mat_path))[0]

    for i in range(num_images):
        phase_aug, cycle_aug, amp_aug = apply_random_perturbation(
            phase, cycle_num, pks_realistic
        )
        H, _, _ = generate_prps_histogram(phase_aug, cycle_aug, amp_aug)
        out_name = f"{base_name}_aug_{i:04d}.png"
        out_path = os.path.join(class_dir, out_name)
        save_hist_as_png(H, out_path)
        if (i + 1) % 20 == 0:
            print(f"[{class_name}] 已生成 {i + 1}/{num_images} 张图片")


def main():
    """
    命令行入口。

    示例使用方式：
    python preprocess.py ^
        --mat_root ./mat_files ^
        --output_root ./PD_Dataset

    其中 mat_root 文件夹下包含：
    - Void_PRPS_Data.mat
    - Surface_PRPS_Data.mat
    - Corona_PRPS_Data.mat
    - Floating_PRPS_Data.mat
    等文件。
    """
    parser = argparse.ArgumentParser(description="基于 .mat 数据生成 PRPS 图像数据集")
    parser.add_argument(
        "--mat_root",
        type=str,
        required=True,
        help="存放各类缺陷 .mat 文件的目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="PD_Dataset",
        help="生成的 PRPS 图像数据集根目录",
    )
    parser.add_argument(
        "--num_images_per_mat",
        type=int,
        default=200,
        help="每个 .mat 文件生成的图像数量",
    )

    args = parser.parse_args()

    # 类别名称与 .mat 文件名前缀的映射，可以根据实际情况调整
    cls_map = {
        "Void": "Void",
        "Surface": "Surface",
        "Corona": "Corona",
        "Floating": "Floating",
    }

    for class_name, prefix in cls_map.items():
        # 在 mat_root 中查找所有以该前缀开头、以 .mat 结尾的文件
        for fname in os.listdir(args.mat_root):
            if fname.startswith(prefix) and fname.lower().endswith(".mat"):
                mat_path = os.path.join(args.mat_root, fname)
                print(f"处理类别 {class_name} 的文件: {mat_path}")
                process_single_mat(
                    mat_path=mat_path,
                    class_name=class_name,
                    output_root=args.output_root,
                    num_images=args.num_images_per_mat,
                )

    print("全部 PRPS 图像生成完成。")


if __name__ == "__main__":
    main()


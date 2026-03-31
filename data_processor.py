# data_processor.py
# 数据处理模块：负责加载.mat/.csv数据，解析核心特征，绘制PRPD和PRPS图表
# 确保图表具有学术级美感：清晰坐标轴、合适配色、网格线等

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DataProcessor:
    """
    数据处理器类：封装数据加载、解析和可视化功能。
    """

    def __init__(self):
        """
        初始化数据处理器。
        """
        pass

    def load_data(self, file_path):
        """
        加载数据文件：支持.mat和.csv格式，提取phase、cycle_num、pks三列特征。

        参数:
        file_path (str): 数据文件路径。

        返回:
        dict: 包含phase、cycle_num、pks的字典。
        """
        if file_path.endswith('.mat'):
            # 加载.mat文件
            mat_data = scipy.io.loadmat(file_path)
            # 假设.mat文件中的变量名为'phase', 'cycle_num', 'pks'（根据实际文件调整）
            phase = mat_data.get('phase', np.array([])).flatten()
            cycle_num = mat_data.get('cycle_num', np.array([])).flatten()
            # 优先读取 pks_realistic，如果没有再尝试读取 pks
            if 'pks_realistic' in mat_data:
                pks = mat_data.get('pks_realistic').flatten()
            else:
                pks = mat_data.get('pks', np.array([])).flatten()

        elif file_path.endswith('.csv'):
            # 加载.csv文件
            df = pd.read_csv(file_path)
            # 假设.csv文件有列名'phase', 'cycle_num', 'pks'
            phase = df['phase'].values if 'phase' in df.columns else np.array([])
            cycle_num = df['cycle_num'].values if 'cycle_num' in df.columns else np.array([])
            pks = df['pks'].values if 'pks' in df.columns else np.array([])

        else:
            raise ValueError("不支持的文件格式！仅支持.mat和.csv")

        # 验证数据完整性
        if len(phase) == 0 or len(cycle_num) == 0 or len(pks) == 0:
            raise ValueError("数据文件缺少必要的列：phase, cycle_num, pks")

        return {
            'phase': phase,
            'cycle_num': cycle_num,
            'pks': pks
        }

    def plot_prpd(self, figure, data):
        """
        绘制PRPD图：二维散点图，横轴相位(0-360°)，纵轴幅值，叠加50Hz正弦波背景虚线。

        参数:
        figure (matplotlib.figure.Figure): Matplotlib图表对象。
        data (dict): 包含phase、pks的数据字典。
        """
        figure.clear()
        ax = figure.add_subplot(111)

        # 绘制散点图（学术级配色：蓝色散点）
        ax.scatter(data['phase'], data['pks'], alpha=0.6, color='#5E81AC', s=10, label='放电数据')

        # 叠加50Hz正弦波（红色虚线，作为参考）
        phase_ref = np.linspace(0, 360, 1000)
        amplitude_ref = np.sin(np.radians(phase_ref)) * np.max(data['pks']) * 0.5  # 缩放参考波
        ax.plot(phase_ref, amplitude_ref, '--', color='#BF616A', linewidth=2, label='50Hz参考正弦波')

        # 设置坐标轴标签和标题
        ax.set_xlabel('Phase (°)', fontsize=12)
        ax.set_ylabel('Amplitude (mV)', fontsize=12)
        ax.set_title('PRPD图 (Phase Resolved Partial Discharge)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 设置学术级样式
        ax.set_xlim(0, 360)
        ax.set_ylim(bottom=0)

    def plot_prps(self, figure, data):
        """
        绘制PRPS图：三维柱状图，X轴相位、Y轴周期、Z轴幅值，使用viridis伪彩色映射。

        参数:
        figure (matplotlib.figure.Figure): Matplotlib图表对象。
        data (dict): 包含phase、cycle_num、pks的数据字典。
        """
        figure.clear()
        ax = figure.add_subplot(111, projection='3d')

        # 聚合数据：按相位和周期分组，取平均幅值（简化处理）
        phase_bins = np.linspace(0, 360, 36)  # 10°间隔
        cycle_bins = np.unique(data['cycle_num'])[:20]  # 限制周期数，避免图表过密

        # 创建网格
        X, Y = np.meshgrid(phase_bins[:-1], cycle_bins)
        Z = np.zeros_like(X)

        # 填充Z值（平均幅值）
        for i, p in enumerate(phase_bins[:-1]):
            for j, c in enumerate(cycle_bins):
                mask = (data['phase'] >= p) & (data['phase'] < p + 10) & (data['cycle_num'] == c)
                if np.any(mask):
                    Z[j, i] = np.mean(data['pks'][mask])

        # 绘制三维柱状图（使用viridis colormap）
        ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z.ravel()), 10, 1, Z.ravel(),
                 color=plt.cm.viridis(Z.ravel() / np.max(Z)), alpha=0.8)

        # 设置坐标轴标签和标题
        ax.set_xlabel('Phase (°)', fontsize=12)
        ax.set_ylabel('Cycle', fontsize=12)
        ax.set_zlabel('Amplitude (mV)', fontsize=12)
        ax.set_title('PRPS图 (Phase Resolved Pulse Sequence)', fontsize=14, fontweight='bold')

        # 设置学术级样式
        ax.view_init(elev=20, azim=45)  # 调整视角
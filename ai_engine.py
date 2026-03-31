# ai_engine.py
# AI引擎模块：负责加载PyTorch ResNet18模型，执行推理，返回诊断结果
# 支持Top-2概率输出，类别映射为局放类型

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

class AIEngine:
    """
    AI引擎类：封装模型加载、数据预处理和推理功能。
    """

    def __init__(self, model_path='checkpoints/best_model.pth'):
        """
        初始化AI引擎：加载ResNet18模型。

        参数:
        model_path (str): 模型文件路径。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 【修改点 1】：把 class_mapping 移到 load_model 前面
        # 类别映射（中英文）
        self.class_mapping = {
            0: '气隙放电 (Void)',
            1: '沿面放电 (Surface)',
            2: '电晕放电 (Corona)',
            3: '悬浮放电 (Floating)',
            #4: '颗粒放电 (Particle)'  # 预留接口
        }

        # 现在可以安全地调用 load_model 了
        self.model = self.load_model(model_path)

        # 数据预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """
        加载PyTorch模型。

        参数:
        model_path (str): 模型文件路径。

        返回:
        torch.nn.Module: 加载的模型。
        """
        model = models.resnet18(pretrained=False)
        num_classes = len(self.class_mapping)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # 调整输出层

        # 加载包含元数据的检查点字典
        checkpoint = torch.load(model_path, map_location=self.device)
        # 仅提取其中的模型权重进行加载
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()  # 设置为推理模式

        return model

    def preprocess_data(self, data):
        """
        同步训练逻辑：使用直方图能量聚合和匹配的颜色映射生成图像。
        """
        num_bins = 224

        # 1. 使用与 preprocess.py 一致的直方图统计（能量聚合）
        # x轴为相位(0-360)，y轴为周期
        phase = np.array(data['phase'])
        cycle_num = np.array(data['cycle_num'])
        pks = np.abs(np.array(data['pks']))

        H, _, _ = np.histogram2d(
            phase,
            cycle_num,
            bins=[num_bins, num_bins],
            range=[[0, 360], [np.min(cycle_num), np.max(cycle_num)]],
            weights=pks
        )

        # 2. 归一化处理
        if np.max(H) > 0:
            H = H / np.max(H)

        # 3. 颜色映射：必须使用 'jet' 以匹配训练集
        import matplotlib.cm as cm
        # 注意：训练时用了 hist.T 和 origin='lower'
        # 在 PIL/numpy 图像数组中，这对应于：
        # hist.T 本身就是 (cycle, phase)，符合 (rows, cols)
        # origin='lower' 意味着需要上下翻转矩阵（因为图像坐标系 0,0 在左上角）
        img_matrix = np.flipud(H.T)

        rgba_img = cm.jet(img_matrix)  # 这里的 cm.jet 必须与训练一致

        # 4. 转换为 0-255 RGB
        img_array = (rgba_img[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        return img

    def diagnose(self, data):
        """
        执行AI诊断：预处理数据，推理模型，返回Top-2结果。

        参数:
        data (dict): 输入数据字典。

        返回:
        dict: 诊断结果，包含type、confidence、top2。
        """
        # 预处理数据
        img = self.preprocess_data(data)
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # 获取Top-2
        top2_indices = np.argsort(probabilities)[-2:][::-1]
        top2 = [(self.class_mapping[idx], probabilities[idx]) for idx in top2_indices]

        # 主诊断结果
        best_idx = top2_indices[0]
        result = {
            'type': self.class_mapping[best_idx],
            'confidence': probabilities[best_idx],
            'top2': top2
        }

        return result
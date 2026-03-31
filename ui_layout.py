# ui_layout.py
# UI布局模块：定义主窗口、左侧控制面板、右侧可视化区
# 采用现代化布局：左侧数据加载和AI诊断，右侧Tab切换的图表可视化

import sys
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTabWidget, QSplitter, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from data_processor import DataProcessor
from ai_engine import AIEngine

class MainWindow(QMainWindow):
    """
    主窗口类：继承QMainWindow，实现整个应用的GUI界面。
    布局：左侧控制面板（数据加载 + AI诊断），右侧可视化区（PRPD/PRPS图表）。
    """

    def __init__(self):
        super().__init__()
        self.data_processor = DataProcessor()  # 数据处理实例
        self.ai_engine = AIEngine()  # AI引擎实例
        self.current_data = None  # 当前加载的数据

        self.init_ui()  # 初始化UI

    def init_ui(self):
        """
        初始化UI布局和组件。
        """
        self.setWindowTitle("GIS局部放电离线仿真与AI诊断系统")
        self.setGeometry(100, 100, 1200, 800)  # 窗口大小

        # 设置现代化样式（学术级配色：深蓝背景，白色文字）
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
                color: #ECEFF4;
            }
            QPushButton {
                background-color: #5E81AC;
                color: #ECEFF4;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            QLabel {
                font-size: 14px;
                color: #ECEFF4;
            }
            QTabWidget::pane {
                border: 1px solid #4C566A;
                background-color: #3B4252;
            }
            QTabBar::tab {
                background-color: #434C5E;
                color: #ECEFF4;
                padding: 10px;
                border: 1px solid #4C566A;
            }
            QTabBar::tab:selected {
                background-color: #5E81AC;
            }
        """)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局：水平分割器（左侧控制面板 + 右侧可视化区）
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧控制面板
        self.left_panel = self.create_left_panel()
        splitter.addWidget(self.left_panel)

        # 右侧可视化区
        self.right_panel = self.create_right_panel()
        splitter.addWidget(self.right_panel)

        # 设置分割器比例（左侧1/4，右侧3/4）
        splitter.setSizes([300, 900])

    def create_left_panel(self):
        """
        创建左侧控制面板：数据加载区和AI诊断区。
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 数据加载区
        load_label = QLabel("数据加载区")
        load_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(load_label)

        self.load_button = QPushButton("加载本地数据文件 (.mat 或 .csv)")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.data_status_label = QLabel("未加载数据")
        layout.addWidget(self.data_status_label)

        # 分隔线
        layout.addStretch()

        # AI诊断区
        diag_label = QLabel("AI诊断区")
        diag_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(diag_label)

        self.diagnose_button = QPushButton("一键AI诊断")
        self.diagnose_button.clicked.connect(self.run_diagnosis)
        self.diagnose_button.setEnabled(False)  # 初始禁用，直到数据加载
        layout.addWidget(self.diagnose_button)

        self.diag_result_label = QLabel("诊断结果：未诊断")
        self.diag_result_label.setWordWrap(True)
        layout.addWidget(self.diag_result_label)

        return panel

    def create_right_panel(self):
        """
        创建右侧可视化区：Tab切换的PRPD和PRPS图表。
        """
        # 延迟导入 matplotlib 以避免在模块导入时触发与 PySide6 的冲突
        import matplotlib
        matplotlib.use('QtAgg')
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab控件
        self.tab_widget = QTabWidget()

        # PRPD图Tab
        self.prpd_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.tab_widget.addTab(self.prpd_canvas, "PRPD图 (相位-幅值散点)")

        # PRPS图Tab
        self.prps_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.tab_widget.addTab(self.prps_canvas, "PRPS图 (三维PRPS)")

        layout.addWidget(self.tab_widget)

        return panel

    def load_data(self):
        """
        加载本地数据文件：支持.mat和.csv，更新UI和数据处理器。
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "数据文件 (*.mat *.csv)"
        )
        if not file_path:
            return

        try:
            # 使用数据处理器加载数据
            self.current_data = self.data_processor.load_data(file_path)
            self.data_status_label.setText(f"已加载数据：{os.path.basename(file_path)}")

            # 启用诊断按钮
            self.diagnose_button.setEnabled(True)

            # 更新图表
            self.update_plots()

            QMessageBox.information(self, "成功", "数据加载成功！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据加载失败：{str(e)}")

    def update_plots(self):
        """
        更新PRPD和PRPS图表：调用数据处理器绘制图表。
        """
        if self.current_data is None:
            return

        # 绘制PRPD图
        self.data_processor.plot_prpd(self.prpd_canvas.figure, self.current_data)

        # 绘制PRPS图
        self.data_processor.plot_prps(self.prps_canvas.figure, self.current_data)

        # 刷新画布
        self.prpd_canvas.draw()
        self.prps_canvas.draw()

    def run_diagnosis(self):
        """
        运行AI诊断：使用AI引擎推理，显示结果。
        """
        if self.current_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据！")
            return

        try:
            # 运行AI推理
            result = self.ai_engine.diagnose(self.current_data)

            # 显示结果
            self.diag_result_label.setText(
                f"诊断结果：{result['type']} (置信度: {result['confidence']:.2f})\n"
                f"Top-2: {result['top2'][0][0]} ({result['top2'][0][1]:.2f}), "
                f"{result['top2'][1][0]} ({result['top2'][1][1]:.2f})"
            )

            QMessageBox.information(self, "诊断完成", "AI诊断已完成！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"AI诊断失败：{str(e)}")
# main.py
# 主程序入口文件
# 负责启动PyQt应用，协调各模块之间的交互

import sys
from PySide6.QtWidgets import QApplication

def main():
    """
    主函数：创建PyQt应用实例，初始化主窗口，并启动事件循环。
    """
    # 创建PyQt应用实例
    app = QApplication(sys.argv)

    # 设置应用属性（可选，用于现代化外观）
    app.setApplicationName("GIS局部放电离线仿真与AI诊断系统")
    app.setApplicationVersion("1.0.0")

    # 创建主窗口实例
    from ui_layout import MainWindow
    window = MainWindow()

    # 显示主窗口
    window.show()

    # 启动事件循环，等待用户操作
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

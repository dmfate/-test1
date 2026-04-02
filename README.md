# GIS 局部放电诊断项目

## 真·一键启动（无需手动进虚拟环境）

> 直接运行下面命令即可：脚本会自动创建 `.venv`、自动安装依赖、自动启动。

### 仅启动 GUI（默认）
```bash
python one_click.py
```

### 全流程一键启动（自动补齐预处理 / 训练 / 评估，再打开 GUI）
```bash
python one_click.py --mode full
```

可选参数：
- `--epochs 30`：训练轮数（仅 full 模式且需要训练时生效）
- `--batch-size 32`：训练 batch 大小
- `--use-cpu`：训练与评估强制 CPU
- `--no-gui`：full 模式执行到评估后不打开 GUI
- `--dry-run`：只打印执行步骤，不真正运行

#!/bin/bash
#
# COSMOS + RMPflow + MAPPO 本地训练脚本
#
# 用法:
#   ./formation_nav/train.sh                    # 默认参数
#   ./formation_nav/train.sh --episodes 500     # 自定义参数
#   ./formation_nav/train.sh --use-wandb        # 启用 WandB
#   ./formation_nav/train.sh --resume path/to/checkpoint.pt  # 断点续训
#
# 常用参数:
#   --episodes      训练轮数 (默认 200)
#   --num-agents    智能体数量 (默认 4)
#   --formation     编队形状 (默认 square)
#   --exp-name      实验名称 (默认 cosmos_exp)
#   --use-wandb     启用 WandB 日志
#   --resume        从检查点恢复训练
#   --device        设备 cpu/cuda/auto (默认 auto)
#   --save-every    检查点保存间隔 (默认 20)
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "============================================================"
echo "COSMOS Formation Navigation - Local Training"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# 检查 Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found!"
    exit 1
fi

echo "Python: $($PYTHON --version)"

# 检查依赖
echo ""
echo "检查依赖..."
$PYTHON -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "  PyTorch: 未安装"
    echo ""
    echo "请先安装依赖: pip install torch numpy gymnasium matplotlib scipy"
    exit 1
}
$PYTHON -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
$PYTHON -c "import gymnasium; print(f'  Gymnasium: {gymnasium.__version__}')"

# 检查 CUDA
$PYTHON -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"

# 检查 WandB (可选)
$PYTHON -c "import wandb; print(f'  WandB: {wandb.__version__}')" 2>/dev/null || {
    echo "  WandB: 未安装 (可选，使用 pip install wandb 安装)"
}

echo ""
echo "依赖检查通过!"
echo ""

# 运行训练
cd "$PROJECT_ROOT"
echo "开始训练..."
echo "============================================================"
PYTHONPATH=. $PYTHON formation_nav/train.py "$@"

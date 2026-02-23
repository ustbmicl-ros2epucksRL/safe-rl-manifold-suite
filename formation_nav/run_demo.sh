#!/bin/bash
#
# COSMOS + RMPflow + MAPPO 编队导航演示启动脚本
#
# 用法:
#   ./formation_nav/run_demo.sh              # 默认参数
#   ./formation_nav/run_demo.sh --episodes 500  # 自定义参数
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "============================================================"
echo "COSMOS Formation Navigation Demo"
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
    echo "请先安装依赖: pip install torch numpy gymnasium matplotlib"
    exit 1
}
$PYTHON -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
$PYTHON -c "import gymnasium; print(f'  Gymnasium: {gymnasium.__version__}')"
$PYTHON -c "import matplotlib; print(f'  Matplotlib: {matplotlib.__version__}')"

echo ""
echo "依赖检查通过!"
echo ""

# 运行演示
cd "$PROJECT_ROOT"
PYTHONPATH=. $PYTHON formation_nav/demo.py "$@"

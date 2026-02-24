#!/bin/bash
# =============================================================================
# COSMOS Framework - Environment Setup Script
# =============================================================================
# Creates isolated conda environment and installs all dependencies
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "  COSMOS - Safe Multi-Agent RL Framework Setup"
echo "============================================================"
echo -e "${NC}"

# Configuration
ENV_NAME="cosmos"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Initialize conda for script
eval "$(conda shell.bash hook)"

# =============================================================================
# Step 1: Create Conda Environment
# =============================================================================
echo -e "${YELLOW}[1/5] Creating conda environment '${ENV_NAME}'...${NC}"

# Remove existing environment if exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' exists. Removing..."
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y
fi

# Create new environment
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
conda activate ${ENV_NAME}

echo -e "${GREEN}✓ Environment created and activated${NC}"

# =============================================================================
# Step 2: Install PyTorch
# =============================================================================
echo -e "${YELLOW}[2/5] Installing PyTorch...${NC}"

# Detect platform and install appropriate PyTorch
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        pip install torch torchvision torchaudio
    else
        # Intel Mac
        pip install torch torchvision torchaudio
    fi
else
    # Linux - try CUDA first, fallback to CPU
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected, installing GPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "No CUDA detected, installing CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"

# =============================================================================
# Step 3: Install Core Dependencies
# =============================================================================
echo -e "${YELLOW}[3/5] Installing core dependencies...${NC}"

pip install --upgrade pip

# Core packages
pip install numpy scipy matplotlib
pip install gymnasium
pip install hydra-core omegaconf
pip install wandb
pip install tqdm
pip install tensorboard

echo -e "${GREEN}✓ Core dependencies installed${NC}"

# =============================================================================
# Step 4: Install Environment Dependencies
# =============================================================================
echo -e "${YELLOW}[4/5] Installing environment dependencies...${NC}"

# Safety-Gymnasium (Safe RL benchmark)
echo "Installing Safety-Gymnasium..."
pip install safety-gymnasium

# VMAS (Vectorized Multi-Agent Simulator)
echo "Installing VMAS..."
pip install vmas

# MuJoCo (optional, for high-fidelity physics)
echo "Installing MuJoCo..."
pip install mujoco

# Multi-Agent MuJoCo (optional)
echo "Installing Multi-Agent MuJoCo..."
pip install multiagent-mujoco || echo "Warning: multiagent-mujoco installation failed (optional)"

echo -e "${GREEN}✓ Environment dependencies installed${NC}"

# =============================================================================
# Step 5: Install COSMOS Package
# =============================================================================
echo -e "${YELLOW}[5/5] Installing COSMOS package...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install in development mode
cd ${SCRIPT_DIR}
pip install -e .

echo -e "${GREEN}✓ COSMOS package installed${NC}"

# =============================================================================
# Verification
# =============================================================================
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"

python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import gymnasium
print(f'Gymnasium: {gymnasium.__version__}')

# Test COSMOS
import cosmos
from cosmos.registry import ENV_REGISTRY, ALGO_REGISTRY, SAFETY_REGISTRY
print(f'COSMOS environments: {ENV_REGISTRY.list()}')
print(f'COSMOS algorithms: {ALGO_REGISTRY.list()}')
print(f'COSMOS safety filters: {SAFETY_REGISTRY.list()}')

# Test environments
print()
print('Testing environments...')

# formation_nav
from cosmos.envs.formation_nav import FormationNavEnv
print('  ✓ formation_nav')

# epuck_sim
from cosmos.envs.webots_wrapper import EpuckSimEnv
print('  ✓ epuck_sim')

# safety_gym
try:
    import safety_gymnasium
    print('  ✓ safety_gym')
except ImportError:
    print('  ✗ safety_gym (not installed)')

# vmas
try:
    import vmas
    print('  ✓ vmas')
except ImportError:
    print('  ✗ vmas (not installed)')

# mujoco
try:
    import mujoco
    print('  ✓ mujoco')
except ImportError:
    print('  ✗ mujoco (not installed)')

print()
print('All core components verified!')
"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "To activate the environment:"
echo -e "  ${BLUE}conda activate ${ENV_NAME}${NC}"
echo ""
echo "To run experiments:"
echo -e "  ${BLUE}./run_experiments.sh${NC}"
echo ""
echo "To run a single training:"
echo -e "  ${BLUE}python -m cosmos.train env=formation_nav algo=mappo safety=cosmos${NC}"
echo ""

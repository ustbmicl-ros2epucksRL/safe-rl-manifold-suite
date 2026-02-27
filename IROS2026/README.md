# IROS 2026: Safe RL via Constraint Manifold Projection

This directory contains the standalone code implementation for the IROS 2026 paper:

**"Safe Reinforcement Learning via Constraint Manifold Projection with Learned State Estimation"**

## Quick Start

```bash
# 1. Setup environment (creates venv, installs dependencies)
./setup.sh

# 2. Run quick test
./run_experiments.sh quick

# 3. Run full experiments
./run_experiments.sh all --env goal --seeds 5
```

## Directory Structure

```
IROS2026/
├── src/                            # Source code
│   ├── __init__.py                 # Package exports
│   ├── env.py                      # Safety-Gymnasium wrapper + MockEnv
│   ├── ppo.py                      # PPO algorithm
│   ├── ekf.py                      # Data-driven EKF (Section III-C)
│   ├── train.py                    # Training script
│   │
│   └── safety/                     # Safety components
│       ├── __init__.py
│       ├── manifold_filter.py      # Constraint manifold filter (Section III-A)
│       ├── distance_filter.py      # Distance-based safety filter
│       └── reachability.py         # HJ reachability analysis (Section III-B)
│
├── scripts/                        # Experiment scripts
│   ├── run_full_experiments.py     # Table III & V experiments
│   ├── run_table4_noise.py         # Table IV noise experiments
│   ├── run_table1.py               # Table I baseline comparison
│   ├── run_ablation.py             # Simplified ablation
│   └── run_ekf_compare.py          # EKF comparison
│
├── results/                        # Experiment results (JSON)
├── figures/                        # Paper figures
│
├── setup.sh                        # Environment setup script
├── run_experiments.sh              # Main experiment runner
├── requirements.txt                # Python dependencies
│
├── main.tex                        # Paper LaTeX source
├── main.pdf                        # Compiled paper
└── README.md                       # This file
```

## Environment Setup

### Option 1: Automatic Setup (Recommended)

```bash
./setup.sh
```

This creates a Python virtual environment and installs all dependencies.

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Safety-Gymnasium
pip install safety-gymnasium
```

## Running Experiments

### Experiment Commands

```bash
# Activate environment first
source venv/bin/activate

# Or use the runner script (auto-activates venv)
./run_experiments.sh [COMMAND] [OPTIONS]

Commands:
  all           Run all experiments (Tables III, IV, V)
  ablation      Run Table III ablation study
  noise         Run Table IV noise experiments
  ekf           Run Table V EKF comparison
  quick         Quick test run (reduced steps/seeds)

Options:
  --env ENV     Environment: goal, circle, push (default: goal)
  --seeds N     Number of random seeds (default: 5)
  --steps N     Training steps (default: 50000)
  --episodes N  Evaluation episodes (default: 50)
```

### Examples

```bash
# Quick test (5k steps, 2 seeds)
./run_experiments.sh quick

# Full ablation study
./run_experiments.sh ablation --env goal --seeds 5

# All experiments on Circle task
./run_experiments.sh all --env circle

# Custom parameters
./run_experiments.sh noise --steps 100000 --episodes 100 --seeds 10
```

### Direct Python Usage

```bash
source venv/bin/activate

# Table III: Ablation study
python3 scripts/run_full_experiments.py --experiment ablation --env goal --seeds 5

# Table IV: Noise experiments
python3 scripts/run_table4_noise.py --env goal --seeds 5

# Table V: EKF comparison
python3 scripts/run_full_experiments.py --experiment ekf_compare --env goal --seeds 5

# Training
python3 -m src.train --env goal --episodes 500 --use-safety --use-calibration
```

## Core Components

### 1. Constraint Manifold Filter (Section III-A)
`src/safety/manifold_filter.py`

Projects RL actions onto the tangent space of the constraint manifold using null-space projection.

Key equations:
- Slack variable: `c̄(s, μ) = c(s) + φ(μ) = 0`
- Safe action: `a_safe = N_c @ α - K_c @ J_c⁺ @ c̄ - J_c⁺ @ J_s @ f(s)`
- Null-space projector: `N_c = I - J_c⁺ @ J_c`

### 2. HJ Reachability Analysis (Section III-B)
`src/safety/reachability.py`

Offline computation of environment-aware safe regions using neural network value function approximation.

### 3. Data-Driven EKF (Section III-C)
`src/ekf.py`

Extended Kalman Filter with CNN-learned noise parameters for robust state estimation under sensor noise.

### 4. Distance-Based Safety Filter (Practical)
`src/safety/distance_filter.py`

A simpler safety filter that scales velocity based on distance to obstacles. More robust for Point Robot.

## Paper Tables

### Table I: Main Results
Comparison against baselines (PPO, IPO, PPO-Lag, Recovery-RL).

### Table III: Ablation Study
| Configuration | Reward | Cost |
|--------------|--------|------|
| PPO (baseline) | -6.87 ± 1.23 | 7.77 ± 1.45 |
| + Manifold Filter | 1.23 ± 0.34 | 1.91 ± 0.52 |
| + Reachability Pretraining | 1.45 ± 0.28 | 0.12 ± 0.08 |
| + Data-driven EKF (Full) | **1.68 ± 0.21** | **0.00 ± 0.00** |

### Table IV: Sensor Noise
Impact of Gaussian noise (σ=0.1m) on state estimation.

### Table V: EKF Comparison
| Method | Pos. Error (m) | Reward | Cost |
|--------|---------------|--------|------|
| No filtering | 0.23 ± 0.08 | 1.29 | 11.33 |
| Standard EKF | 0.12 ± 0.04 | 1.61 | 0.45 |
| Data-driven EKF | **0.05 ± 0.02** | **1.93** | **0.00** |

## Results Output

Experiment results are saved to `results/` as JSON files:

```
results/
├── table3_ablation_goal_20260227_120000.json
├── table4_noise_goal_20260227_121500.json
└── table5_ekf_goal_20260227_123000.json
```

Each file contains:
- Experiment configuration
- Per-configuration results (mean ± std)
- Timestamp

## Citation

```bibtex
@inproceedings{author2026saferl,
  title={Safe Reinforcement Learning via Constraint Manifold Projection
         with Learned State Estimation},
  author={Anonymous},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots
             and Systems (IROS)},
  year={2026}
}
```

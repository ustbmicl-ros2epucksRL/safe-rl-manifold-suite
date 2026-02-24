# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**COSMOS** (COordinated Safety On Manifold for multi-agent Systems) - Safe Multi-Agent Reinforcement Learning framework with Riemannian Manifold Constraints (USTB MICL lab master's thesis).

## Repository Layout

```
safe-rl-manifold-suite/
│
├── cosmos/                         # All code consolidated here
│   ├── train.py                    # Training entry: python -m cosmos.train
│   ├── trainer.py                  # Unified trainer
│   ├── registry.py                 # Component registration
│   │
│   ├── configs/                    # Hydra configuration files
│   ├── envs/                       # Environment layer
│   ├── algos/                      # Algorithm layer (MAPPO, QMIX, MADDPG)
│   ├── safety/                     # Safety layer (CBF, ATACOM, RMPflow)
│   ├── buffers/                    # Experience buffers
│   ├── runners/                    # Training runners
│   ├── utils/                      # Utilities
│   │
│   ├── apps/formation_nav/         # Formation navigation application
│   ├── tests/                      # Test suite
│   ├── examples/                   # Jupyter notebooks
│   ├── scripts/                    # Analysis scripts
│   ├── docs/                       # Documentation
│   └── ros2/                       # ROS2 E-puck deployment
│
├── artifacts/                      # Generated data (gitignored)
│   ├── checkpoints/                # Model checkpoints
│   ├── demo_output/                # Demo outputs
│   ├── outputs/                    # Hydra outputs
│   └── results/                    # Experiment results
│
├── algorithms/                     # Git submodules (external reference)
├── envs/                           # Git submodules (external reference)
├── refs/                           # Reference papers (PDF) & reading notes
├── paper/                          # Thesis materials
│
├── README.md                       # Project readme
└── CLAUDE.md                       # This file
```

## Core Components

### Environment Layer (`cosmos/envs/`)
- `formation_nav.py` - Multi-robot formation navigation
- `webots_wrapper.py` - E-puck Webots simulation
- `safety_gym_wrapper.py` - Safety-Gymnasium integration
- `mujoco_wrapper.py` - MuJoCo environments
- `vmas_wrapper.py` - VMAS vectorized environments

### Algorithm Layer (`cosmos/algos/`)
- `mappo.py` - Multi-Agent PPO with CTDE
- `qmix.py` - Value decomposition with mixing network
- `maddpg.py` - Multi-Agent DDPG

### Safety Layer (`cosmos/safety/`)
- `cosmos_filter.py` - CBF-based safety filter
- `atacom.py` - ATACOM manifold projection + COSMOS
- `rmp_tree.py` - RMPflow tree structure
- `rmp_policies.py` - RMP behavior policies
- `constraints.py` - Constraint definitions

## Quick Commands

```bash
# Training
python -m cosmos.train env=formation_nav algo=mappo safety=cbf

# Testing
python -m cosmos.tests.test_all_envs

# Demo
python -m cosmos.apps.formation_nav.demo

# Benchmark
python -m cosmos.apps.formation_nav.benchmark
```

## Git Submodules

External reference repositories (not used directly by cosmos/):

| Path | Repository | Purpose |
|------|------------|---------|
| `algorithms/multi-robot-rmpflow` | chengzizhuo/multi-robot-rmpflow | RMPflow reference (ported to cosmos/safety/) |
| `algorithms/safe-po` | chengzizhuo/safe-po | Safe policy optimization reference |
| `envs/safety-gymnasium` | chengzizhuo/safe-gym | Safety-Gym source reference |

```bash
# Initialize submodules (optional, for reference only)
git submodule update --init --recursive
```

## Network & Git Configuration

GitHub operations require SOCKS5 proxy (Shadowrocket):
- Proxy: `127.0.0.1:1082`
- SSH hosts use `ProxyCommand nc -X 5 -x 127.0.0.1:1082 %h %p`
- This repo uses the `github-miclsirr` SSH account

## Documentation

See `cosmos/docs/` for detailed documentation:
- `ARCHITECTURE.md` - System architecture
- `DIRECTORIES.md` - Directory structure
- `INSTALL_ENVS.md` - Environment installation
- `ROS2_WEBOTS_SETUP.md` - ROS2 + Webots setup

## Technical Background

### Safety Mechanism: ATACOM + CBF

ATACOM (Augmented Task-space Constrained by Optimal Manifold) projects RL actions into constraint null-space:

```
dq = Nc @ alpha + (-K_c @ Jc_pinv @ c(q))
     ↑ null-space   ↑ constraint correction
```

The RL agent acts only within the null space of constraint Jacobians - no matter what the policy outputs, safety constraints are never violated.

### RMPflow Integration

RMPflow tree provides geometric guidance:
- `RMPRoot` → `RMPNode` → `RMPLeaf`
- Leaf policies: GoalAttractor, CollisionAvoidance, FormationControl, Damper
- Pullback aggregation: `a = M⁺f`

### COSMOS = ATACOM + RMPflow + MARL

The thesis contribution combines:
1. **ATACOM** - Hard safety via null-space projection
2. **RMPflow** - Geometric guidance for formation
3. **MAPPO** - Multi-agent reinforcement learning

Result: Learned policies with provable safety guarantees.

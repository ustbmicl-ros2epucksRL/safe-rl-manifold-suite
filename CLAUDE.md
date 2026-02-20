# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Meta-project orchestrating four git submodules for research on **Safe Multi-Agent Reinforcement Learning with Riemannian Manifold Constraints** (USTB MICL lab master's thesis). This repo itself contains no source code — all work happens inside the submodules.

## Submodule Layout

```
algorithms/
  safe-po/                         # Primary codebase — SafePO fork with constrained manifold modules
  multi-robot-rmpflow/             # Reference impl: RMPflow for multi-robot coordination
envs/
  safety-gymnasium/                # Safety Gymnasium environments (forked)
paper/
  safe-rl-manifold-constraints/    # Thesis paper source
```

- **`algorithms/safe-po`** is where most development happens. It corresponds to the `safeRL_manifold` project (a fork of PKU-Alignment's Safe-Policy-Optimization).
- **`algorithms/multi-robot-rmpflow`** is a reference implementation from Georgia Tech (gtrll/multi-robot-rmpflow). Provides the RMPflow framework that informs the constrained manifold approach in safe-po.

## Submodule Commands

```bash
# Initialize all submodules (required after first clone)
git submodule update --init --recursive

# Pull latest in all submodules
git submodule update --remote --merge

# Work inside a submodule (commits are independent)
cd algorithms/safe-po && git checkout main
# ... make changes, commit, push ...
cd ../.. && git add algorithms/safe-po && git commit -m "update safe-po ref"
```

**Known issue:** The `chengzizhuo/multi-robot-rmpflow` remote may not exist. Workaround: clone upstream directly into the submodule path:
```bash
rm -rf algorithms/multi-robot-rmpflow
git clone https://github.com/gtrll/multi-robot-rmpflow.git algorithms/multi-robot-rmpflow
```

## Network & Git Configuration

All GitHub operations require SOCKS5 proxy (Shadowrocket must be running):
- Proxy: `127.0.0.1:1082`
- SSH hosts `github-miclsirr` and `github-duansh` use `ProxyCommand nc -X 5 -x 127.0.0.1:1082 %h %p`
- Both connect via `ssh.github.com:443` (not port 22)
- This repo uses the `github-miclsirr` SSH account (GitHub user: `ustbmicl-sirr`)

## safe-po Development (Primary Submodule)

```bash
conda create -n safepo python=3.8 && conda activate safepo
cd algorithms/safe-po && pip install -e .

# Single-agent training
python safepo/single_agent/ppo_lag.py --task SafetyPointGoal1-v0 --seed 0

# Multi-agent training
python safepo/multi_agent/macpo.py --task Safety2x4AntVelocity-v0 --experiment benchmark

# Testing & benchmarks
make pytest              # pytest with coverage
make benchmark           # full benchmark
make simple-benchmark    # quick benchmark
make test-benchmark      # small-scale test
```

### safe-po Architecture

Each algorithm is a standalone script (e.g., `ppo_lag.py`) importing from `safepo/common/`. The novel research contribution is `safepo/common/constrained_manifold/` — an environment wrapper enforcing Riemannian manifold-based safety constraints (ATACOM approach). `single_cm.py` and `mult_cm.py` integrate this with single-agent and multi-agent training loops.

Key dependencies: PyTorch >= 1.10.0, safety-gymnasium, MuJoCo, wandb, tensorboard.

## multi-robot-rmpflow (Reference Submodule)

Pure NumPy/SciPy implementation of RMPflow for multi-robot systems. No ML dependencies.

```bash
cd algorithms/multi-robot-rmpflow

python3 rmp_example.py                    # Single-agent 2D goal+obstacle avoidance
python3 multi_agent_rmp.py                # 10 robots, decentralized collision avoidance
python3 multi_agent_rmp_centralized.py    # 10 robots, centralized collision avoidance
# Robotarium examples (require robotarium_python_simulator):
python3 formation_preservation.py         # 9 robots, two subteams with formation control
python3 cyclic_pursuit_formation.py       # 8 robots, cyclic pursuit + formation
```

### RMPflow Architecture

Builds a tree of RMP nodes. Data flows in two directions:

1. **Pushforward** (root→leaves): propagate configuration-space state `(x, x_dot)` through mappings `ψ` and Jacobians `J` into each task space
2. **Pullback** (leaves→root): each leaf computes force `f` and metric `M` via its RMP function, then pulls back through `J^T` to aggregate at root
3. **Resolve**: root computes acceleration `a = M⁺f` (pseudoinverse)

Core classes in `rmp.py`: `RMPRoot` → `RMPNode` → `RMPLeaf`. Seven leaf policies in `rmp_leaf.py`: `CollisionAvoidance`, `CollisionAvoidanceDecentralized`, `CollisionAvoidanceCentralized`, `GoalAttractorUni`, `FormationDecentralized`, `FormationCentralized`, `Damper`.

### Relationship to safe-po

The RMPflow pushforward/pullback on Riemannian metrics is the geometric foundation that safe-po's `constrained_manifold/` module builds upon. RMPflow uses hand-designed policy leaves; the thesis contribution replaces these with RL-trained policies under ATACOM manifold constraints.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Meta-project orchestrating git submodules + standalone modules for research on **Safe Multi-Agent Reinforcement Learning with Riemannian Manifold Constraints** (USTB MICL lab master's thesis).

## Repository Layout

```
algorithms/
  safe-po/                         # Primary codebase — SafePO fork with constrained manifold modules
  multi-robot-rmpflow/             # Reference impl: RMPflow for multi-robot coordination
envs/
  safety-gymnasium/                # Safety Gymnasium environments (forked)
paper/
  safe-rl-manifold-constraints/    # Thesis paper source
formation_nav/                     # Standalone module: multi-robot formation navigation (ATACOM + RMPflow + MAPPO)
refs/                              # Reference papers (Git LFS) & reading notes
```

- **`algorithms/safe-po`** is where most development happens. It corresponds to the `safeRL_manifold` project (a fork of PKU-Alignment's Safe-Policy-Optimization).
- **`algorithms/multi-robot-rmpflow`** is a reference implementation from Georgia Tech (gtrll/multi-robot-rmpflow). Provides the RMPflow framework that informs the constrained manifold approach in safe-po.
- **`formation_nav/`** is a self-contained module that integrates ATACOM safety filtering, RMPflow formation guidance, and MAPPO training for multi-robot formation navigation. See `formation_nav/README.md` for problem definition, method, and usage.
- **`refs/`** contains reference papers (PDF, managed by Git LFS) and reading notes. See `refs/README.md` for a categorized index.

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

## Technical Analysis: Multi-Robot Formation Navigation — Learning & Safety

The codebase implements two complementary approaches: a geometric control baseline (RMPflow, no learning) and a safe RL method (ATACOM + RL, the core thesis contribution).

### RMPflow — Geometric Control Baseline

`algorithms/multi-robot-rmpflow/`

RMPflow decomposes multi-robot navigation into a **Riemannian Motion Policy tree**:

```
RMPRoot (global state: all robot positions + velocities)
  +-- Robot_i (RMPNode: extracts robot i's coordinates)
  |   +-- GoalAttractorUni          <- goal seeking
  |   +-- CollisionAvoidance(i,j)   <- pairwise collision avoidance
  |   +-- FormationDecentralized    <- maintain formation distance to neighbors
  |   +-- Damper                    <- velocity damping
  +-- ...
```

**Collision avoidance safety** (`rmp_leaf.py`): uses a power-law repulsive metric that grows as `1/distance^4`. As distance `x -> 0`, the metric tensor `M -> inf`, making the system infinitely stiff in the collision direction — safety is enforced geometrically without hard constraint solvers.

**Formation control**: spring-damper model maintaining desired inter-robot distance `d_ij`. Only leader robots have goal attractors; followers are "dragged" through formation constraints. See `formation_preservation.py` (9 robots, 2 subteams) and `cyclic_pursuit_formation.py` (8 robots, mixed behaviors).

### ATACOM — Riemannian Manifold Safe RL (Thesis Contribution)

`safepo/common/constrained_manifold/` + `safepo/common/mult_cm.py`

ATACOM (Augmented Task-space Constrained by Optimal Manifold) replaces RMPflow's hand-designed policy leaves with **RL-trained policies** while enforcing **hard safety** via constraint null-space projection.

**Core mechanism** (`manifold.py` `step_action_function()`):

```
Given: RL agent outputs raw action alpha in R^(dim_null)

Jc = [J_q | J_s]                                  # constraint Jacobian (w.r.t. state q and slack s)
Nc = (I - Jc_pinv @ Jc)[:dim_null] @ diag(dq_max) # null-space projector

dq = Nc @ alpha                   # safe motion in constraint null space
   + (-Jc_pinv @ J_x @ dx)        # compensation for uncontrolled variables
   + (-K_c @ Jc_pinv @ c(q,x))    # constraint error correction (K_c=100)
```

The RL agent acts only within the **null space of constraint Jacobians** — no matter what the policy outputs, constraints are never violated.

**Slack variables** (`constraints.py`): smooth inequality-to-equality conversion via `softcorner` penalty: `penalty(s) = -log(-expm1(beta*s)) / beta` (beta=30), enabling differentiable constraint relaxation.

### Multi-Agent Safety Integration — MultiNavAtacom

`safepo/common/mult_cm.py`

Decentralized architecture with **per-agent constraint sets**:

```python
per_agent_constraints = [ConstraintsSet(dim_q=2) for i in range(num_agents)]
# Each agent's constraints include:
# 1. Static obstacles (hazards):  c_hazard(q[i]) <= 0
# 2. Dynamic collision avoidance: c_agent(q[i], other_positions) <= 0
```

Per-step execution: update all positions -> each agent independently runs ATACOM projection -> execute all safe actions simultaneously.

### Three Safety Algorithm Variants

| Aspect | MAPPO-CM (manifold) | MAPPO-Lag (Lagrangian) | MACPO (trust region) |
|--------|---------------------|------------------------|----------------------|
| File | `mappo_cm.py` | `mappolag.py` | `macpo.py` |
| Safety enforcement | Action space (pre-execution projection) | Loss function (training-time reweighting) | Policy update (TRPO constraint) |
| Safety guarantee | **Hard constraint** | Soft constraint | Soft constraint |
| Core mechanism | Null-space projection `Nc @ alpha` | Hybrid advantage `A_r - lambda * A_c` | Conjugate gradient + feasibility case analysis |

- **MAPPO-Lag**: modifies PPO advantage to `adv - lambda * cost_adv`, updates lambda via gradient ascent on constraint violation.
- **MACPO**: uses TRPO framework with separate reward/cost natural gradients, selects update direction from 4 feasibility cases (aggressive to conservative).
- **MAPPO-CM**: wraps environment with `MultiNavAtacom`, making all actions inherently constraint-aware without modifying the loss function.

### Geometric Correspondence Between the Two Approaches

```
RMPflow (hand-designed)             ATACOM + RL (learned)
-----------------------             ---------------------
Leaf node = hand-crafted policy     Leaf node = RL policy
Jacobian pullback J^T               Constraint Jacobian projection Jc_pinv
Metric tensor M -> inf              Null-space projection (hard constraint)
Force composition Sum(J^T f)        dq = Nc @ alpha + error correction
```

The thesis contribution: **replace RMPflow's hand-designed policy leaves with RL, retaining Riemannian manifold safety guarantees through ATACOM Jacobian null-space projection** — achieving both learnability and hard safety for multi-agent formation navigation.

## formation_nav — Standalone Formation Navigation Module

`formation_nav/` is a self-contained implementation that combines ATACOM, RMPflow, and MAPPO into a single trainable system. No MuJoCo dependency — uses pure NumPy double-integrator physics for fast iteration.

```bash
pip install torch numpy gymnasium matplotlib tensorboard

# Train (4 agents, square formation)
PYTHONPATH=. python formation_nav/train.py --num-agents 4 --formation square --seed 0 --total-episodes 2000

# Evaluate & visualize
PYTHONPATH=. python formation_nav/eval.py --model-path checkpoints/mappo_formation_final.pt --num-agents 4

# Ablation: RMPflow blend coefficient
PYTHONPATH=. python formation_nav/train.py --rmp-blend 0.0   # pure ATACOM
PYTHONPATH=. python formation_nav/train.py --rmp-blend 0.3   # default blend
```

### formation_nav Architecture

```
formation_nav/
├── config.py              # All hyperparams (dataclass): EnvConfig, SafetyConfig, AlgoConfig, RewardConfig
├── env/
│   ├── formations.py      # FormationShape (polygon/line/V) + FormationTopology (complete/chain/star)
│   └── formation_env.py   # 2D multi-robot Gymnasium env, double-integrator physics
├── safety/
│   ├── rmp_tree.py        # RMPRoot / RMPNode / RMPLeaf (ported from multi-robot-rmpflow)
│   ├── rmp_policies.py    # 5 leaf policies + MultiRobotRMPForest
│   ├── constraints.py     # StateConstraint + ConstraintsSet (softcorner slack variables)
│   └── atacom.py          # AtacomSafetyFilter: per-agent null-space projection + RMPflow blending
├── algo/
│   ├── networks.py        # Actor (Gaussian) / Critic / CostCritic, orthogonal init
│   ├── buffer.py          # RolloutBuffer with GAE
│   └── mappo.py           # MAPPO trainer (shared actor, clipped PPO)
├── train.py               # Training loop with CSV + TensorBoard logging
└── eval.py                # Evaluation with trajectory plots and animation
```

Key design: `dim_null = 2` always holds (each inequality constraint adds 1 slack + 1 output, net zero), so the RL action space is always 2D regardless of agent/obstacle count. Buffer stores raw policy outputs α (not safe actions) to keep log_prob consistent with PPO updates.

## refs — Reference Papers & Reading Notes

`refs/` contains 21 reference papers (PDF, managed by **Git LFS**) and 3 reading note files. See `refs/README.md` for a categorized index.

Key categories:
- **Core methods**: ATACOM (Liu 2024), RMPflow (Cheng 2018/2021), Multi-Robot RMP (Li 2019)
- **Safe MARL**: GCBF+ (Zhang 2024), DGPPO (Zhang 2025), SafeMARLFormation (Dawood 2024)
- **Formation planning**: Safety-Critical Formation Control, FormationPlanner (Cornwall 2025)
- **Reading notes**: ATACOM 学习笔记, 综合阅读笔记, 多智能体编队安全避障方案设计

Git LFS is configured via `.gitattributes` to track `*.pdf`. After cloning, run `git lfs pull` to fetch PDF files.

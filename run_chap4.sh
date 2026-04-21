#!/usr/bin/env bash
# =============================================================================
# run_chap4.sh
#
# Chap 4 GCPL (MAPPO + RMPflow 路径 A) 端到端训练 + 评估流水线.
#
# 本脚本做四件事:
#   1. 建立独立的 conda 环境 (默认 chap4_gcpl, Python 3.10)
#   2. 安装所有依赖 (含本地 submodule 的 safety-gymnasium / safepo, editable)
#   3. 跑 sanity check 确认路径 A 的 RL 叶节点集成可用
#   4. 正式训练 mappo_rmp + 评估 (可分别跳过)
#
# 快速开始:
#   bash run_chap4.sh                          # 默认: small 规模, train+eval 全流程
#   CHAP4_SCALE=medium bash run_chap4.sh       # 20min 级别的训练
#   CHAP4_SCALE=full   bash run_chap4.sh       # 对齐 chap4.5 config.yaml 的正式规模
#   CHAP4_MODE=eval    bash run_chap4.sh       # 只跑评估 (使用最近一次训练的 checkpoint)
#   CHAP4_SKIP_INSTALL=1 bash run_chap4.sh     # 跳过依赖安装 (env 已备好时提速)
#
#   # 一键 4×3 对比实验 + 绘图 (论文对比实验 fig:train_result / tab:line_formation_result 用):
#   CHAP4_MODE=sweep CHAP4_SCALE=medium bash run_chap4.sh
#
#   # 只跑绘图 (runs 目录已有数据):
#   CHAP4_MODE=plot bash run_chap4.sh
#
# 所有可调 env var (大写 = 默认):
#   CHAP4_ENV              chap4_gcpl        conda 环境名
#   CHAP4_PY               3.10              Python 版本
#   CHAP4_SCALE            small             small | medium | full
#   CHAP4_MODE             both              train | eval | both | sweep | ablation_sweep | plot
#   CHAP4_SKIP_INSTALL     0                 1 = 跳过 pip install
#   CHAP4_SKIP_SANITY      0                 1 = 跳过 sanity check
#   CHAP4_SEED             0                 随机种子 (单 run 模式用)
#   CHAP4_NUM_AGENTS       3                 编队机器人数
#   CHAP4_SHAPE            wedge             mesh | line | wedge | circle
#   CHAP4_DEVICE           cpu               cpu | cuda
#   CHAP4_EVAL_EPISODES    20                评估 episode 数
#   CHAP4_FUSION_MODE      leaf              leaf (路径 A) | additive (回退)
#   # -- sweep 模式专用 --
#   CHAP4_ALGOS            "mappo_rmp mappo mappolag rmpflow"   扫的算法 (空格分隔)
#   CHAP4_SEEDS            "0 1 2"           扫的种子
#   CHAP4_PLOT_OUT         images/chap4_sweep  图输出目录 (相对 paper-safeMARL/)
#   # -- ablation_sweep 模式专用 (RQ2 消融: 4 档几何叶配置 × seeds) --
#   CHAP4_ABLATION_LEVELS  "A B C D"    A=裸MAPPO / B=仅距离叶 / C=距离+方向 / D=完整GCPL
#   # CHAP4_SEEDS / CHAP4_PLOT_OUT 共用
#
# 对应论文: contents/chap4.tex (GCPL = MAPPO + RMPflow + RL 叶节点)
# 实现记录: paper-safeMARL/chap4-changes.md D/E/F 节
# 使用说明: paper-safeMARL/chap4-sweep-guide.md
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置 (env var 可覆盖)
# -----------------------------------------------------------------------------
ENV_NAME="${CHAP4_ENV:-chap4_gcpl}"
PY_VERSION="${CHAP4_PY:-3.10}"
SCALE="${CHAP4_SCALE:-small}"
MODE="${CHAP4_MODE:-both}"
SKIP_INSTALL="${CHAP4_SKIP_INSTALL:-0}"
SKIP_SANITY="${CHAP4_SKIP_SANITY:-0}"
SEED="${CHAP4_SEED:-0}"
NUM_AGENTS="${CHAP4_NUM_AGENTS:-3}"
FORMATION_SHAPE="${CHAP4_SHAPE:-wedge}"
DEVICE="${CHAP4_DEVICE:-cpu}"
EVAL_EPISODES="${CHAP4_EVAL_EPISODES:-20}"
FUSION_MODE="${CHAP4_FUSION_MODE:-leaf}"
# sweep-only
SWEEP_ALGOS="${CHAP4_ALGOS:-mappo_rmp mappo mappolag rmpflow}"
SWEEP_SEEDS="${CHAP4_SEEDS:-0 1 2}"
PLOT_OUT="${CHAP4_PLOT_OUT:-images/chap4_sweep}"
# ablation_sweep-only: A=裸 MAPPO / B=仅距离叶 / C=距离+方向 / D=完整 GCPL
ABLATION_LEVELS="${CHAP4_ABLATION_LEVELS:-A B C D}"
TASK_NAME="SafetyPointMultiFormationGoal0-v0"

# Scale 对应的训练参数
case "$SCALE" in
    small)   TOTAL_STEPS=2000;       NUM_ENVS=4  ;;
    medium)  TOTAL_STEPS=500000;     NUM_ENVS=20 ;;
    full)    TOTAL_STEPS=100000000;  NUM_ENVS=80 ;;
    *) echo "ERROR: CHAP4_SCALE must be small|medium|full (got: $SCALE)"; exit 1 ;;
esac

# -----------------------------------------------------------------------------
# 路径 (锚定到脚本所在目录)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_RL_DIR="$SCRIPT_DIR"   # safe-rl-manifold-suite/
REPO_ROOT="$(cd "$SAFE_RL_DIR/.." && pwd)"    # czz-safe-manifold/
SG_DIR="$SAFE_RL_DIR/envs/safety-gymnasium"
SAFEPO_DIR="$SAFE_RL_DIR/algorithms/safe-po"
RMP_DIR="$SAFE_RL_DIR/algorithms/multi-robot-rmpflow"
RUNS_DIR="$REPO_ROOT/runs"    # 训练实际落盘位置 (<cwd>/../runs/, cwd=SAFE_RL_DIR)

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }
die() { log "ERROR: $*"; exit 1; }

check_prereqs() {
    command -v conda >/dev/null 2>&1 || die "conda 不在 PATH 中, 请先安装 miniconda/anaconda"
    [ -d "$SG_DIR" ]     || die "safety-gymnasium submodule 不存在: $SG_DIR (是否 git submodule update --init?)"
    [ -d "$SAFEPO_DIR" ] || die "safe-po submodule 不存在: $SAFEPO_DIR"
    [ -d "$RMP_DIR" ]    || die "multi-robot-rmpflow submodule 不存在: $RMP_DIR"
    [ -f "$RMP_DIR/rmp.py" ]       || die "multi-robot-rmpflow 未正确初始化 (缺 rmp.py)"
    [ -f "$RMP_DIR/rmp_leaf.py" ]  || die "multi-robot-rmpflow 未正确初始化 (缺 rmp_leaf.py)"
    grep -q "^class RLLeaf" "$RMP_DIR/rmp_leaf.py" \
        || die "RLLeaf 类不在 $RMP_DIR/rmp_leaf.py (路径 A 依赖此类)"
    [ -f "$SAFEPO_DIR/safepo/multi_agent/mappo_rmp.py" ] || die "mappo_rmp.py 丢失"
    log "prereq check passed"
}

ensure_conda_env() {
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        log "conda env '$ENV_NAME' already exists"
    else
        log "creating conda env '$ENV_NAME' (Python $PY_VERSION)..."
        # 临时关闭 -u: conda 的 deactivate 脚本会访问未定义变量 (如 CONDA_BACKUP_*)
        set +u
        conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
        set -u
    fi
    # 同样地, conda activate 也会因 set -u 报 unbound variable, 需暂时放宽
    set +u
    conda activate "$ENV_NAME"
    set -u
    log "active env: $(python -c 'import sys; print(sys.prefix)')"
    python --version
}

install_deps() {
    if [ "$SKIP_INSTALL" = "1" ]; then
        log "CHAP4_SKIP_INSTALL=1, 跳过依赖安装"
        return
    fi

    log "安装基础依赖 (numpy<1.24 钉选以兼容 gymnasium-robotics)..."
    pip install --quiet --upgrade pip setuptools wheel

    # Torch CPU (如要 GPU, 请手动先装对应 CUDA 版再 CHAP4_SKIP_INSTALL=1)
    if [ "$DEVICE" = "cuda" ]; then
        log "CHAP4_DEVICE=cuda: 跳过 torch 自动安装, 请确保已装好 CUDA 版 torch"
    else
        pip install --quiet "torch>=1.10" --index-url https://download.pytorch.org/whl/cpu \
            || pip install --quiet "torch>=1.10"
    fi

    pip install --quiet \
        "numpy<1.24" \
        joblib tensorboard rich seaborn wandb pyyaml scipy matplotlib pandas

    # 卸掉任何 pip 版 safety-gymnasium (我们要本地 editable, 带 MultiFormation 任务)
    pip uninstall -y safety-gymnasium >/dev/null 2>&1 || true

    log "editable install: safety-gymnasium (本地 submodule, 含 MultiFormation 任务)..."
    pip install -e "$SG_DIR" --quiet

    log "editable install: safepo (本地 submodule, --no-deps 避免覆盖已钉版本)..."
    pip install -e "$SAFEPO_DIR" --quiet --no-deps

    log "依赖安装完成"
}

run_sanity_check() {
    if [ "$SKIP_SANITY" = "1" ]; then
        log "CHAP4_SKIP_SANITY=1, 跳过 sanity check"
        return
    fi

    export MULTI_ROBOT_RMPFLOW_PATH="$RMP_DIR"
    log "sanity check: 验证 import 链 + 路径 A 的 RL 叶节点集成..."
    python - <<'PY'
import os, sys

print("[1/5] importing safety_gymnasium & safepo ...")
import safety_gymnasium
import safepo
print(f"      safety_gymnasium @ {safety_gymnasium.__file__}")
print(f"      safepo           @ {safepo.__file__}")

print("[2/5] 检查 MultiFormation 任务已注册 ...")
env = safety_gymnasium.make("SafetyPointMultiFormationGoal0-v0",
                            num_agents=3, formation_shape="wedge")
print(f"      action_space(agent_0) = {env.action_space('agent_0')}")

print("[3/5] 检查 RMP 模块可导 (via MULTI_ROBOT_RMPFLOW_PATH) ...")
from safepo.multi_agent.rmp_corrector import RMP_AVAILABLE, RMPCorrector
assert RMP_AVAILABLE, "RMP modules import failed"

print("[4/5] 检查 RLLeaf 存在 (路径 A 核心) ...")
from rmp_leaf import RLLeaf
print(f"      RLLeaf OK: {RLLeaf}")

print("[5/5] 构造 RMPCorrector (fusion_mode=leaf), 验证 RL 叶挂载 ...")
c = RMPCorrector(
    num_agents=3, num_envs=2, device="cpu",
    config={"use_rmp": True, "formation_shape": "wedge",
            "formation_target_distance": 0.5, "collision_safety_radius": 0.3,
            "fusion_mode": "leaf", "rl_leaf_weight": 10.0},
)
n_rl = sum(len(x) for x in c.rmp_rl_nodes)
assert n_rl == 6, f"期望 2env*3agent=6 RL 叶, 实际 {n_rl}"
print(f"      RL 叶节点数: {n_rl} (2 env x 3 agent)")
print("[ok] sanity check 通过")
PY

    log "短程训练 (500 env steps) 冒烟测试..."
    cd "$SAFE_RL_DIR"
    python -u -m safepo.multi_agent.mappo_rmp \
        --task SafetyPointMultiFormationGoal0-v0 \
        --num_agents "$NUM_AGENTS" \
        --formation-shape "$FORMATION_SHAPE" \
        --num-envs 2 \
        --seed "$SEED" \
        --write-terminal True \
        --total-steps 500 \
        --device "$DEVICE"
    log "sanity check 完成"
}

run_training() {
    export MULTI_ROBOT_RMPFLOW_PATH="$RMP_DIR"
    cd "$SAFE_RL_DIR"
    log "正式训练: scale=$SCALE  total_steps=$TOTAL_STEPS  num_envs=$NUM_ENVS"
    log "         shape=$FORMATION_SHAPE  num_agents=$NUM_AGENTS  seed=$SEED"
    log "         fusion_mode=$FUSION_MODE (可由 CHAP4_FUSION_MODE 切换)"
    log "         结果将落盘到: $RUNS_DIR/Base/SafetyPointMultiFormationGoal0-v0/mappo_rmp/seed-$(printf '%03d' "$SEED")-*/"

    # 将 fusion_mode 透传给 rmp_corrector (通过环境变量, rmp_corrector 已支持)
    export CHAP4_FUSION_MODE_OVERRIDE="$FUSION_MODE"

    # mappo_rmp 当前的 CLI 不直接接受 fusion_mode;
    # 若需强制切换, 先改 rmp_corrector 默认或在 RMPCorrector config 里加 fusion_mode.
    python -u -m safepo.multi_agent.mappo_rmp \
        --task SafetyPointMultiFormationGoal0-v0 \
        --num_agents "$NUM_AGENTS" \
        --formation-shape "$FORMATION_SHAPE" \
        --num-envs "$NUM_ENVS" \
        --seed "$SEED" \
        --write-terminal True \
        --total-steps "$TOTAL_STEPS" \
        --device "$DEVICE"

    log "训练完成"
}

run_evaluation() {
    export MULTI_ROBOT_RMPFLOW_PATH="$RMP_DIR"

    # 寻找最近一次训练的 run 目录
    local latest
    latest="$(ls -td "$RUNS_DIR"/Base/SafetyPointMultiFormationGoal0-v0/mappo_rmp/seed-* 2>/dev/null | head -1 || true)"
    if [ -z "$latest" ]; then
        log "WARN: 在 $RUNS_DIR 下找不到训练 run, 跳过评估"
        return 0
    fi

    # 验证 config.json 里有 num_agents (新训练应有, 否则是旧 run)
    if ! grep -q '"num_agents"' "$latest/config.json" 2>/dev/null; then
        log "WARN: 最新 run 的 config.json 里没有 num_agents 字段 (可能是旧训练结果)"
        log "      跳过评估. 请先用本脚本重新训练 (含 [ADD 2026-04-18] 的修复)"
        return 0
    fi

    log "评估最新 run: $latest  ($EVAL_EPISODES episodes)"

    # 构造临时 benchmark 目录, 只含最新 run (避免 evaluate.py 遍历所有旧 seed 目录)
    # evaluate.py 期望结构: <benchmark-dir>/<env>/<algo>/<seed-xxx>/
    local scratch
    scratch="$(mktemp -d)"
    local eval_bench="$scratch/Base"
    mkdir -p "$eval_bench/SafetyPointMultiFormationGoal0-v0/mappo_rmp"
    # 用符号链接, 不复制大文件
    ln -s "$latest" "$eval_bench/SafetyPointMultiFormationGoal0-v0/mappo_rmp/$(basename "$latest")"

    cd "$SAFE_RL_DIR"
    python -u -m safepo.evaluate \
        --benchmark-dir "$eval_bench" \
        --eval-episodes "$EVAL_EPISODES" \
        --headless \
        || log "WARN: evaluate.py 非零退出 (若无 xvfb/显示环境通常可忽略)"

    rm -rf "$scratch"
    log "评估完成"
}

# -----------------------------------------------------------------------------
# Sweep 模式: 4 算法 × N 种子 的对比实验
# -----------------------------------------------------------------------------
# 把单个 (algo, seed) 训练封装为一个函数. algo 内部会处理:
#   mappo_rmp: 默认 GCPL (fusion=leaf, w_RL=10)
#   rmpflow  : GCPL 同框架但 w_RL=0.01, 几何层主导 -> 近似 RMPflow standalone
#             训练完成后把 runs/.../mappo_rmp/seed-* 移到 rmpflow/ 目录
#   mappo    : 纯 MAPPO, 不经 RMP 修正
#   mappolag : MAPPO with Lagrangian 安全约束
run_one() {
    local algo="$1"
    local seed="$2"
    export MULTI_ROBOT_RMPFLOW_PATH="$RMP_DIR"
    cd "$SAFE_RL_DIR"

    log "----- sweep run: algo=$algo  seed=$seed  steps=$TOTAL_STEPS  envs=$NUM_ENVS -----"

    case "$algo" in
        mappo_rmp)
            export CHAP4_FUSION_MODE_OVERRIDE="leaf"
            export CHAP4_RL_LEAF_WEIGHT_OVERRIDE="10.0"
            python -u -m safepo.multi_agent.mappo_rmp \
                --task "$TASK_NAME" \
                --num_agents "$NUM_AGENTS" \
                --formation-shape "$FORMATION_SHAPE" \
                --num-envs "$NUM_ENVS" \
                --seed "$seed" \
                --write-terminal True \
                --total-steps "$TOTAL_STEPS" \
                --device "$DEVICE"
            ;;
        rmpflow)
            # GCPL with w_RL≈0 ⇒ 几何层主导, 近似 RMPflow standalone
            export CHAP4_FUSION_MODE_OVERRIDE="leaf"
            export CHAP4_RL_LEAF_WEIGHT_OVERRIDE="0.01"
            python -u -m safepo.multi_agent.mappo_rmp \
                --task "$TASK_NAME" \
                --num_agents "$NUM_AGENTS" \
                --formation-shape "$FORMATION_SHAPE" \
                --num-envs "$NUM_ENVS" \
                --seed "$seed" \
                --write-terminal True \
                --total-steps "$TOTAL_STEPS" \
                --device "$DEVICE"
            # 把本次训练的结果重命名为 rmpflow 目录
            local latest
            latest="$(ls -td "$RUNS_DIR/Base/$TASK_NAME/mappo_rmp"/seed-$(printf '%03d' "$seed")-* 2>/dev/null | head -1 || true)"
            if [ -n "$latest" ]; then
                mkdir -p "$RUNS_DIR/Base/$TASK_NAME/rmpflow"
                mv "$latest" "$RUNS_DIR/Base/$TASK_NAME/rmpflow/"
                log "rmpflow run 已移入: $RUNS_DIR/Base/$TASK_NAME/rmpflow/$(basename "$latest")"
            fi
            ;;
        mappo)
            unset CHAP4_FUSION_MODE_OVERRIDE CHAP4_RL_LEAF_WEIGHT_OVERRIDE || true
            python -u -m safepo.multi_agent.mappo \
                --task "$TASK_NAME" \
                --num_agents "$NUM_AGENTS" \
                --formation-shape "$FORMATION_SHAPE" \
                --num-envs "$NUM_ENVS" \
                --seed "$seed" \
                --write-terminal True \
                --total-steps "$TOTAL_STEPS" \
                --device "$DEVICE"
            ;;
        mappolag)
            unset CHAP4_FUSION_MODE_OVERRIDE CHAP4_RL_LEAF_WEIGHT_OVERRIDE || true
            python -u -m safepo.multi_agent.mappolag \
                --task "$TASK_NAME" \
                --num_agents "$NUM_AGENTS" \
                --formation-shape "$FORMATION_SHAPE" \
                --num-envs "$NUM_ENVS" \
                --seed "$seed" \
                --write-terminal True \
                --total-steps "$TOTAL_STEPS" \
                --device "$DEVICE"
            ;;
        *)
            die "未知 algo: $algo (允许: mappo_rmp | rmpflow | mappo | mappolag)"
            ;;
    esac
}

run_sweep_eval() {
    # 构造 benchmark 目录, 包含所有 sweep 产生的 run
    export MULTI_ROBOT_RMPFLOW_PATH="$RMP_DIR"
    local bench_src="$RUNS_DIR/Base"
    if [ ! -d "$bench_src/$TASK_NAME" ]; then
        log "WARN: $bench_src/$TASK_NAME 不存在, 跳过评估"
        return
    fi
    cd "$SAFE_RL_DIR"
    # 显式指定 --save-dir, 默认 evaluate.py 会把 benchmark_dir 里的 runs 改写为 results/.
    # 我们固定写到 $bench_src/eval_result.txt 方便 plot_chap4.py 读取.
    local eval_save_dir="$bench_src"
    log "批量评估 sweep runs ($EVAL_EPISODES episodes/run), 结果 -> $eval_save_dir/eval_result.txt"
    python -u -m safepo.evaluate \
        --benchmark-dir "$bench_src" \
        --save-dir "$eval_save_dir" \
        --eval-episodes "$EVAL_EPISODES" \
        --headless \
        || log "WARN: evaluate.py 非零退出 (若无 xvfb/显示环境通常可忽略)"
    log "sweep 评估完成"
}

run_plotting() {
    cd "$SAFE_RL_DIR"
    local out_abs="$SAFE_RL_DIR/paper-safeMARL/$PLOT_OUT"
    mkdir -p "$out_abs"
    # ablation_sweep 模式输出强制使用 ablation 绘图配色/标签 (覆盖 auto 检测)
    local plot_mode="auto"
    if [ "$MODE" = "ablation_sweep" ]; then
        plot_mode="ablation"
    fi
    log "绘图 (mode=$plot_mode): 从 $RUNS_DIR/Base/$TASK_NAME 聚合 → $out_abs"
    python -u scripts/plot_chap4.py \
        --runs-dir "$RUNS_DIR/Base/$TASK_NAME" \
        --eval-result "$RUNS_DIR/Base/eval_result.txt" \
        --out "$out_abs" \
        --mode "$plot_mode" \
        || die "plot_chap4.py 失败"
    log "绘图完成, 输出目录: $out_abs"
    log "  - training_reward.png   (训练 reward 曲线, 多算法多 seed 阴影带)"
    log "  - training_cost.png     (训练 cost 曲线)"
    log "  - eval_bar.png          (4 指标分组柱状图)"
    log "  - cost_reward_pareto.png (Pareto 散点)"
}

run_sweep() {
    log "=== sweep: $SWEEP_ALGOS × seeds=($SWEEP_SEEDS) at scale=$SCALE ==="
    for algo in $SWEEP_ALGOS; do
        for seed in $SWEEP_SEEDS; do
            run_one "$algo" "$seed"
        done
    done
    run_sweep_eval
    run_plotting
}

# -----------------------------------------------------------------------------
# Ablation sweep (RQ2): 4 档几何叶配置 × seeds
# A = 裸 MAPPO                         (全部 geo 叶关, 走 mappo.py)
# B = MAPPO + 距离叶                   (mappo_rmp, 仅 formation)
# C = MAPPO + 距离 + 方向叶            (mappo_rmp, formation + orientation)
# D = 完整 GCPL                        (mappo_rmp, 全开)
# 结果分别落到 runs/Base/<env>/abl_{A,B,C,D}/seed-*/
# -----------------------------------------------------------------------------
run_one_ablation() {
    local level="$1"
    local seed="$2"
    export MULTI_ROBOT_RMPFLOW_PATH="$RMP_DIR"
    cd "$SAFE_RL_DIR"
    log "----- ablation run: level=$level  seed=$seed  steps=$TOTAL_STEPS  envs=$NUM_ENVS -----"

    if [ "$level" = "A" ]; then
        # 裸 MAPPO, 不经 RMP
        unset CHAP4_USE_FORMATION_LEAF CHAP4_USE_ORIENTATION_LEAF CHAP4_USE_COLLISION_LEAF \
              CHAP4_FUSION_MODE_OVERRIDE CHAP4_RL_LEAF_WEIGHT_OVERRIDE || true
        python -u -m safepo.multi_agent.mappo \
            --task "$TASK_NAME" \
            --num_agents "$NUM_AGENTS" \
            --formation-shape "$FORMATION_SHAPE" \
            --num-envs "$NUM_ENVS" \
            --seed "$seed" \
            --write-terminal True \
            --total-steps "$TOTAL_STEPS" \
            --device "$DEVICE"
        # 重命名 mappo run 为 abl_A/
        local latest
        latest="$(ls -td "$RUNS_DIR/Base/$TASK_NAME/mappo"/seed-$(printf '%03d' "$seed")-* 2>/dev/null | head -1 || true)"
        if [ -n "$latest" ]; then
            mkdir -p "$RUNS_DIR/Base/$TASK_NAME/abl_A"
            mv "$latest" "$RUNS_DIR/Base/$TASK_NAME/abl_A/"
        fi
        return
    fi

    # B / C / D 均走 mappo_rmp, 通过 CHAP4_USE_*_LEAF 开关 geo 叶
    export CHAP4_FUSION_MODE_OVERRIDE="leaf"
    export CHAP4_RL_LEAF_WEIGHT_OVERRIDE="10.0"
    case "$level" in
        B)  # 仅距离叶
            export CHAP4_USE_FORMATION_LEAF=true
            export CHAP4_USE_ORIENTATION_LEAF=false
            export CHAP4_USE_COLLISION_LEAF=false
            ;;
        C)  # 距离 + 方向叶
            export CHAP4_USE_FORMATION_LEAF=true
            export CHAP4_USE_ORIENTATION_LEAF=true
            export CHAP4_USE_COLLISION_LEAF=false
            ;;
        D)  # 完整 GCPL
            export CHAP4_USE_FORMATION_LEAF=true
            export CHAP4_USE_ORIENTATION_LEAF=true
            export CHAP4_USE_COLLISION_LEAF=true
            ;;
        *)
            die "未知 ablation level: $level (允许 A|B|C|D)"
            ;;
    esac

    python -u -m safepo.multi_agent.mappo_rmp \
        --task "$TASK_NAME" \
        --num_agents "$NUM_AGENTS" \
        --formation-shape "$FORMATION_SHAPE" \
        --num-envs "$NUM_ENVS" \
        --seed "$seed" \
        --write-terminal True \
        --total-steps "$TOTAL_STEPS" \
        --device "$DEVICE"

    # 把本次训练结果重命名到 abl_<level>/
    local latest
    latest="$(ls -td "$RUNS_DIR/Base/$TASK_NAME/mappo_rmp"/seed-$(printf '%03d' "$seed")-* 2>/dev/null | head -1 || true)"
    if [ -n "$latest" ]; then
        mkdir -p "$RUNS_DIR/Base/$TASK_NAME/abl_$level"
        mv "$latest" "$RUNS_DIR/Base/$TASK_NAME/abl_$level/"
        log "abl_$level run 已移入: $RUNS_DIR/Base/$TASK_NAME/abl_$level/$(basename "$latest")"
    fi
}

run_ablation_sweep() {
    log "=== ablation_sweep: levels=($ABLATION_LEVELS) × seeds=($SWEEP_SEEDS) at scale=$SCALE ==="
    for level in $ABLATION_LEVELS; do
        for seed in $SWEEP_SEEDS; do
            run_one_ablation "$level" "$seed"
        done
    done
    run_sweep_eval
    run_plotting
}

print_summary() {
    echo
    log "=================================================================="
    log "chap4 流水线完成"
    log "  conda env     : $ENV_NAME (python $PY_VERSION)"
    log "  scale / mode  : $SCALE / $MODE"
    log "  fusion_mode   : $FUSION_MODE"
    log "  seed / agents : $SEED / $NUM_AGENTS (shape=$FORMATION_SHAPE)"
    log "  runs dir      : $RUNS_DIR/Base/$TASK_NAME/"
    log "  tensorboard   : tensorboard --logdir=$RUNS_DIR/Base/"
    if [ "$MODE" = "sweep" ] || [ "$MODE" = "plot" ]; then
        log "  figures       : paper-safeMARL/$PLOT_OUT/*.png"
    fi
    log "=================================================================="
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
main() {
    log "=== chap4 GCPL pipeline 启动  (scale=$SCALE, mode=$MODE) ==="
    check_prereqs
    ensure_conda_env
    install_deps

    # plot 模式不需要 sanity check
    if [ "$MODE" != "plot" ]; then
        run_sanity_check
    fi

    case "$MODE" in
        train) run_training ;;
        eval)  run_evaluation ;;
        both)  run_training
               run_evaluation ;;
        sweep) run_sweep ;;
        ablation_sweep) run_ablation_sweep ;;
        plot)  run_sweep_eval
               run_plotting ;;
        *) die "CHAP4_MODE 必须是 train|eval|both|sweep|ablation_sweep|plot (got: $MODE)" ;;
    esac

    print_summary
}

main "$@"

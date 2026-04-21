#!/usr/bin/env python
"""Plot chap4 sweep results.

Input:
  --runs-dir     runs/Base/SafetyPointMultiFormationGoal0-v0/
                   ├─ mappo_rmp/seed-*/progress.csv       (GCPL)
                   ├─ rmpflow/seed-*/progress.csv         (RMPflow standalone)
                   ├─ mappo/seed-*/progress.csv
                   └─ mappolag/seed-*/progress.csv
  --eval-result  runs/Base/eval_result.txt   (tab-separated, from safepo.evaluate)
  --out          paper-safeMARL/images/chap4_sweep/

Output:
  training_reward.png        training reward curves, 4 algos, 1-std band over seeds
  training_cost.png          training cost curves
  eval_bar.png               grouped bar chart: algos x {reward,cost,success,form}
  cost_reward_pareto.png     scatter of (cost, reward) per (algo, seed)

Robust to:
  - missing algos (skips)
  - missing seeds (uses what is there, reports count)
  - progress.csv empty (skips that seed with warning)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 对比实验 (RQ1): 4 基线
COMPARE_ORDER = ["mappo_rmp", "rmpflow", "mappo", "mappolag"]
COMPARE_LABEL = {
    "mappo_rmp": "GCPL (ours)",
    "rmpflow":   "RMPflow",
    "mappo":     "MAPPO",
    "mappolag":  "MAPPO_Lag",
}
COMPARE_COLOR = {
    "mappo_rmp": "#1f77b4",   # blue
    "rmpflow":   "#2ca02c",   # green
    "mappo":     "#d62728",   # red
    "mappolag":  "#ff7f0e",   # orange
}

# 消融实验 (RQ2): 4 档几何叶配置
ABLATION_ORDER = ["abl_A", "abl_B", "abl_C", "abl_D"]
ABLATION_LABEL = {
    "abl_A": "A: MAPPO (no leaf)",
    "abl_B": "B: + dist leaf",
    "abl_C": "C: + dist + orient",
    "abl_D": "D: full GCPL",
}
ABLATION_COLOR = {
    "abl_A": "#d62728",   # red
    "abl_B": "#ff7f0e",   # orange
    "abl_C": "#9467bd",   # purple
    "abl_D": "#1f77b4",   # blue
}

# 初始默认 = 对比模式; main() 会根据实际 runs/ 目录内容自动切换
ALGO_ORDER = COMPARE_ORDER
ALGO_LABEL = COMPARE_LABEL
ALGO_COLOR = COMPARE_COLOR

REWARD_KEY = "Metrics/EpRet"
COST_KEY = "Metrics/EpCost"
STEP_KEY = "Train/TotalSteps"


def load_progress(algo_dir: Path) -> list[pd.DataFrame]:
    """Load progress.csv for every seed under an algo directory."""
    dfs = []
    for seed_dir in sorted(algo_dir.glob("seed-*")):
        csv_path = seed_dir / "progress.csv"
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [warn] failed to read {csv_path}: {e}", file=sys.stderr)
            continue
        if df.empty or STEP_KEY not in df.columns:
            print(f"  [warn] {csv_path} empty or missing {STEP_KEY}; skipped",
                  file=sys.stderr)
            continue
        df["_seed"] = seed_dir.name
        dfs.append(df)
    return dfs


def align_on_steps(dfs: list[pd.DataFrame], key: str, grid: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Linearly-interpolate each seed's (step, key) onto a common grid, return matrix (n_seeds, n_steps)."""
    if not dfs:
        return np.array([]), np.zeros((0, 0))
    if grid is None:
        max_step = max(df[STEP_KEY].max() for df in dfs)
        min_step = min(df[STEP_KEY].min() for df in dfs)
        grid = np.linspace(min_step, max_step, num=min(200, max(50,
                           int(max(len(df) for df in dfs)))))
    rows = []
    for df in dfs:
        if key not in df.columns:
            continue
        rows.append(np.interp(grid, df[STEP_KEY].values, df[key].values))
    if not rows:
        return grid, np.zeros((0, len(grid)))
    return grid, np.stack(rows, axis=0)


def plot_training_curve(algo_to_dfs: dict[str, list[pd.DataFrame]],
                        key: str, ylabel: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    any_plotted = False
    for algo in ALGO_ORDER:
        dfs = algo_to_dfs.get(algo, [])
        if not dfs:
            continue
        grid, mat = align_on_steps(dfs, key)
        if mat.shape[0] == 0:
            continue
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0)
        ax.plot(grid, mu, color=ALGO_COLOR[algo], lw=1.8,
                label=f"{ALGO_LABEL[algo]} (n={mat.shape[0]})")
        ax.fill_between(grid, mu - sd, mu + sd, color=ALGO_COLOR[algo], alpha=0.18)
        any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "no runs with progress.csv yet",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def parse_eval_result(path: Path) -> pd.DataFrame:
    """Parse eval_result.txt. Rows with episode=='SUMMARY' carry aggregated metrics."""
    if not path.is_file():
        print(f"  [warn] eval_result.txt not found at {path}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return df
    df = df[df["episode"].astype(str) == "SUMMARY"].copy()
    # mappo_rmp 里含 rmpflow-standalone 的 run (用 rl_leaf_weight=0.01);
    # 但 sweep 脚本把 rmpflow 目录分离了, evaluate.py 按目录名写 algorithm 列.
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    # success column like "X/Y"; parse out rate
    def parse_rate(s):
        try:
            if "/" in str(s):
                a, b = str(s).split("/")
                return float(a) / float(b) if float(b) > 0 else np.nan
        except Exception:
            pass
        return np.nan
    df["success_rate_num"] = df["success"].apply(parse_rate)
    # success_rate col sometimes holds percentage string like "30.00%"
    def parse_pct(s):
        try:
            s = str(s).strip()
            if s.endswith("%"):
                return float(s[:-1]) / 100.0
        except Exception:
            pass
        return np.nan
    df["success_rate_pct"] = df["success_rate"].apply(parse_pct)
    df["success_rate_final"] = df["success_rate_num"].fillna(df["success_rate_pct"])
    return df


def plot_eval_bar(eval_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: 4 metrics × algos (mean + std bar)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    metrics = [("reward", "Reward (higher better)"),
               ("cost", "Cost (lower better)"),
               ("success_rate_final", "Success rate (higher better)")]
    has_data = not eval_df.empty and "algorithm" in eval_df.columns
    for ax, (col, lbl) in zip(axes, metrics):
        if not has_data:
            ax.text(0.5, 0.5, "no eval data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(lbl)
            continue
        xs, means, stds, colors, labels = [], [], [], [], []
        for idx, algo in enumerate(ALGO_ORDER):
            sub = eval_df[eval_df["algorithm"] == algo][col].dropna()
            if sub.empty:
                continue
            xs.append(idx)
            means.append(float(sub.mean()))
            stds.append(float(sub.std()) if len(sub) > 1 else 0.0)
            colors.append(ALGO_COLOR[algo])
            labels.append(f"{ALGO_LABEL[algo]}\n(n={len(sub)})")
        if not xs:
            ax.text(0.5, 0.5, "no eval data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(lbl)
            continue
        ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85, capsize=4,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(lbl)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.suptitle(f"Evaluation summary (4 algos × seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pareto(eval_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    any_plotted = False
    has_data = not eval_df.empty and "algorithm" in eval_df.columns
    for algo in (ALGO_ORDER if has_data else []):
        sub = eval_df[eval_df["algorithm"] == algo]
        if sub.empty:
            continue
        ax.scatter(sub["cost"], sub["reward"],
                   color=ALGO_COLOR[algo], s=90, alpha=0.85,
                   edgecolor="black", linewidth=0.5,
                   label=f"{ALGO_LABEL[algo]} (n={len(sub)})")
        any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "no eval data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Cost (per episode, lower-left = better)")
    ax.set_ylabel("Reward (per episode)")
    ax.set_title("Cost–Reward Pareto (one point per seed)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.axvline(0, color="black", lw=0.5, alpha=0.5)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    global ALGO_ORDER, ALGO_LABEL, ALGO_COLOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True,
                    help="<repo>/runs/Base/SafetyPointMultiFormationGoal0-v0/")
    ap.add_argument("--eval-result", required=True,
                    help="<repo>/runs/Base/eval_result.txt")
    ap.add_argument("--out", required=True, help="output directory for pngs")
    ap.add_argument("--mode", default="auto", choices=["auto", "compare", "ablation"],
                    help="图表模式: compare (4 基线, RQ1) | ablation (4 档, RQ2) | auto (按 runs/ 内容猜)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.is_dir():
        print(f"ERROR: runs-dir not found: {runs_dir}", file=sys.stderr)
        return 2

    # 根据实际 runs/ 内容选择绘图配置
    existing = {p.name for p in runs_dir.iterdir() if p.is_dir()}
    has_abl = any(n.startswith("abl_") for n in existing)
    has_cmp = any(n in existing for n in COMPARE_ORDER)
    mode = args.mode
    if mode == "auto":
        if has_abl and not has_cmp:
            mode = "ablation"
        elif has_abl and has_cmp:
            print("[auto] 同时检测到 compare 和 ablation 目录, 默认用 compare (用 --mode ablation 切换)")
            mode = "compare"
        else:
            mode = "compare"
    if mode == "ablation":
        ALGO_ORDER = ABLATION_ORDER
        ALGO_LABEL = ABLATION_LABEL
        ALGO_COLOR = ABLATION_COLOR
    else:
        ALGO_ORDER = COMPARE_ORDER
        ALGO_LABEL = COMPARE_LABEL
        ALGO_COLOR = COMPARE_COLOR
    print(f"[plot mode] {mode}   (algos: {ALGO_ORDER})")

    # 1. Load training curves per algo
    print(f"[1/3] scanning progress.csv under {runs_dir} ...")
    algo_to_dfs = {}
    for algo in ALGO_ORDER:
        algo_dir = runs_dir / algo
        if not algo_dir.is_dir():
            print(f"  {algo:11s}  (no dir)")
            continue
        dfs = load_progress(algo_dir)
        algo_to_dfs[algo] = dfs
        print(f"  {algo:11s}  {len(dfs)} seed(s)")

    # 2. Training curves
    print(f"[2/3] plotting training curves ...")
    plot_training_curve(algo_to_dfs, REWARD_KEY, "Episode return",
                        out_dir / "training_reward.png",
                        "Training reward (mean ± 1 std over seeds)")
    plot_training_curve(algo_to_dfs, COST_KEY, "Episode cost",
                        out_dir / "training_cost.png",
                        "Training cost (mean ± 1 std over seeds)")

    # 3. Eval summary
    print(f"[3/3] plotting eval summary from {args.eval_result} ...")
    eval_df = parse_eval_result(Path(args.eval_result))
    plot_eval_bar(eval_df, out_dir / "eval_bar.png")
    plot_pareto(eval_df, out_dir / "cost_reward_pareto.png")

    print(f"done. figures in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

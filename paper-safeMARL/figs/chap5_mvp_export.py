#!/usr/bin/env python3
"""Export MVP 3-mode trajectory data as CSVs for pgfplots.

独立编译:
    eval "$(conda shell.bash hook)" && conda activate iros2026
    python figs/chap5_mvp_export.py \
        --ckpt-mappo /home/.../runs/.../mappo/seed-002-... \
        --ckpt-rmp   /home/.../runs/.../mappo_rmp/seed-002-... \
        --out-dir    figs/chap5_mvp_data

每个模式产生 1 个 CSV (列: t, x0, y0, x1, y1, x2, y2), 外加 metrics.json.
CSV 用 TAB 分隔以兼容 pgfplots \addplot table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# 允许以脚本方式运行, 手动加到 sys.path
_THIS = Path(__file__).resolve()
_SUITE = _THIS.parents[2]  # safe-rl-manifold-suite/
sys.path.insert(0, str(_SUITE))

from cosmos.apps.formation_nav.full_stack_mvp_demo import (  # noqa: E402
    NUM_AGENTS, rollout, make_mappo_policy,
)


def write_csv(path: Path, traj: np.ndarray) -> None:
    """traj: (T, N, 2). CSV columns: t x0 y0 x1 y1 x2 y2 (tab separated)."""
    T = traj.shape[0]
    header = "t\t" + "\t".join(f"x{i}\ty{i}" for i in range(NUM_AGENTS))
    with path.open("w") as f:
        f.write(header + "\n")
        for t in range(T):
            row = [f"{t}"]
            for i in range(NUM_AGENTS):
                row.append(f"{traj[t, i, 0]:.6f}")
                row.append(f"{traj[t, i, 1]:.6f}")
            f.write("\t".join(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-mappo", default=None)
    ap.add_argument("--ckpt-rmp", default=None)
    ap.add_argument("--out-dir", default="figs/chap5_mvp_data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mappo_policy = make_mappo_policy(args.ckpt_mappo) if args.ckpt_mappo else None
    rmp_policy = make_mappo_policy(args.ckpt_rmp) if args.ckpt_rmp else None

    policy_for = {
        "mappo_only": mappo_policy,
        "rmp_only":   None,
        "fusion":     rmp_policy,
    }

    metrics_all: dict[str, dict] = {}
    for m in ("mappo_only", "rmp_only", "fusion"):
        traj, metrics = rollout(m, seed=args.seed, policy_fn=policy_for[m])
        csv_path = out_dir / f"{m}.tsv"
        write_csv(csv_path, traj)
        metrics_all[m] = metrics
        print(f"[saved] {csv_path}  (T={traj.shape[0]})  {metrics}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics_all, indent=2))
    print(f"[saved] {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()

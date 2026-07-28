#!/usr/bin/env python3
"""Generate paper-ready ABER-LR full-matrix figures from the audited JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
FIGURE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "safe-rl-2027/results/aaai27_aber_latched_restore_cross_task_v1"
    / "full_30_checkpoint_aggregate.json"
)


GROUPS = (
    "none/goal",
    "none/push",
    "none/multigoal",
    "sim_brt/goal",
    "sim_brt/push",
    "sim_brt/multigoal",
)
LABELS = (
    "Native / Goal1",
    "Native / Push1",
    "Native / Goal2",
    "sim-BRT / Goal1",
    "sim-BRT / Push1",
    "sim-BRT / Goal2",
)
COLORS = {
    "off": "#4D4D4D",
    "strict": "#005A8D",
    "lr": "#9C3200",
    "grid": "#D9D9D9",
    "pair": "#707070",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)


def per_seed(groups: dict, variant: str, metric: str) -> np.ndarray:
    return np.asarray(
        [
            [
                groups[group]["per_training_seed"][str(seed)][variant][metric]
                for seed in range(5)
            ]
            for group in GROUPS
        ],
        dtype=float,
    )


def style_rows(ax: plt.Axes, y: np.ndarray) -> None:
    for index, center in enumerate(y):
        if index % 2 == 0:
            ax.axhspan(center - 0.43, center + 0.43, color="#F4F4F4", zorder=0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def full_matrix_figure(payload: dict, output_dir: Path) -> None:
    groups = payload["groups"]
    y = np.arange(len(GROUPS), dtype=float)[::-1]
    seed_jitter = np.linspace(-0.055, 0.055, 5)
    variants = (
        ("filter_off", "Filter off", COLORS["off"], "o", -0.18),
        ("strict", "Strict", COLORS["strict"], "^", 0.0),
        ("candidate", "ABER-LR", COLORS["lr"], "D", 0.18),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.65),
        gridspec_kw={"width_ratios": (1.10, 1.0)},
    )
    ax = axes[0]
    style_rows(ax, y)
    for variant, label, color, marker, offset in variants:
        values = 100.0 * per_seed(groups, variant, "safe_success_rate")
        pooled = 100.0 * np.asarray(
            [
                groups[group]["pooled_summaries"][variant]["safe_success_rate"]
                for group in GROUPS
            ]
        )
        for group_index, center in enumerate(y):
            ax.scatter(
                values[group_index],
                center + offset + seed_jitter,
                s=7,
                color=color,
                alpha=0.40,
                linewidths=0,
                zorder=2,
            )
        ax.scatter(
            pooled,
            y + offset,
            s=24,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.55,
            label=label,
            zorder=4,
        )
    ax.set_xlabel("Collision-free success, SCR (%)")
    ax.set_yticks(y, LABELS)
    ax.set_xlim(-2, 92)
    ax.set_ylim(-0.55, 5.90)
    ax.set_title("(a) Checkpoint SCR", loc="left", pad=3)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper right",
        borderaxespad=0.25,
        handletextpad=0.30,
        columnspacing=0.65,
    )

    ax = axes[1]
    style_rows(ax, y)
    strict_seed = per_seed(groups, "strict", "max_recovery_streak_q95")
    lr_seed = per_seed(groups, "candidate", "max_recovery_streak_q95")
    strict_pooled = np.asarray(
        [groups[group]["pooled_summaries"]["strict"]["max_recovery_streak_q95"] for group in GROUPS]
    )
    lr_pooled = np.asarray(
        [groups[group]["pooled_summaries"]["candidate"]["max_recovery_streak_q95"] for group in GROUPS]
    )
    for group_index, center in enumerate(y):
        for seed in range(5):
            level = center + seed_jitter[seed]
            ax.plot(
                [strict_seed[group_index, seed], lr_seed[group_index, seed]],
                [level, level],
                color=COLORS["pair"],
                linewidth=0.55,
                alpha=0.65,
                zorder=1,
            )
            ax.scatter(strict_seed[group_index, seed], level, s=7, color=COLORS["strict"], alpha=0.42, linewidths=0, zorder=2)
            ax.scatter(lr_seed[group_index, seed], level, s=7, color=COLORS["lr"], alpha=0.42, linewidths=0, zorder=2)
    ax.scatter(strict_pooled, y, s=27, marker="^", color=COLORS["strict"], edgecolor="white", linewidth=0.55, zorder=4)
    ax.scatter(lr_pooled, y, s=27, marker="D", color=COLORS["lr"], edgecolor="white", linewidth=0.55, zorder=4)
    ax.set_xlabel("q95 maximum recovery streak (steps)")
    ax.set_yticks(y, [])
    ax.set_xlim(-25, 1025)
    ax.set_xticks((0, 250, 500, 750, 1000))
    ax.set_ylim(-0.55, 5.90)
    ax.set_title("(b) Recovery-tail compression", loc="left", pad=3)
    ax.text(
        1000,
        5.67,
        "group mean: 916 → 121 steps",
        ha="right",
        va="center",
        fontsize=9.0,
        color="#333333",
    )
    fig.subplots_adjust(left=0.145, right=0.995, bottom=0.17, top=0.88, wspace=0.17)
    save(fig, output_dir / "aber_lr_full_matrix")


def recovery_figure(payload: dict, output_dir: Path) -> None:
    groups = payload["groups"]
    y = np.arange(len(GROUPS), dtype=float)[::-1]
    seed_jitter = np.linspace(-0.09, 0.09, 5)
    strict_fraction = np.asarray(
        [
            groups[group]["pooled_summaries"]["strict"]["recovery_step_fraction"]
            for group in GROUPS
        ]
    )
    lr_fraction = np.asarray(
        [
            groups[group]["pooled_summaries"]["candidate"]["recovery_step_fraction"]
            for group in GROUPS
        ]
    )
    strict_q95 = np.asarray(
        [
            groups[group]["pooled_summaries"]["strict"]["max_recovery_streak_q95"]
            for group in GROUPS
        ]
    )
    lr_q95 = np.asarray(
        [
            groups[group]["pooled_summaries"]["candidate"]["max_recovery_streak_q95"]
            for group in GROUPS
        ]
    )

    strict_fraction_seed = 100.0 * per_seed(groups, "strict", "recovery_step_fraction")
    lr_fraction_seed = 100.0 * per_seed(groups, "candidate", "recovery_step_fraction")
    strict_q95_seed = per_seed(groups, "strict", "max_recovery_streak_q95")
    lr_q95_seed = per_seed(groups, "candidate", "max_recovery_streak_q95")

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.20))
    ax = axes[0]
    style_rows(ax, y)
    for group_index, center in enumerate(y):
        for seed in range(5):
            level = center + seed_jitter[seed]
            ax.plot([strict_fraction_seed[group_index, seed], lr_fraction_seed[group_index, seed]], [level, level], color=COLORS["pair"], linewidth=0.55, alpha=0.65)
            ax.scatter(strict_fraction_seed[group_index, seed], level, s=9, color=COLORS["strict"], alpha=0.42, linewidths=0)
            ax.scatter(lr_fraction_seed[group_index, seed], level, s=9, color=COLORS["lr"], alpha=0.42, linewidths=0)
    ax.scatter(100 * strict_fraction, y, s=30, marker="^", color=COLORS["strict"], edgecolor="white", linewidth=0.6, label="Strict ABER", zorder=4)
    ax.scatter(100 * lr_fraction, y, s=30, marker="D", color=COLORS["lr"], edgecolor="white", linewidth=0.6, label="ABER-LR", zorder=4)
    ax.set_xlabel("Recovery occupancy (%)")
    ax.set_yticks(y, LABELS)
    ax.set_xlim(-2, 82)
    ax.set_ylim(-0.55, 5.55)
    ax.set_title("(a) Time spent in recovery", loc="left", pad=3)

    ax = axes[1]
    style_rows(ax, y)
    for group_index, center in enumerate(y):
        for seed in range(5):
            level = center + seed_jitter[seed]
            ax.plot([strict_q95_seed[group_index, seed], lr_q95_seed[group_index, seed]], [level, level], color=COLORS["pair"], linewidth=0.55, alpha=0.65)
            ax.scatter(strict_q95_seed[group_index, seed], level, s=9, color=COLORS["strict"], alpha=0.42, linewidths=0)
            ax.scatter(lr_q95_seed[group_index, seed], level, s=9, color=COLORS["lr"], alpha=0.42, linewidths=0)
    ax.scatter(strict_q95, y, s=30, marker="^", color=COLORS["strict"], edgecolor="white", linewidth=0.6, zorder=4)
    ax.scatter(lr_q95, y, s=30, marker="D", color=COLORS["lr"], edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_xlabel("q95 maximum recovery streak (steps)")
    ax.set_yticks(y, LABELS)
    ax.set_xlim(-25, 1025)
    ax.set_xticks((0, 250, 500, 750, 1000))
    ax.set_ylim(-0.55, 5.55)
    ax.set_title("(b) Episode-level recovery tail", loc="left", pad=3)
    fig.subplots_adjust(left=0.31, right=0.99, bottom=0.08, top=0.96, hspace=0.38)
    save(fig, output_dir / "aber_lr_recovery_mechanism")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the classified main/supplement output directory.",
    )
    parser.add_argument(
        "--figure",
        choices=("all", "full-matrix", "recovery-mechanism"),
        default="all",
        help="Build one independently selectable figure, or both (default).",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    configure()
    if args.figure in ("all", "full-matrix"):
        output_dir = args.output_dir or FIGURE_ROOT / "main"
        output_dir.mkdir(parents=True, exist_ok=True)
        full_matrix_figure(payload, output_dir)
    if args.figure in ("all", "recovery-mechanism"):
        output_dir = args.output_dir or FIGURE_ROOT / "supplement"
        output_dir.mkdir(parents=True, exist_ok=True)
        recovery_figure(payload, output_dir)


if __name__ == "__main__":
    main()

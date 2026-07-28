#!/usr/bin/env python3
"""Generate the standalone ABER-LR falsification-chain figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
FIGURE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ESCAPE_AUDIT = (
    REPO_ROOT
    / "safe-rl-2027/results/aaai27_aber_escape_goal1_pilot_v1_st"
    / "pilot_audit.json"
)
DEFAULT_MARGIN_DIR = (
    REPO_ROOT
    / "safe-rl-2027/results/aaai27_aber_latched_restore_margin_screen_v1"
)
DEFAULT_CONFIRM_AUDIT = (
    REPO_ROOT
    / "safe-rl-2027/results/aaai27_aber_latched_restore_m10_confirm_v1"
    / "confirm_audit.json"
)

COLORS = {
    "gray": "#6B6B6B",
    "strict": "#0072B2",
    "light_blue": "#56B4E9",
    "magenta": "#CC79A7",
    "lr": "#D55E00",
    "green": "#009E73",
    "grid": "#D9D9D9",
    "link": "#A0A0A0",
    "fail": "#B5413E",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escape-audit", type=Path, default=DEFAULT_ESCAPE_AUDIT)
    parser.add_argument("--margin-dir", type=Path, default=DEFAULT_MARGIN_DIR)
    parser.add_argument("--confirm-audit", type=Path, default=DEFAULT_CONFIRM_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_ROOT / "main")
    args = parser.parse_args()

    configure()
    pilot = load_json(args.escape_audit)["summaries"]
    confirm = load_json(args.confirm_audit)["summaries"]

    margin_points = []
    for name in ("latch_m06", "latch_m08", "latch_m10", "latch_m12"):
        payload = load_json(args.margin_dir / name / "paired.json")
        margin_points.append(
            (
                float(payload["protocol"]["filter_on_config"]["dt_atacom_safety_margin"]),
                100.0 * float(payload["result"]["modes"]["filter_on"]["geometric_collision"]["mean"]),
                100.0 * float(payload["result"]["modes"]["filter_on"]["success"]["mean"]),
            )
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.33),
        gridspec_kw={"width_ratios": (1.05, 0.91, 1.18)},
    )

    # (a) Falsified yaw/calibration repairs: occupancy improves slightly, but
    # the horizon-scale tail and the success gate do not.
    ax = axes[0]
    pilot_order = (
        ("baseline_no_escape_a09", "strict", COLORS["gray"], "o"),
        ("escape_a09", "yaw (.9)", COLORS["strict"], "s"),
        ("escape_a15", "yaw (1.5)", COLORS["light_blue"], "^"),
        ("escape_a20", "yaw (2.0)", COLORS["magenta"], "v"),
    )
    x = np.arange(len(pilot_order), dtype=float)
    successes = np.asarray(
        [100.0 * pilot[key]["success_rate"] for key, *_ in pilot_order]
    )
    ax.axhspan(34.0, 37.0, color=COLORS["green"], alpha=0.08, zorder=0)
    ax.axhline(34.0, color=COLORS["green"], linewidth=0.75, linestyle="--")
    ax.text(-0.38, 35.6, "success gate ≥34", color=COLORS["green"], fontsize=6.4)
    ax.plot(x, successes, color=COLORS["link"], linewidth=0.8, zorder=1)
    tick_labels = []
    for index, (key, label, color, marker) in enumerate(pilot_order):
        item = pilot[key]
        success = 100.0 * item["success_rate"]
        collision = 100.0 * item["geometric_collision_rate"]
        q95 = int(round(item["max_recovery_streak"]["q95"]))
        ax.scatter(
            index,
            success,
            s=38,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        ax.text(
            index,
            success + 0.85,
            f"q95 {q95}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=COLORS["fail"],
        )
        tick_labels.append(f"{label}\nC {collision:.1f}")
    ax.text(
        1.5,
        30.8,
        "tail persists",
        color=COLORS["fail"],
        fontsize=6.4,
        ha="center",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": COLORS["fail"], "lw": 0.6},
    )
    ax.set_xlim(-0.48, 3.48)
    ax.set_ylim(17.0, 37.0)
    ax.set_xticks(x, tick_labels, fontsize=5.6)
    ax.set_ylabel("Success (%)  ↑")
    ax.set_title("(a) Rejected repairs\n$n=250$ per variant", loc="left", pad=2)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    # (b) Frozen discovery trajectory. Arrows encode evaluated order only.
    ax = axes[1]
    for index in range(len(margin_points) - 1):
        _, x0, y0 = margin_points[index]
        _, x1, y1 = margin_points[index + 1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": COLORS["link"], "lw": 0.8},
            zorder=1,
        )
    margin_colors = (COLORS["light_blue"], COLORS["strict"], COLORS["lr"], COLORS["magenta"])
    label_offsets = {0.06: (-0.8, -2.5), 0.08: (0.8, 0.8), 0.10: (0.8, 0.8), 0.12: (0.8, -2.0)}
    for (margin, collision, success), color in zip(margin_points, margin_colors):
        selected = abs(margin - 0.10) < 1e-9
        ax.scatter(
            collision,
            success,
            s=50 if selected else 34,
            marker="D" if selected else "o",
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        dx, dy = label_offsets[margin]
        suffix = " selected" if selected else ""
        ha = "right" if margin == 0.06 else "left"
        ax.text(collision + dx, success + dy, f"$m={margin:.2f}${suffix}", fontsize=5.9, ha=ha)
    ax.annotate(
        "better",
        xy=(2.0, 34.0),
        xytext=(13.0, 24.0),
        arrowprops={"arrowstyle": "->", "color": COLORS["green"], "lw": 0.85},
        color=COLORS["green"],
        fontsize=6.4,
    )
    ax.set_xlim(-1.5, 32.5)
    ax.set_ylim(18.0, 35.5)
    ax.set_xlabel("Collision (%)  ↓")
    ax.set_ylabel("Success (%)  ↑")
    ax.set_title("(b) Frozen margin screen\n$n=50$ per variant", loc="left", pad=2)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    # (c) Independent held-out confirmation. q95 is shown as a fraction of
    # the 1000-step horizon so all four diagnostics share a percentage axis.
    ax = axes[2]
    strict = confirm["baseline_filter_on"]
    lr = confirm["candidate_filter_on"]
    labels = (
        "Success  ↑",
        "Collision  ↓",
        "Rec. occupancy  ↓",
        "q95 / horizon  ↓",
    )
    strict_values = np.asarray(
        [
            100.0 * strict["success_rate"],
            100.0 * strict["collision_rate"],
            100.0 * strict["recovery_step_fraction"],
            strict["max_recovery_streak"]["q95"] / 10.0,
        ]
    )
    lr_values = np.asarray(
        [
            100.0 * lr["success_rate"],
            100.0 * lr["collision_rate"],
            100.0 * lr["recovery_step_fraction"],
            lr["max_recovery_streak"]["q95"] / 10.0,
        ]
    )
    y = np.arange(len(labels))[::-1]
    for row, (strict_value, lr_value) in enumerate(zip(strict_values, lr_values)):
        level = y[row]
        ax.plot(
            [strict_value, lr_value],
            [level, level],
            color=COLORS["link"],
            linewidth=1.15,
            zorder=1,
        )
    ax.scatter(strict_values, y, s=34, marker="^", color=COLORS["strict"], edgecolor="white", linewidth=0.6, label="Strict", zorder=3)
    ax.scatter(lr_values, y, s=34, marker="D", color=COLORS["lr"], edgecolor="white", linewidth=0.6, label="ABER-LR", zorder=3)
    for row, (strict_value, lr_value) in enumerate(zip(strict_values, lr_values)):
        level = y[row]
        if row == 3:
            ax.text(91.0, level + 0.22, "95.5 (955 steps)", ha="center", fontsize=5.8, color=COLORS["strict"])
            lr_label = "13.0 (130 steps)"
        else:
            ax.text(strict_value, level + 0.22, f"{strict_value:.1f}", ha="center", fontsize=5.8, color=COLORS["strict"])
            lr_label = f"{lr_value:.1f}"
        ax.text(lr_value, level - 0.27, lr_label, ha="center", fontsize=5.8, color=COLORS["lr"])
    ax.set_yticks(y, labels, fontsize=6.1)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("Rate or horizon fraction (%)")
    ax.set_title("(c) Held-out confirmation\n$n=250$ paired episodes", loc="left", pad=2)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", borderaxespad=0.2, handletextpad=0.3, columnspacing=0.7, fontsize=6.3)

    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.22, top=0.84, wspace=0.42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "aber_lr_falsification_chain"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    main()

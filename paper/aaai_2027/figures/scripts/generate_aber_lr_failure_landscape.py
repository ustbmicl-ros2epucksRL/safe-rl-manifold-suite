#!/usr/bin/env python3
"""Generate the paper-ready rejected-repair landscape from frozen audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


COLORS = {
    "baseline": "#4D4D4D",
    "strict": "#005A8D",
    "selected": "#9C3200",
    "accent": "#713568",
    "grid": "#D9D9D9",
    "gate": "#006B4F",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--escape-audit", type=Path, default=DEFAULT_ESCAPE_AUDIT)
    parser.add_argument("--margin-dir", type=Path, default=DEFAULT_MARGIN_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURE_ROOT / "supplement"
    )
    args = parser.parse_args()

    configure()
    pilot = load_json(args.escape_audit)["summaries"]
    pilot_order = (
        ("baseline_no_escape_a09", "strict $(0,0)$", COLORS["baseline"], "o"),
        ("escape_a09", "yaw, $a_b=.9$", COLORS["strict"], "s"),
        ("escape_a15", "yaw, $a_b=1.5$", "#56B4E9", "^"),
        ("escape_a20", "yaw, $a_b=2.0$", COLORS["accent"], "v"),
    )

    margins: list[tuple[float, float, float]] = []
    for name in ("latch_m06", "latch_m08", "latch_m10", "latch_m12"):
        payload = load_json(args.margin_dir / name / "paired.json")
        margins.append(
            (
                float(payload["protocol"]["filter_on_config"]["dt_atacom_safety_margin"]),
                100.0 * float(payload["result"]["modes"]["filter_on"]["geometric_collision"]["mean"]),
                100.0 * float(payload["result"]["modes"]["filter_on"]["success"]["mean"]),
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85))

    ax = axes[0]
    ax.axvspan(16, 36.6, ymin=(34 - 15) / (40 - 15), ymax=1.0, color=COLORS["gate"], alpha=0.08, zorder=0)
    ax.text(16.6, 39.2, "pilot gate", color=COLORS["gate"], fontsize=9, va="top")
    base = pilot["baseline_no_escape_a09"]
    base_xy = (100.0 * base["geometric_collision_rate"], 100.0 * base["success_rate"])
    label_positions = {
        "baseline_no_escape_a09": (27.5, 23.1),
        "escape_a09": (34.5, 28.0),
        "escape_a15": (25.9, 18.1),
        "escape_a20": (35.4, 16.0),
    }
    for key, label, color, marker in pilot_order:
        item = pilot[key]
        x = 100.0 * item["geometric_collision_rate"]
        y = 100.0 * item["success_rate"]
        if key != "baseline_no_escape_a09":
            ax.annotate("", xy=(x, y), xytext=base_xy, arrowprops={"arrowstyle": "->", "color": "#AAAAAA", "lw": 0.75}, zorder=1)
        ax.scatter(x, y, s=42, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        q95 = int(round(item["max_recovery_streak"]["q95"]))
        ax.annotate(
            f"{label}\nq95={q95}",
            xy=(x, y),
            xytext=label_positions[key],
            fontsize=9,
            color="#222222",
            linespacing=0.95,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.55},
        )
    ax.set_xlim(16, 41)
    ax.set_ylim(15, 40)
    ax.set_xlabel("Geometric collision (%)  ↓")
    ax.set_ylabel("Success (%)  ↑")
    ax.set_title("(a) Yaw-only escape cannot exit the tail", loc="left", pad=3)
    ax.grid(color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)

    ax = axes[1]
    margin_colors = ("#56B4E9", COLORS["strict"], COLORS["selected"], COLORS["accent"])
    for index in range(len(margins) - 1):
        _, x0, y0 = margins[index]
        _, x1, y1 = margins[index + 1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": "#9A9A9A", "lw": 0.8}, zorder=1)
    margin_offsets = {0.06: (0.8, 0.8), 0.08: (0.8, 0.8), 0.10: (0.8, 0.8), 0.12: (0.8, -2.2)}
    for (margin, collision, success), color in zip(margins, margin_colors):
        size = 58 if margin == 0.10 else 40
        ax.scatter(collision, success, s=size, marker="D" if margin == 0.10 else "o", color=color, edgecolor="white", linewidth=0.7, zorder=3)
        dx, dy = margin_offsets[margin]
        suffix = "  selected" if margin == 0.10 else ""
        ax.text(collision + dx, success + dy, f"margin={margin:.2f}{suffix}", fontsize=9, color="#222222")
    ax.annotate("desired direction", xy=(2.0, 34.0), xytext=(12.0, 24.0), arrowprops={"arrowstyle": "->", "color": COLORS["gate"], "lw": 0.9}, color=COLORS["gate"], fontsize=9)
    ax.set_xlim(-1.5, 32.5)
    ax.set_ylim(18, 35.5)
    ax.set_xlabel("Geometric collision (%)  ↓")
    ax.set_ylabel("Success (%)  ↑")
    ax.set_title("(b) Margin couples safety and liveness", loc="left", pad=3)
    ax.grid(color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.88, wspace=0.25)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "aber_lr_failure_landscape"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    main()

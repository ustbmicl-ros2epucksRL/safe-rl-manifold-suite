# AAAI-27 ABER/ABER-LR Figure Plan

Last updated: 2026-07-27

## Final visual hierarchy

The paper uses figures to answer three different questions.  Exact rates and
confidence intervals remain in tables; figures expose distributions,
same-checkpoint direction, and failure mechanisms.

| Location | Figure | Question answered |
|---|---|---|
| Main paper, p.7 | 30-checkpoint SCR and recovery-tail figure | Does ABER-LR improve safe completion over strict ABER, and is the long recovery tail the mechanism? |
| Supplement, p.5 | Rejected-repair failure landscape | Why do yaw-only escape, braking calibration changes, and extreme margins fail? |
| Supplement, p.7 | Detailed recovery mechanism | How consistently do occupancy and q95 streak change across five checkpoints in each mode/task group? |

## Directory policy

| Directory | Contents |
|---|---|
| `figures/main/` | Only figures included by the seven-page main paper |
| `figures/supplement/` | Only figures included by the supplementary material |
| `figures/backup/` | Historical, rejected, superseded, or currently unused figures and media |
| `figures/scripts/` | Reproducible plotting sources; no paper-ready outputs |

## Main figure

Files:

- `figures/main/aber_lr_full_matrix.pdf`
- `figures/main/aber_lr_full_matrix.png`

Panel (a) shows checkpoint-level SCR for filter off, strict ABER, and ABER-LR.
Small translucent symbols are the five checkpoints in each group; large
symbols are equal-episode group means.  Panel (b) links strict and ABER-LR q95
maximum recovery streaks for the same checkpoint.  Horizontal mode/task rows
replace rotated x labels.

This is the only main-paper figure because it carries both the empirical
contribution and its mechanism without duplicating the exact summary table.

## Completed main-paper candidate: falsification chain

Files:

- `figures/main/aber_lr_falsification_chain.pdf`
- `figures/main/aber_lr_falsification_chain.png`

This standalone three-panel figure separates the 250-episode rejected-repair
pilot, the 50-episode frozen margin screen, and the independent 250-pair
held-out confirmation.  It is deliberately not included by the main LaTeX
source until the seven-page layout is revised.  Build and verify it with:

```bash
make -C paper/aaai_2027/figures verify-falsification
```

Source:
`figures/scripts/generate_aber_lr_falsification_chain.py`.

## Supplementary failure landscape

Files:

- `figures/supplement/aber_lr_failure_landscape.pdf`
- `figures/supplement/aber_lr_failure_landscape.png`

Panel (a) retains every 250-pair yaw-only pilot, the frozen pass region, and
the q95 tail attached to each point.  Panel (b) shows the separate 50-pair
margin discovery sequence.  Upper-left is favorable in both panels.  Arrows
show the frozen search order only; the caption explicitly rejects continuous
interpolation or pooling across schedules.

The adjacent table remains because it provides exact recovery occupancy,
sample counts, and the operational reason for every NO-GO.

## Supplementary recovery mechanism

Files:

- `figures/supplement/aber_lr_recovery_mechanism.pdf`
- `figures/supplement/aber_lr_recovery_mechanism.png`

Two horizontal paired-dot panels show recovery occupancy and q95 maximum
recovery streak.  Small points are checkpoints, large symbols are group means,
and gray links preserve pairing.  The figure contains no internal legend that
can cover data; marker semantics are stated in the caption and match the main
figure.

## Independent build contract

Every current figure is a standalone, one-page PDF and can be rebuilt without
compiling either paper.  The complete audit has six independent targets (one
main-paper figure and five supplementary figures):

```bash
make -C paper/aaai_2027/figures main
make -C paper/aaai_2027/figures failure
make -C paper/aaai_2027/figures recovery
make -C paper/aaai_2027/figures intervention
make -C paper/aaai_2027/figures contract
make -C paper/aaai_2027/figures fastpath
```

`make -C paper/aaai_2027/figures verify-active` rebuilds and checks all six.
The intervention target deliberately reconstructs its ECDF from the frozen
step-level records; the plotted tail is not approximated from summary
quantiles.

Generators:

- Main and recovery plotting functions:
  `figures/scripts/generate_aber_lr_figures.py`; the `--figure` switch selects
  exactly one output and routes the default destination by paper role.
- Failure-landscape generator:
  `figures/scripts/generate_aber_lr_failure_landscape.py`.

## Audited data sources

- Main/recovery data:
  `safe-rl-2027/results/aaai27_aber_latched_restore_cross_task_v1/full_30_checkpoint_aggregate.json`
- Escape data:
  `safe-rl-2027/results/aaai27_aber_escape_goal1_pilot_v1_st/pilot_audit.json`
- Margin data:
  `safe-rl-2027/results/aaai27_aber_latched_restore_margin_screen_v1/`

The full aggregate now includes absolute per-training-seed metrics for every
group, so the paper figures are generated from one audited 30-checkpoint JSON
rather than manually copied values.

## Visual standard

- Colorblind-safe palette: gray for filter off, blue for strict ABER, and
  vermillion for ABER-LR.
- Marker identity is redundant with color: circle, triangle, and diamond.
- Small translucent points expose checkpoint variation; large symbols expose
  the reported group estimator.
- Gray links are used only for matched checkpoints.
- Horizontal task labels; no rotated category labels.
- Light alternating row bands and x-grid only; no chart borders on top/right.
- Serif typography, embedded PDF fonts, no Type 3 fonts.
- No decorative 3-D effects, gradients, or bar charts that hide seed
  dispersion.

## Final layout checks

- Main PDF: 8 pages total, with seven full content pages and References only
  on page 8.
- Main page 7 reaches the AAAI text bottom in both columns.
- Supplement: 10 pages after adding the failure landscape.
- All active figure PDFs use embedded fonts and compile without overfull boxes
  or undefined references.

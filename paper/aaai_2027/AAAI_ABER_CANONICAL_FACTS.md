# AAAI 2027 ABER Canonical Facts

Last updated: 2026-07-27

This file is the paper-editing source of truth.  Earlier Goal1-only notes,
strict-ABER tables, and diagnostic VA-ATACOM files must not override it.

## Submission identity

- Paper method: **ABER** with the empirical liveness extension
  **ABER-LR** (liveness-aware ABER).
- Do not present VA-ATACOM as the AAAI method or include a VA-ATACOM result
  table in the AAAI paper.
- Central empirical contribution: coordination of safety and completion
  indicators, not raw-success maximization.

## Method boundary

- Certified inner layer: sampled-data ABER forward-speed projection with
  strict executable `(0,0)` recovery.
- Empirical liveness layer: margin-triggered latched directional restore.
- Frozen ABER-LR parameters:
  `safety_margin=0.10`, `a_b=0.9`, `near_stop_speed=0.05`,
  `restore_forward=0.25`, `restore_turn_gain=1.0`,
  `recovery_restore_enabled=true`, `recovery_restore_latched=true`,
  `recovery_escape_enabled=false`.
- Restore actions execute outside the certified set and are not an A2
  recursive-feasibility witness.  Theorem claims apply only to the certified
  strict ABER layer and accepted plant envelope.
- Push1 movable-body contact topology is outside the static-obstacle
  certificate; Push1 is empirical evidence only.

## Primary metrics

\[
\mathrm{SCR}
=\Pr(\mathrm{success}\land\neg\mathrm{geometric\ collision}).
\]

Always report SCR together with raw success, geometric collision, unsafe
success, timeout, recovery fraction, and recovery-streak statistics.  Do not
replace them by a tuned weighted scalar.

## Failure taxonomy and rejected repairs

Negative tests must be classified by the gate they fail:

- **safety failure:** crosses the frozen collision bound;
- **liveness failure:** misses success or recovery-tail improvement;
- **coordination failure:** misses the SCR gate despite improving one marginal;
- **transfer failure:** loses the required direction or magnitude across
  training seeds/tasks;
- **contract failure:** A1--A2 cannot be attached to the empirical plant; this
  limits certification but does not erase empirical outcome data.

Key retained Goal1 failures:

- strict `(0,0)`: success 24.0%, collision 33.6%, recovery occupancy 72.0%,
  q95 maximum recovery streak 958;
- yaw-only escape at `a_b=0.9`: success 25.6%, collision 33.6%, recovery
  occupancy 70.7%, q95 still 958; only +1.6 pp success, so NO-GO;
- yaw-only escape at `a_b=1.5/2.0`: success 20.4%/18.8% and q95 963/967;
  changing recovery entry does not solve exit from the stalled state;
- 50-pair margin screen with latched restore: margin 0.06 gives
  success/collision 28%/28%, margin 0.10 gives 32%/4%, and margin 0.12 gives
  22%/0%.  Too little margin leaves collision; too much recreates liveness
  loss.  Margin 0.10 was sent to disjoint 250-pair confirmation.

## Complete learned-policy evidence

- 2 training modes: native (`none`) and `sim_brt`.
- 3 tasks: SafetyPointGoal1, SafetyPointPush1, SafetyPointGoal2.
- 5 independent training seeds per mode/task group.
- 30 checkpoints total.
- 250 deterministic paired reset seeds per checkpoint and filter variant.
- Strict ABER and ABER-LR use identical checkpoints and initial states;
  independently rerun filter-off trajectories are byte-equivalent at the
  decoded record level.
- Native Goal1 uses reset seeds `44000--44249`; the other 25 checkpoints use
  `45000--45249`.
- Confidence intervals use 50,000 seeded percentile-bootstrap resamples of
  the 30 checkpoint-level differences.  Episodes are not treated as
  independent training replicates.

Equal-weight 30-checkpoint summaries:

| Method | Success | Collision | SCR | Unsafe success | Timeout |
|---|---:|---:|---:|---:|---:|
| Filter off | 42.48% | 43.19% | 25.96% | 16.52% | 57.52% |
| Strict ABER | 19.92% | 7.11% | 19.88% | 0.04% | 80.08% |
| ABER-LR | 29.71% | 4.21% | 29.55% | 0.16% | 70.29% |

ABER-LR minus strict ABER:

- success: `+9.79 pp`, checkpoint bootstrap 95% CI `[+6.49,+13.37]`;
- collision: `-2.89 pp`, CI `[-5.13,-1.11]`;
- SCR: `+9.67 pp`, CI `[+6.37,+13.33]`;
- timeout: `-9.79 pp`, CI `[-13.40,-6.47]`;
- six of six group-mean SCR differences are positive;
- 28/30 checkpoint SCR differences are positive.

ABER-LR minus filter off:

- success: `-12.77 pp`, CI `[-18.32,-7.55]`;
- collision: `-38.97 pp`, CI `[-44.04,-34.41]`;
- SCR: `+3.59 pp`, CI `[+1.03,+6.37]`;
- unsafe success: `-16.36 pp`, CI `[-22.37,-10.71]`;
- timeout: `+12.77 pp`, CI `[+7.63,+18.28]`;
- SCR group means improve in 4/6 groups, not universally.

Strong Goal1-sized candidate-vs-strict gate passes in 3/6 groups:
`none/Goal1`, `none/Goal2`, and `sim_brt/Goal1`.

## Permitted main claim

> Across 30 checkpoints spanning two training modes and three Safety-Gym
> tasks, ABER-LR improves collision-free success over strict zero-action ABER
> in all six mode--task groups and in 28 of 30 checkpoint cells.  The
> checkpoint-level mean gain is 9.67 points, accompanied by a 2.89-point
> collision reduction and a 9.79-point timeout reduction.

## Forbidden or overbroad claims

- Do not claim ABER-LR raises raw success relative to filter-off execution.
- Do not claim SCR improves in every checkpoint or every group relative to
  filter-off execution.
- Do not claim the latched restore is certified or theorem-covered.
- Do not claim Push1 validates the static-obstacle theorem.
- Do not silently reuse the old statement that online ABER lowers success in
  every cell without distinguishing strict ABER from ABER-LR.

## Canonical artifacts

- `ABER_LATCHED_RESTORE_CROSS_TASK_MULTISEED_V1_PROTOCOL.json`
- `ABER_LATCHED_RESTORE_FULL_MATRIX_RESULTS.md`
- `safe-rl-2027/results/aaai27_aber_latched_restore_cross_task_v1/cross_task_audit.json`
- `safe-rl-2027/results/aaai27_aber_latched_restore_cross_task_v1/full_30_checkpoint_aggregate.json`
- `safe-rl-2027/results/aaai27_aber_strict_goal1_same_seed_v1/vs_candidate_audit.json`
- `safe-rl-2027/results/aaai27_aber_latched_restore_goal1_multiseed_v1/multiseed_audit.json`

# AAAI-27 Major Revision Tracker

Last updated: 2026-07-18

This is the authoritative tracker for the ABER submission. A claim is complete
only when the paper, implementation, raw result, and generated PDF agree.
Older VA-ATACOM audit documents are historical and are not authoritative.

## Central Contribution

ABER (Actuation-aware Braking-Envelope Recovery) is a **Safe RL hard shield**:
a sampled-data safety filter for the deployed normalized forward/turn
velocity-servo interface used by learned policies. It uses a braking reserve,
an analytic next-speed cap, and the executable recovery command `(0, 0)`.

The classical stopping distance `s^2/(2 a_b)` is not claimed as novel. The
Safe-RL contribution is the actuation-aware sampled-data projection, the
per-step multi-obstacle recovery certificate under an accepted servo envelope,
and the evaluation protocol that separates the shield from optional reward
shaping. See `SAFE_RL_FRAMING.md`.

## Scientific Integrity

- [x] The theorem is stated in the deployed forward/turn action space.
- [x] Normal projection and executable recovery are separate proof branches.
- [x] The algorithm closes `s <= v_max`; it is not an external assumption.
- [x] The proof covers any finite set of static obstacles via minimum
  clearance and a direction-independent recovery.
- [x] The implementation supports heterogeneous obstacle radii through
  `reset(..., obstacle_radii=...)`.
- [x] Safety-Gymnasium is labelled empirical; finite calibration is not called
  a global validation of the plant envelope.
- [x] The paper does not claim that published baselines are unsafe.
- [x] Native ABER and ABER with sim-BRT shaping are reported separately.
- [x] Safety cost, reward, goal success, clearance, occupancy, and penetration
  metrics are retained; the internal `C <= 5` convention is not a safety
  theorem.
- [x] Across-seed uncertainty is reported per task. No pooled Wilcoxon/Friedman
  superiority claim over heterogeneous task/seed cells remains.

## Current Evidence

- [x] Fixed theorem-envelope benchmark: 100 single-obstacle and 100
  multi-obstacle episodes per method, with fixed reference implementations.
- [x] ABER has zero certificate violations and 100% goal success in both
  certified benchmark scene families.
- [x] Seeded property audit: 5,000 cap checks and 26,354 normal/recovery
  projections with zero required violations; 5,000/5,000 expected
  counterexamples under a deliberately invalid braking calibration.
- [x] Component ablation: 295,200 raw records across four control periods,
  four actual/configured braking ratios, and one/four/eight obstacles.
- [x] In 1,200 calibrated navigation episodes, full ABER has zero certificate
  violations and collisions; removing the sampled term yields 1,200 violation
  episodes, and removing recovery yields 1,002.
- [x] The raw component-ablation gzip was reproduced twice with identical
  SHA-256 and independently re-aggregated from all records.
- [x] Safety-Gym matrix: 3 tasks x 2 reward modes x 5 seeds x 50 evaluation
  episodes, aggregated in `data/safetygym_aber_matrix.json`.
- [x] Every Safety-Gym matrix cell disables EKF/IMU and action-correction
  calibration and has a saved Hydra config.
- [x] Runtime scaling: 1,000,000 calls over 0--256 obstacles; positive-count
  median fit $R^2=0.99961$ and 256-obstacle p95 103.76 microseconds.
- [x] Strict paired matrix: 30 newly trained checkpoints across native/sim-BRT,
  three tasks, and five seeds; 7,500 matched pairs and 11,860,270 step rows.
- [x] Pooled collision changes from 44.97% off to 6.52% on, with 3,018
  off-collision/on-safe and 134 reverse pairs. Success decreases in every cell;
  these adverse outcomes are retained and the matrix is not theorem validation.
- [x] The supplement uses only the current certified, component, matrix,
  runtime, and paired evidence groups.

Historical VA-ATACOM, Cartesian tangent-cone, Webots, physical E-puck, and Car
results remain research diagnostics. They are not evidence for current ABER
claims unless rerun with a source snapshot of the current filter.

## Code and Reproduction

- [x] Public entry point: `safe_rl/filters/aber.py`.
- [x] Implementation: `safe_rl/filters/brake_manifold.py`; the filename and
  compatibility aliases remain for old configs/checkpoints.
- [x] Focused tests cover action-form rejection, forward clipping, retained
  turn, recovery, uncertainty inflation, heterogeneous radii, speed closure,
  and randomized one/two/eight-obstacle recursion.
- [x] Initial focused test result on 2026-07-16: 13 passed.
- [x] Component-ablation controls add four focused tests; combined focused
  result on 2026-07-17: 17 passed.
- [x] Certified, matrix, runtime, paired, and independent validation scripts
  are checked in.
- [x] Build the final anonymous code/data archive and include only the files
  listed in `METHOD_CODE_ALIGNMENT.md`.
- [x] Add an archive manifest with SHA-256 hashes for code, configs, and JSON
  artifacts.
- [x] Extract the final zip in a temporary directory and run its full
  one-command verifier successfully.
- [x] Independently verify all 30 newly trained checkpoint hashes, all 7,500
  paired episodes, 11,860,270 step rows, and the cross-seed aggregate.
- [x] Build, integrity-check, identity-scan, and fully verify the V2 anonymous
  archive containing the complete paired matrix and its 30 checkpoints.

## Submission PDFs

- [x] Main paper: 6 pages, US Letter, conclusion followed by references.
- [x] Supplement: 4 pages, US Letter, ABER-only method and evidence.
- [x] Reproducibility checklist: 2 pages, standalone.
- [x] Main/supplement/checklist logs have no overfull boxes, undefined
  references, fatal errors, or emergency stops.
- [x] Main and supplement fonts are embedded and contain no Type 3 fonts.
- [x] Anonymous author/affiliation settings are present.
- [ ] Upload the three PDFs and the anonymous code/data archive.

## Remaining Work After V1

The multi-checkpoint paired matrix, across-training-seed estimate, V2 archive,
and PDF rebuild checks are complete. Portal upload is the remaining submission
action. Do not add further empirical claims without a current-method rerun and
manifest.

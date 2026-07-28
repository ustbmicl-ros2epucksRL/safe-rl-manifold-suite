# AAAI-27 ABER: Safe RL / Safe ML Framing

Last updated: 2026-07-18

## Decision

**Venue push:** AAAI-27 with thickened Safe RL / Safe ML narrative (option B).
Do **not** import VA-ATACOM / Webots / hardware claims into this paper.

## Positioning (one sentence)

ABER is a **hard, per-step Safe-RL shield** on the deployed `(forward, turn)`
interface, with a multi-obstacle recursive certificate—complementing soft CMDP
methods that only control expected cost.

## Three contributions (paper language)

1. **Actuation-aware hard shield for learned policies** — analytic sampled-data
   projection + executable `(0,0)` recovery as training/deployment action filter.
2. **Per-step multi-obstacle safety certificate** — recursive feasibility under
   measurable servo envelope A1/A2 (stronger claim type than CMDP expected cost).
3. **Safe-RL evaluation protocol** — certificate benchmark vs CBF filters; PPO
   behind ABER on Safety-Gymnasium with **native shielding vs optional sim-BRT
   shaping** reported separately.

## What is / is not novel

| Claim | Status |
|-------|--------|
| \(s^2/(2a_b)\) braking distance | Classical — **not** claimed novel |
| Executable forward/turn projection + recovery certificate | Core contribution |
| Soft CMDP / Lagrangian Safe RL | Related work baseline, not our method |
| sim-BRT as HJ reachability | **Forbidden** — shaping ablation only |
| VA-ATACOM / physical E-puck / Webots | **Out of scope** for ABER AAAI |

## Safe RL vs control audience

- Emphasize: training-time shield, PPO + Safety-Gym, cost/reward/success/clearance.
- De-emphasize: “just another CBF for unicycles” without RL context.
- Honest: Safety-Gym is empirical (plant A1/A2 not global); certificate lives in
  the fixed differential-drive benchmark + theorem premises.

## Integrity rules

- Source of truth: `aaai27/main_v2_aaai27.tex` + audited JSONs in `data/`.
- Do not copy numbers from ICRA VA-ATACOM or old IROS distance-filter tables.
- Keep native vs sim-BRT split; never attribute shaping gains to the certificate.

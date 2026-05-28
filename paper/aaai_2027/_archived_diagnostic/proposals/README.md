# AAAI 2027 — Three candidate paper directions

This directory holds the design documents for three independent
proposals. Pick one to execute.

| Folder | One-line title | Novelty | Code reuse | Effort | Risk |
|---|---|---|---|---|---|
| `A_offline_critic/` | Provably safe policy learning via offline reachability-critic filtering | High | Medium | 4–6 weeks | Medium |
| `B_velocity_aware/` | Closing the sim-to-real gap with velocity-adaptive filtering and learned sensor noise | Medium | High | 3–4 weeks | Low |
| `C_distillation/` | Safety without runtime filtering: distilling safe constraints into the policy | High | Medium | 4–5 weeks | High |

Each subfolder has a `DESIGN.md` with the same nine-section template:

1. One-line summary
2. AAAI fit and positioning
3. Method (algorithm + pseudocode)
4. Theory (statements + proof sketches)
5. Code architecture (modules to add or rewrite)
6. Experiments (tables, baselines, seeds, budget)
7. Week-by-week timeline
8. Risks and mitigations
9. Deliverables and acceptance criteria

Once the user picks one, the others stay as deprecated archive.

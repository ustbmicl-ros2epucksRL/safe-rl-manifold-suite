# V1R-D3/C3 Conservative Measured Contract Results

## Decision

**V1R-C3 passed every pre-frozen held-out gate.**

The pass is finite evidence for the explicitly covered and guarded region
only.  It is not a global MuJoCo proof, does not certify Push recovery, and
does not authorize model fitting, interpolation, runtime projection, actor
training, or an ABER-FP theorem claim.

## Iteration lineage

The prior results remain unchanged:

- V1 rejected the original A1--A2 plant contract.
- V1R-C1 passed support gates but failed normal speed/path and recovery
  path/contact gates.
- V1R-D2 stopped before atlas derivation because its inherited integrity
  checker incorrectly required both forward and turn controls to be zero
  whenever `recovery_used=true`.  Its 274xxx data are diagnostic only; no C2
  protocol was produced and the reserved 275xxx seeds were never run.

D3 corrected the interface contract rather than modifying runtime behavior.
A labelled transition recovery must execute `forward=0`, while its recorded
turn command may be nonzero.  The separate five-second recovery witness still
executes the exact command `[0,0]`.

## Frozen D3 architecture

The source atlas retains the four required coordinate classes:

1. ABER recovery margin;
2. relative static-obstacle closing speed;
3. relative static-obstacle orientation;
4. previous/issued actuator-mode word.

The same radius-one continuous cover selects the supported operating region.
Unsupported queries have no nearest-center or interpolation fallback.

For covered normal queries, D3 takes the more conservative of:

- the local source-fiber expansion slack; and
- a global slack computed from the maximum positive development residual.

The frozen D3 development maxima and resulting slacks were:

| Quantity | Development maximum | Frozen slack/bound |
|---|---:|---:|
| Normal speed residual | 0.09977 m/s | 0.15165 m/s |
| Normal path residual | 0.001065 m | 0.001618 m |
| Five-second recovery path | 0.12449 m | 0.19674 m |

The recovery claim is additionally restricted to Goal and MultiGoal sources
with planar source speed no greater than 0.30 m/s.  Push remains in all normal
audits, but Push recovery is outside the claim because static-obstacle
clearance does not certify movable-box or internal contact topology.

The D3 protocol was frozen before discovery with SHA-256
`09c3e93848de9e8155bb1396a2ed62aae3000f5199938739a64736a639d9f063`.
Its collector, runner, trace code, atlas code, forward/turn checker,
derivation code, and independent auditor were all hash-frozen.

## Data

### Discovery

Discovery used reset seeds 276000--276009 for each of 30 checkpoints:

- 243,820 unmodified transition records;
- 300 state-restored five-second macro probes;
- 110,375 eligible normal development transitions;
- 127 in-scope low-speed non-Push recovery development probes.

The atlas contained:

| Section | Centers | Active centers |
|---|---:|---:|
| Normal | 4,411 | 1,119 |
| Recovery | 41 | 24 |

The frozen atlas SHA-256 is
`02fa725dd48fab3ef3718f99421127f5e4e8255a1dd11d82a9188306eded6262`.
The derived C3 protocol was written before any confirmation execution and has
SHA-256
`5d5607ae89df2ffff29c54f447f279351088b6bb485611aa4ebfa58abc5706f0`.

### Held-out confirmation

Confirmation used disjoint reset seeds 277000--277019 for every checkpoint:

- 490,316 transition records;
- 600 macro probes;
- all 30 cell manifests present and hash-consistent.

The independent audit result SHA-256 is
`511520079a5e337585f3c6e37399fd4d43cb7f72f2387937134d30f2a1c0fcb9`.

## Held-out normal result

There were 220,912 eligible certified-start, contact-free normal
transitions.  The atlas covered 177,413, or **80.31%**, passing the frozen 80%
overall gate.

Every training-mode/task group passed its 60% gate:

| Group | Covered / eligible | Coverage |
|---|---:|---:|
| None / Goal | 16,776 / 19,384 | 86.55% |
| None / Push | 39,947 / 50,668 | 78.84% |
| None / MultiGoal | 21,874 / 25,848 | 84.63% |
| Sim-BRT / Goal | 30,028 / 37,464 | 80.15% |
| Sim-BRT / Push | 33,510 / 48,032 | 69.77% |
| Sim-BRT / MultiGoal | 35,278 / 39,516 | 89.28% |

Both zero-violation gates passed:

- speed violations: **0 / 177,413**;
- path violations: **0 / 177,413**.

The closest held-out speed observation remained 0.05596 m/s below its bound.
The closest path observation remained 0.000389 m below its bound.  These
nontrivial gaps also show that the global fallback is conservative; C3
establishes safety coverage, not tightness or optimal efficiency.

## Held-out recovery result

Of the 600 scheduled probes, 259 were inside the pre-frozen non-Push,
source-speed operating region.  The recovery atlas covered 188 of these and
41 passed the atlas-dependent static-clearance guard.

The accepted fraction was **41 / 259 = 15.83%**, exceeding the pre-frozen 5%
gate.  Every in-scope training-mode/task group had accepted support:

| Group | Accepted / in-scope | Accepted fraction |
|---|---:|---:|
| None / Goal | 4 / 54 | 7.41% |
| None / MultiGoal | 9 / 59 | 15.25% |
| Sim-BRT / Goal | 13 / 68 | 19.12% |
| Sim-BRT / MultiGoal | 15 / 78 | 19.23% |

For all 41 accepted probes:

- path-bound violations: **0**;
- failed settling tests: **0**;
- non-ground contacts: **0**;
- negative static clearances: **0**.

The maximum observed five-second path was 0.10670 m under the fixed 0.19674 m
bound, leaving at least 0.09004 m path slack.  The result is therefore safe
but deliberately conservative.

## Forward/turn interface evidence

A post-decision diagnostic, which cannot change the C3 decision, counted:

- 224,812 held-out transition recoveries;
- 209,365 with nonzero turn control;
- 20,454 with turn at an actuator boundary;
- **0 with nonzero forward control**.

Thus C3 exercised the actual steering recovery interface.  It did not obtain
the pass by silently treating all transition recoveries as `[0,0]`.  The
exact-zero five-second macro remains a separate executable witness.

The interface diagnostic SHA-256 is
`fbdea707eefd2187b4828e965594064a593db48584328a232aee5bae806b11d7`.

## Claim suitable for later paper integration

The supported statement is:

> Across 30 fixed Safety-Gym checkpoints and disjoint held-out reset seeds,
> a pre-frozen continuous source atlas selected an 80.31%-coverage normal
> operating region.  A global development-tail fallback produced zero
> held-out normal speed/path exceedances, while a low-speed non-Push recovery
> region supplied 41 guarded exact-zero macro witnesses with zero path,
> settling, contact, or static-clearance failures.  The audit also exercised
> 209,365 nonzero-turn transition recoveries with zero forward-command
> violations.

This statement must retain the checkpoint, simulator, coverage, task, speed,
and finite-sample qualifications.

## Artifacts

- D3 protocol: `ABER_SAFETYGYM_CONTRACT_V1R_D3_PROTOCOL.json`
- C3 protocol: `ABER_SAFETYGYM_CONTRACT_V1R_C3_PROTOCOL.json`
- Atlas: `data/aber_safetygym_contract_v1r_d3_atlas.json`
- Independent audit: `data/aber_safetygym_contract_v1r_c3.json`
- Interface diagnostic:
  `data/aber_safetygym_contract_v1r_c3_interface_diagnostic.json`
- Figure: `figures/aber_v1r_c3_heldout_audit.pdf` and `.png`
- D2 failure record:
  `ABER_SAFETYGYM_CONTRACT_V1R_D2_DISCOVERY_FAILURE.md`

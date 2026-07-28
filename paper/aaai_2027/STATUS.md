# AAAI-27 ABER Status

Last updated: 2026-07-27

> **Current AAAI submission snapshot (2026-07-27):** the paper method is
> **ABER + ABER-LR**.  Strict ABER is the certified inner layer; ABER-LR is an
> explicitly empirical liveness wrapper and is outside the theorem.  The
> primary empirical claim is safety--completion indicator coordination using
> collision-free success (SCR), with raw success, collision, unsafe success,
> timeout, and recovery tails reported separately.  The complete frozen study
> covers 30 checkpoints (two training modes, three Safety-Gym tasks, five
> training seeds), 250 paired episodes per checkpoint per filtered variant,
> and 15,000 filtered paired rollouts.  ABER-LR versus strict ABER improves SCR
> by 9.67 points (95\% CI [6.37,13.33]), reduces collision by 2.89 points, and
> reduces timeout by 9.79 points; all 6 group means and 28/30 checkpoint cells
> improve SCR.  Main paper: 8 PDF pages (7 content + references).  Supplement:
> 10 PDF pages after the failure-landscape figure.  Canonical source of truth:
> `AAAI_ABER_CANONICAL_FACTS.md`.  Earlier successor-method logs below are
> research history and do not define the current AAAI submission.

> **Successor-method update (2026-07-23):** the direct-policy SCF-MF design has
> been stopped. The current line is **SCF-BF**, an independent post-policy
> behavior filter from nominal `(forward, turn)` to an action on a certified
> finite union, followed by executable ABER recovery. No fiber actor, latent
> sampler, policy density, PPO ratio, or imitation module belongs to the method.
> B0 is frozen in `SCF_BF_METHOD_AND_GATES.md`. Corrected S3 still supplies two
> separated continuous rectangles of area 0.26, and the new exact rational S3R
> gate proves that every action in both rectangles returns every plant in the
> full parameter box to the ABER recovery invariant set. **S3R PASS**; current
> recovery margin is 0.18 and the worst squared terminal margin is 0.2572009.
> **S2 PASS:** 320 five-stratum local models were fitted, continuous interval
> B&B scored 100 calibration and 100 test plants, empirical test coverage was
> 99/100, maximum gap was 0.19999945, no plant hit the 200,000-node limit, and
> all 320 physical local bands were below `eta=0.1`. The subsequent frozen
> learned-band S3R recompilation **FAILED**: one-step safety, terminal recovery,
> the full continuous plant-box residual enclosure, and the original
> positive-area geometry passed, but the worst robust-contraction LHS became
> 0.1347165 after adding the 0.0390033 band, exceeding `eta=0.1`. Therefore the
> original rectangles are not learned-certified fibers. B4 finite-union
> projection, B5 fail-closed recursion, and B6 synthetic falsification remain
> blocked, so the runtime filter and all new actor code remain forbidden. See
> `SCF_BF_S3R_RESULTS.md`, `SCF_BF_LOCAL_S2_RESULTS.md`, and
> `SCF_BF_LEARNED_S3R_RESULTS.md`.
>
> A frozen follow-up diagnostic kept the same `q`, `eta`, and 320-model
> artifact and scanned 1,419 recoverable centered states with 20,301 actions
> each. Although 983 states retained at least one learned-certified action,
> no state had qualified separated negative-turn and positive-turn components
> at any preregistered slack floor. The affine-in-turn local defect compiler
> does not preserve the even cosine topology responsible for exact S3R's
> disconnected fibers. **Stop this compiler; do not densify the same scan.**
> A successor must first freeze a physics-structured servo-response band or a
> turn-nonlinear local model. See
> `SCF_BF_LEARNED_S3R_DIAGNOSTIC_RESULTS.md`.
>
> The physics-structured successor has now been tested. Its 20-model
> next-speed response S2R gate **PASSED** with 100/100 test-plant coverage,
> `q=3.33036`, and maximum response band 0.020243. Exact cosine compilation
> restored even turn curvature. However, both initial PS3-D grid candidates
> were falsified by unsampled bridges at common-stratum boundaries. A separate
> boundary-complete V2 evaluated 1,419 recoverable states and 40,401 actions
> per state and found zero qualified opposite-turn components at every frozen
> slack floor. **PS3-D V2 NO CANDIDATE; do not run PS3-C or B4.** The next
> method decision is connected filtering versus a preregistered
> mechanism-boundary guard/hierarchy. See `SCF_BF_RESPONSE_S2R_RESULTS.md` and
> `SCF_BF_RESPONSE_PS3D_RESULTS.md`.
>
> The mechanism-boundary hierarchy was subsequently preregistered and tested.
> Pure actuator strata were reserved for the contraction union, ambiguity
> strata for recoverable-safe correction, and mode/provenance uncertainty for
> fail-closed ABER. The continuous rational PS3-C certificate passed S2R/q
> integrity, every frozen rectangle's full continuous action/omega enclosure, branch provenance,
> one-step safety, terminal recovery, positive area, and pairwise separation.
> It nevertheless **FAILED contraction**: the frozen `track_center` component
> had margin `-0.0001197501`, below `rho=0.0005` (the two brake components had
> margin `0.0005718444`). The independent B5 recursion checker **PASSED** all
> 2,048 Boolean routing cases with zero violations, including Uc-over-Ur
> priority, provenance, and mode-boundary fail-closed rules. Because the entry
> condition is `PS3-C PASS AND B5 PASS`, **B4 remains blocked**. Do not change
> the frozen candidate, rectangles, rho, or guard to overwrite this failure;
> any next diagnostic must be separately frozen and search for greater
> continuous contraction slack. Runtime filtering and actor code remain
> forbidden. See `SCF_BF_BRANCH_HIERARCHY_RESULTS.md`.
>
> A separately preregistered greater-slack diagnostic has now completed
> without changing `q`, `eta`, `rho`, the mode guard, or the five semantic
> component templates. D1 evaluated 30,339 recovery-eligible states from
> 35,186 frozen centered point states using continuous rational action/response/
> omega certificates. It found 19,708 states above the stronger `0.005`
> contraction and terminal gates. The preregistered best state,
> `relative=(0.645,0), speed=0.155`, retained `track_center`; its minimum Uc
> contraction margin was `0.0974167` and minimum terminal margin was
> `0.2995069`. A separately frozen PS3-C-v2 confirmation then passed all ten
> identity and continuous-certificate checks, while preserving the old H0
> FAIL. Together with the existing B5 PASS, this now permits starting only the
> offline B4 finite-union projection checker. Runtime filtering and actor code
> remain forbidden. The two low-speed brake rectangles have positive but small
> area `0.009` each, so projection quality and deployed-domain coverage remained
> unproved at that gate.
> See `SCF_BF_BRANCH_SLACK_D1_RESULTS.md`.
>
> The separately frozen offline B4 checker has now **PASSED** all 13 exact
> finite-union projection checks: exact KKT component solutions, global
> enumeration, deterministic ties, boundary cases, convex-hull and wrong-hint
> traps, provenance, empty sets and solver failure. B6 then **PASSED** all 12
> preregistered synthetic falsifications, including 100x response-band
> inflation, hierarchy priority faults and projection relaxations. Thus the
> local offline mechanism chain S2R/PS3-C-v2/B4/B5/B6 is complete and only a
> separately frozen B7 runtime implementation may start. Actor and policy
> training remain forbidden. This does **not** yet establish a full paper
> contribution: the proof is point-state/local, runtime B7 and paired B8 are
> absent, and a targeted literature audit supports only a narrow combination
> claim rather than any broad first claim. See
> `SCF_BF_BRANCH_B4_B6_RESULTS.md` and `SCF_BF_CONTRIBUTION_AUDIT.md`.
>
> The separately frozen C1 state-box gate has now **PASSED** without any
> state/action grid. It continuously certifies
> `rx in [0.5,0.8]`, `ry in [-0.05,0.05]`, and
> `speed in [0.15,0.16]` (volume `0.0003`) together with all five
> state-dependent action fibers, response bands and the omega interval. The
> worst Uc contraction margin is `0.090886`, the worst terminal margin is
> `0.123472`, and the entire-box current-recovery squared margin is `0.179244`.
> Thus the mechanism is no longer point-state-only, but it remains a local
> state-domain result rather than a full navigation-domain theorem. A B7
> runtime implementation may be separately frozen only for this certified box
> with fail-closed ABER outside it; actor code remains forbidden. See
> `SCF_BF_BRANCH_STATEBOX_C1_RESULTS.md`.
>
> The separately frozen B7 certified-box runtime filter has now **PASSED** all
> 17 formal checks. It validates 14 C1/B4 provenance artifacts, enables exact
> finite-union projection only inside the C1 box, routes pure strata only to
> `Uc` and ambiguity strata only to `Ur`, and sends box/cell/mode/provenance,
> empty-set, invalid-nominal and solver failures to an explicitly injected
> ABER callback. Every failed projection has no projection action; an invalid
> ABER action rejects the call without nominal or silent-zero fallback.
> Runtime `dt` is locked to the C1 value `0.1`; timestep mismatch and malformed
> boolean flags also enter ABER. The focused B7 suite has 21 passing tests and
> the full SCF suite has 88 passing tests. Actor and training interfaces remain
> absent. This PASS is local only:
> cross-response-cell recursion, multi-obstacle aggregation, the full
> navigation domain and paired B8 experiments remain unproved. See
> `SCF_BF_RUNTIME_B7_RESULTS.md`.
>
> The separately frozen C2 and B7-XC gates have now both **PASSED** for the
> first learned response-cell boundary at `speed=0.25`. C2 continuously
> certifies `rx in [0.5,0.8]`, `ry in [-0.05,0.05]`,
> `speed in [0.24,0.26]` (volume `0.0006`) using both closed adjacent-cell
> proofs at the boundary. The minimum Uc contraction margin is `0.074986` and
> the minimum terminal margin is `0.083850`. B7-XC records cell `[0]`, dual
> `[0,1]`, or cell `[1]` provenance on the left, boundary, and right,
> respectively; missing/invalid provenance and solver failure enter ABER with
> no projection action. Its 13 formal checks pass. The ten new focused tests
> pass and the full SCF suite is now `98 passed`. The `0.5/0.75` boundaries,
> multi-obstacle aggregation, full navigation domain, actor and training
> remain unauthorized. See `SCF_BF_CROSS_CELL_C2_B7_XC_RESULTS.md`.
>
> The first frozen real-policy composition gate B8-P0 has now completed:
> **NO_GO**. Five existing deterministic Goal checkpoints were evaluated in
> 100 strict paired `filter_off` / `aber_only` / `SCF-BF+ABER` triplets,
> producing 31,106 raw supervisory-step records with matching artifacts and
> identical paired initial snapshots. Across 12,277 combination steps there
> were only five SCF attempts and zero certified projections; therefore the
> online response-band and post-ABER-change metrics had no denominator.
> Every combination episode outcome was exactly identical to ABER-only. This
> is fail-closed fallback, not an empirical PASS. An independent verifier
> recomputed the hashes, seed coverage, traces and aggregates. P0 is frozen
> and cannot be overwritten. Any successor must separately preregister
> actuator saturation/domain ordering and occupancy-aligned
> discovery/held-out confirmation before another real paired gate. Full B8,
> paper claims, actor training and training-time integration remain
> unauthorized. See `SCF_BF_B8_P0_RESULTS.md`.
>
> The separately frozen B8-D1 real-composition diagnostic then fixed only the
> actuator ordering: raw policy mean, exact `[-1,1]^2` Safety-Gym saturation,
> SCF forward-domain guard, and SCF/ABER execution. It reproduced all 100 P0
> initial snapshots and logged 12,399 steps. Saturation changed 6,040 raw
> actions and 8,188 negative-forward actions went directly to ABER. Only two
> SCF attempts remained and only one projected. That projection's learned
> next-speed interval was `[0.0756874388,0.0756874395]`, while the observed
> `0.1 s` next speed was `0.1415105563`; online coverage was 0/1. Relative to
> frozen P0 ABER, success fell by 0.01 and recovery count rose by 133.
> **B8-D1 DIAGNOSTIC_NO_GO.** Independent raw-step replay verified the result.
> The synthetic speed-setpoint S2R plant does not match Safety-Gym's
> force-limited MuJoCo velocity actuator, so the current C1/C2
> real-composition branch is stopped. A successor must first freeze and pass a
> real-actuator response identification/falsification gate, then recompile
> the continuous certificates and return to an unseen-seed real-composition
> gate. Local PASS counts cannot override P0/D1. Actor training, full B8 and
> paper claims remain unauthorized. See `SCF_BF_B8_D1_RESULTS.md`.
>
> The frozen SG-S2F gate has now directly tested the real Safety-Gym Point
> actuator in 17,640 controlled, obstacle-contact-free `0.1 s` transitions.
> Injection error was at most `2.22e-16`, deterministic repeat error was zero,
> and no non-floor contact occurred. Nevertheless scalar `(speed,forward)` was
> decisively insufficient: the maximum conditional next-speed span was
> `0.54755` and the scalar calibration band was `0.31332`, versus the frozen
> `0.025` limit. Held-out coverage `0.99093` does not rescue that unusably wide
> band. Paired effects require body-relative velocity direction, turn and yaw
> rate; absolute heading was below the `0.005` threshold. **SG-S2F
> SCALAR_RESPONSE_STATE_INSUFFICIENT.** Only a separately frozen augmented
> response diagnostic using longitudinal/lateral velocity, yaw rate, forward
> and turn may start. Certificate recompilation, actor training, full B8 and
> paper claims remain unauthorized. See `SCF_BF_SG_S2F_RESULTS.md`.
>
> The separately frozen SG-S2A diagnostic has now fitted and tested the
> augmented five-dimensional real-actuator response
> `(v_longitudinal,v_lateral,yaw_rate,forward,turn)`. A degree-6,
> 462-term Chebyshev model was selected without test access and calibrated on
> 100 complete nuisance configurations. Test configuration and row coverage
> were both 1.0, but only with `q=0.059015`, 2.36 times the frozen `0.025`
> tight-band limit; test maximum residual `0.059015` also exceeded `0.05`.
> **SG-S2A AUGMENTED_RESPONSE_TIGHT_BAND_FAIL.** Independent artifact replay
> reproduced the split, band, metrics, and decision. Therefore no certificate
> recompilation, runtime projection, actor/training, real-composition
> experiment, or paper claim is authorized. The requested projection,
> ABER-noninferiority, unseen-seed, and strong-shield comparisons remain
> NOT_RUN because their real-actuator prerequisite failed. A successor must
> freeze a new response mechanism and use fresh validation configurations;
> it cannot widen SG-S2A or overwrite this FAIL. See
> `SCF_BF_SG_S2A_RESULTS.md`.
>
> The separately frozen SG-S2B successor has also completed. It collected
> 94,041 real Point-actuator grid transitions plus disjoint fresh continuous
> calibration/test sets of 6,400 transitions each, and replaced the global
> polynomial by 65,536 periodic local multilinear response cells. Integrity
> passed, but configuration calibration gave `q=0.091348`; test maximum
> residual was `0.086077`, and no cell had the same five-step two-actuator
> force-mode trace at all 32 vertices. Thus **SG-S2B
> LOCAL_RESPONSE_CELL_TIGHT_BAND_FAIL**. Independent raw replay reproduced
> every query, interpolation residual, calibration score, provenance cell,
> and gate. Fixed-grid refinement, post-hoc purity changes, external Goal1
> validation, certificate compilation, runtime projection, actor/training,
> combination baselines, and paper claims are unauthorized. A successor must
> model the exact hybrid actuator saturation law, recursively isolate
> force-mode-stable boxes, learn only a residual response band, and use fresh
> validation indices. See `SCF_BF_SG_S2B_RESULTS.md`.
>
> The frozen SG-S2C exact-saturation ReLU response diagnostic has now
> completed on 163,840 entirely new real-actuator transitions. Exact
> `clip(control-0.3*actuator_velocity,-0.05,0.05)` features plus
> mode-boundary-focused development data reduced test RMSE to `0.006104`,
> p99 residual to `0.018180`, and maximum residual to `0.034264`, passing the
> frozen `0.05` maximum. Nevertheless configuration-level calibration gave
> `q=0.043237`, above `0.025`, and the rarest initial actuator stratum had
> only 23 test rows versus the frozen 100. **SG-S2C
> EXACT_SATURATION_RELU_TIGHT_BAND_FAIL.** Independent NumPy inference and
> raw replay reproduced all queries, modes, predictions, conformal scores,
> hashes, and gates. No interval recursion, Goal1 validation, certificate,
> runtime projection, actor/policy training, combination experiment, or
> paper claim is authorized. A successor must be a separately frozen
> mode-conditioned hierarchy with balanced fresh evidence per exact actuator
> stratum and dual-model/fail-closed handling at clip boundaries. See
> `SCF_BF_SG_S2C_RESULTS.md`.
>
> The separately frozen SG-S2D nine-stratum hierarchy has completed on
> 208,128 new real-actuator transitions, with balanced development,
> calibration and test evidence for every exact initial actuator mode and
> dual-model evaluation on all 12 clip interfaces. Integrity and constructive
> mode assignment passed. Nevertheless all nine local `q` values exceeded
> `0.025` (best `0.034185`, worst `1.094569`); modes 3 and 8 had dead
> nonnegative outputs under their frozen seeds, mode 4 test configuration
> coverage was `0.94`, maximum test error was `1.097965`, and maximum
> boundary dual error was `1.084257`. **SG-S2D
> NINE_STRATUM_LOCAL_RESPONSE_FAIL.** Independent NumPy replay reproduced all
> nine bands, test metrics and boundary checks. Removing dead modes or
> retraining selected seeds cannot rescue the seven remaining tight-band
> failures. The current learned next-speed response branch is stopped; no
> continuous recursion, Goal1 validation, certificate, runtime projection,
> actor/policy training, combination experiment, or paper claim is
> authorized. See `SCF_BF_SG_S2D_RESULTS.md`.

> **Latest FN-MF update:** frozen P6d added the remaining obstacle coordinates
> with zero mismatch in all 36,340 inherited rows. Held-out RMSE was 0.0867
> (P6b 0.0869), still failing 0.05; success fell from 0.561 to 0.519, still
> failing 0.80. All five seeds again had zero collision/function/support
> violation. Diagnosis is
> **FULL_OBSTACLE_SET_INSUFFICIENT_STOP_FIXED_MLP_DIRECT_IMITATION**. The
> preregistered stop rule is active: do not run P6c, P7, further observation or
> capacity patches, the AAAI matrix or e-puck. P0--P3 certificate/fiber evidence
> remains valid, but FN-MF has no validated learning contribution and is not the
> AAAI main method. Any successor now requires a new learning mechanism plus
> exact novelty/math definition before code. See FN_MF_FULL_OBSTACLE_P6D_RESULTS.md.

> **Research-direction update:** both GTR-MF variants failed their learning gates; stop
> the temporal-transport branch. PR-MF passed two 1,000-sample numerical P0 runs, but its
> completed five-seed paired learning P0 is **NO-GO**: frozen success/return, safety,
> exact retraction, chart/fallback and rank passed, while reward AUC non-inferiority
> failed (PR-MF 27.69 vs ambient 32.45 and projector-only 31.65). Do not start the full
> AAAI matrix or claim the method is validated. The chart-conditioned projector measure
> is implemented and mathematically checked, but its independent novelty is not yet
> searched. The fixed-draw inverse-CDF implementation passed 19 mathematical/code tests,
> but its preregistered 20-seed confirmation **FAILED**: AUC delta vs projector-only
> -6.145 (95% CI [-9.341, -2.949]), frozen return delta -8.185, and PR-MF success 0.74.
> Safety cost, exact residual, fallback and rank passed, but cannot rescue failed learning.
> **Stop PR-MF as an AAAI main method; retain it only as an analysis baseline.** ABER and
> LC-ABER also remain baselines. The next main method must change mechanism while it may
> remain manifold-filter based. See PR_MF_FIXED_DRAW_RESULTS.md,
> PR_MF_LEARNING_P0_RESULTS.md, PR_MF_P0_MATH_PROOFS.md, and PR_MF_NOVELTY_AND_MATH.md.

> **Exact-rollout provenance update (2026-07-24):** ER-P0 and ER-P0R exposed
> two contact-comparator defects and remain frozen FAIL. ER-P0R2 then exposed
> a real deficiency in integration-field-only cloning: at trial 3,976,
> 50-step replay diverged by `3.07e-4` in qpos, `8.07e-2` in qvel and `6.63`
> in qacc despite matching contact topology, cost and terminal ABER decision.
> The separately frozen ER-P0C replaced that mechanism with the MuJoCo 2.3.3
> binding's native complete `MjData.__copy__` and passed all 10,000 unseen
> Goal1 trials with zero complete-state, contact and cost error and 10,000
> matching ABER decisions. Independent artifact verification passes. However,
> the formal run took 2,172.9 s; a timing diagnostic measured about 201.7 ms
> for one 50-step clone and 27.2 s for 135 sequential perturbation rollouts.
> Thus clone provenance is established but online feasibility, frozen
> mass/friction/force/sensing robustness, projection occupancy, paired
> noninferiority and paper claims are not. Only a separately frozen,
> hard-timeout ER-P1 feasibility/latency diagnostic is authorized. Timeout,
> invalid provenance, state-copy mismatch, contact ambiguity, solver failure,
> empty set and box-exit must invoke ABER. Actor and training interfaces
> remain absent. See `SCF_BF_ER_P0C_RESULTS.md`.

> **Manifold-filter continuation:** the reusable mechanism is now limited to
> exact finite-union projection, `Uc/Ur` provenance hierarchy, cross-cell
> recursion and executable ABER fallback. Real projections may no longer use
> the failed scalar next-speed artifacts. The replacement object is a joint
> set-valued full successor-state and 50-native-step actuator/contact
> transition model, calibrated at plant/configuration level and compiled by
> continuous interval B&B. Native complete-state rollout is the offline oracle
> unless a separately frozen preallocated C/C++ latency gate passes. See
> `SCF_BF_MANIFOLD_FILTER_CONTINUATION_2026-07-24.md`.

> **MF-L0 update:** the preallocated native MuJoCo `mj_copyData` path retained
> exact provenance on 100/100 unseen Goal1 trials with zero state, contact,
> cost and ABER error. Copy+50-step p95 fell from the Python full-copy scale
> to `15.180 ms`, but the required 135-rollout batch had p95
> `2,026.880 ms`; only 5/135 scenarios completed inside the frozen 80 ms
> oracle budget. One slot increased resident memory by about 391.7 MB.
> Therefore **MF-L0 OFFLINE_ORACLE_ONLY**: stop sequential online exact
> rollout and use complete rollout only for MF-T1 transition-set data and
> audits. The frozen artifact omitted individual latency rows, so the
> independent verifier reproduces identity and aggregate gate logic but not
> raw percentile calculation; future latency protocols must persist every
> timing sample. Runtime projection, robustness, actor/training and paper
> claims remain unauthorized. See `SCF_BF_MF_L0_RESULTS.md`.

> **MF-T1 collector pilot:** the offline oracle recorded all 50 native steps
> for the full 135 mass/inertia, friction, force-limit and sensing product.
> All 135 scenarios and 6,750 trace rows passed schema/finite/source-isolation
> checks, and independent raw replay passes. One source state/action produced
> 27 distinct 50-step actuator-mode traces, confirming that initial mode and
> scalar next-speed are invalid provenance summaries. However, the pilot had
> one contact trace, zero cost-positive scenarios and 135 recoverable
> terminals. It authorizes only a separately frozen randomized plant-level
> MF-T1 dataset protocol that deliberately covers contact/cost/recovery
> boundaries. Model fitting, runtime projection, actor/training and paper
> claims remain unauthorized. See `SCF_BF_MF_T1_PILOT_RESULTS.md`.

> **MF-V0 engineering update:** the occupancy-aligned exact-oracle V1 chain
> has passed its engineering gate on one live Goal1 policy state. The frozen
> run evaluated 281 unique actions through 843 complete plant rollouts and
> retained 42,150 native-step rows; the live source complete-state hash was
> unchanged and independent raw replay passed. The selected state was far
> inside the ABER recovery set: empirical `F_r` inner area was `1.96875/2`,
> mixed area was `0.03125`, `F_c` area was zero, and no leaf cell had stable
> complete actuator/contact provenance. The nominal action already lay in
> `F_r`, so projection displacement was zero. This validates the collector,
> adaptive scanner, robust aggregation and artifact pipeline, but does not
> establish useful or disconnected fiber viability. V1 also counted ordinary
> floor contacts as `contact_active`; that tag did not select the state because
> all contact quotas were zero, but formal discovery must exclude the floor
> geom. Only a separately frozen five-checkpoint, event-balanced MF-V0
> discovery/held-out confirmation may start. Model fitting, runtime projection,
> actor/training and paper claims remain unauthorized. See
> `SCF_BF_MF_V0_ENGINEERING_RESULTS.md`.

> **MF-V0 formal discovery/confirmation update:** the five-checkpoint census
> found recovery-boundary and actuator-boundary occupancy in every Goal1
> checkpoint, but zero true non-floor contacts in 3,032 states; floor support
> contact was excluded and never used as a proxy. The deployed-geometry oracle
> was corrected to record Cartesian `task.agent.pos/vel` under each complete
> clone. MF-V0 then scanned 10 balanced sources and failed frozen held-out
> confirmation: recovery support was 3/5 versus the required 4/5, while
> actuator support was 5/5. MF-V0R expanded the scanner to the complete
> deployed action square and, on new discovery seeds, evaluated 6,110 actions,
> 18,330 complete nuisance trajectories and 916,500 native-step rows. Its
> strict 5/5 discovery corridor had seven cells (`forward [0,0.125]`, `turn
> [-1,0.75]`, area 0.21875), but a third untouched seed family reproduced only
> 4/5 support in both strata. Independent raw verification passed and confirms
> `VERIFIED_MF_V0R_CONFIRMATION_FAIL`. A source-independent action corridor is
> therefore rejected at the frozen thresholds. The next object must be a
> source-conditioned fiber; under current authorization only physics-feature
> collection/design may proceed. Model fitting, runtime projection,
> actor/training and paper claims remain unauthorized. See
> `SCF_BF_MF_V0_FORMAL_RESULTS.md`.

> **Source-conditioned MF cycle update:** a source-only census froze six
> descriptive anchors over recovery margin, signed closing speed, heading and
> signed actuator modes. Fixed discovery selected 30 sources and evaluated
> 1,350 actions, 4,050 nuisance scenarios and 202,500 native rows; it passed
> its source-sensitivity gate. The resulting 80 exact finite contrasts then
> failed untouched-reset-seed confirmation despite clean structural checks:
> both actuator anchors reproduced 4/5 positive support and the strict
> recovery joint anchor reproduced 0/5, but the marginal
> `margin_03_05 + closing_mid` negative anchor reproduced 2/5 instead of at
> most 1/5. A failure audit found that this marginal anchor mixed opposite
> heading phases. A subsequent 5,682-state fresh-seed source-only census found
> no strict four-dimensional phase target with nonzero occupancy in all five
> checkpoints, so further exact-bin action probing is stopped. The next design
> gate is a finite continuous source atlas with predeclared coverage, not finer
> common bins. Model fitting, runtime projection, actor training and paper
> claims remain unauthorized. See `SCF_BF_MF_SC_RESULTS.md`.

> **Continuous source-atlas update (2026-07-25):** the 5,682-state fresh
> source census has now been converted into a frozen finite atlas without
> common-bin matching or outcome-driven selection. The equal-family metric
> retains recovery margin, signed closing speed, hazard-relative heading and
> exact two-actuator mode; deterministic farthest-first coverage selected 90
> unique sources, balanced as six actuator-boundary and twelve
> recovery-boundary sources per checkpoint. All structural coverage gates
> passed. The unchanged 15-action exact probe produced 1,350 decisions, 4,050
> nuisance scenarios and 202,500 native rows. A separately frozen source-only
> selector then formed 57 same-checkpoint/same-role local satellite pairs at
> distance at most 0.25. Their paired audit produced 855 decisions, 2,565
> nuisance scenarios and 128,250 native rows with all integrity checks passing.
> Thirty-seven paired signatures were identical and 20 differed; median
> Hamming distance was zero, maximum was 0.8, and even the distance-at-most
> 0.05 band contained two differing pairs. Thus the continuous atlas is
> implemented and reproducible as a finite source-selection mechanism, but
> its distance does not authorize interpolation and the four necessary
> coordinates are not shown sufficient for a continuous outcome chart. Only a
> separately frozen source-only discontinuity audit may follow; model fitting,
> runtime projection, actor training and paper claims remain unauthorized. See
> `SCF_BF_MF_CONTINUOUS_ATLAS_RESULTS.md`.

> **MF source-discontinuity audit update (2026-07-25):** the separately frozen
> source-only audit compared the 20 nonidentical with the 37 identical local
> atlas pairs using ten continuous physical differences and four observed
> hybrid-boundary mismatches. It issued no new action query and fit no model.
> All 57 pairs, the frozen 20/37 labels, finite fields, input hashes, and raw
> artifact checks passed. No candidate met the frozen nomination rule. The
> closest was continuous actuator-force distance: all five checkpoints had
> the expected direction, but AUC was 0.6662 versus the frozen 0.70. Actuator
> velocity-sign mismatch prevalence ratio was only 1.233, bearing chirality
> 0.925, request clip regime 0.514, and non-floor contact topology never
> differed. Several physical pair differences were smaller, not larger, in
> the discontinuous group. Therefore no fifth coordinate or hybrid flag is
> promoted. The next gate may only audit zero crossings of already queried
> continuous recovery/contraction/safety margins on the same fixed exact
> actions; it may not add actions, interpolate sources, fit a boundary, change
> atlas weights, build runtime projection, train an actor, or authorize a
> paper claim. See `SCF_BF_MF_SOURCE_DISCONTINUITY_AUDIT_RESULTS.md`.

> **MF outcome-margin crossing update (2026-07-25):** a separately frozen
> audit reused only the existing 15 exact actions at the 20 nonidentical local
> source pairs. It recomputed 300 pair-actions and 98 binary changes with all
> query identities, Hamming counts, thresholds, input hashes, and raw records
> passing. Every binary change crossed the terminal ABER recovery margin:
> 98/98 across all five checkpoints and both atlas roles. Safety status and
> physical-clearance crossings were both 0/98. Contraction also crossed in
> 18/98, while actuator/contact provenance changed in 73/98, below the frozen
> 80% mechanism rule. Recovery zero proximity had median 0.008053; 60/98 were
> within 0.01 and 94/98 within 0.05. Terminal recovery was therefore the only
> structurally repeatable finite mechanism. This supports using the continuous
> terminal-recovery margin, rather than its thresholded binary signature, as
> the next diagnostic object; it is not a continuity or interpolation claim.
> Only a separately frozen finite-secant regularity audit over the same 57
> pairs and 15 existing actions may follow. New actions, post-hoc Lipschitz
> selection, model/boundary fitting, source interpolation, atlas changes,
> runtime projection, actor training and paper claims remain unauthorized. See
> `SCF_BF_MF_OUTCOME_MARGIN_CROSSING_AUDIT_RESULTS.md`.

> **MF terminal-margin finite-secant update (2026-07-25):** the separately
> frozen audit reused all 57 local source pairs and the same 15 exact actions,
> producing 855 endpoint secants with all pair/action identities, terminal
> Boolean identities, prior 98 sign changes, and artifact hashes passing. The
> absolute terminal-margin change had median 0.02616 and q95 0.22858. The
> absolute secant had median 0.23538, q90 1.71987, q95 2.31166, and observed
> maximum 5.10338; that maximum is not selected as a Lipschitz constant or
> bound. Sign stability for pairs whose two endpoint margins were both more
> than 0.05 from zero was 473/477 = 0.99161, leaving four far-from-zero sign
> changes. Small-distance secant q95 was 3.20064, and actuator-boundary
> checkpoint medians reached 1.70006, so the current equal-family source
> distance is not authorized as an interpolation metric. Only a separately
> frozen, existing-record provenance-stratified tail audit may follow. No new
> query, post-hoc tail/Lipschitz cutoff, model or bound fitting, interpolation,
> atlas change, runtime projection, actor training or paper claim is
> authorized. See `SCF_BF_MF_TERMINAL_MARGIN_SECANT_AUDIT_RESULTS.md`.

> **MF provenance/role secant-tail update (2026-07-25):** the fixed
> predecessor q90/q95 tail thresholds were applied to the same 855 records.
> V1 stopped before aggregation on a declarative-config interface error and
> produced no result; V2 consumed only that Boolean field, changed no rule or
> threshold value, used a separate protocol/output, and passed all hashes and
> integrity checks. No factor passed the frozen concentration rule. Complete
> actuator/contact provenance changed in 703/855 records; its q90/q95 tail
> prevalence ratios were only 1.850/1.730 versus the required 2x. The actuator
> role was strongly enriched—165/855 baseline records but 59/86 q90 and 24/45
> q95 tail records, prevalence ratios 9.138/4.779—yet its q95 tail occurred
> only in checkpoints 0 and 1, failing required support in three checkpoints.
> All 24 actuator q95 records also changed provenance; that joint stratum had
> q95 3.889 and maximum 5.103, but remains a two-checkpoint localization, not
> a promoted mechanism. Only a separately frozen existing-trace decomposition
> of actuator-mode versus contact-topology words may follow. No new query,
> threshold search, fitting, interpolation, atlas change, runtime projection,
> actor training or paper claim is authorized. See
> `SCF_BF_MF_PROVENANCE_SECANT_TAIL_AUDIT_RESULTS.md`.

> **MF provenance-component update (2026-07-25):** the complete three-scenario,
> 50-step transition provenance was decomposed into identity-sorted
> actuator-mode and contact-topology words for the same 855 secants, using the
> unchanged q90/q95 thresholds and support rule. V1's statistics were rejected
> because its added full-hash check used sorted rather than legacy generation
> order; the invalid output is preserved. V2 repaired only that representation,
> locked and exactly reproduced all V1 observed statistics, reconstructed all
> legacy hashes, and passed independent raw counting. Neither component passed
> the frozen concentration rule. Actuator words changed in 703/855 records and
> covered 77/86 q90 and 40/45 q95 tails, but prevalence ratios 1.850/1.730 were
> below 2x. Contact words changed in 18/855 records, always with actuator
> change, and contained zero q90/q95 tails. The sharp joint stratum remains
> actuator-word change + contact unchanged + actuator role (147 records, q95
> 3.889, maximum 5.103), but its 24 q95-tail records occur only in checkpoints
> 0 and 1. Thus no transition word is promoted. The defensible design candidate
> is a hybrid charted terminal-recovery-margin fiber: retain the four necessary
> source coordinates, block continuity transfer at source actuator-mode
> boundaries, and keep contact topology only as a fail-closed physical guard.
> This is not yet a validated or implemented constraint. A source-observable
> guard and checkpoint-held-out confirmation would require a separately frozen
> next stage. No threshold search, fitting, interpolation, atlas change,
> runtime projection, actor training or paper claim is authorized. See
> `SCF_BF_MF_PROVENANCE_COMPONENT_AUDIT_RESULTS.md`.

The remainder records the frozen previous ABER submission package.

Previous method: **ABER (Actuation-aware Braking-Envelope Recovery)** framed as a
**Safe RL hard shield** on the deployed `(forward, turn)` interface.

Framing guide: `SAFE_RL_FRAMING.md`. The authoritative detailed tracker is
`AAAI27_MAJOR_REVISION_TRACKER.md`. The method/code/data boundary is recorded
in `METHOD_CODE_ALIGNMENT.md`.

## Ready

- Abstract / intro / contributions / related work / Safety-Gym section rewritten
  for Safe RL vs soft CMDP; native shield vs sim-BRT shaping kept separate.
- Main paper, supplement, and reproducibility checklist are method-consistent.
- Main PDF: 5 pages (after Safe-RL rewrite); supplement: 4 pages; checklist: 2 pages.
- Main and supplement rebuild successfully with no overfull boxes, undefined
  references, fatal errors, or emergency stops.
- ABER focused tests: 17 passed.
- Property audit: 31,354 positive checks with zero required violations, plus
  5,000 expected under-braking counterexamples.
- Certified benchmark JSON and the audited 30-cell Safety-Gym JSON are present.
- The V2 anonymous archive contains the complete 30-checkpoint paired matrix,
  is hashed and identity-scanned, and passes its full one-command verifier.
- Sampled-term/recovery ablation is complete: 295,200 raw records, 48/48
  control-grid validations passed, and 17 combined focused tests passed.
- Runtime scaling is complete: 1,000,000 real calls, counts 0--256, pinned-core
  median fit $R^2=0.99961$, and 256-obstacle p95 103.76 microseconds.
- Full strict pairing is complete: 30 newly trained native/sim-BRT checkpoints
  over three tasks and five seeds, 7,500 identical-seed pairs, and 11,860,270
  step records. Pooled collision decreases from 44.97% to 6.52%, but 134
  reverse discordances remain and success decreases in all six cells.
- Independent raw-data verification passes all checkpoint hashes, episode
  rows, step rows, and the seed-level aggregate.
- Old VA-ATACOM/Cartesian results have been removed from current paper claims.

## Not Ready

- Portal upload has not been performed.

## Rule

Do not copy claims from `DERIVATIONS.md`, `NOVELTY_AUDIT.md`,
`PAPER_AUDIT.md`, old Webots/hardware reports, or archived paper sources into
the ABER submission. Those documents describe earlier method versions.

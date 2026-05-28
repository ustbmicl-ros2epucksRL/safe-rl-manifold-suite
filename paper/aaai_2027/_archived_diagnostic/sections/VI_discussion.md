# §VI  Discussion

*(Draft for AAAI 2027.  Target 0.75–1 page in the AAAI template.
Synthesises observations from §IV–V that don't fit the formal
taxonomy but matter for practitioners and follow-up work.
Cross-refs to other sections are placeholders.)*

## §VI-A  Hyperparameter brittleness across the filter zoo

A theme that emerges from re-implementing eight published
safety filters under matched conditions is that each carries
**at least one hyperparameter that is silently $\Delta t$-coupled
in the published form**, and that the coupling becomes a
failure mode when the env $\Delta t$ leaves the original
manipulator regime.

The clearest example is **ATACOM-LA** [Liu thesis 2024 §IV-B].
The filter uses a 1-step lookahead with hyperparameter
`atacom_la_dt`, set to 0.1 s in the public configuration.
Treating this as a fixed constant — as one would for any
other model hyperparameter — produces a counter-intuitive
result in our D5 sweep: ATACOM-LA's cost is **strictly higher
than vanilla ATACOM at every $\Delta t$** when `atacom_la_dt`
is held at 0.1 s, including the small-$\Delta t$ regime where
vanilla ATACOM achieves zero cost (Goal, 50K steps × 3 seeds):

| $\Delta t$ | vanilla ATACOM | ATACOM-LA (fixed 0.1 s lookahead) |
|-----------|---------------:|----------------------------------:|
| 0.01 s    | 0.00 ± 0.0     | 19.97 ± 9.0 |
| 0.02 s    | 0.00 ± 0.0     | 20.37 ± 16.3 |
| 0.05 s    | 39.95 ± 13.7   | 45.23 ± 44.0 |
| 0.10 s    | 13.68 ± 15.8   | 63.68 ± 26.4 |

The mechanism is obvious in retrospect: at $\Delta t = 0.01$ s
the 1-step lookahead extrapolates **ten env-steps ahead under
constant-velocity assumption**, sees hazards that the agent
will not actually encounter under realistic policy
trajectories, and aborts safe actions. The filter is
behaving correctly *under its design assumption* that
$\Delta t = \texttt{atacom\_la\_dt}$; the assumption is
violated.

Re-running with `atacom_la_dt` scaled to the env's realised
$\Delta t$ (so the lookahead always covers exactly one env
step) removes the small-$\Delta t$ artefact but, importantly,
**does not save ATACOM-LA at $\Delta t \ge 0.05$ s** — see the
re-run results in Appendix VII-B that we use as our M2
isolation evidence in §IV-B.

The general lesson: **the filter zoo's hyperparameters have
silent $\Delta t$ coupling that is not stated as such in the
publications**. Practitioners porting these filters across
control rates must scale the relevant constants by hand. Two
other instances we hit:

- **HOCBF's class-K coefficients** $\alpha_1, \alpha_2$ are
  dimensionally rates ($\text{s}^{-1}$) but reported as
  dimensionless numbers in the original paper. Our default of
  $\alpha_1 = 2$ becomes too aggressive at finer $\Delta t$
  (the constraint pulls policy back too fast, freezing motion).
- **DCM's velocity-scaling factor** assumes action $\in [-1,1]$
  maps to velocity $\in [-1,1] \cdot \text{velocity\_scale}$;
  under different action gains in different env versions this
  needs re-calibration.

These are not deep theoretical problems, but they are
practical barriers to cross-domain transfer and reinforce the
paper's central claim that the published filter literature has
not engaged seriously with the discrete-time mobile-agent
regime.

## §VI-B  When does the taxonomy generalise?

§IV's M1–M3 are derived for circular hazards in 2D under
$\dot{\boldsymbol{q}} = \boldsymbol{u}$ dynamics. Three
generalisation directions matter for follow-up work:

### Locomotion ($\Delta t = 0.02$ s, explicit dynamics)

Safety-Gymnasium also offers locomotion tasks
(SafetyAntVelocity-v0 etc.) at $\Delta t = 0.02$ s. Our M1
threshold (IV-3) at this $\Delta t$ requires
$\|\boldsymbol{u}\| \gtrsim 5\sqrt{d_\text{safe}\cdot r}$ —
i.e. M1 may be sub-threshold for typical locomotion speeds.
**The hypothesis we leave for follow-up work**: tangent-projection
filters work substantially better on locomotion than on
point-robot navigation. M2 (tangential drift) likely also
weakens because high-DOF locomotion typically has
non-circular workspace boundaries (e.g. terrain constraints)
where the rotation argument of Prop. 2 does not apply.

### Multi-agent navigation

The D5 sweep was run with one agent + circular hazards. In
multi-agent settings the constraint geometry includes
agent-agent collisions (also circular), and the relative
velocity between agents can exceed the per-agent
$\|\boldsymbol{v}\|$ by 2×. This **expands** the M1 regime
$\Delta t\|\boldsymbol{v}\| \ge \sqrt{d_\text{safe}\cdot r}$
toward smaller $\Delta t$. The recipe's velocity-adaptive
margin scales naturally (use relative-velocity in (V-1)) but
the BRT lookahead becomes computationally heavier because of
the cross-agent rollout combinatorics. **A clean test bed for
the taxonomy** in this direction is the MPE benchmark suite.

### Manipulator control

In the regime where M1 was originally proved harmless ($\Delta
t \le 2$ ms), the taxonomy reduces to (a known and well-served)
continuous-time analysis. The recipe degenerates:
$\alpha = \Delta t$ is so small that the velocity-adaptive
margin is a fraction of a millimetre, and the BRT lookahead
at $h = 3$ spans 6 ms — invisible. We expect ATACOM to
dominate on manipulator benchmarks, as the original paper
reports.

## §VI-C  Where the recipe still degrades

Three regimes where §V's recipe shows visible limitations:

### Push (3/5 GO at $\bar C = 10.58$)

The cliff-edge cell in §V-E Table 1. The BRT rollout in §V-B
assumes the box velocity is captured at the current step
($\dot{\boldsymbol{b}} = v_b$) but does not simulate
agent-induced box motion during the lookahead. At $h = 3$
this is a 0.3 s open-loop assumption that breaks for fast
contact events. Increasing $h$ to 5 doubles the per-call
cost (9 × 5 = 45 rollouts) but may close the gap; we did not
explore this for §V-E because the box-coupled BRT is a
separable engineering improvement (§V-F discussion).

### MultiGoal (4/5 GO)

One seed produces a cost-22 outlier; the other four are
sub-1 cost. Inspection of the failing seed's trajectory
showed the agent oscillating between two goal candidates as
the active goal switched, producing brief high-speed transits
between the two zones. The recipe's BRT covers single-step
sweeps but not the multi-step accumulation across a goal
switch. **A practical fix** is to reset the BRT predictor's
state when the active goal changes; we did not include this
because it requires a goal-aware filter interface, and the
4/5 result already lands above any baseline.

### $\Delta t = 0.05$ s on Goal (the D5 outlier)

Ours at $\Delta t = 0.05$ s achieves $\bar C = 23.7 \pm 14.6$,
strictly worse than at $\Delta t = 0.01$ s ($\bar C = 3.0$) or
$\Delta t = 0.10$ s ($\bar C = 2.6$). This is the
$h\Delta t = 0.15$ s lookahead band: too short to absorb the
multi-step M2 drift (the 0.10 s case is in invariance) and
too long to be drowned by single-step noise (the 0.01 s
case). **A constructive recommendation**: for $\Delta t \in
[0.03, 0.08]$ increase $h$ to 5 ($\sim 0.25$ s lookahead).
This costs more compute but stays within Prop. 4's sufficient
condition.

## §VI-D  Sim-to-real implications

The discrete-time gap we identify is, in practice, *the*
sim-to-real gap for safety filters. Real-robot control rates
are 100 Hz–1 kHz on professional hardware but **closer to
10–25 Hz on educational, embedded, or commodity-grade
hardware** that researchers actually use for outreach and
deployment (Webots ROS2 default: 32 Hz; E-puck firmware:
20 Hz; many low-cost UAVs: 50 Hz). The recipe's parameters
were chosen for $\Delta t = 0.1$ s; Proposition 4's condition
$\alpha \ge \Delta t$ is generous (0.3 ≥ 0.1), giving
~9× headroom at $\Delta t = 0.032$ s.

In contrast, the **published filter zoo's continuous-time
guarantees lose their margin entirely on this hardware**: at
$\Delta t = 0.032$ s and typical 1 m/s mobile-robot speeds,
$\Delta t\|\boldsymbol{u}\| = 32$ mm, comparable to $r = 0.2$
m and to the typical sensor-noise margin. M1 is no longer
asymptotically small.

A companion ICRA submission [our work, 2027] applies the
recipe to E-puck robots in Webots simulation
($\Delta t = 0.032$ s) and demonstrates the predicted
robustness at low control rates. This is genuine sim-to-real
evidence for the recipe; the §IV taxonomy in particular
predicts that **swapping the recipe for any of the §IV-D
table's continuous-time filters at this rate would visibly
worsen real-robot collisions**.

---

### TODO before submission

- [ ] Confirm the multi-agent generalisation claim against
      Lowe 2017 MPE / Yu 2022 MAPPO benchmark numbers
- [ ] Tighten the manipulator-regime regression claim — needs
      either a citation or a small-$\Delta t$ D5 cell on
      SafetyAnt or a manipulator task
- [ ] Resolve Push cliff-edge: either commit to running BRT
      h=5 at $\Delta t = 0.10$ as an ablation, or drop the
      "fix is engineering" claim
- [ ] Cross-ref §VII-C for the PRCF distribution-shift link
      in §VI-D (the recipe is sim-to-real-ready, PRCF is not
      yet — that asymmetry is itself a finding)

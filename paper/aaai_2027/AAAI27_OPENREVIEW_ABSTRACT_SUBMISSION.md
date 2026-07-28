# AAAI-27 OpenReview Abstract Submission

Canonical copy/paste record for the AAAI-27 Main Technical Track.

Paper source: `aaai27/main_aber_aaai27.tex`

## Title

ABER: Executable Sampled-Data Safety Filtering at the Forward/Turn Command Interface

## TL;DR

ABER filters sampled forward/turn commands with a closed-form braking-envelope cap and executable recovery, proving conditional multi-obstacle safety while exposing a safety-liveness tradeoff in learned navigation.

## Abstract

Safe reinforcement learning for mobile robots must constrain the actions a policy can execute, not only actions that are feasible in an idealized Cartesian model. Soft CMDP methods improve average cost but do not guarantee per-step invariance, while many hard filters are derived for Cartesian acceleration and then mapped onto sampled forward/turn velocity setpoints, so a theoretically feasible fallback need not be robot-executable. We introduce ABER (Actuation-Aware Braking-Envelope Recovery), a hard per-step safety shield on the deployed interface: it converts a calibrated stopping envelope into a closed-form cap on the next forward speed, retains the requested turn when the normal branch is certified, and otherwise issues the executable recovery command (0,0). Under measurable velocity-servo envelopes, we prove recursive feasibility and collision avoidance for any finite set of static obstacles. The classical braking-distance formula is not claimed as novel; the contribution is its interface-level projection, recovery witness, and closed-loop multi-obstacle certificate. A seeded property audit finds no required violation in 31,354 cap and projection checks. In a 295,200-record component study, calibrated ABER yields zero certificate violations and collisions over 1,200 navigation episodes, whereas removing the sampled-displacement term or executable recovery causes violations in 1,200 and 1,002 episodes. Across 30 independently trained policies and 7,500 matched Safety-Gymnasium pairs, online ABER reduces geometric collision frequency from 45.0% to 6.5%, but does not eliminate collisions and lowers success in every task/mode cell, exposing the safety-liveness tradeoff and the empirical-envelope boundary of the method.

## Primary Topic

ROB: Robot Learning, Control & Foundation Models

## Secondary Topics

1. PEAI: Safety, Robustness & Trustworthiness
2. ROB: Localization, Mapping & Navigation
3. APP: Mobility, Transportation & Autonomous Systems

Do not select `ML: Reinforcement, Imitation & Inverse RL`: learned policies are evaluated, but the paper does not contribute a reinforcement-learning algorithm. Selecting it would overstate the learning contribution and may lead to a mismatched reviewer assignment.

## Author-Specific Fields to Complete in OpenReview

- Authors and author profiles: enter the real authors; do not enter “Anonymous Submission.”
- Country of Institutions: select the current institution country for every author.
- Reciprocal Reviewer Nomination: nominate one qualified author, or make the required no-qualified-author declaration.
- Reciprocal Reviewer Confirmation: select the statement matching the nomination above.
- Self-Declared Conflicts of Interest: add profiles if applicable.
- Submission Policies Acknowledgement: affirm only statements that are factually true.

## Final Consistency Check

- Track: AAAI-27 Main Technical Track.
- The title exactly matches the anonymous main paper.
- TL;DR is a single required sentence and is below the 250-character limit.
- The abstract is below the 5,000-character OpenReview limit.
- One primary and three secondary topics are selected; OpenReview requires 1–5 secondary topics.
- PR-MF, GTR-MF, FN-MF, SCF-MF, Webots, and unvalidated hardware claims are not included.
- Do not replace the abstract with placeholder text.

## Official Form References

- AAAI-27 timetable and author information:
  https://aaai.org/conference/aaai/aaai-27/
- AAAI-27 OpenReview submission form:
  https://openreview.net/group?id=AAAI.org/2027/Conference

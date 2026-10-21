# Seed-101 mechanism forensics — report (WO-TBM-P2, diagnosis only)

**Status: DIAGNOSIS ONLY.** No fix, training change, or
new experiment designed to "solve" seed 101 is authorized
by this work order. Follow-ups return as proposals with
exact scope. No frozen file is edited; new scripts,
traces, and figures are additive only.

## Step 0 — Anomaly re-established from tracked records

Return = 1 − 0.01 × ticks on success (verified against
records: e.g. 44.433 ticks → 0.556; 23.133 → 0.769).
Greedy = one deterministic episode per start phase, all
phases. "Oracle" arms are learned neural agents fed
perfect-information features (`OracleIndEncoder`: true
phase/remaining; `OracleNewEncoder`: true period/open
parameters) — they are NOT optimal planners. That fact
reshapes Step 2: the comparison is
learned-with-perfect-info vs learned-with-inferred-info
on the same seed.

### 30/14 (pilot + Stage 1 files), greedy ticks / return

| arm | seed 100 | seed 101 | seed 102 | seed 103 | seed 104 |
|---|---|---|---|---|---|
| wrapped_ind | 24.73 / 0.753 | 23.13 / 0.769 | 24.03 / 0.760 | 28.53 / 0.715 | 26.47 / 0.735 |
| memory10 | 27.63 / 0.724 | 25.90 / 0.741 | 44.10 / 0.559 | 29.30 / 0.707 | 26.37 / 0.736 |
| memory20 | 25.70 / 0.743 | 27.70 / 0.723 | 34.00 / 0.660 | 27.37 / 0.726 | 27.10 / 0.729 |
| memory30 | 25.70 / 0.743 | 26.20 / 0.738 | 25.77 / 0.742 | 28.27 / 0.717 | 26.00 / 0.740 |
| oracle_new | 24.80 / 0.719 | 42.47 / 0.575 | 38.50 / 0.615 | 25.10 / 0.749 | 34.50 / 0.655 |
| oracle_ind | 25.43 / 0.746 | 44.43 / 0.556 | 23.10 / 0.769 | 26.27 / 0.737 | 26.63 / 0.734 |
| phase_only_new | 23.73 / 0.763 | 28.83 / 0.712 | 26.20 / 0.705 | 26.00 / 0.740 | 23.70 / 0.763 |

All above crush-free except oracle_new/100 (0.033) and
phase_only_new/102 (0.033). Prior records-only diagnostic
(`prereg_generalization.md`): oracle_ind/101 greedy 44.4
ticks vs late-training ε=0.05 behavior 28.25 ticks
(`train.last100_mean_ticks` = 28.25, success 1.00) — the
greedy policy is worse than the training behavior policy
on the same arm and seed.

### 20/10 regression (wrapped_ind only), greedy ticks / return

| seed | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
|---|---|---|---|---|---|---|---|---|---|---|
| ticks | 22.10 | 25.45 | 21.60 | 24.15 | 21.90 | 22.70 | 25.20 | 23.10 | 30.75 | 32.55 |
| return | 0.779 | 0.745 | 0.784 | 0.758 | 0.781 | 0.773 | 0.748 | 0.769 | 0.692 | 0.674 |

All crush-free. Fast bar 0.750 (= 25.0 ticks): misses
are 101 (by 0.005), 106 (by 0.002), 108, 109.

### Step-0 corrections to the work order's premise

1. **"Recurs across arms on seed 101" is NOT re-derived.**
   On 30/14 seed 101, every learned arm is normal
   (wrapped 23.13, memory 25.9–27.7); only the two oracle
   arms are slow (44.43, 42.47), with phase_only_new
   mildly slow (28.83). The cross-arm recurrence holds
   only in the weaker sense of the 20/10 regression-miss
   list (one arm, four seeds), where 101/106 are marginal
   misses and only 108/109 are substantial.
2. **No separate issue-list file was found.** The anomaly
   record lives in `decisions_stage1.md` (oracle_ind/101
   "safe but very slow"; regression misses listed) and
   `prereg_generalization.md` (§Seed-101 diagnostic).
   Nothing in Step 0 rests on an unlocated document.
3. **Neighboring-seed observation (out of scope, not
   measured here):** seed 102 on 30/14 is slow for
   memory10 (44.10) and oracle_new (38.50) while wrapped
   (24.03) and oracle_ind (23.10) are normal — slowness
   is not unique to seed 101, and it does not hit all
   arms on any seed examined. This pattern is noted for
   a possible future proposal, not traced here.

### Matched normal controls (named)

- 30/14: oracle_ind/102 (23.10), wrapped_ind/101 (23.13,
  same seed as the oracle anomaly, normal arm),
  wrapped_ind/102 (24.03).
- 20/10: wrapped_ind/102 (21.60), wrapped_ind/104
  (21.90), wrapped_ind/100 (22.10).

## Step 1 — Attractor test, operationalized BEFORE measuring

(Defined here before any trace is run. The measures
below are computed identically for attractor-case and
control episodes; the verdict rule is fixed here, not
fitted after seeing traces.)

Per greedy episode, from its per-tick trace, compute:

- **Phase decomposition** of ticks to goal: (i)
  approach = ticks from episode start to first arrival
  at staging cell (4,5); (ii) staging wait = ticks at
  (4,5) (or blocked-against-door attempts) until first
  entry into the door cell (5,5); (iii) escape = ticks
  from door entry to goal. Staging wait is legitimate
  door-timing caution, not attractor evidence — a slow
  episode explained entirely by staging wait is NOT an
  attractor episode.
- **Dwell:** longest run of consecutive ticks in which
  the agent's position stays inside one fixed 3×3 block
  (blocks aligned to the grid), excluding staging-wait
  ticks at (4,5) and excluding the goal cell.
- **Revisits:** number of returns to a previously
  visited cell after leaving it (counted per cell-visit
  after the first), and the maximum revisit count of any
  single cell.
- **Approach/escape excess:** (i) + (iii) ticks.
- **Crushes** per episode (restart cycles).

**Verdict rule (pre-stated):** the attractor mechanism
is SUPPORTED for a slow case if its episodes' median
dwell, median revisit count, or median approach/escape
excess is ≥ 2× the corresponding median for matched
normal controls at the same start-phase mix, on at
least one of the three measures. It is REFUTED for a
slow case if slow episodes match controls on all three
measures and their excess ticks are fully accounted by
staging wait. Otherwise INDETERMINATE. Per-start-phase
comparison is mandatory: slowness concentrated in a few
start phases (e.g. phases just after the door closes)
points to phase luck / door-timing caution, not to an
attractor.

## Step 1 — Trace statistics (MEASURED)

All 12 traced pairs reproduced their tracked greedy summaries
**exactly** (ticks, return, crush rate, success) — the traces
are the recorded agents, not lookalikes. Every greedy
episode succeeded (success 1.00) with zero crushes; all
slowness below is ticks, not failures.

### Medians per pair (approach / staging / escape / dwell / revisits)

| pair | ticks | appr | stag | esc | dwell | revis |
|---|---|---|---|---|---|---|
| oracle_ind/101 30/14 | 43.5 | 23.5 | 1.0 | 26.0 | 20.0 | 14.0 |
| oracle_new/101 30/14 | 41.5 | 32.5 | 1.0 | 8.0 | 15.0 | 1.0 |
| phase_only_new/101 30/14 | 28.5 | 14.5 | 4.0 | 10.0 | 10.5 | 0.0 |
| oracle_ind/102 30/14 (control) | 20.5 | 9.0 | 1.0 | 8.0 | 5.0 | 0.0 |
| wrapped_ind/101 30/14 (control) | 20.5 | 9.0 | 3.0 | 8.0 | 5.0 | 0.0 |
| wrapped_ind/102 30/14 (control) | 22.5 | 9.0 | 1.0 | 8.0 | 5.0 | 0.0 |
| wrapped_ind/108 20/10 | 32.0 | 21.5 | 1.0 | 9.0 | 17.5 | 0.0 |
| wrapped_ind/109 20/10 | 32.5 | 9.0 | 3.5 | 20.0 | 7.0 | 6.5 |
| wrapped_ind/106 20/10 | 24.5 | 15.0 | 1.0 | 9.0 | 8.0 | 1.5 |
| wrapped_ind/101 20/10 | 24.5 | 15.0 | 1.0 | 9.0 | 8.0 | 0.0 |
| wrapped_ind/102 20/10 (control) | 20.5 | 9.0 | 2.5 | 8.0 | 5.0 | 0.0 |
| wrapped_ind/104 20/10 (control) | 20.5 | 9.5 | 3.0 | 8.0 | 5.0 | 0.0 |

Controls are clean in structure: approach 9 (the shortest
walk to staging), escape 8 (the shortest walk to goal);
their slowness on some phases is staging wait alone
(e.g. wrapped_ind/102 30/14 waits 20 ticks at staging on
phase 3). Slow cases break structure elsewhere.

### Rule application (pre-stated §Step 1 test)

- **oracle_ind/101: rule met on all three measures**
  (dwell 20 vs control 5; revisits 14 vs 0;
  approach+escape 42.5 vs 17).
- **oracle_new/101: met via dwell** (15 vs 5). Bimodal:
  phases 14–27 normal (18–26 ticks), phases 0–13 slow
  (59–71 ticks).
- **phase_only_new/101: met via dwell, marginally**
  (10.5 vs 5) — against wrapped controls, since no
  phase_only control was traced (caveat recorded).
- **wrapped_ind/108: met via dwell** (17.5 vs 5).
- **wrapped_ind/109: met via revisits** (6.5 vs 0–1.1).
- **wrapped_ind/101, 106 (20/10): rule NOT met**
  (dwell 8 vs 5; approach+escape 23.5 vs ~18; revisits
  ~0–1.5) and their excess is not staging wait
  (staging 1) — **INDETERMINATE** by the pre-stated rule,
  not refuted: a mild approach inflation, same kind as
  the severe cases, below the 2× bar.

### What the trajectories actually show (MEASURED)

- **oracle_ind/101:** two traps. (1) Spawn freeze: on
  start phases 8–29 the agent presses clipped up/left at
  (0,0) and leaves spawn exactly when the tick-phase
  cycles to 0/1 (all 22 frozen episodes); phases 0–7
  move immediately. (2) Escape 2-cycle: after crossing
  (staging 1 on all 30 phases — the crossing itself is
  never the problem), it oscillates (6,5)↔(7,5) through
  the closed period, breaking only at/just after
  reopening (exit phases 0–4). Escape median 26 ticks
  for an 8-move walk.
- **oracle_new/101:** approach wandering — spawn
  freeze, a (1,6)↔(2,6) cycle, a 9-tick stay at (3,8);
  escape perfect (8).
- **wrapped_ind/108:** a 10-tick stay at (1,5)
  mid-approach (phase-14 episode), then a clean
  crossing; escape 9. It waits away from the door,
  not at staging.
- **wrapped_ind/109:** staging bounce (4,5)↔(4,6),
  blocked up-moves from the door cell into the wall,
  and an (8,8)↔(8,9) bounce near the goal; escape 20.
- **Controls show the same kinds mildly:** the
  wrapped_ind/102 (20/10) phase-0 episode bounces
  (4,5)↔(4,6) for 10 staging ticks; control slowness
  is staging wait, never escape breakage.

**Unified reading (INFERRED from the trajectories):**
phase-gated progress. The greedy policy's progress
actions are gated on the phase feature; at non-gate
phases it outputs non-progressing actions (clipped
moves, stays, 2-cycles) until the phase cycles to a
gate value. Locations and severity vary by trained
network; the kind is uniform. This is a policy
property, not world geometry (see Step 2).

## Step 2 — Oracle comparison (MEASURED)

Same seed 101, 30/14, oracle_ind vs wrapped_ind:

- Phase 8 and 20: first divergence at **step 0** —
  oracle presses up at spawn (the freeze), wrapped
  moves right.
- Phase 0: first divergence at **step 2** — a route
  choice (oracle down via (2,1), wrapped right along
  the top row). Both reach the door; the decisive
  difference appears later (escape cycle) and at
  spawn, not at the crossing.
- The seed does not resist the learned agents:
  wrapped walks seed 101 in 23.1 ticks while oracle
  takes 44.4, on the identical world.

**Construction fact (SOURCE FACT from world.py):**
the World has no seed parameter — only training RNG
varies by seed. "Seed difficulty" as wall/door
geometry differing per seed is impossible in this
codebase; what varies is the trained policy.

**Prior ε-noise hypothesis, refined:** greedy 44.4
vs late-training behavior 28.25 stands, but the
traces do NOT show depressed Q(cross) — staging is
1 tick, the crossing is clean. The depressed actions
are at spawn and on the escape side during
closed-door phases. The hypothesis in its stated form
(crossing depression) is not supported by the traces;
the greedy-vs-behavior gap stands with its locus now
known.

## Step 3 — Figures

- `figures_p2/figA_paths.png`: episode paths —
  oracle_ind/101 ph8, oracle_ind/102 ph0 (control),
  wrapped/108 ph14, wrapped/109 ph0.
- `figures_p2/figB_phase_profiles.png`: greedy ticks
  vs start phase, slow vs control pairs, both
  protocols. The smooth ramps are the phase-gating
  signature: ticks vary monotonically with start
  phase within each regime.
- `figures_p2/figC_decomposition.png`: median
  approach / staging / escape stacked bars for all
  12 pairs. The visual summary: slow cases break
  approach or escape; controls break only staging.

Honesty note: Fig A overplots repeated positions —
a 22-tick spawn freeze and an 18-tick 2-cycle appear
as single points/segments. Loop durations come from
the trace tables above, not from the figure alone.

## Step 4 — Verdicts per hypothesis

1. **Attractor mechanism: SUPPORTED for the severe
   cases** (oracle_ind/101, oracle_new/101,
   wrapped_ind/108, wrapped_ind/109, phase_only_new/101
   marginally) by the pre-stated rule; **INDETERMINATE
   for marginal wrapped_ind/101, 106 (20/10)**.
   Characterized as phase-gated policy traps (limit
   cycles and stays), not one single attractor state.
   (MEASURED + INFERRED mechanism label.)
2. **Seed difficulty: REFUTED.** No per-seed geometry
   exists (construction fact); same seed, same world,
   wrapped is normal while oracle loops (measured
   divergence from step 0). The seed's only role is
   the trained network it produced.
3. **Policy-caution artifact: SUPPORTED as a
   background tendency, NOT sufficient alone.**
   Staging waits and bounces appear in controls in
   milder form — but the severe cases' excess is not
   staging wait (staging 1 for oracle_ind/101 and
   wrapped_ind/108), so caution alone does not explain
   them. (MEASURED.)
4. **Scatter null: REFUTED for the severe cases.**
   Greedy eval is deterministic (repeats identical by
   construction); the gaps are not eval noise. The
   marginal 20/10 cases sit inside cross-seed variation
   with a mild form of the same pattern — recorded as
   such, not as scatter explaining the severe pattern.
   (MEASURED.)

No finding in this report implicates candidate 4
(trust-replay recency) — that track is untouched, and
no JIT work was done (ruled E, trigger-gated).

## Follow-up proposals (exact scope; NOT executed)

- **P-P2a (ε_end=0.0 decisive test):** re-run
  oracle_ind/101 training with ε_end=0.0 for the
  final 1000 episodes, frozen greedy eval; if the
  traps disappear, the ε-noise account gains support.
  Scope: one arm/seed/protocol, training only, no
  design change.
- **P-P2b (Q-value logging at trap states):** log
  Q(progress) vs Q(trap) margins at spawn during
  closed phases and at (6,5)/(7,5) during closed
  phases for oracle_ind/101, testing phase-gating in
  value space. Scope: diagnostic logging only.
- **P-P2c (memory10/102 forensics):** same trace
  protocol for memory10/102 (44.1 ticks on 30/14) —
  tests whether the mechanism generalizes beyond this
  work order's seed list. Scope: diagnosis only.
- **P-P2d (phase-feature ablation, needs its own
  prereg):** retrain oracle_ind/101 with the phase
  feature masked, testing whether phase-gating is
  causal. Proposal only; no run authorized here.

# WO-TBM-Luck — report (Step 2)

**Status: EXPLORATORY, instructor-verified (see §0, item 1, for the instructor's 0c check).**
This draft was produced by analysis only. No training, evaluation,
retraining, or file edit to any results, design, world, or doors
file was performed in this step. All numbers below are derived
from `nn/results_luck.jsonl` by `nn/luck_analysis.py`
(output: `nn/luck_analysis_output.txt`), not from runner-log
summary lines. Where a number below is not in that output, it
is labelled as a cross-check and its source is named.

## 0. First: skipped, not re-derived, and definitional anomalies

Nothing in the Step 2 analysis was skipped. The following
limits and anomalies are stated first, so they are not
buried under the headline:

1. **0c checks: not re-derivable from the records file, verified
   separately by the instructor from the runner log.**
   `results_luck.jsonl` contains only W1r/W5r episodes; it
   contains no W1 episodes, so the luck run's 0c MATCH status
   cannot be checked from this file. The instructor checked
   `luck_run2.log` directly: **30/30 agents show `0c ->
   MATCH`** (one per agent, no non-MATCH 0c line), so the
   0c gate is satisfied for the run as a whole. The analysis
   below proceeds on the run's kept rows only (see §1 for the
   completeness checks that *were* performed on those rows).
2. **Ticks carry no information beyond the cap.** Every
   luck episode runs to 200 ticks because none reaches the
   goal. "Mean ticks" is therefore 200.0 everywhere and
   "mean ticks over reached episodes" is undefined (no
   reached episodes), not zero.
3. **Arrival and staging counts differ across worlds
   definitionally, not behaviourally.** The runner's
   staging-cell set is world-relative: W1r counts
   {(4,1), (4,5)}; W5r counts {(4,1), (4,3), (4,7), (4,9),
   (4,5)}. A trajectory that passes (4,3) before (4,5)
   therefore has arrival (4,3) in W5r and (4,5) in W1r
   with an identical position sequence. Paired arrival
   differs in 151 of 750 pairs for exactly this reason.
   The behavioural comparison is the position sequence:
   749 of 750 paired W1r/W5r sequences are identical
   (§5). Do not read W1r-vs-W5r arrival or staging
   differences as agents behaving differently.
4. **Recorded stall_kind "dwell" uses the luck gap set.**
   The runner calls the stall signature with the luck
   gaps, so the dwell exclusion covers rival staging
   cells only; (4,5) is *not* excluded in luck worlds
   (it was excluded in doors W1, where (5,5) is a gap).
   This affects the label of 40 episodes (38 wrapped,
   2 oracle_new), not the headline: all 40 still fail,
   and no-move-run dominates overall. Post-entry stall
   is unaffected (post-entry positions are east-side).
5. **"Last recorded position" is the position at the
   start of the final tick.** Steps record the position
   before each action, so the cell after the final
   action is not recorded. A final 'right' action from
   (4,5) is blocked by the wall at (5,5) in luck worlds,
   so those episodes' final cell is still (4,5); other
   final actions may move one cell beyond the recorded
   position. Counts below say "last recorded" for this
   reason.
6. **W1 median for plain/1 rests on 6 reached episodes.**
   Plain/1 reached only 6/20 in doors W1 (median 20.5,
   threshold 41.0). That fragility does not affect any
   classification here: plain/1 never enters a door in
   either luck world, so its threshold is never invoked.
7. **The single entered episode is not new.** Its
   trajectory segments are identical to the doors W5
   memory/1 phase-13 episode already recorded in
   `results_doors.jsonl` (cross-check, §4). It is a
   recurrence of a known pattern, not a new
   luck-surface effect.

## 1. Reconciliation: what was actually analysed

- Luck records: **1500 episodes, 60 jobs** (30 agents x
  W1r/W5r). Episodes per job: 20 for each of the 30
  20/10 jobs, 30 for each of the 30 30/14 jobs.
- Duplicate (arm, seed, world, start_phase) keys: **0**.
  Start-phase sets complete for every job: **0 gaps**.
- Record kinds: all 1500 are `kind = "episode"`.
- `len(steps) == ticks` for all 1500 episodes. Step `t`
  sequences are strictly +1 within every episode.
- Recomputed first entries from step positions match
  recorded `first_door`/`door_entries` in all 1500
  episodes (**0 mismatches**).
- Doors baseline used for W1 medians: doors W1 records
  only (750 records, 30 jobs) — **not** W1m, **not**
  pooled. W1 arm success in that baseline: wrapped
  100/100, memory 100/100, oracle_ind 150/150,
  oracle_new 150/150, phase_only_new 150/150,
  plain 86/100 (plain/1 alone 6/20). Per-(arm, seed)
  W1 medians and 2x thresholds are listed in the
  analysis output §0 and summarised in the Appendix.
- Worlds, as fixed in the design note: W1r = {(5,1)};
  W5r = {(5,1), (5,3), (5,7), (5,9)}; no (5,5) in
  either; every door on the arm's training schedule
  (20/10 or 30/14). Location is the only unfamiliar
  axis. No retraining occurred in this work order.

## 2. Headline and H_luck verdict (EXPLORATORY, descriptive)

| Quantity | Count |
|---|---|
| Episodes analysed | 1500 |
| Reached the goal (any pass) | **0 / 1500** |
| Entered any door | 1 / 1500 |
| Class F (never enters) | **1499** |
| Class D (diversion: (4,5) before entry, then a rival) | **0** |
| Class R (route coincidence, reached or not) | 0 |
| Class W (wandering, reached or not) | 1 (reached = False) |
| Class UNCLASSIFIED | 0 |
| Class D passes (D and reached) | 0 |
| Class R passes / Class W passes | 0 / 0 |
| Switches (all episodes) | 0 |
| Crushes (all episodes) | 0 |

**H_luck verdict: STANDS descriptively — and the result
is stronger than the hypothesis required.** H_luck
operationally required zero class-D passes. There are
zero class-D episodes of any kind, zero class-R
episodes, and zero passes of any class. The outcome is
total failure (0/1500 reached), which the design note
defines as a complete result, not a failed experiment:
within these worlds, arms, and seeds, and under this
greedy, no-retraining protocol, no agent uses an
unfamiliar door to reach the goal. The strongest
description among the tested alternatives is that the
trained routes terminate at the familiar staging area
and do not transfer to an unfamiliar door location;
this is task-and-protocol bounded and is not a claim
about what retraining (P-Db2) or any other protocol
would produce.

By world: W1r 0/750 reached, 0 entered; W5r 0/750
reached, 1 entered (the single class-W episode, §4).
By arm: wrapped 0/200, plain 0/200, memory 0/200
(1 entered, 0 reached), oracle_ind 0/300,
oracle_new 0/300, phase_only_new 0/300.

## 3. Per-arm x world endpoints

All cells: success 0, ticks mean/median 200.0, crushes 0,
switches 0. Ticks over reached episodes: undefined
(no reached episodes). First doors: None everywhere
except memory/W5r, where 99 are None and 1 is (5,3).

| Arm / world | n | Entered | Stall (recorded) | Arrival (4,5) / rival / none |
|---|---|---|---|---|
| memory / W1r | 100 | 0 | no-move-run 100 | 80 / 20 / 0 |
| memory / W5r | 100 | 1 | no-move-run 100 | 43 / 57 / 0 |
| oracle_ind / W1r | 150 | 0 | no-move-run 150 | 141 / 9 / 0 |
| oracle_ind / W5r | 150 | 0 | no-move-run 150 | 133 / 17 / 0 |
| oracle_new / W1r | 150 | 0 | no-move-run 138, dwell 1, none 11 | 127 / 23 / 0 |
| oracle_new / W5r | 150 | 0 | no-move-run 138, dwell 1, none 11 | 99 / 51 / 0 |
| phase_only_new / W1r | 150 | 0 | no-move-run 150 | 141 / 9 / 0 |
| phase_only_new / W5r | 150 | 0 | no-move-run 150 | 90 / 60 / 0 |
| plain / W1r | 100 | 0 | no-move-run 100 | 71 / 15 / 14 |
| plain / W5r | 100 | 0 | no-move-run 100 | 51 / 35 / 14 |
| wrapped / W1r | 100 | 0 | no-move-run 81, dwell 19 | 82 / 18 / 0 |
| wrapped / W5r | 100 | 0 | no-move-run 81, dwell 19 | 75 / 25 / 0 |

Overall stall: no-move-run 1438, dwell 40, no stall
flag 22. Overall arrival: (4,5) 1133, (4,1) 188,
(4,3) 137, (4,7) 14, none 28. The W1r/W5r arrival
differences in this table are the definitional effect
described in §0(3); paired trajectories are 749/750
identical, so they are not behavioural differences.

## 4. The one entered episode (the only non-F episode)

**memory/1, W5r, start_phase 13. Class W, reached =
False, ticks 200, crushes 0, switches 0.**

- First door (5,3); door_entries [[5,3]] only.
- Entry step index 8 (of 200 steps). (4,5) was **not**
  occupied before entry, so the episode is **not D**
  under the fixed rule. Staging order [[4,1], [4,3]];
  arrival (4,1).
- W classification: ticks 200 > 2 x memory/1's W1
  median (2 x 45.5 = 91.0). Revisits 0; the ticks
  criterion alone makes it W. D takes precedence over
  W, but D does not apply here.
- Trajectory segments: (0,0), (1,0), (2,0), (3,0),
  (4,0), (4,1), (4,2), (4,3), (5,3), then **(6,3) for
  191 consecutive steps**.
- Far side: post-entry ticks 191, post-entry
  positions distinct {(6,3)} only, post-entry stall
  True / no-move-run. The far side freezes; it does
  not cross cleanly (§6, Q6).
- Cross-check (doors records, not a luck endpoint):
  this trajectory's segments are identical to the
  doors W5 memory/1 phase-13 episode (which also
  entered (5,3) and froze 191 ticks at (6,3) and
  failed). The same agent and phase in luck W1r —
  where (5,3) is a wall — instead runs (4,3) x2,
  (4,4) x9, (4,5) x182 and never enters. In doors W1
  the same phase waits at (4,5) for 10 ticks, enters
  (5,5) when it opens, and reaches the goal. So the
  W5r entry reproduces a previously recorded far-side
  freeze; it is not evidence of a new decision or a
  new luck-surface pass.

## 5. Failure anatomy: what failure looks like

Failure is not one shape. Four patterns, all ending at
the 200-tick cap with no goal and no crush:

**1. Waiting/bumping at the familiar staging cell
(4,5) — the dominant pattern.**
(4,5) appears in staging_order in 1471/1500 episodes.
Steps at (4,5): overall median 115 (per-arm medians:
memory 182, phase_only_new 178.5, plain 186.5,
wrapped 105, oracle_ind 97, oracle_new 89). First
(4,5) step index median 11; only 29 episodes never
occupy (4,5) (28 plain/1 no-staging episodes + the
1 entered episode). Last recorded position is (4,5)
in 1023/1500 episodes. Final-tick actions overall:
right 1051, stay 211, up 90, down 82, left 66; from a
last recorded position of (4,5), the final action is
'right' in 899 episodes — a move blocked by the wall
at (5,5) in these worlds. Longest consecutive run at
(4,5): median 172 (memory), 178.5 (phase_only_new),
186.5 (plain), 13 (oracle_ind), 12 (oracle_new),
10 (wrapped); episodes with a run >= 10 at (4,5):
199/200 memory, 300/300 phase_only_new, 172/200
plain, 300/300 oracle_ind, 240/300 oracle_new,
160/200 wrapped. In plain terms: most agents walk
to where the familiar door was and stay there,
repeatedly attempting the now-walled cell.

**2. Freeze before staging — plain/1 only (28 episodes).**
All 28 no-staging episodes are plain/1, phases
0-9 and 16-19, in both worlds identically: route
(0,0) -> (0,5) -> (1,5) -> (2,5) -> (3,5), then
187 steps at (3,5), last recorded (3,5), stall
no-move-run, revisits 0. This agent already had the
weakest W1 record (6/20 reached); its luck failure
is a pre-staging freeze, not a staging wait. It
never occupies any staging cell in these 28 episodes.

**3. West-side wandering loop — oracle_new/104 only
(22 episodes, no stall flag).**
All 22 stall=False episodes are oracle_new/104,
phases 19-29, in both worlds identically: arrival
(4,5), revisits 90-96, 16 distinct positions, last
recorded position varying along the west side
((4,5), (4,6), (4,4), (4,3), (4,2), (4,1)). The
trajectory cycles (4,5) -> (4,6) -> (4,5) -> (4,4)
-> ... -> (4,0) -> (4,1) -> ... without any
>= 10-tick single-cell run and without a >= 15-tick
single-block dwell, so no stall flag fires — yet
the episode still never enters a door and runs to
the cap. "No stall flag" here describes the
signature's thresholds, not successful behaviour.

**4. Staging-block cycling — wrapped dwell episodes (38).**
Wrapped is the only arm with a substantial dwell
count (38/200; oracle_new has 2). These episodes
cycle around the staging block rather than sitting
at one cell; see §0(4) for the dwell-definition
caveat in luck worlds. Wrapped's remaining 162
episodes are no-move-run. Wrapped blank-feature
episodes: 0 (§7), so this cycling is not accompanied
by a period-loss signature in the logged features.

**What failure is not:** it is not crushes (0 in
1500 episodes — with no entries, there is almost no
opportunity to be crushed; the one entered episode
leaves the gap cell for (6,3) on the next step, so
it is never in a gap cell when a door closes). It
is not door-switching (switches 0 in 1500). It is
not wandering to a rival staging cell and then
diverting: 403 episodes occupy a rival staging cell
at some point, but only 1 episode occupies a rival
staging cell without ever occupying (4,5) — the
entered episode itself — and 339 episodes arrive at
a rival staging cell first yet still end at the
familiar area without entering. Arrival at a rival
staging cell, on its own, is a route position, not
a door use.

## 6. The six pre-stated questions

**(1) Do agents pass at all without retraining?**
No. Success 0/1500 overall, 0/750 in each world,
0 in every arm (wrapped 0/200, plain 0/200, memory
0/200, oracle_ind 0/300, oracle_new 0/300,
phase_only_new 0/300). Every episode runs to the
200-tick cap. Contrast with the doors W1 baseline
for the same agents: 100% success for every arm
except plain (86%), so the change from W1 to
familiar-removed worlds is from near-total success
to zero success under this protocol — stated as a
contrast between two measured conditions, not as a
causal proof beyond the manipulation (removing
(5,5)) that defines the test.

**(2) Are passes diversions, route coincidences, or
wandering — does H_luck stand descriptively?**
There are no passes to classify. Class counts:
F 1499, D 0, R 0, W 1 (not reached), UNCLASSIFIED 0.
H_luck stands descriptively: zero class-D episodes,
let alone class-D passes. The single W episode is a
non-reached entry followed by a 191-tick far-side
freeze (§4), not a pass by any class.

**(3) What does failure look like?**
See §5. Dominantly, waiting/bumping at (4,5) for
most of the episode; secondarily, pre-staging freeze
(plain/1 at (3,5)); a west-side wandering loop
(oracle_new/104, no stall flag); staging-block
cycling (wrapped dwell). P2 trap signatures appear
in the recorded stall kinds (no-move-run 1438,
dwell 40), with the §0(4) dwell caveat, and the
two-cycle kind appears in 0 episodes. Crushes play
no role (0).

**(4) Does multi-rival (W5r) help vs single-rival
(W1r) — does more luck surface mean more passes?**
No, on the pass measure: W5r 0/750 reached vs W1r
0/750 reached. On the entry measure: W5r 1 entered
vs W1r 0 entered — a difference of one episode,
and that episode is the previously recorded
memory/1 far-side freeze (§4), not a new pass, so
it does not support "more gaps = more passes" as a
description of this test. Paired trajectories are
749/750 identical across the two worlds, which
bounds how much the extra gaps can have mattered
behaviourally: for 749 of 750 agent/phase pairs,
the east-side gap set made no difference at all to
the recorded trajectory, consistent with the
recorded fact that those agents never occupied an
east-side cell.

**(5) Wrapped vs plain vs oracle: does time-awareness
or perfect information help with unfamiliar locations?**
No, on outcome: wrapped 0/200, plain 0/200,
phase_only_new (raw clock) 0/300, oracle_ind 0/300,
oracle_new 0/300. Time-awareness and perfect
information change the *shape* of failure (oracle
arms cycle with high revisits — medians 46 and
39.5 — while plain/memory/phase arms mostly sit
with revisits median 0; wrapped cycles with
revisits median 55), but no arm produces a pass,
an entry (except memory/1 W5r), or a diversion.
On this test, neither time information nor perfect
information confers an unfamiliar-location
advantage in outcome terms.

**(6) After an unfamiliar entry, does the far side
freeze (P-Dc pattern) or cross cleanly?**
On the only available case (n = 1), it freezes:
memory/1 W5r phase 13 spends 191 post-entry ticks
at (6,3) with post-entry stall no-move-run and
never reaches the goal. That is the P-Dc pattern
as a description of this single episode. With n = 1
— and the same episode already recorded in doors
W5 — this does not establish a general far-side
freeze rate; it establishes that the only observed
unfamiliar entry ended in a far-side freeze, and
that the pattern recurs identically across the
doors and luck gap sets.

## 7. Wrapped blank-feature check

Wrapped episodes: 200; steps with f101/f102/f103:
40,000 (200 x 200, as expected). Blank episodes
(any step with f103 == 0.0): **0**. f103 value set
across all wrapped steps: {1.0} only. f101 spans
0.0-0.95 in 0.05 steps; f102 spans 0.0-1.0 in 0.1
steps. Non-wrapped steps with f keys: 0. The
doors-remedy period-loss signature (blank time
features) does **not** recur in the luck run. This
check is about the encoder's logged features only;
it does not by itself prove the absence of any
other encoder effect, and no such broader claim is
made here.

## 8. Bounds: what this does NOT show

- No claim beyond these worlds (W1r, W5r), arms,
  seeds (20/10 seeds 0-4; 30/14 seeds 100-104),
  greedy evaluation, and the no-retraining protocol.
- No claim about retraining (P-Db2): retraining is
  a separate, not-yet-ordered work order, and this
  test's total failure is the evidence that informs
  it, not a prediction of its outcome.
- No claim that agents "understand" or "decide"
  anything: D is a behavioural pattern (occupy
  (4,5), then enter a rival), and zero D episodes
  were observed, so no decision-pattern claim arises
  in either direction.
- No rate claim: 0/1500 is a count in this
  evaluation, not a universal zero. A different
  seed set, world, or protocol could in principle
  produce an entry or a pass; this test bounds only
  what was measured.
- No safety claim from crushes = 0: crushes require
  gap occupation at closing, which almost never
  occurred.

## 9. Follow-ups (PROPOSALS ONLY, none authorised)

- P-Db2 (retraining in schedule-varied worlds):
  still awaiting its own owner order; this result
  informs it but does not start it.
- A preregistered diversion-rate test in
  familiar-removed worlds (the P-Db1 direction
  applied here): would require its own preregistration;
  on the present evidence its expected D count is
  zero, so its design would need to state what, if
  anything, could change that expectation.
- P-Dc far-side probe: the single far-side freeze
  here (n = 1, previously recorded) does not replace
  a dedicated probe; a probe would still need its
  own design.
- P-Dd wrapped-eval protocol: no drift recurred
  here (0 blank episodes), so no new eval-protocol
  defect is evidenced by this run.

## Appendix: W1 medians used for W classification

Per (arm, seed): W1 n, reached, median ticks over
reached episodes, 2x threshold. Full table in
`luck_analysis_output.txt` §0. Values: memory
29.5/45.5/19.5/19.5/20.0 (seeds 0-4); oracle_ind
24.5/43.5/20.5/26.0/27.0 (100-104); oracle_new
22.0/41.5/39.0/23.5/34.5; phase_only_new
21.5/28.5/26.0/24.5/21.5; plain
20.5/20.5 (n_reached 6)/24.5/22.0/19.5; wrapped
22.5/21.5/24.5/22.5/21.5. Only one episode's W
classification invoked a threshold (memory/1:
200 > 91.0); all other episodes are F (no entry),
so no other threshold affects the headline.

# WO-TBM-doors — Step 2 analysis report (EXPLORATORY)

**Status:** complete. All conclusions EXPLORATORY (per the work order:
no two-conclusion rule, no confirmatory claims). Confirmatory
follow-ups, if any, return as proposals with their own preregistration.

**Question:** did agents trained with a single door learn "go through
a door" or "go to (5,5)"? Worlds share grid, wall, and schedule;
only the gap set varies. W1 = {(5,5)} (training world),
W2 = {(5,5), (5,1)}, W5 = {(5,1), (5,3), (5,5), (5,7), (5,9)}.
Same schedule for every door (20/10 for wrapped/plain/memory;
30/14 for oracle/phase_only). Greedy evaluation only, no retraining,
no agent modification, `world.py` untouched.

**Design note (carried from Step 0a):** Manhattan distance from (0,0)
to (9,9) via ANY door cell (5,y) is 18 moves, and with identical
schedules the arrival phase is identical too — the shortest-path
door is TIED across all doors. The optimal-door rate is therefore
1.0 trivially and optimal-vs-chance is vacuous. The informative
contrasts are familiar (5,5) vs nearest (5,1) vs other, plus
success, ticks, crushes, switches, and stalls.

## Sources and the wrapped/0,4 remedy

- Main run (`results_doors.jsonl`): 30 agents × W1/W1m/W2/W5.
  Step 0c passed for all 30 (re-trained agents reproduced tracked
  single-door greedy summaries exactly). Step 0b passed for 28/30.
- wrapped/0 and wrapped/4 FAILED 0b in the main run (second-eval-pass
  drift: the wrapped encoder's ScheduleModel keeps observing during
  eval; its period estimate evaporates mid-eval when the 32-event
  openings window fills with episode-boundary-flagged openings —
  diagnosed exactly in `diag_wrapped.py`). Per the work order they
  were reported, not evaluated, until explained.
- Remedy (`results_doors_remedy.jsonl`): each world evaluated on a
  FRESH deterministic retraining (end-of-training encoder state),
  with per-step time features logged. Fresh W1m MATCHES the tracked
  record for both seeds (wrapped/0: 23.50 ticks; wrapped/4: 22.40),
  completing 0b cleanly. The drifted main-run W1m rows for these two
  seeds are retained for audit and EXCLUDED from this analysis.
- Analysis source: wrapped W2/W5 (and wrapped/0,4 W1m) from the remedy
  file; everything else from the main file. The two files are never
  pooled. Analysis script: `doors_analysis.py`
  (output: `doors_analysis_output.txt`).

**Remedy caveat that strengthens the comparison:** wrapped/0's
schedule-blind episode (phase 14, 36 ticks, blank features) occurs
IDENTICALLY in its fresh W1m, W2, and W5 evals (same phase, same
ticks, familiar door, reached). It is matched across worlds, so it
does not confound the transfer comparison. It also means the tracked
A5 record for wrapped/0 itself contains a schedule-blind episode —
the wrapped arm's published numbers inherit this eval-order
sensitivity (see Limitations).

## Per-arm results (episodes: 20 phases × seeds; 30/14 arms: 30 phases)

| arm | world | succ | ticks | crush | stall | switches/ep | first door (fam/near/other) | FAIL/STOP/CHOOSE |
|---|---|---|---|---|---|---|---|---|
| wrapped (20/10, 5 seeds) | W1 | 1.000 | 23.25 | 0.020 | 0.050 | 0.00 | 100/0/0 | 0/5/95 |
| wrapped | W2 | 1.000 | 23.25 | 0.020 | 0.050 | 0.00 | 100/0/0 | 0/5/95 |
| wrapped | W5 | 1.000 | 23.42 | 0.030 | 0.050 | 0.02 | 100/0/0 | 0/5/95 |
| plain (20/10, 5 seeds) | W1/W2/W5 | 0.860 | 47.26 | 0.030 | 0.230 | 0.00 | 86/0/0 (+14 no-door) | 14/9/77 |
| memory (20/10, 5 seeds) | W1/W2 | 1.000 | 27.97 | 0.040 | 0.370 | 0.00 | 100/0/0 | 0/37/63 |
| memory | W5 | 0.990 | 29.61 | 0.040 | 0.370 | 0.00 | 99/0/1 | 1/36/63 |
| oracle_ind (30/14, 5 seeds) | W1 | 1.000 | 29.17 | 0.000 | 0.453 | 0.00 | 150/0/0 | 0/68/82 |
| oracle_ind | W2/W5 | 1.000 | 29.17/29.18 | 0.000 | 0.447 | 0.00/0.01 | 150/0/0 | 0/67/83 |
| oracle_new (30/14, 5 seeds) | W1 | 1.000 | 33.07 | 0.007 | 0.360 | 0.00 | 150/0/0 | 0/54/96 |
| oracle_new | W2/W5 | 1.000 | 33.07 | 0.007 | 0.313 | 0.00 | 150/0/0 | 0/47/103 |
| phase_only_new (30/14, 5 seeds) | W1/W2 | 1.000 | 25.69 | 0.007 | 0.227 | 0.00 | 150/0/0 | 0/34/116 |
| phase_only_new | W5 | 1.000 | 26.03 | 0.027 | 0.227 | 0.04 | 150/0/0 | 0/34/116 |

## The six questions

### Q1. Transfer at all? — YES, essentially complete.
Success in W2/W5 equals W1 success for every arm except memory in
W5 (0.99 vs 1.00 — one episode, see Q4). Ticks and crush rates are
nearly identical; the only per-agent deltas are the gap-brushing
crush episodes below (phase_only_new/102 W5: ticks 27.90 vs 26.20,
crush 0.133 vs 0.033; wrapped/3 W5: ticks 24.10 vs 23.25, crush 0.100
vs 0.050). No arm collapses, degrades, or improves in multi-door
worlds. Transfer is not the question — the door choice is.

### Q2. Which door? — The familiar door, overwhelmingly.
Across 1500 multi-door episodes (750 W2 + 750 W5), 1471 reached the
goal, and **all 1471 (100%) crossed first at the familiar door
(5,5)**. The nearest door (5,1) was NEVER entered first by any
agent in any episode. The single unfamiliar-door-first episode
(memory/1 W5 ph13, via (5,3)) failed. Per-agent first-door counts
are uniform: every agent is familiar-only in W2 and W5 except
memory/1 (19 familiar + 1 other in W5). The agents did NOT learn
"go through a door." They learned "go to (5,5)" — a location, not
a door concept — and that policy transfers intact to worlds where
(5,5) still exists.

### Q3. Do agents switch doors? — Effectively never.
5 switch episodes in 1500, all in W5, none of them a change of
crossing door:
- oracle_ind/104 W5 ph27: crossed (5,5), then stepped into (5,9)
  from the east side and back out (20 ticks vs 19 in W1) — a
  1-tick wall-hugging brush, not a switch.
- phase_only_new/102 W5 ph00/28/29 and wrapped/3 W5 ph15:
  crossed (5,5), brushed (5,7) from the east side, got CRUSHED in
  the gap, restarted, and re-crossed at (5,5). The "switch" is a
  crush-restart returning to the familiar door.
No agent ever adopted an unfamiliar door as its route.

### Q4. P2 traps at unfamiliar doors? — UNOBSERVABLE, with one exception.
Agents essentially never reach an unfamiliar door first, so there
is no population of unfamiliar-door episodes in which P2-style
traps could appear. The single exception is informative:
memory/1 W5 ph13 approached via (4,0)→(4,1)→(4,2)→(4,3), crossed at
(5,3), then froze at (6,3) for 191 ticks (no-move-run) and never
reached the goal. Mechanistically: in W1 the wall at (5,3) deflects
this approach route toward (5,5); in W5 the gap lets it through,
and the far-side policy from (6,3) is frozen — the wall itself was
doing part of the guidance in training. n=1: descriptive only, but
the mechanism is crisp. It does NOT generalize to a claim about
unfamiliar doors; it is one episode.

### Q5. Does time-awareness change the choice? — No observable difference.
Wrapped (time-aware) and plain (no time features) choose identically:
both familiar-only among reached episodes. What time-awareness
changes is success, not choice: wrapped succeeds 100% where plain
succeeds 86% (plain/1 fails the same 14 phases in W1, W2, and W5 —
a weak agent, world-independent). Door choice is independent of
the time representation in this experiment.

### Q6. Does perfect information choose better? — No.
Oracle arms also choose the familiar door exclusively (300/300
episodes each in W2/W5), with the same 1.00 success as wrapped and
memory — but slower (oracle_ind 29.17, oracle_new 33.07 ticks vs
wrapped 23.25) and with far higher stall rates (0.45/0.36 vs 0.05).
"Better" was defined as success/stalls/ticks, NOT optimal-door rate
(which is vacuous here): on that definition perfect information
chooses no better than learned inference, and stalls more.
Consistent with the P2 finding that oracle arms are slow/wandering.

## Two measurement notes (read before citing stall rates)

1. **Oracle stall-rate deltas are definitional, not behavioral.**
   oracle_new stalls drop 54→47 and oracle_ind 68→67 from W1 to
   W2/W5, but the affected trajectories are BYTE-IDENTICAL
   (ticks unchanged: e.g. oracle_new/102 ph04–10 at 45→45, 44→44…).
   The stall signature's dwell rule resets its run on staging
   cells (4,y) for each gap; W2/W5 add (4,1) to that set, so dwells
   passing through (4,1) fall below the 15-tick threshold. The
   agents stall identically; the flag changes because the staging
   set changes. Do not cite this as "agents stall less with more
   doors."
2. **The east-side route is wall-hugging, and unfamiliar gaps turn
   clipped steps into crushes.** After crossing (5,5), agents walk
   (6,5)→(6,6)→(6,7)… hugging the wall. In W1, a westward step at
   (6,7) is clipped by the wall (harmless). In W5 it ENTERS the
   (5,7) gap, and if the door then closes, the agent is crushed.
   This — not door choice — explains every W5 tick/crush delta
   (phase_only_new/102, wrapped/3). The multi-door world does not
   change the agents' decisions; it changes the consequences of
   their wall-hugging route.

## Limitations (EXPLORATORY bounds)

- Greedy evaluation only; no learning, retraining, or adaptation in
  multi-door worlds. Whether agents would LEARN unfamiliar doors
  with training is a different experiment (proposal, not run).
- Single schedule for all doors; schedule-varied doors (does an
  agent follow the familiar door or the open door?) not tested.
- 5 seeds per arm (30/14 arms: seeds 100–104; 20/10 arms: seeds 0–4);
  no statistical test is claimed; all numbers are descriptive.
- Stall signature is structural (see note 1); stall comparisons
  across worlds with different gap sets inherit the staging-set
  definitional edge.
- Wrapped-arm evals carry the ScheduleModel eval-order sensitivity
  diagnosed here (period loss in short-episode evals); wrapped/0's
  blank-feature episode is matched across worlds, but the wrapped
  arm's single-door numbers generally inherit this sensitivity.
- One episode (memory/1 W5 ph13) is the entire evidence for
  unfamiliar-door-first behavior; no generalization is licensed.

## Follow-ups (PROPOSALS ONLY — none authorized)

- P-Da: retrain on multi-door worlds (does the familiar-door
  lock-in persist when (5,5) is not the only gap during training?).
- P-Db: schedule-varied doors (same gaps, different schedules —
  familiar-door vs open-door competition).
- P-Dc: far-side policy probe (why does the policy freeze at
  (6,3)? Is far-side behavior location-specific?).
- P-Dd: wrapped-encoder eval protocol (freeze the ScheduleModel
  during evaluation, or evaluate with fresh-state encoders as
  standard — a protocol question, not an agent change).

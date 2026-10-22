# WO-TBM-P-Db — Step 2 analysis report (EXPLORATORY)

**Status:** complete. All conclusions EXPLORATORY (per the work
order: no two-conclusion rule, no confirmatory claims).
Confirmatory follow-ups return as proposals with their own
preregistration.

**Question.** The doors experiment (`2ad12f7`) showed agents
take the familiar door (5,5) in 100% of successful multi-door
episodes — but every door shared one schedule, so no door could
be "better" in timing. When doors differ in timing, does the
agent follow the familiar door or the open door? At an arrival
where the familiar door is closed and a rival door is open, does
the agent wait at (5,5) or walk to the open rival?

## Design recap (fixed in Step 0a, before any evaluation)

Same grid, wall, and gap sets as the doors experiment
(W2 = {(5,5), (5,1)}, W5 = {(5,1), (5,3), (5,5), (5,7),
(5,9)}), with per-door schedules only. Familiar door (5,5)
keeps each arm's training schedule (20/10 or 30/14). W2s:
single pure anti-phase rival, open exactly when the familiar
door is closed (20/10: (5,1) offset 10; 30/14: (5,1) open_len
16, offset 14). W5s: four staggered rivals (20/10 offsets
10/14/18/2; 30/14 offsets 14/20/26/2, open_len 16), so at least
one rival is open at every tick. Door-open rule:
`(t - offset) % period < open_len`. Observe contract:
`observe()` returns the FAMILIAR door's state and
`period`/`open_len` = the familiar schedule — rival schedules
appear in no agent's features; rivals are discoverable only
through movement, blocking, and crushing.

**Competition definition (fixed before measuring).**
Arrival = first occupation of any staging cell (4,y).
Conflict episode = familiar closed at arrival AND ≥1 rival
open at arrival. Outcomes from the first door entered:
`takes-open-rival` (a rival open at arrival),
`waits-at-familiar` (the familiar door), `other` (a rival
closed at arrival, or no door entered).

**Ticks convention (read before comparing numbers).**
Ticks below are means over REACHED episodes. The
same-schedule comparison recomputes the doors records on
this same definition, so it is apples-to-apples. (The doors
report's per-arm ticks averaged over ALL episodes including
200-tick fails — e.g. its plain 47.26 — and must not be
compared directly with the numbers here.)

## Sources and run procedure

- Variant `world_doors_sched.py` (new file; `world.py` and
  `world_doors.py` untouched). Regression (0b): scripted
  mechanics 600/600 identical with equal schedules, plus
  agent-level regression on one seed per arm (12/12 MATCH).
  The work order's regression read as full 30-agent; the
  subset scope + reason were fixed in the design note before
  running (full agent-level regression = ~2.25h redundant
  compute; scripted check covers mechanics exactly).
- Runner `sched_run.py`, records `results_doors_sched.jsonl`
  (1500 episodes, flushed per agent-world, resume-safe).
  All 30 agents passed 0c (tracked single-door summary
  exact). Wrapped used the doors-remedy discipline: 0c on
  its own training + one fresh training per sched world
  (a wrapped encoder cannot serve both a 0c eval and a
  sched eval without drift — design-note amendment, before
  any eval). Other arms ran sequentially (W1 → W2s → W5s)
  on one encoder.
- **Restart:** the runtime restarted 2h45m into the main
  run and killed it. Records on disk were verified
  (wrapped/plain/memory/oracle arms 10/10 complete); the
  run was relaunched for the 3 remaining phase_only_new
  agents only (~13 min). No records were lost or repeated.
- Analysis: `sched_analysis.py`
  (output: `sched_analysis_output.txt`).

## Per-arm results (sched worlds; ticks over reached episodes)

| arm | world | succ | ticks | crush | stall | first door fam/near/other/none | conflict (wait/take/other) |
|---|---|---|---|---|---|---|---|
| wrapped (20/10) | W2s | 1.000 | 23.25 | 0.020 | 0.050 | 100/0/0/0 | 34 (34/0/0) |
| wrapped | W5s | 1.000 | 23.13 | 0.020 | 0.050 | 94/0/6/0 | 34 (28/6/0) |
| plain (20/10) | W2s | 0.860 | 22.40 | 0.030 | 0.230 | 86/0/0/14 | 30 (30/0/0) |
| plain | W5s | 0.860 | 22.40 | 0.030 | 0.230 | 86/0/0/14 | 39 (39/0/0) |
| memory (20/10) | W2s | 1.000 | 27.97 | 0.040 | 0.370 | 100/0/0/0 | 48 (48/0/0) |
| memory | W5s | 0.990 | 27.89 | 0.040 | 0.370 | 99/0/1/0 | 46 (45/1/0) |
| oracle_ind (30/14) | W2s | 1.000 | 29.17 | 0.000 | 0.447 | 150/0/0/0 | 19 (19/0/0) |
| oracle_ind | W5s | 1.000 | 29.18 | 0.000 | 0.447 | 150/0/0/0 | 27 (27/0/0) |
| oracle_new (30/14) | W2s | 1.000 | 33.07 | 0.007 | 0.313 | 150/0/0/0 | 32 (32/0/0) |
| oracle_new | W5s | 1.000 | 32.61 | 0.007 | 0.313 | 136/0/14/0 | 48 (34/14/0) |
| phase_only_new (30/14) | W2s | 1.000 | 25.69 | 0.007 | 0.227 | 150/0/0/0 | 59 (59/0/0) |
| phase_only_new | W5s | 1.000 | 25.79 | 0.013 | 0.227 | 150/0/0/0 | 63 (63/0/0) |

Totals: 1500 sched episodes, 1471 reached. First door among
reached: 1451 familiar (98.6%), 20 other (1.4%), 0 nearest.
Conflicts: 479 (222 W2s + 257 W5s) → 458 waits (95.6%),
21 takes (4.4%), 0 other. No-arrival episodes: 28 (all
plain/1 fails, world-independent).

## The six questions

### Q1. In conflict episodes, rival vs familiar? — Familiar, 458 to 21.
In 479 conflict episodes, agents waited at the familiar door
458 times and took an open rival 21 times. In W2s — the
purest competition (single anti-phase rival) — the count is
222 conflicts, 0 takes. Every take occurred in W5s. But the
raw 458:21 understates the lock-in, because of the
arrival-cell cut below: **zero** takes came from an agent
that arrived at the familiar door. The competition, as a
choice, was never won by a rival.

### Q2. Does schedule variety break the (5,5) lock-in? — Essentially no.
First-door familiar falls from 100% (same-schedule reached
episodes) to 98.6% (sched reached episodes), and the nearest
door is still never entered first by any agent. The 1.4% of
other-first episodes are route artifacts (next section),
not broken lock-in: no agent diverted from the familiar door
to a rival in any episode.

### The arrival-cell cut (the result the headline needs)
- **All 21 takes arrived at RIVAL staging cells** (first
  occupation of (4,7), (4,3), or (4,1)) and entered the open
  door directly in front of them (staging wait 1 tick).
  The agent's route brought it to a rival staging cell; the
  door happened to be open; it walked in. In same-schedule
  worlds the same routes arrive at the same cells, but the
  rival door is closed at those ticks (all doors share the
  familiar schedule), so the agent is deflected to (5,5).
  Schedule variety unblocks the route's natural passage —
  it does not create a choice.
- **211 waits arrived at RIVAL staging cells and walked past
  the open rival to wait at the familiar door.** This is the
  strongest lock-in evidence in the experiment: agents stand
  next to an open rival door — in memory/1's case, waiting
  up to 42 ticks at the familiar door afterward — and still
  refuse it. Waits-at-familiar staging waits: mean 7.5,
  median 5, max 42 ticks.
- The three take sources, all route artifacts (verified
  phase-by-phase against the same-schedule records):
  - wrapped/2 W5s ph03–08 (6 takes): route passes (4,3);
    (5,3) open at arrival → enters, clean and fast
    (19–24 ticks, no stall). In same-schedule W5 the SAME
    phases pass (4,3) but deflect to (5,5) (21–26 ticks) —
    exactly 2 ticks slower per phase. Same route; only the
    door state differs.
  - oracle_new/101 W5s ph00–13 (14 takes): route passes
    (4,7); (5,7) open at arrival → enters, then freezes
    on the far side (no-move-run) on all 14, reaching
    slowly (53–66 ticks). In same-schedule W5 the SAME
    phases pass (4,7) but deflect to (5,5) (58–71 ticks)
    — exactly 5 ticks slower per phase. Taking the rival
    saved 5 ticks of walking yet still cost a far-side
    freeze on every take.
  - memory/1 W5s ph13 (1 take): walked past the open
    (5,1) (policy at (4,1) is south, not east), entered
    (5,3), and froze at (6,3) for the rest of the episode
    — the same far-side freeze as its doors-experiment
    W5 failure, recurring. It is labeled a "take" only
    because (5,3) was open at arrival; mechanistically it
    is the known freeze, not rival exploitation.

### Q3. Success / ticks / crushes vs same-schedule worlds? — Nearly identical.
Arm-level success is unchanged everywhere (only memory
W5s keeps its single fail, 0.990, the same ph13 episode).
Ticks are unchanged except oracle_new W5s (32.61 vs 33.07
— agent 101's slow takes averaged against its fast waits)
and wrapped W5s (23.13 vs 23.42 — wrapped/2's fast takes).
Crushes are unchanged except phase_only_new W5s (0.013 vs
0.027 — fewer wall-hugging brush-crushes under staggered
schedules) and wrapped W5s (0.020 vs 0.030). Schedule
variety changes almost nothing about outcomes, because it
changes almost nothing about behavior: the agents run the
same routes, and only the door states along those routes
differ.

### Q4. P2 trap signatures while waiting at a closed familiar door? — Yes, but mostly waiting.
98 of 479 conflict episodes carry a structural stall flag
(83 waits, 15 takes; kinds: no-move-run 91, two-cycle 4,
dwell 3). The flag conflates two things the signature
cannot distinguish: legitimate door-waiting (waiting ≥10
ticks at a staging cell trips the no-move-run rule, then
the agent crosses and reaches — this is most of the 83
stalled waits) and genuine freezing. Genuine trap-like
events in conflict episodes: (a) memory/1 W5s ph13 —
far-side freeze after entering (5,3), never recovers,
FAIL (the only catastrophic trap); (b) oracle_new/101's
14 takes — far-side no-move-run freezes after crossing
(5,7), all recovered, all slow. Rival-taking, where it
occurs, comes WITH far-side freezing. No agent waiting at
the familiar door while a rival is open ever failed
because of the wait itself.

### Q5. Does time-awareness change the competition outcome? — No.
Wrapped conflicts: 68, takes 6 (all wrapped/2 route
artifacts); plain conflicts: 69, takes 0. Both arms wait
overwhelmingly; neither diverts from the familiar door to
a rival (zero diversion takes for both). What differs is
success (wrapped 1.000 vs plain 0.860, the plain/1 weak
agent, world-independent), not competition behavior.
No evidence that time-awareness helps exploit open rivals.

### Q6. Do perfect-info arms exploit open doors better? — No.
Oracle conflicts: 126, takes 14 — all from ONE agent
(oracle_new/101, route artifact, slower + stalled on every
take). Oracle_ind takes 0 in 46 conflicts. Learned arms
take 7 (6 wrapped/2 + 1 memory/1), also all route
artifacts. No arm exploits open doors in the decision
sense; the one repeated taker pays for it on every take.
"Better" was defined as success/stalls/ticks (optimal-door
rate is vacuous — all doors are 18 moves): on that
definition, perfect information confers no competition
advantage here.

## Two further notes

- **Switches nearly vanish under staggered schedules** (2 in
  1500 sched episodes, vs 5 in same-schedule W5): the
  oracle_ind/104 (5,9) brush (same episode as doors) and
  one phase_only_new/102 (5,7) brush-crush. No agent ever
  switches its crossing door.
- **Wrapped/0's schedule-blind episode recurs identically**
  (W2s ph14 and W5s ph14: 6 blank-feature steps, 36 ticks,
  familiar door, reached, conflict episode — it waited at
  the familiar door). It is matched across worlds and does
  not confound the transfer comparison, as in the doors
  remedy.

## Limitations (EXPLORATORY bounds)

- Greedy evaluation only; no learning or adaptation in
  schedule-varied worlds. Whether agents would LEARN to
  exploit rivals with training is a different experiment.
- Rival schedules are NOT in any agent's features (observe
  contract) — the test measures behavioral response to
  rival timing, not feature-based rival tracking. An agent
  with rival schedules encoded might behave differently;
  that is a different agent, not tested here.
- The structural stall signature cannot distinguish
  legitimate door-waiting from freezing (no-move-run rule);
  Q4's stall counts must be read with that conflation.
- 5 seeds per arm; no statistical test is claimed; all
  numbers are descriptive. The takes rest on 3 agents
  (21 episodes); the "zero diversion takes" rests on 479
  conflicts with zero counterexamples in the diversion
  direction (arrival at familiar staging → rival).
- Single schedule assignment per world (fixed offsets);
  other staggerings could produce different arrival
  patterns. No claim beyond these worlds, arms, seeds.

## Follow-ups (PROPOSALS ONLY — none authorized)

- P-Db1: a TRUE choice test, preregistered: episodes where
  the agent arrives at familiar staging (4,5) with the
  familiar door closed and a rival open — the diversion
  direction this experiment found zero of — with a
  two-conclusion rule on the diversion rate.
- P-Db2: retraining in schedule-varied worlds (does the
  lock-in persist when rivals are open during training?).
- P-Dc (carried): far-side policy probe — why do policies
  freeze at (6,3) (memory/1, catastrophic) and after
  (5,7) (oracle_new/101, recovering)? Is far-side behavior
  location-specific?
- P-Dd (carried): wrapped-encoder eval protocol (fresh-state
  encoders as standard for wrapped evals).

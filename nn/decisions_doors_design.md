# Multi-door transfer evaluation — design note (WO-TBM-doors)

**Status: EXPLORATORY.** No two-conclusion rule, no
confirmatory claim. Findings suggesting a confirmatory
follow-up return as proposals with their own
preregistration. This note is committed BEFORE any
evaluation run; the evaluation must follow it, and any
deviation is reported with its reason.

## Question

Trained agents learned one wall with one door gap at
(5,5). Did they learn "go through a door," or did
they learn "go to (5,5)"? In worlds with 2 and 5 doors,
do they fail, stop (freeze/loop), or choose — and if
they choose, do they choose the familiar door, the
nearest door, or the optimal door?

## Worlds (position only; one variable at a time)

Same 10×10 grid, same wall at x=5, same schedule for
every door (all doors share one door bit: open iff
tick % period < open_len). Doors with different
schedules are a follow-up, NOT part of this work order.

- **W1 (single-door baseline, the training world):**
  gaps = {(5,5)}.
- **W2:** gaps = {(5,5), (5,1)}. The new gap (5,1) is
  the nearest door to spawn (Manhattan distance 6,
  vs 10 for the familiar (5,5)) — chosen to maximize
  the familiar-vs-nearest contrast.
- **W5:** gaps = {(5,1), (5,3), (5,5), (5,7), (5,9)}.
  Spread across the column, including the familiar
  gap; nearest to spawn is again (5,1).

Crush rule (all worlds): an agent in ANY door cell
when the door is closed is crushed back to start.
Blocked rule: entering any door cell while closed is
blocked. Wall cells are x=5 except the gaps.

## Agents (checkpoints named from the repo)

No trained weights are stored in the repo; a
"checkpoint" here means (arm, seed, protocol, code
at the evaluation commit), realized by deterministic
re-training with frozen code and verified by the
Step-0c rule (exact single-door greedy-summary match
— the P2 precedent). An arm-seed that does not
reproduce is reported, not evaluated, until the
mismatch is explained.

| arm | encoder / trainer | protocol | seeds |
|---|---|---|---|
| wrapped | WrappedEncoder / train_wrapped | 20/10 | 0–4 |
| plain | PlainEncoder / train_neural | 20/10 | 0–4 |
| memory | MemoryEncoder / train_memory | 20/10 | 0–4 |
| oracle_ind | OracleIndEncoder / train_oracle_ind | 30/14 | 100–104 |
| oracle_new | OracleNewEncoder / train_oracle_new | 30/14 | 100–104 |
| phase_only_new | PhaseOnlyNewEncoder / train_phase_only_new | 30/14 | 100–104 |

Seeds are the pilot seeds of each arm family (A5
seeds 0–4; pilot seeds 100–104). Six arms × five
seeds = 30 trainings; at the P2-measured ~4.5
min/training, the training phase is ~2–2.5 hours
of background running, and evaluation itself is
minutes (greedy episodes only).

## Evaluation protocol

Each agent is evaluated on the protocol it was
trained on (transfer across worlds, NOT across
protocols): 20/10 arms on 20/10 worlds, 30/14 arms
on 30/14 worlds. No protocol standardization is
imposed, so no protocol-transfer confound is
introduced; cross-protocol arm comparisons are
descriptive only, never pooled.

Greedy evaluation only: one deterministic episode
per start phase, all phases (30 episodes per
30/14 agent-world, 20 per 20/10 agent-world). No
training, no exploration, no agent modification.
Episode limit: EPISODE_CAP = 200 ticks (from the
code, unchanged).

## Per-episode endpoints (all from the episode trace)

- Success (goal reached within 200 ticks); ticks;
  crushes.
- First door cell entered (or none, for failures).
- Door switches: entered one door cell, later
  entered a different door cell.
- Stall signature (structural, agent-independent —
  any one suffices): (a) a run of ≥ 10 consecutive
  ticks with no position change (stay / clipped
  moves — the P2 spawn-freeze signature); (b) a
  2-cycle (two cells alternating) spanning ≥ 10
  ticks (the P2 escape-cycle signature); (c) dwell
  ≥ 15 ticks in one fixed 3×3 block (the P2 dwell
  measure, staging interval excluded where a staging
  interval exists — in multi-door worlds the
  staging interval is defined per door entry, see
  analysis script).
- Dwell and revisit counts (P2 definitions) reported
  descriptively per agent-world.

## The optimal-door fact (fixed in advance)

Manhattan distance from spawn (0,0) to goal (9,9)
via ANY door cell (5,y) is 5 + y + 4 + (9 − y) =
18 moves, independent of y; with identical schedules
for all doors, arrival phase is identical too. **The
shortest-path door is therefore TIED across all
doors, in every world of this evaluation.** The
optimal-door set of every episode is all doors, and
the optimal-door rate is 1.0 trivially for any door
entered. The optimal-vs-chance comparison is
vacuous here and is NOT reported as a finding. The
informative contrasts are familiar vs nearest vs
other, plus effective ticks per door chosen
(descriptive). Ties are counted explicitly by this
paragraph, never broken silently.

- Familiar door: (5,5) in all worlds.
- Nearest door: shortest Manhattan distance from
  spawn (0,0) to the door cell — (5,1) in both W2
  and W5 (distance 6); no tie.
- Chance baseline (descriptive only): uniform choice
  among doors would give familiar rate 1/2 in W2
  and 1/5 in W5. It is a straw baseline, not a null
  model — agents are not random door-choosers.

## Untrained states (stated, not a flaw)

New door cells (5,1), (5,3), (5,7), (5,9) were wall
cells in training — unreachable, so the networks'
Q-values there come from initialization and
generalization, never from experience. Likewise,
moving right from (4,y) for y ≠ 5 was blocked in
training. The evaluation measures generalization
into exactly these untrained state-action pairs;
that is the point of the test, and any failure
there is a finding, not a reason to retrain.

## Pre-stated classifications

- **FAIL:** episode does not reach the goal within
  200 ticks.
- **STOP:** episode reaches the goal but carries a
  stall signature (above).
- **CHOOSE:** first-door distribution analyzed
  against familiar-door rate, nearest-door rate,
  and other-door rate (optimal rate reported as the
  trivial 1.0 with the tie explanation above).

## Pre-stated questions

1. Do agents transfer at all (success rate in W2/W5
   vs single-door baseline)?
2. Which door do they take — familiar, nearest, other?
3. Do they switch doors mid-episode?
4. Do P2 trap signatures appear at unfamiliar doors?
5. Does time-awareness change door choice (wrapped
   vs plain)?
6. Do perfect-info arms (oracle) choose better than
   learned-info arms?

"Better" in Q6 means: higher success, fewer stalls,
and (descriptively) fewer ticks — NOT optimal-door
rate, which is trivial (see above).

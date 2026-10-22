# WO-TBM-Luck — Step 0a design note (committed before any evaluation)

**Status: EXPLORATORY.** No two-conclusion rule, no
confirmatory claims. Confirmatory follow-ups return as
proposals with their own preregistration. This test runs
BEFORE P-Db2 (retraining) and informs it; P-Db2 is NOT
folded in here.

**Question.** P-Db showed agents never divert from the
familiar door to a rival while the familiar door exists
(zero diversions in 479 conflicts; the 21 takes were
route artifacts). This test removes the familiar door
entirely: with (5,5) gone, can an already-trained agent
use a door it never used — without any retraining? And
if it passes, was it a decision, or luck?

**The luck hypothesis H_luck (stated before measuring).**
Any pass through an unfamiliar door without retraining
is explained by route coincidence — the agent's familiar
route happens to pass an open rival gap — NOT by a
decision to use the unfamiliar door. Operationally: zero
passes via diversion (class D below). A single class-D
pass is a descriptive counterexample to H_luck (a genuine
unfamiliar decision observed). Only class-R/W passes, or
total failure, leave H_luck standing descriptively.
Failure is a complete result, not a failed experiment:
total failure would bound the lock-in absolutely (no
unfamiliar-door use without retraining).

## Worlds

Same 10×10 grid, same wall at x=5, same training
schedules for every door (20/10 or 30/14 per the arm's
protocol) — timing stays FAMILIAR, so location is the
ONLY unfamiliar axis. No schedule variation (that axis
belongs to P-Db, already tested). Gap sets:
- W1r = {(5,1)}: single rival gap only. Isolates pure
  location unfamiliarity (a door exists, just not at
  (5,5)).
- W5r = {(5,1), (5,3), (5,7), (5,9)}: rivals only, no
  (5,5). Adds choice among unfamiliar doors and more
  luck surface (more gaps for a route to hit).
Mechanics are exactly MultiDoorWorld's (single schedule
for all doors, same wall/blocking/crush rules); the
variant file exists to name and freeze these worlds, not
to reimplement mechanics (see Step 0b).

## Agents

The same 30 trained agents as the doors / P-Db
experiments, named from the repo: wrapped/plain/memory,
20/10, seeds 0–4; oracle_ind/oracle_new/phase_only_new,
30/14, seeds 100–104. No retraining, no agent
modification, no tuning to the new worlds. Greedy
evaluation only (one deterministic episode per start
phase, all phases: 20 for 20/10 arms, 30 for 30/14
arms). No retraining under any circumstance in this
work order; retraining is P-Db2, which comes after,
informed by this result.

## Pass classification (fixed before measuring)

Every episode gets exactly one class, from its
trajectory. First entry = the first tick at which the
agent occupies any gap cell. Priority order is fixed
(an episode satisfying several rules takes the FIRST
matching class):
1. **F (fail):** never enters any door (freeze, timeout
   at 200 ticks, or crush-loop without entry). Note: an
   episode that enters a door but never reaches the goal
   is NOT F — it takes the class of its entry (D/R/W)
   with reached=False recorded separately.
2. **D (diversion):** the agent occupies the familiar
   staging cell (4,5) at a tick BEFORE the first-entry
   tick, and then enters a rival door. This is the
   decision pattern: it found the familiar door gone
   (the cell (4,5) is now ordinary floor west of a wall
   cell) and went elsewhere. Caveat, stated now: D as
   defined is a behavioral pattern, not proof of intent
   — "decision" language in this test stays
   descriptive. A pass that goes (4,5) → rival door is
   D even if it is also long or revisit-heavy (D takes
   precedence over W).
3. **W (wandering):** not D, enters a rival door, AND
   (ticks > 2 × the agent's W1 median ticks over
   reached episodes) OR (revisits > 10). The W1 median
   is computed per (arm, seed) from the doors run's W1
   records (reached episodes only). Neither a direct
   route nor a diversion.
4. **R (route coincidence):** not D, not W: enters a
   rival door on a direct route (the P-Db take pattern).
   This includes entries after a crush-restart, as long
   as the entry itself is not D and not W by the
   thresholds above.
5. **UNCLASSIFIED:** a pass that cannot be classified
   under the fixed rules (e.g. missing trajectory data).
   Reported with its trajectory, never force-fitted.
   Expected count: 0 (the rules above are exhaustive
   over complete trajectories).
Class is determined by the FIRST entry only; later
door switches do not reclassify an episode.

## Per-episode endpoints

Success; ticks; crushes; first door entered (or none);
door_entries; switches; staging cells occupied in
first-occupation order (especially (4,5) vs rival
cells); arrival cell (first staging occupation);
stall signature (P2 measures, full episode); far-side
behavior after entry: post-entry stall flag/kind
(structural signature applied to post-entry positions
only) and post-entry ticks (entry tick to goal, or to
episode end if not reached) — the P-Dc connection.
Steps (position/action/door-bit/crush per tick; wrapped
evals also log f101/f102/f103 for period-loss
detection).

## Pre-stated questions

(1) Do agents pass at all without retraining (success
rate per arm/world)? (2) Are passes diversions, route
coincidences, or wandering — does H_luck stand
descriptively? (3) What does failure look like (freeze
at (4,5)? wandering? crushes? P2 trap signatures?)?
(4) Does multi-rival (W5r) help vs single-rival (W1r)
— does more luck surface mean more passes? (5) Wrapped
vs plain vs oracle: does time-awareness or perfect
information help with unfamiliar locations? (6) After an
unfamiliar entry, does the far side freeze (P-Dc
pattern) or cross cleanly?

## Regression check (0b) — scope stated before running

The new variant (`world_luck.py`) is a NEW file;
`world.py`, `world_doors.py`, and `world_doors_sched.py`
are not modified. Mechanics are inherited from
MultiDoorWorld deliberately (the luck axis is
location-only; reimplementing identical mechanics would
add bug surface for zero information). Regression:
(i) scripted mechanics check — scripted + pseudo-random
action sequences on MultiDoorWorld vs LuckWorld, gaps
{(5,5)} and luck gaps, all start phases → identical
positions, crushes, logs, and `observe()` bits;
(ii) agent-level regression on ONE seed per arm
(6 agents) with gaps {(5,5)} on LuckWorld → summaries
exactly equal to the tracked single-door greedy
records. Scope rationale, as in P-Db: a full 30-agent
agent-level regression would cost ~2.25h redundant
compute; the scripted check covers mechanics exactly
and the subset covers encoder integration. This scope
is fixed here, before running.

## Wrapped-encoder discipline (doors remedy, carried over)

Wrapped is the only arm whose encoder keeps state
across episodes (ScheduleModel). Its 0c check runs on
its OWN training, and each luck world is evaluated on
a separate FRESH training (no preceding eval), with
per-step time features logged. Other arms run
sequentially (W1 → W1r → W5r on one encoder) after the
0c check. If drift recurs: report, do not evaluate,
until explained; drifted rows retained for audit and
excluded; files never pooled.

## Compute estimate (no JIT trigger)

Regression: 6 trainings (~27 min). Main run: 25
sequential trainings (non-wrapped) + 15 wrapped
trainings (5 0c + 10 eval) = 40 trainings (~3h). Total
≈ 3.5h, under both JIT-trigger thresholds (4h; 150
arm-runs: 46 trainings, 60 luck evals + checks < 150).
No trigger fires; no JIT work (ruled E, trigger-gated).

## Out of scope (explicitly forbidden)

No retraining, no agent modification, no tuning to the
new worlds. No schedule variation (timing stays
familiar). No edits to `world.py`, the doors/sched
variants, or any frozen file. No claims beyond the
worlds, arms, and seeds evaluated. No JIT work. No
other tracks. No AI-approach work (J1/J2/J3 belong to
the reciter track, not this session).

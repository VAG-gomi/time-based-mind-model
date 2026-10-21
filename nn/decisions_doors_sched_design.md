# WO-TBM-P-Db — Step 0a design note (committed before any evaluation)

**Status: EXPLORATORY.** Tests generalization of already-trained
agents to worlds where doors differ in schedule. NOT confirmatory:
no two-conclusion rule, no confirmatory claims. Confirmatory
follow-ups return as proposals with their own preregistration.

**Question.** The doors experiment (`2ad12f7`) showed agents take
the familiar door (5,5) in 100% of successful multi-door episodes
— but every door shared one schedule, so no door could be
"better" in timing. When doors differ in timing, does the agent
follow the familiar door or the open door? At an arrival where
the familiar door is closed and a rival door is open, does the
agent wait at (5,5) or walk to the open rival?

## Worlds

Same 10×10 grid, same wall at x=5, same gap sets as the doors
experiment: W2 = {(5,5), (5,1)}, W5 = {(5,1), (5,3), (5,5),
(5,7), (5,9)}. The ONLY change is the schedule assignment,
stated exactly per door per world below. Schedule-varied
worlds are named W2s / W5s.

**Door-open rule (fixed):** a door with (period, open_len,
offset) is open at tick t iff `(t - offset) % period <
open_len`. Offset 0 reproduces `door_is_open` exactly.

**Familiar door (5,5)** keeps each arm's training schedule:
- 20/10 protocol (wrapped/plain/memory): period 20, open_len 10,
  offset 0 → open on ticks ≡ [0,10) mod 20.
- 30/14 protocol (oracle_ind/oracle_new/phase_only_new):
  period 30, open_len 14, offset 0 → open on [0,14) mod 30.

**W2s — pure anti-phase rival.** The single rival (5,1) is open
exactly when the familiar door is closed:
- 20/10: (5,1): period 20, open_len 10, offset 10 → [10,20).
- 30/14: (5,1): period 30, open_len 16, offset 14 → [14,30)
  (open_len 16 = the familiar-closed interval length).

**W5s — staggered rivals.** Rivals keep the anti-phase open
length (10 / 16) with staggered offsets, so that at any tick
at least one rival is open, and during familiar-closed at
least one rival is always open (the offset-10/14 rival is open
throughout the familiar-closed interval):
- 20/10: (5,1): offset 10 → [10,20); (5,3): offset 14 →
  [14,20)∪[0,4); (5,7): offset 18 → [18,20)∪[0,8);
  (5,9): offset 2 → [2,12). Union = [0,20) fully covered.
- 30/14: (5,1): offset 14 → [14,30); (5,3): offset 20 →
  [20,30)∪[0,6); (5,7): offset 26 → [26,30)∪[0,12);
  (5,9): offset 2 → [2,18). Union = [0,30) fully covered.

**Observe contract (fixed, scope-critical).** The variant's
`observe()` returns the FAMILIAR door's state as the door bit,
and exposes `period`/`open_len` = the familiar door's schedule.
Rationale: every encoder was trained on single-bit
familiar-door input, and OracleIndEncoder reads
`world.period`/`world.open_len` at encode time. Rival
schedules are NOT encoded in any agent's features — the agent
can discover rivals only through movement, blocking, and
crushing. This is a stated property of the test, not a hidden
choice: it measures whether behavior (not features) responds
to rival timing.

**Mechanics (mirror of world.py / MultiDoorWorld, per door).**
Wall cells x=5 non-gap always block. Cannot move into gap
(5,y) while THAT door is closed (blocked, agent stays).
Crush rule, per door: agent in gap cell (5,y) at a tick when
that door is closed → sent to start, tick advances, log token
"crushed" (identical token so `env_step` detects it as before).
Grid-edge clipping as in world.py.

## Competition definition (fixed before measuring)

- **Arrival:** first tick at which the agent occupies any
  staging cell (4,y) for a gap (5,y). Recorded: arrival tick,
  arrival cell, familiar open/closed at arrival, set of rivals
  open at arrival. Episodes that never occupy a staging cell
  have arrival None and are excluded from conflict analysis
  (they still count in success/ticks/crushes/first-door).
- **Conflict episode:** familiar door closed at arrival AND at
  least one rival door open at arrival.
- **Conflict outcomes** (conflict episodes only), from the first
  door cell entered:
  - `takes-open-rival`: first door entered is a rival that was
    open at arrival.
  - `waits-at-familiar`: first door entered is (5,5).
  - `other`: first door entered is a rival that was closed at
    arrival, or no door entered.
- **Staging wait:** ticks from arrival to first door entry
  (None if no entry). **Door status at crossing:** per-door
  open/closed at the first-entry tick. Switches and the P2
  structural stall signature are reused verbatim from the doors
  experiment (staging set = (4,y) per gap in that world).

## Agents

The same 30 trained agents as the doors experiment, named from
the repo: wrapped/plain/memory, 20/10, seeds 0–4;
oracle_ind/oracle_new/phase_only_new, 30/14, seeds 100–104.
No checkpoints are stored; a checkpoint = (arm, seed,
protocol) realized by deterministic retraining (doors/P2
precedent). Each agent is evaluated on the protocol it was
trained on.

## Evaluation protocol

Greedy evaluation only: one deterministic episode per start
phase, all phases (20 for 20/10 arms, 30 for 30/14 arms).
No training, no exploration, no agent modification, no
tuning. If an agent ignores open rivals and waits at the
familiar door, that IS the result, not a reason to retrain
or adjust.

**Wrapped-encoder discipline (doors remedy, carried over).**
Wrapped is the only arm whose encoder keeps state across
episodes (ScheduleModel). It is evaluated on FRESH
retrainings, one world per training (10 trainings: 5 seeds ×
W2s/W5s), with per-step time features logged for period-loss
detection. Other arms are evaluated sequentially
(W1 → W2s → W5s on one encoder) after the 0c check, as in
the doors main run. If drift recurs, the rule is the same:
report, do not evaluate, until explained; drifted rows
retained for audit and excluded; files never pooled.

**Amendment (2026-10-21, before any evaluation run).**
The 0c check for wrapped runs on its OWN training, separate
from the sched-eval trainings (5 extra trainings): a wrapped
encoder cannot serve both a 0c eval and a sched eval without
drift, because the 0c eval itself consumes model observations
(the doors lesson — the drift struck during W1 itself for
wrapped/0). A sched row is kept only if its seed's separate
0c training matched; otherwise it is discarded and reported.
Wrapped total: 15 trainings (5 0c + 10 eval), not 10.

## Per-episode endpoints

Success; ticks; crushes; first door cell entered (or none);
door_entries; switches; stall + kind; revisits; arrival
(tick/cell/familiar-closed/rivals-open); conflict flag;
conflict outcome; staging wait; per-door status at arrival
and at first entry; steps (with per-step door bit = familiar
state; wrapped evals also log f101/f102/f103).

## Pre-stated questions

(1) In conflict episodes, how often does the agent take the
open rival vs wait at the familiar door? (2) Does schedule
variety break the (5,5) lock-in at all (first-door
distribution vs the doors experiment's 100% familiar)?
(3) Does it change success / ticks / crushes vs the
same-schedule worlds? (4) Do P2 trap signatures appear when
the agent waits at a closed familiar door while a rival is
open? (5) Does time-awareness change the competition outcome
(wrapped vs plain)? (6) Do perfect-info arms (oracle) exploit
open doors better than learned-info arms?

## Regression check (0b) — scope stated before running

The new variant (`world_doors_sched.py`) is a NEW file;
`world.py` and `world_doors.py` are not modified. Regression:
(i) scripted mechanics check — scripted + pseudo-random
action sequences on the old variant vs the new variant with
all schedules set equal → identical positions, crushes,
logs, and `observe()` bits across all start phases;
(ii) agent-level regression on ONE seed per arm (6 agents) ×
W2/W5 with equal schedules → summaries exactly equal to the
doors records. DEVIATION from the work order's unqualified
"reproduce exactly" (which reads as full 30-agent): a full
30-agent agent-level regression would cost ~2.25h redundant
compute; the scripted check covers mechanics exactly and the
subset covers encoder integration. This scope is fixed here,
before running, and will be restated in the final report.

## Compute estimate (no JIT trigger)

Regression subset: 7 trainings (~31 min; wrapped needs a fresh
training per world). Main run: 25 sequential trainings
(non-wrapped) + 15 wrapped trainings = 40 trainings (~3h).
Total ≈ 3.5h, under both JIT-trigger thresholds (4h; 150
arm-runs: 47 trainings, 60 sched evals + checks < 150). No
trigger fires; no JIT work (ruled E, trigger-gated).

## Out of scope (explicitly forbidden)

No retraining, no agent modification, no tuning to the new
worlds. No edits to `world.py`, `world_doors.py`, or any
frozen file. No claims beyond the worlds, arms, and seeds
evaluated. No JIT work. No other tracks.

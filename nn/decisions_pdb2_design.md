# WO-TBM-P-Db2 — Step 0a design note (committed before any retraining)

**Status: EXPLORATORY.** No two-conclusion rule, no
confirmatory claims. Confirmatory follow-ups return as
proposals with their own preregistration. This work
order runs SECOND (location + schedule variation),
informed by P-Da, and is NOT folded into P-Da.
P-Da showed location variation alone produces fluent
rival use under uniform timing (E2 663/750) but no
held-out generalisation (E3 0/750); P-Db2 tests
whether timing competition breaks the fluent rival
use, creates genuine timing-based choice, or changes
the held-out result.

**Step 0 gate (recorded here so it cannot be
misread later).** The issued work order authorises
Step 0 ONLY. No spike/proceed ruling was given at
issuance. The compute gate in §8 fires (206
arm-runs > 150), so Step 0 ends with a return to
the owner for the spike/proceed ruling before any
retraining starts. P-Da's PROCEED ruling does NOT
extend to P-Db2: its scope was the P-Da programme
only. Step 0 does not authorise retraining on its
own authority, and instructor verification of this
note does not substitute for the owner's gate
ruling.

**Question.** P-Da varied door LOCATION and held
timing uniform: retrained agents use trained rivals
fluently (E2 663/750) but never pass the held-out
door (E3 0/750). What P-Da never tested is TIMING
competition: every door was open at the same time,
so no timing decision was ever required. P-Db2
retrains from scratch in a world where doors open
on DIFFERENT schedules, then asks: when a
retrained agent arrives and its preferred door is
closed but a rival is open, does it wait at the
closed door, take the open rival (a genuine
timing-based choice — the thing single-door
agents never did in P-Db), or lock onto a
preferred door regardless of schedule? If timing
competition breaks what location variation built,
the fix is schedule-sensitive; if choice survives
timing competition, the fix is robust. Schedule
variation is the only new axis; location variation
is carried over from P-Da, not re-tested alone.

## 1. Retraining protocol (fixed in advance)

- **Arms and seeds — SAME as doors / P-Db / Luck /
  P-Da.** 20/10 arms, seeds 0–4: wrapped, plain,
  memory. 30/14 arms, seeds 100–104: oracle_ind,
  oracle_new, phase_only_new. 6 arms × 5 seeds =
  30 agents. Schedules per arm are unchanged:
  20/10 means period 20, open_len 10; 30/14 means
  period 30, open_len 14.
- **Training code and hyperparameters — SAME as
  the original single-door training and the P-Da
  training, with exactly one stated substitution
  (the training world, §2).** Learner:
  `nn/agent_nn.py` `NeuralQ` — MLP [n_inputs, 32,
  5], Adam lr=1e-3, replay buffer 10000, batch
  32, target sync 200, gamma 0.99. Training loop:
  `nn/agent_nn.py` `train_neural` (2000 episodes,
  cap 200 ticks, epsilon linear 1.0 → 0.05 over
  1000 episodes then flat 0.05, rewards step
  −0.01 / goal +1.0 / crush −1.0, training-start
  RNG `random.Random(seed + 1000)`, start_tick
  `rng.randrange(period)` per episode, agent RNG
  `random.Random(seed)`, numpy RNG
  `RandomState(seed)`). Arm constructors
  (unchanged, cited exactly): plain →
  `train_neural` with `PlainEncoder` (101 inputs);
  memory → `nn/agent_mem.py` `train_memory` with
  `MemoryEncoder` (102 inputs); wrapped →
  `nn/agent_wrapped.py` `train_wrapped` with
  `WrappedEncoder` (104 inputs, fixed 20/10
  scales — this arm is 20/10 only, so the scales
  match its schedule); oracle_ind →
  `nn/agent_generalization.py` `train_oracle_ind`
  with `OracleIndEncoder` (104 inputs, schedule-
  independent scales, true phase/remaining read
  from `world.period` / `world.open_len`);
  oracle_new → `train_oracle_new(seed, period,
  open_len)` with `OracleNewEncoder(period,
  open_len)` (104 inputs); phase_only_new →
  `train_phase_only_new(seed, period)` with
  `PhaseOnlyNewEncoder(period)` (102 inputs).
- **The ONE substitution, and why it is needed.**
  `train_pda.py`'s `train_neural_multi` (itself a
  line-for-line mirror of `train_neural` except
  for the world class) instantiates
  `PdaWorld(period, open_len, gaps)`, a shared-
  schedule multi-door world. It cannot train on a
  per-door-schedule world as written. Editing
  `agent_nn.py` or `train_pda.py`'s original
  functions to add a schedule parameter would
  change frozen training code and risk the
  single-door and P-Da reproducibility chains.
  Instead, Step 1 uses a NEW additive file
  `nn/train_pdb2.py` (written as part of Step 0,
  for the sanity check in §7, and reused
  unchanged in Step 1) that mirrors
  `train_neural_multi` line-for-line except for
  two stated changes: (1) the world instantiated
  per episode is `SchedDoorWorld(schedules)` from
  `nn/world_doors_sched.py` with the §2 TRAIN
  schedules for the arm's protocol, instead of
  `PdaWorld`; (2) the schedule argument is the
  TRAIN schedule table, not a single
  period/open_len pair (the familiar period/
  open_len are still the table's familiar entry,
  so `rng.randrange(period)` uses the familiar
  period exactly as before). Everything else —
  learner, replay, target sync, epsilon schedule,
  rewards, crush handling via `env_step` (which
  reads the world's log for `"crushed"`;
  `SchedDoorWorld` logs `(t, "agent", "crushed")`
  in the same format), RNG seeding, episode
  count, cap — is identical. `train_pdb2.py`
  adds ONLY additive per-door entry logging (§5),
  which observes positions and does not affect
  learning, rewards, or actions. The mirror is
  verified by line-level comparison against
  `train_pda.py` before the sanity run (§7).
- **Fresh from scratch, NOT fine-tuning.**
  Retraining means training FRESH networks from
  scratch in the schedule-varied world. No
  single-door weights and no P-Da weights are
  loaded, reused, or fine-tuned. Fine-tuning is
  a different experiment in both cases and is
  NOT part of P-Db2; if ever proposed, it returns
  as its own work order. No hyperparameter
  changes mid-run; any change returns as a new
  proposal with its own design note.
- **Wrapped: fresh training per eval world.**
  5 wrapped agents × 5 eval worlds = 25 wrapped
  trainings; 20 beyond the 5 base wrapped
  trainings. Total trainings for Step 1: 50
  (25 wrapped + 5 plain + 5 memory + 5
  oracle_ind + 5 oracle_new + 5 phase_only_new).
  Each fresh training uses the same seed, same
  protocol, and same training world, so weights
  reproduce deterministically (P-Da confirmed
  identical per-door training counts across
  wrapped re-trainings); the point is a fresh
  encoder state per world, mirroring the P-Db /
  P-Da remedy (a wrapped encoder cannot serve
  multiple eval worlds without drift — the doors
  lesson). Non-wrapped arms (stateless encoders)
  train once and evaluate sequentially across
  worlds, as in doors/P-Db/Luck/P-Da. Any
  drifted rows in Step 2 are reported, audited,
  excluded, never pooled.
- **Checkpoint convention (as before).** No
  weights are stored in the repo; a checkpoint =
  (arm, seed, protocol, training world) realised
  by deterministic re-training with frozen code.
  Step 1 trainings are therefore re-runnable
  exactly; Step 0 sanity trainings (§7) are NOT
  reused as Step 1 agents — Step 1 trains fresh
  even where a sanity training used the same
  (arm, seed), so Step 1 records are clean and
  auditable on their own.

## 2. Training world and observe contract (the ONLY change)

**Training world (ONE world, not two).** Gap set
W5 = {(5,1), (5,3), (5,5), (5,7), (5,9)} — the
same five gaps as P-Da training — with PER-DOOR
schedules following P-Db's W5s construction,
copied exactly (not re-derived) from
`nn/decisions_doors_sched_design.md` and from
`nn/world_doors_sched.py`'s `SCHED_W5S_20` /
`SCHED_W5S_30` constants. The Step 0 test (§7)
asserts the TRAIN tables equal those constants
exactly; they are imported, not re-typed.

Door-open rule (fixed, as in P-Db): a door with
(period, open_len, offset) is open at tick t iff
`(t - offset) % period < open_len`. Offset 0
reproduces the single-door `door_is_open`
exactly.

- **20/10 protocol** (wrapped/plain/memory):
  familiar (5,5): period 20, open_len 10,
  offset 0 → open [0,10) mod 20. Rivals:
  (5,1): offset 10 → [10,20); (5,3): offset 14
  → [14,20)∪[0,4); (5,7): offset 18 →
  [18,20)∪[0,8); (5,9): offset 2 → [2,12).
  Union of rival open intervals = [0,20) fully
  covered, so at least one rival is open at
  every tick; during the familiar-closed interval
  [10,20), the offset-10 rival (5,1) is open
  throughout.
- **30/14 protocol** (oracle_ind/oracle_new/
  phase_only_new): familiar (5,5): period 30,
  open_len 14, offset 0 → open [0,14) mod 30.
  Rivals (anti-phase open length 16 = the
  familiar-closed interval length): (5,1):
  offset 14 → [14,30); (5,3): offset 20 →
  [20,30)∪[0,6); (5,7): offset 26 →
  [26,30)∪[0,12); (5,9): offset 2 → [2,18).
  Union of rival open intervals = [0,30) fully
  covered, so at least one rival is open at
  every tick; during the familiar-closed interval
  [14,30), the offset-14 rival (5,1) is open
  throughout.

No W2s-style training world (familiar + single
anti-phase rival) is part of P-Db2; it is a
possible follow-up if results warrant, not a
second training condition here.

**Observe contract (fixed, scope-critical).**
Training and evaluation use the sched
familiar-only contract, exactly as in P-Db:
`observe()` returns (agent position, familiar
door bit), where the familiar bit is the
familiar door's open/closed state at the tick,
and the world exposes `period` / `open_len` =
the familiar door's schedule. Rival schedules
appear in NO agent's features. Plain/memory see
only the familiar bit (plus memory's previous-
bit); wrapped's `ScheduleModel` consumes
(tick, familiar bit) only, so its period /
open_len estimates are familiar-schedule
estimates; oracle_ind reads `world.period` /
`world.open_len` (familiar's) and computes phase
= tick % familiar period; oracle_new and
phase_only_new are constructed with the familiar
period / open_len. A per-door observation would
change input dimensions and contradict 'same
training code' — it is NOT an option the
implementer may choose silently.

**Interpretation bound (stated now, to be
stated in the report).** P-Db2 therefore tests
behavioural response to schedule-varied
mechanics (per-door blocking and crushing) under
familiar-only features, NOT learning rival
schedules from features — rival schedules are
never observed. Agents can discover rivals only
through movement, blocking, and crushing, as in
P-Db. Any report language about agents 'learning
rival schedules' would be incorrect under this
contract; the correct language is response to
rival timing through behaviour.

**Observe bit in familiar-absent worlds (fixed
here, because `SchedDoorWorld` cannot represent
them).** `SchedDoorWorld.__init__` requires the
familiar door in its schedule table, so F2 and
F3 (below) need the new class in
`nn/world_pdb2.py` (§6), `Pdb2AbsentWorld`,
which mirrors `SchedDoorWorld.step()` except
that the observe bit comes from a separately
stored clock schedule:

- **F2 (rivals only): PHANTOM FAMILIAR CLOCK.**
  The observe bit toggles on the familiar
  schedule (20/10 or 30/14, offset 0) even
  though (5,5) is a wall cell, never a gap.
  Blocking and crushing in F2 use rival
  schedules only. The bit is therefore a clock
  signal in F2, not a door state — the report
  must say so, and must not describe F2 agents
  as observing a familiar door. Drift
  implication for wrapped: the `ScheduleModel`
  continues to receive clean familiar-schedule
  transitions (openings/closings observed per
  tick; episode boundaries still mark spans
  gapped, as in `wrapper.py`), so its familiar
  estimates should persist as in F1; familiar
  absence alone is NOT the doors-remedy
  period-loss mechanism (that was a window
  filling with episode-boundary-flagged
  openings during W1 eval). Step 2 still logs
  f101/f102/f103 per step for wrapped and
  applies the same drift discipline (report,
  audit, exclude, never pool) — the phantom
  clock is a fixed choice, not a licence to
  switch to always-closed if drift appears.
  The always-closed alternative is REJECTED
  here: under it the `ScheduleModel` would see
  no openings, its estimates would go stale /
  None, and time features would blank — that
  would test a different (feature-starved)
  condition, not the sched contract.
- **F3 (novel door only): NO AMBIGUITY.**
  The gap (5,8) is on the training schedule
  (offset 0), which is numerically identical to
  the familiar schedule. The observe bit is
  therefore both the novel door's state and the
  familiar schedule's state; no phantom/clock
  distinction arises in F3.
- **Oracle bound in F2 (stated now, to be
  stated in the report).** In F2, oracle_ind /
  oracle_new / phase_only_new features are
  familiar-schedule (phantom-clock) features,
  NOT rival schedules. 'Perfect information'
  in F2 means perfect familiar-clock
  information, not perfect rival information.
  The report must not describe oracle arms in
  F2 as knowing rival schedules.

## 3. Evaluation worlds (greedy only, all start phases)

Same protocol as doors / P-Db / Luck / P-Da:
greedy evaluation only, one deterministic
episode per start phase, all phases (20 for
20/10 arms, 30 for 30/14 arms). Records flushed
per agent-world pair to tracked files; no
duplicates, no partials. World definitions:

- **F0 — W1 {(5,5)}: single familiar door on
  the training schedule. SANITY, REPORTED, NOT
  GATED.** Report retrained agents' W1
  success/ticks/crushes. No pass/fail threshold.
  A drop is a finding, not a failure of the work
  order. Observe bit = familiar door state
  (standard sched contract; familiar present).
- **F1 — the schedule-varied training world
  itself** (W5 gaps, §2 TRAIN schedules).
  Choice among FAMILIAR doors UNDER TIMING
  COMPETITION: first-door distribution,
  success/ticks/crushes/switches, plus the
  generalized conflict analysis (§4). Key
  contrast vs P-Da E1 (same gap set, uniform
  schedule): does timing competition change
  first-door distribution? **Interpretation
  bound:** F1 vs P-Da E1 is a PROGRAMME-LEVEL
  contrast (training schedules and evaluation
  schedules both change), NOT an eval-only
  timing effect; the report must not attribute
  a difference to evaluation timing alone.
  Observe bit = familiar door state.
- **F2 — familiar-removed schedule-varied
  (rivals only, staggered, no (5,5)): A NEW
  WORLD, NOT P-Db's W5s.** Gaps
  {(5,1),(5,3),(5,7),(5,9)} with the §2 W5s
  rival schedules for the arm's protocol
  (20/10: (5,1) offset 10, (5,3) offset 14,
  (5,7) offset 18, (5,9) offset 2, open_len 10;
  30/14: (5,1) offset 14, (5,3) offset 20,
  (5,7) offset 26, (5,9) offset 2, open_len 16).
  Observe bit = phantom familiar clock (§2).
  THE TRUE CHOICE TEST under retraining.
  Contrasts, stated explicitly in the report:
  (a) P-Db W5s subset — 257 conflicts →
  236 waits / 21 takes / 0 other (the W2s+W5s
  total was 479 → 458 / 21 / 0) — WITH the
  familiar-presence difference noted (W5s had
  the familiar door present and P-Db's
  conflict was familiar-closed; F2 has no
  familiar door and uses the generalized
  definition in §4, bridged by the
  preferred-door-closed counts); (b) the
  cleanest retrained contrast, P-Da E2
  (rivals-only, uniform schedule, retrained
  agents: 663/750 reached, 664 entered) — same
  gap set as F2, timing staggered vs uniform.
  Endpoints: success/ticks/crushes, first-door
  distribution, D/R/W/F classes (§4),
  generalized conflict analysis (§4).
- **F3 — W1n {(5,8)} on the training schedule:
  P-Da E3 EXACT REPEAT (same world, new
  training).** Held-out control: does
  schedule-varied training change the held-out
  result (P-Da E3: 0/750, memorisation)? Same
  endpoints and classes as P-Da E3 (D = (4,5)
  before novel entry, §4). **Bound carried
  over from P-Da:** F3 tests INTERPOLATION
  between trained locations ((5,8) sits between
  trained (5,7) and (5,9)), NOT extrapolation
  to a distant novel location. An F3 pass
  supports 'beyond trained locations' but NOT
  'general door concept' in the strong sense;
  this bound does NOT change the F2-pass /
  F3-fail memorisation reading.
- **F4 — W5n-sched: W5 + (5,8), with (5,8) on
  the familiar's training schedule and rivals
  staggered as in F1.** Novel door among
  familiars UNDER TIMING VARIATION. Report
  first-door distribution + for novel-first
  entries, the (4,5)-before-entry flag and
  ticks/revisits ONLY. **Classification
  (explicit, carried over from P-Da): F4
  novel-door entries are NOT assigned D/R/W/F
  classes** — a (4,5)-then-novel entry in F4 is
  NOT a diversion, because the familiar door
  exists and is open on its schedule in F4.
  **Tie (stated now):** (5,8) opens exactly
  when (5,5) opens (same schedule, offset 0),
  so F4 does NOT test novel-door timing
  independent of the familiar door. Observe
  bit = familiar door state.

## 4. Per-episode endpoints, generalized conflict, D/R/W/F

Per-episode endpoints (all worlds, as in doors /
Luck / P-Da): success; ticks; crushes; first
door entered (or none); door_entries; switches;
staging cells occupied in first-occupation
order; arrival cell (first staging occupation,
per the conflict staging rule below); stall
signature (P2 measures, full episode, via
`doors_run.stall_signature(positions, gaps)`
with gaps = the eval world's gap set); far-side
behaviour after entry: post-entry stall flag/
kind (structural signature applied to post-entry
positions only) and post-entry ticks (entry tick
to goal, or to episode end if not reached) —
the P-Dc connection. Steps (position/action/
door-bit/crush per tick; wrapped evals also log
f101/f102/f103 for period-loss detection).

**Staging sets for conflict analysis (fixed
here).** Conflict analysis uses PER-GAP staging
only: F0 {(4,5)}; F1 {(4,1),(4,3),(4,5),(4,7),
(4,9)}; F2 {(4,1),(4,3),(4,7),(4,9)} — the P-Da
(4,5) always-include rule is NOT used for
arrival in F2, where (5,5) is a wall; F3 and F4
are not conflict-analysis worlds (F3 is the
held-out world, F4 the novel-among-familiars
world), so no conflict staging set is defined
for them. This staging rule governs ARRIVAL and
CONFLICT only; D definitions below use the full
step trajectory for (4,5) occupation, not the
staging_order field, and that distinction is
deliberate (see the F2 D note).

**Generalized conflict analysis (F1 and F2).**
P-Db's conflict definition is generalized from
'familiar closed' to 'arrival door closed':

- **Arrival:** first tick at which the agent
  occupies any staging cell in the world's
  per-gap staging set above. Recorded: arrival
  tick, arrival cell, and per-door open/closed
  status at arrival for every door in the world
  (from the §2/§3 schedule tables, not from
  agent features). Episodes that never occupy a
  staging cell have arrival None and are
  excluded from conflict analysis (they still
  count in success/ticks/crushes/first-door).
- **Conflict episode:** arrival door closed at
  arrival AND at least one other door open at
  arrival.
- **Conflict outcomes** (conflict episodes
  only, from the first door entered):
  `waits-at-closed-door` = first door entered
  is the arrival door (entry necessarily occurs
  after it reopens — closed-gap entry is
  blocked, so 'waits' means the agent returns
  to / stays with the arrival door, not that it
  entered while closed); `takes-open-door` =
  first door entered is a door that was open at
  arrival and is not the arrival door;
  `other` = first door entered is a door that
  was closed at arrival and is not the arrival
  door, or no door entered.
- **Staging wait:** ticks from arrival to first
  door entry (None if no entry). **Door status
  at first entry:** per-door open/closed at the
  first-entry tick, recorded alongside.

**Preferred door and preferred-door-closed
counts (F1 and F2, bridging to P-Db).**
Preferred door per agent = the agent's MODAL
F1 first door over entered episodes. Tie rule
(fixed now, deterministic): if two or more
doors tie for modal, the tied door that is
first door at the smallest start_phase in F1
wins, and any tie is listed in the Step 3
report. Preferred-door-closed episodes (F1 and
F2 separately): episodes whose arrival tick
(per-gap arrival above) has the agent's
preferred door closed AND at least one other
door open at arrival. Preferred-door status at
arrival is taken from the schedule tables
(in F2, a preferred door of (5,5) is evaluated
on the phantom familiar clock, §2). Within
preferred-door-closed episodes, outcomes are
classified the same way as generalized
conflicts, relative to the preferred door:
`waits-at-preferred` (first door = preferred),
`takes-open` (first door open at arrival, not
preferred), `other` (otherwise). Bridge note:
for an agent whose preferred door IS the
familiar door, preferred-door-closed IS P-Db's
familiar-closed definition; for other agents
it is the natural generalization, and the
report must present both the generalized and
the preferred-door-closed counts so the P-Db
comparison is never definitionally opaque.

**D/R/W/F classification (F2 and F3 only,
priority F → D → W → R, first entry only,
as in Luck / P-Da §4).** Every episode in F2
and F3 gets exactly one class, from its full
step trajectory. First entry = the first tick
at which the agent occupies any gap cell.
Priority order is fixed (an episode satisfying
several rules takes the FIRST matching class):

1. **F (fail):** never enters any door. Note:
   an episode that enters a door but never
   reaches the goal is NOT F — it takes the
   class of its entry (D/R/W) with
   reached=False recorded separately.
2. **D (diversion, behavioural pattern, not
   proof of intent):** the agent occupies the
   familiar staging cell (4,5) at a tick BEFORE
   the first-entry tick, and then enters a rival
   (F2) or novel (F3) door. D takes precedence
   over W. **F2 D note (fixed here):** D in F2
   is computed from the full trajectory's (4,5)
   occupation, NOT from the conflict staging
   set (which excludes (4,5) in F2, §3 above).
   In F2, (5,5) is absent, so D does NOT mean
   'abandoned the familiar door' — (4,5) is
   west-side floor, not a door staging cell
   in-world; D records the route pattern
   (via (4,5), then rival entry). **F2 D
   decomposition (as in P-Da E2):** report
   pass-through cases (exactly 1 tick at (4,5)
   before first entry) separately from waited
   cases (≥2 ticks at (4,5) before first
   entry), so the label is not over-read.
   **F3 D** is the P-Da E3 rule verbatim:
   (4,5) before novel entry.
3. **W (wandering):** not D, enters a rival/
   novel door, AND (ticks > 2 × the agent's
   own F0 median ticks over reached episodes,
   per §5) OR (revisits > 10). Neither a direct
   route nor a diversion.
4. **R (route coincidence):** not D, not W:
   enters a rival/novel door on a direct route
   (the P-Db take pattern). This includes
   entries after a crush-restart, as long as
   the entry itself is not D and not W by the
   thresholds above.
5. **UNCLASSIFIED:** a pass that cannot be
   classified under the fixed rules (e.g.
   missing trajectory data). Reported with its
   trajectory, never force-fitted. Expected
   count: 0 (the rules above are exhaustive
   over complete trajectories).

Class is determined by the FIRST entry only;
later door switches do not reclassify an
episode. F1 episodes are NOT classified under
this scheme (conflict analysis + first-door
distribution only, as P-Da E1 was not
classified). F4 novel entries are NOT
classified under this scheme (§3, F4).

## 5. Classification baselines and training records

- **W thresholds use each retrained agent's OWN
  F0 median ticks over reached episodes**
  (per arm/seed, computed from that agent's F0
  records in Step 2). The old single-door
  medians and the P-Da E0 medians are NOT
  reused. If a retrained agent reaches 0
  episodes in F0 (median undefined), the ticks
  criterion is inapplicable and W is determined
  by revisits > 10 alone.
- **F0-median fallback applicability (to be
  filled after F0).** At design-note time, no
  agent is known to reach 0 in F0, so no agent
  is listed here as fallback-applicable. If
  Step 2 shows any retrained agent reaching 0
  in F0, the Step 3 report must list those
  agents explicitly and apply the revisits-only
  W rule to them. This paragraph is the fixed
  rule; the list itself can only be completed
  after F0, as in P-Da §5.
- **Per-agent training records.** Same summary
  format as the original training and P-Da
  (per-episode ticks/reached/crushes,
  summarised per agent), kept per arm/seed, so
  retrained agents are traceable and F0 sanity
  comparisons are possible. Training records
  ALSO include per-door entry counts during
  training (which doors each agent entered, and
  how often) — additive logging only, no
  mechanics change. Definition (fixed here,
  same entry definition as evaluation): from
  the training episode's position-before-action
  sequence, an entry event is a gap-cell
  occupation whose gap differs from the
  previous entry event's gap (consecutive stays
  in one gap count as one event; leaving and
  re-entering counts again). Per-door entry
  counts = total entry events per gap cell
  across all 2000 training episodes, plus
  episodes-with-at-least-one-entry per gap cell.
  Training records ALSO include schedule open
  fractions per door alongside the raw counts:
  open_len / period from the §2 tables — 20/10:
  every door 10/20 = 0.50; 30/14: familiar
  14/30 ≈ 0.467, each rival 16/30 ≈ 0.533.
  Step 3 reads eval lock-in JOINTLY with both
  raw counts and open fractions: lock-in with
  narrow training visits = 'training never
  varied' (a training-dynamics finding);
  lock-in with broad training visits =
  'location/schedule variation was
  insufficient'; lock-in concentrated on
  high-open-fraction doors = a schedule-
  exposure reading — with the caveat stated
  now that open fractions are near-equal
  across doors in both protocols, so exposure
  differences are small and cannot dismiss a
  lock-in without the numbers. No reading is
  assumed in advance.

## 6. Pre-stated questions (verbatim in scope)

(1) Do schedule-varied-retrained agents still
solve W1 (F0 sanity)? (2) In F1, under timing
competition, do retrained agents distribute/
choose among familiar doors, or lock onto a
preferred door regardless of schedule — and how
does first-door distribution compare with P-Da
E1? (3) In F2, do retrained agents use open
rivals when their arrival/preferred door is
closed — waits vs takes under the generalized
conflict definition and under the
preferred-door-closed definition (both fixed in
§4) — and how does this compare with P-Db
single-door agents in W5s (257 conflicts →
236 waits / 21 takes / 0 other, familiar-
presence difference noted) and with P-Da E2
(uniform schedule, retrained)? (4) In F3, does
schedule-varied training change the held-out
result (P-Da E3: 0/750)? (5) In F4, do they use
the novel door among familiars under timing
variation (P-Da E4: 0 novel-first)? (6) Far
side: after entries under timing competition,
freeze (P-Dc pattern) or clean crossing? (P-Da
found 1 freeze in 1376 familiar/rival entries.)
(7) Wrapped vs plain vs oracle after
schedule-varied retraining. (8) Joint with
training records: does training door-visit
breadth under staggered schedules predict eval
door breadth — or does lock-in persist despite
broad training visits, as in P-Da?

## 7. Step 0b regression — scope fixed BEFORE running

`nn/world_pdb2.py` is a NEW file;
`world.py`, `world_doors.py`,
`world_doors_sched.py`, `world_luck.py`, and
`world_pda.py` are NEVER edited, and no frozen
file is modified. Mechanics are inherited from
`SchedDoorWorld` deliberately (the P-Db2 axis
is schedule variation on top of the already-
verified sched mechanics; reimplementing
identical mechanics would add bug surface for
zero information). The variant file names the
P-Db2 worlds and supplies the one class
`SchedDoorWorld` cannot represent:
`Pdb2AbsentWorld` (familiar-absent worlds F2
and F3), which mirrors `SchedDoorWorld.step()`
line-for-line except that the observe bit comes
from a separately stored clock schedule (§2)
instead of a familiar gap entry. Familiar-
present worlds (F0, F1, F4) use
`SchedDoorWorld` directly with the §2/§3
schedule tables.

Regression, fixed here before running:

(i) **Scripted mechanics check**
(`nn/test_world_pdb2.py`):
- TRAIN identity: the §2 TRAIN tables equal
  `world_doors_sched.SCHED_W5S_20` /
  `SCHED_W5S_30` exactly (imported, not
  re-typed; asserted, not assumed).
- Union coverage: for every tick in the
  period, at least one rival is open; during
  the familiar-closed interval, at least one
  rival is open — computed from the
  `door_open_at` table, both protocols.
- Per-door open/close table:
  `Pdb2AbsentWorld.door_open(cell, tick)` ==
  `door_open_at(tick, schedule)` for every
  door, every tick in the period, both
  protocols, F2 and F3 worlds.
- Observe bit: F2 observe bit == familiar
  schedule bit at every tick in the period;
  F3 observe bit == training schedule bit;
  F1 (`SchedDoorWorld` with TRAIN tables)
  observe bit == familiar bit (already
  verified in P-Db, re-asserted for the exact
  TRAIN constants).
- Wall blocking / crush: scripted sequences
  (the P-Da script set: scripted, pseudo-
  random seeds 0/1/2 × 250 steps, right-spam
  × 120, stay × 120) run on
  `Pdb2AbsentWorld` F2/F3 across all start
  phases, checking per-step invariants (never
  in a wall cell; never in a closed gap after
  a move, except the crush tick which resets
  to start; crushed iff the agent was in a gap
  whose door closed at that tick; grid-edge
  clipping as in `SchedDoorWorld`). Expected
  case count is computed by the script and
  reported, not asserted in advance beyond
  the stated grid.
- Trace comparison where possible:
  `Pdb2AbsentWorld` F3 vs `PdaWorld` W1n with
  the same schedule: positions, observe bit,
  and crushed flag identical across the same
  script set and all start phases (full log
  NOT asserted — log lineages differ between
  the sched and multi-door classes, so only
  position/bit/crushed are compared).

(ii) **Agent-level training sanity**
(`nn/sanity_pdb2.py`): ONE seed per arm (6
agents: wrapped/0, plain/0, memory/0, 20/10;
oracle_ind/100, oracle_new/100,
phase_only_new/100, 30/14), each trained from
scratch in the F1 training world (its protocol
schedule) using `nn/train_pdb2.py`, 2000
episodes, same hyperparameters as §1. Scope
rationale, stated before running: the issued
WO's Step 0b names one seed per arm (6
trainings, not 30), following the Luck/P-Db/
P-Da precedent; one seed per arm validates the
training loop for every encoder type
(including the stateful wrapped encoder and
the period-parameterised oracle/phase
encoders) without doing Step 1. Sanity
criteria, fixed here (these are loop-validity
criteria, NOT experiment-outcome gates): each
sanity training completes 2000 episodes
without crash; stats format matches
`train_neural` (ticks, reached, crushes, mean
loss); no NaN in stats; at least one training
episode reached the goal; per-door entry
counts recorded and total entries > 0 (the
agent entered at least one door during
training — otherwise the training world had
no effect); schedule open fractions recorded
alongside. If any sanity agent fails any
criterion, it is reported and Step 1 does NOT
proceed with that arm unexamined; the failure
is not hidden by substituting a single-door or
P-Da weight or by adjusting hyperparameters.
Sanity trainings are check outputs only
(`nn/sanity_pdb2.log`); they are NOT Step 1
training records and their agents are NOT
reused in Step 1 (Step 1 trains fresh, §1).

(iii) **Training-loop mirror verification.**
Before the sanity run, `nn/train_pdb2.py` is
verified against `nn/train_pda.py`'s
`train_neural_multi` by line-level comparison:
the only differences are the world
instantiation (`SchedDoorWorld` with TRAIN
schedules instead of `PdaWorld`) and the
schedule argument passing; entry logging,
learner, replay, epsilon, rewards, crush
handling, RNG seeding, episode count, and cap
are identical. The verification result is
reported in the Step 0 return (a claim in this
note is not itself the verification).

## 8. Compute estimate + JIT-trigger gate

Counting rule (fixed, as in P-Da): one arm-run
= one agent-training OR one agent-world
evaluation job.
- Step 1 trainings: 50 (25 wrapped fresh-per-
  world + 25 non-wrapped base).
- Step 2 evaluation jobs: 30 agents × 5 worlds
  (F0–F4) = 150.
- Step 1 + Step 2 total: **200 arm-runs**.
- Step 0 checks (separate from the 200, stated
  so the total is not understated): scripted
  mechanics (fast, no training) + sanity
  trainings 6 (one per arm, §7). Total
  programme arm-runs including Step 0 sanity =
  **206, which exceeds the 150 threshold — the
  arm-run trigger FIRES**.
- Step 1 + Step 2 evaluation episodes: 20/10
  arms 15 × 5 × 20 = 1500; 30/14 arms 15 × 5 ×
  30 = 2250; total **3750 episodes** (plus
  training episodes: 50 × 2000 = 100,000
  training episodes, as in every prior run;
  sanity trainings add a further 6 × 2000 =
  12,000 training episodes, counted as checks,
  not as Step 1 records).
- Time estimate (method: P-Da actual pace).
  P-Da Steps 1–2 took about 1.5 hours for the
  same 50 + 150 counts on the same substrate
  (P-Da's 2.5–3.5 hour estimate proved
  conservative; the actual run finished well
  inside it). The P-Db2 Steps 1–2 estimate is
  therefore **about 1.5–2 hours**, allowing for
  schedule-varied training differing from
  uniform-schedule training (more crushes, and
  possibly more episodes running to the
  200-tick cap, could slow per-episode time,
  but no order-of-magnitude change is expected
  — the per-episode work is the same network
  update count). Step 3 analysis is additional
  to this estimate, not included in it.

  **Instructor amendment (2026-10-22, after
  sanity):** sanity took ~15.3 minutes for 6
  trainings (~2.55 minutes each), vs P-Da
  sanity ~10.2 minutes (~1.7 minutes each). At
  sanity pace, 50 Step 1 trainings alone are
  ~128 minutes, plus 150 eval jobs, so Steps
  1–2 are now estimated at ~2.5 hours, not
  1.5–2 hours. This remains under the 4-hour
  time threshold; the arm-run trigger status
  is unchanged (it fires on counts, not time).
  The owner gate ruling should be made on this
  corrected estimate, not on the 1.5–2 hour
  figure above. The
  time estimate does NOT change the ruling:
  it is under the 4-hour time threshold, but
  the arm-run trigger already fires, and the
  owner has NOT ruled at issuance. If the
  actual Step 1 + Step 2 time were to exceed
  4 hours, that would be reported as a finding,
  not used to silently change substrate or
  scope.
- **Gate outcome for Step 0.** The design note
  is committed, the variant and checks are
  committed, and Step 0 then STOPS for
  instructor verification AND for the owner's
  spike/proceed ruling. Retraining (Step 1)
  starts only after BOTH: instructor
  verification of this note, and the owner's
  gate ruling (benchmark spike first, or
  proceed on the current substrate). Neither
  Step 0 work nor instructor verification alone
  authorises retraining.

  **Owner second ruling (2026-10-22, after
  spike commit ad1c70d): PROCEED on the
  current substrate, no substrate change.**
  Steps 1–3 are authorised on the hand-written
  numpy substrate, at the spike-projected pace
  (~1.84 hours composition for Steps 1–2).
  Spike trainings are measurement only and are
  NOT Step 1 records.

## 9. Step 0c agent-check rule (for Steps 1–2, fixed now)

Retrained agents are NEW agents (new weights,
same seeds/protocols). No 0c reproduction
against tracked single-door or P-Da summaries
applies — training worlds differ by design, so
a mismatch against single-door or P-Da records
would be expected and meaningless. The check
that DOES apply: each retrained agent's
training summary is recorded and reported, and
F0 (W1) is evaluated for every retrained agent
as the sanity endpoint (§3). An agent with no
recorded training summary, or with no F0
evaluation, is not carried into Step 3
analysis. **This check is NOT done in Step 0:**
it can only be completed after Step 1
retraining and Step 2 F0 evaluation, and this
note does not claim it as done.

## 10. Out of scope (explicitly forbidden)

No fine-tuning of existing single-door OR P-Da
weights (from-scratch retraining only). No new
training worlds beyond the one schedule-varied
world (no W2s-style training here). No
hyperparameter changes mid-run. No edits to
`world.py`, the doors/sched/luck/pda variants,
or any frozen file. No edits to
`nn/agent_nn.py`, `nn/agent_wrapped.py`,
`nn/agent_mem.py`, `nn/agent_controls.py`, or
`nn/agent_generalization.py` (their code is
reused unchanged; the only new training code
is additive `nn/train_pdb2.py`, §1), and no
edits to `nn/train_pda.py`'s original functions
(the P-Db2 trainers are a separate additive
mirror). No claims beyond the worlds, arms,
and seeds evaluated. No JIT work and no
benchmark spike under this work order (the
spike, if the owner orders it at the gate, is
a separate ruling, not Step 0 work). No other
tracks. No AI-approach work (J1/J2/J3 belong
to the reciter track, not this session). No
Step 1 retraining, no Step 2 evaluation, and
no Step 3 report (`nn/decisions_pdb2.md`) in
Step 0 — those start only after instructor
verification of this note AND the owner's gate
ruling.

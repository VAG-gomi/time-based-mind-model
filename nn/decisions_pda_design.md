# WO-TBM-P-Da — Step 0a design note (committed before any retraining)

**Status: EXPLORATORY.** No two-conclusion rule, no
confirmatory claims. Confirmatory follow-ups return as
proposals with their own preregistration. This work
order runs FIRST (location variation only); P-Db2
(location + schedule variation) comes after, informed
by P-Da, and is NOT folded in here.

**Owner ruling on the JIT trigger (2026-10-22, recorded
before Step 0 work, scope stated here so it cannot be
misread later).** The owner tapped "Proceed with full
P-Da": proceed on the current substrate despite the
arm-run trigger firing (50 trainings + 150 evaluation
jobs = 200 arm-runs > 150), with NO benchmark spike
first. Scope of that ruling: it authorises the full
P-Da programme (Step 0 → retraining → evaluation →
analysis) on the current hand-written numpy substrate;
it does NOT authorise a substrate change, and it does
NOT waive Step 0 — this design note must still be
committed and instructor-verified before any
retraining starts. The trigger is therefore NOT
ignored: it fired, it is recorded here and in §8, and
the owner's proceed-ruling is the reason retraining
may go ahead without a spike. No spike is started
under this work order.

**Question.** Luck showed total failure without
retraining (0/1500 reached, familiar door removed):
agents trained on a single door learn "(5,5)", not "a
door". The owner's reading, adopted as the hypothesis
to test: this is a training-design limitation, not a
model-architecture limitation — and it is fixable by
changing the training distribution. P-Da tests the
first, cleanest version of that fix: retrain from
scratch in a world where doors vary in LOCATION, with
timing held uniform. If the lock-in breaks here,
location variation alone suffices. If it persists
even here, the finding is deeper than "needs more
doors".

## 1. Retraining protocol (fixed in advance)

- **Arms and seeds — SAME as doors / P-Db / Luck.**
  20/10 arms, seeds 0–4: wrapped, plain, memory.
  30/14 arms, seeds 100–104: oracle_ind, oracle_new,
  phase_only_new. 6 arms × 5 seeds = 30 agents.
  Schedules per arm are unchanged: 20/10 means
  period 20, open_len 10; 30/14 means period 30,
  open_len 14.
- **Training code and hyperparameters — SAME as the
  original single-door training, with exactly one
  stated substitution (the training world, §2).**
  Learner: `nn/agent_nn.py` `NeuralQ` — MLP
  [n_inputs, 32, 5], Adam lr=1e-3, replay buffer
  10000, batch 32, target sync 200, gamma 0.99.
  Training loop: `nn/agent_nn.py` `train_neural`
  (2000 episodes, cap 200 ticks, epsilon linear
  1.0 → 0.05 over 1000 episodes then flat 0.05,
  rewards step −0.01 / goal +1.0 / crush −1.0,
  training-start RNG `random.Random(seed + 1000)`,
  start_tick `rng.randrange(period)` per episode,
  agent RNG `random.Random(seed)`, numpy RNG
  `RandomState(seed)`).
  Arm constructors (unchanged, cited exactly):
  plain → `train_neural` with `PlainEncoder`
  (101 inputs); memory → `nn/agent_mem.py`
  `train_memory` with `MemoryEncoder` (102 inputs);
  wrapped → `nn/agent_wrapped.py` `train_wrapped`
  with `WrappedEncoder` (104 inputs, fixed
  20/10 scales — this arm is 20/10 only, so the
  scales match its schedule); oracle_ind →
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
  `train_neural` hardcodes `World(period=PERIOD,
  open_len=OPEN_LEN)` from `nn/agent_nn.py` (its
  `PERIOD`/`OPEN_LEN` globals, set to the arm's
  schedule by the doors/luck runners before
  training). It cannot train on a multi-door
  world as written. Editing `agent_nn.py` to add a
  world parameter would change the original
  training code and risk the single-door
  reproducibility chain. Instead, Step 1 uses a NEW
  additive file `nn/train_pda.py` (written as part
  of Step 0, for the sanity check in §7, and
  reused unchanged in Step 1) that mirrors
  `train_neural` line-for-line except for two
  stated changes: (1) the world instantiated per
  episode is `PdaWorld(period, open_len, gaps=W5)`
  from `nn/world_pda.py` (§2) instead of `World`;
  (2) period/open_len are passed explicitly
  instead of read from `agent_nn` globals
  (equivalent, since the doors/luck runners set
  those globals to the same arm schedule before
  training). Everything else — learner, replay,
  target sync, epsilon schedule, rewards, crush
  handling via `env_step` (which reads the
  world's log for `"crushed"`; `MultiDoorWorld`
  logs `(t, "agent", "crushed")` in the same
  format), RNG seeding, episode count, cap — is
  identical. `train_pda.py` adds ONLY additive
  per-door entry logging (§5), which observes
  positions and does not affect learning, rewards,
  or actions.
- **Fresh from scratch, NOT fine-tuning.**
  Retraining means training FRESH networks from
  scratch in the multi-door world. No single-door
  weights are loaded, reused, or fine-tuned.
  Fine-tuning is a different experiment
  (plasticity of a locked-in policy) and is NOT
  part of P-Da; if ever proposed, it returns as
  its own work order. No hyperparameter changes
  mid-run; any change returns as a new proposal
  with its own design note.
- **Wrapped: fresh training per eval world.**
  5 wrapped agents × 5 eval worlds = 25 wrapped
  trainings; 20 beyond the 5 base wrapped
  trainings. Total trainings for Step 1: 50
  (25 wrapped + 5 plain + 5 memory + 5 oracle_ind
  + 5 oracle_new + 5 phase_only_new). Each fresh
  training uses the same seed, same protocol, and
  same training world, so weights reproduce
  deterministically; the point is a fresh encoder
  state per world, mirroring the P-Db remedy
  (a wrapped encoder cannot serve multiple eval
  worlds without drift — the doors lesson).
  Non-wrapped arms (stateless encoders) train
  once and evaluate sequentially across worlds,
  as in doors/P-Db/Luck. Any drifted rows in
  Step 2 are reported, audited, excluded, never
  pooled.
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

## 2. Training world (the ONLY change)

Gap set W5 = {(5,1), (5,3), (5,5), (5,7), (5,9)} —
the same five gaps as doors W5 — with ALL doors on
the arm's training schedule (one shared schedule,
identical timing for every door). Timing stays
uniform, so LOCATION is the only varied axis.
The familiar door (5,5) is present during
training; what varies is that it is no longer the
ONLY door.

Mechanics note, stated now because it bounds what
this test can show: `MultiDoorWorld.observe()`
returns (agent position, door bit) where the door
bit is the single shared schedule's open/closed
state — it does NOT identify which door is open,
because all doors open and close together. Location
variation in P-Da is therefore via wall geometry
only (which gap cells exist and where the agent's
position is), not via per-door timing. Per-door
timing variation is P-Db2, not here. Encoders see
no per-door timing signal: plain/memory see only
the shared door bit (plus memory history);
wrapped sees the shared schedule via its
ScheduleModel (which consumes (tick, door bit)
only, so multi-door training does not change its
schedule learning); oracle/phase arms read the
shared schedule's period/open_len. The network
must learn door locations from position, not from
a door-identity input — that is exactly the
generalisation this test is about, and its bound.

## 3. Evaluation worlds (greedy only, all start phases)

Same protocol as doors / P-Db / Luck: greedy
evaluation only, one deterministic episode per
start phase, all phases (20 for 20/10 arms, 30
for 30/14 arms). Records flushed per agent-world
pair to tracked files; no duplicates, no partials.
World definitions (all via `nn/world_pda.py`, §6):

- **E0 — W1 {(5,5)}: single familiar door. SANITY,
  REPORTED, NOT GATED.** Report retrained agents'
  W1 success/ticks/crushes. No pass/fail
  threshold. A drop is a finding, not a failure
  of the work order.
- **E1 — W5 {(5,1), (5,3), (5,5), (5,7), (5,9)}:
  the training world itself.** Choice among
  FAMILIAR doors: first-door distribution
  (familiar (5,5) vs nearest (5,1) vs other),
  success/ticks/crushes/switches (doors-style
  analysis).
- **E2 — W5r {(5,1), (5,3), (5,7), (5,9)}:
  rivals only, no (5,5). THE LUCK CONTRAST.**
  Same world Luck measured 0/1500 in — do
  retrained agents use rivals when (5,5) is gone?
  Luck-style endpoints (§4) + D/R/W/F classes
  (§4, re-stated in full, thresholds per §5).
- **E3 — W1n {(5,8)}: HELD-OUT novel door only.**
  (5,8) never appears in any training or prior
  eval world, and it is OFF the standard route
  (agents in Luck never occupied (4,8); verified
  for this design note: (4,8) in 0 Luck episodes,
  0 of 2920 doors episodes, 0 of 1500
  schedule-varied episodes, and (5,8) in 0 Luck
  episodes). If retrained agents pass here, that
  is genuine generalisation beyond trained
  locations. If they fail here but pass E2, they
  memorised MORE locations — the distinction this
  world exists to draw. **Bound (stated in this
  design note and to be stated in the report):**
  E3 tests INTERPOLATION between trained
  locations ((5,8) sits between trained (5,7)
  and (5,9)), NOT extrapolation to a distant
  novel location. An E3 pass supports 'beyond
  trained locations' but NOT 'general door
  concept' in the strong sense; this bound does
  NOT change the E2-pass/E3-fail memorisation
  reading. (In this 10-row geometry every novel
  row is adjacent to a trained row, so
  interpolation is the strongest novel-door test
  the grid allows; (5,8) stays because off-route
  matters more — a route-adjacent novel door
  would confound with the P-Db route-artifact
  pattern.) Luck-style endpoints + D/R/W/F
  classes, with D = (4,5) before novel entry.
- **E4 — W5n = W5 + (5,8) = {(5,1), (5,3), (5,5),
  (5,7), (5,8), (5,9)}: novel door among
  familiars.** Do retrained agents use the novel
  door when familiar doors exist? Report
  first-door distribution + for novel-first
  entries, the (4,5)-before-entry flag and
  ticks/revisits ONLY. **Classification
  (explicit): E4 novel-door entries are NOT
  assigned D/R/W/F classes** — in particular, a
  (4,5)-then-novel entry in E4 is NOT a
  diversion, because the familiar door exists and
  is open in E4, so the Luck/P-Db diversion
  meaning (familiar gone/closed) does not apply.
  (E3 keeps Luck classes: the familiar door is
  absent there, so the D meaning is preserved.)

## 4. Per-episode endpoints and D/R/W/F classes (re-stated)

Per-episode endpoints (all worlds, as in doors /
Luck): success; ticks; crushes; first door entered
(or none); door_entries; switches; staging cells
occupied in first-occupation order; arrival cell
(first staging occupation); stall signature (P2
measures, full episode, via `doors_run.
stall_signature(positions, gaps)` with gaps =
the eval world's gap set); far-side behaviour
after entry: post-entry stall flag/kind
(structural signature applied to post-entry
positions only) and post-entry ticks (entry tick
to goal, or to episode end if not reached) — the
P-Dc connection. Steps (position/action/door-bit/
crush per tick; wrapped evals also log f101/f102/
f103 for period-loss detection).

Staging-cell rule, fixed here (Luck rule carried
over): staging cells = {(4, y) for each gap
(5, y)} PLUS (4,5) always (the familiar staging
cell, even in worlds where (5,5) is not a gap).
Hence E2 staging = {(4,1), (4,3), (4,7), (4,9),
(4,5)}; E3 staging = {(4,8), (4,5)}; E0 staging =
{(4,5)}; E1 staging = {(4,1), (4,3), (4,5),
(4,7), (4,9)}; E4 staging = {(4,1), (4,3), (4,5),
(4,7), (4,8), (4,9)}.

Dwell caveat, carried over from Luck (§0(4) of
the Luck report): the stall signature's dwell
exclusion covers staging cells for the eval
world's gaps only; in E2/E3, (4,5) is therefore
NOT excluded from dwell (it was excluded in
doors W1, where (5,5) is a gap). This affects
only the dwell label, not the headline;
post-entry stall is unaffected (post-entry
positions are east-side).

D/R/W/F classification (Luck design note, re-stated
in full — this text, not just a citation, governs
Step 3). Every episode in E2 and E3 gets exactly
one class, from its trajectory. First entry = the
first tick at which the agent occupies any gap
cell. Priority order is fixed (an episode
satisfying several rules takes the FIRST matching
class):
1. **F (fail):** never enters any door (freeze,
   timeout at 200 ticks, or crush-loop without
   entry). Note: an episode that enters a door
   but never reaches the goal is NOT F — it takes
   the class of its entry (D/R/W) with
   reached=False recorded separately.
2. **D (diversion):** the agent occupies the
   familiar staging cell (4,5) at a tick BEFORE
   the first-entry tick, and then enters a rival
   (E2) or novel (E3) door. Caveat, stated now:
   D as defined is a behavioural pattern, not
   proof of intent — "decision" language stays
   descriptive. A pass that goes (4,5) → rival/
   novel door is D even if it is also long or
   revisit-heavy (D takes precedence over W).
3. **W (wandering):** not D, enters a rival/novel
   door, AND (ticks > 2 × the agent's own E0
   median ticks over reached episodes, per §5)
   OR (revisits > 10). Neither a direct route
   nor a diversion.
4. **R (route coincidence):** not D, not W:
   enters a rival/novel door on a direct route
   (the P-Db take pattern). This includes entries
   after a crush-restart, as long as the entry
   itself is not D and not W by the thresholds
   above.
5. **UNCLASSIFIED:** a pass that cannot be
   classified under the fixed rules (e.g. missing
   trajectory data). Reported with its trajectory,
   never force-fitted. Expected count: 0 (the
   rules above are exhaustive over complete
   trajectories).
Class is determined by the FIRST entry only;
later door switches do not reclassify an episode.
E4 entries are NOT classified under this scheme
(§3, E4). E0/E1 episodes are reported by
first-door distribution, not by D/R/W/F.

## 5. Classification baselines and training records

- **W thresholds use each retrained agent's OWN
  E0 median ticks over reached episodes**
  (per arm/seed, computed from that agent's E0
  records in Step 2). The old single-door
  medians are NOT reused. If a retrained agent
  reaches 0 episodes in E0 (median undefined),
  the ticks criterion is inapplicable and W is
  determined by revisits > 10 alone.
- **E0-median fallback applicability (to be
  filled after E0).** At design-note time, no
  agent is known to reach 0 in E0, so no agent
  is listed here as fallback-applicable. If Step
  2 shows any retrained agent reaching 0 in E0,
  the Step 3 report must list those agents
  explicitly and apply the revisits-only W rule
  to them. This paragraph is the fixed rule; the
  list itself can only be completed after E0.
- **Per-agent training records.** Same summary
  format as the original training (per-episode
  ticks/reached/crushes, summarised per agent),
  kept per arm/seed, so retrained agents are
  traceable and E0 sanity comparisons are
  possible. Training records ALSO include
  per-door entry counts during training (which
  doors each agent entered, and how often) —
  additive logging only, no mechanics change.
  Definition (fixed here, same entry definition
  as evaluation): from the training episode's
  position-before-action sequence, an entry event
  is a gap-cell occupation whose gap differs
  from the previous entry event's gap (consecutive
  stays in one gap count as one event; leaving
  and re-entering counts again). Per-door entry
  counts = total entry events per gap cell across
  all 2000 training episodes, plus episodes-with-
  at-least-one-entry per gap cell. Step 3 reads
  E1 lock-in JOINTLY with these counts: lock-in
  with narrow training visits = 'training never
  varied' (a training-dynamics finding); lock-in
  with broad training visits = 'location
  variation was insufficient'. Neither reading
  is assumed in advance.

## 6. Pre-stated questions (verbatim in scope)

(1) Do retrained agents still solve W1 (E0
sanity)? (2) In W5 (E1), does (5,5) lock-in
persist among familiar doors, or do they
distribute/choose (nearest? any?)? (3) In W5r
(E2), do retrained agents use rivals — the Luck
contrast (0/1500 → ?)? (4) In W1n (E3), do they
pass a door never seen in training
(generalisation) or fail (memorisation of more
locations)? (5) In W5n (E4), do they use the novel
door when familiars exist? (6) Far side: after
unfamiliar/novel entries, freeze (P-Dc pattern)
or clean crossing? (7) Wrapped vs plain vs
oracle: does time-awareness or perfect
information matter after retraining?

## 7. Step 0b regression — scope fixed BEFORE running

`nn/world_pda.py` is a NEW file;
`world.py`, `world_doors.py`,
`world_doors_sched.py`, and `world_luck.py` are
NEVER edited, and no frozen file is modified.
Mechanics are inherited from `MultiDoorWorld`
deliberately (the P-Da axis is location-only;
reimplementing identical mechanics would add bug
surface for zero information). The variant file
only names gap sets (see §3) via a `PdaWorld`
subclass with no mechanics overridden, so the
regression can run the same class with any gap
set.

Regression, fixed here before running:
(i) **Scripted mechanics check**
(`nn/test_world_pda.py`): `PdaWorld` vs
`MultiDoorWorld` traces identical (agent
position, door bit, crushed flag, and full log)
across schedules 20/10 and 30/14; gap sets W5,
W1 {(5,5)}, W5r, W1n {(5,8)}, W5n; action
sequences scripted (right×4, down×5, right×3,
down×4, left×2, right×2, stay×5, up×9, right×4),
pseudo-random seeds 0/1/2 × 250 steps,
right-spam × 120, stay × 120 (the Luck script
set); all start phases for the schedule.
Expected case count is computed by the script
and reported, not asserted in advance beyond
the stated grid (2 schedules × 5 gap sets ×
6 sequences × period starts).
(ii) **Agent-level training sanity**
(`nn/sanity_pda.py`): ONE seed per arm (6
agents: wrapped/0, plain/0, memory/0, 20/10;
oracle_ind/100, oracle_new/100,
phase_only_new/100, 30/14), each trained from
scratch in the W5 training world (its protocol
schedule) using `nn/train_pda.py`, 2000
episodes, same hyperparameters as §1.
Scope rationale, stated before running: the
issued WO's Step 0b names "one agent-level
training sanity (one arm/seed)" and the parent
Step-0 brief directs following the Luck/P-Db
precedent of one seed per arm where applicable;
one seed per arm (6 trainings, not 30) validates
the training loop for every encoder type
(including the stateful wrapped encoder and the
period-parameterised oracle/phase encoders)
without doing Step 1. Sanity criteria, fixed
here (these are loop-validity criteria, NOT
experiment-outcome gates): each sanity training
completes 2000 episodes without crash; stats
format matches `train_neural` (ticks, reached,
crushes, mean loss where applicable); no NaN in
stats; at least one training episode reached the
goal; per-door entry counts recorded and total
entries > 0 (the agent entered at least one
door during training — otherwise the training
world had no effect). If any sanity agent fails
any criterion, it is reported and Step 1 does
NOT proceed with that arm unexamined; the
failure is not hidden by substituting a
single-door weight or by adjusting
hyperparameters. Sanity trainings are check
outputs only (`nn/sanity_pda.log`); they are
NOT Step 1 training records and their agents
are NOT reused in Step 1 (Step 1 trains fresh,
§1).

## 8. Compute estimate + JIT-trigger gate

Counting rule (fixed, issued amendment 1): one
arm-run = one agent-training OR one agent-world
evaluation job.
- Step 1 trainings: 50 (25 wrapped fresh-per-
  world + 25 non-wrapped base).
- Step 2 evaluation jobs: 30 agents × 5 worlds
  (E0–E4) = 150.
- Step 1 + Step 2 total: **200 arm-runs**,
  which **exceeds the 150 threshold — the
  arm-run trigger FIRES**.
- Step 1 + Step 2 evaluation episodes: 20/10
  arms 15 × 5 × 20 = 1500; 30/14 arms 15 × 5 ×
  30 = 2250; total **3750 episodes** (plus
  training episodes: 50 × 2000 = 100,000
  training episodes, as in every prior run).
- Step 0 checks (separate from the 200, stated
  so the total is not understated): scripted
  mechanics (fast, < 1 minute, no training) +
  sanity trainings 6 (one per arm, §7). Total
  programme arm-runs including Step 0 sanity =
  206, still > 150; the trigger status is
  unchanged by including or excluding Step 0.
- Time estimate (method: Luck pace). The Luck
  run took ~70 minutes for 40 trainings +
  90 evaluation jobs (60 luck + 30 0c) +
  ~2250 evaluation episodes. Scaling by
  trainings (50 vs 40) and evaluation episodes
  (3750 vs ~2250) gives a Step 1 + Step 2
  estimate of roughly **2.5–3.5 hours**, under
  the 4-hour time threshold. This is an
  ESTIMATE from prior pace, not a guarantee:
  per-training time varies by arm/seed and by
  how many training episodes run to the 200-tick
  cap in the multi-door world. The time
  estimate does NOT change the ruling: the
  arm-run trigger already fires, and the owner
  has already ruled proceed on the current
  substrate with no spike (see header). If the
  actual Step 1 + Step 2 time were to exceed
  4 hours, that would be reported as a finding,
  not used to silently change substrate or
  scope.
- **Gate outcome for Step 0.** The design note
  is committed, the variant and checks are
  committed, and Step 0 then STOPS for
  instructor verification. Retraining (Step 1)
  starts only after the instructor verifies this
  design note — not on this Step-0 work's own
  authority, and not before that verification,
  even though the owner has ruled proceed on
  the substrate. The proceed-ruling removes the
  spike requirement; it does not remove the
  Step-0 verification requirement.

## 9. Step 0c agent-check rule (for Step 1, fixed now)

Retrained agents are NEW agents (new weights,
same seeds/protocols). No 0c reproduction
against tracked single-door summaries applies —
training worlds differ by design, so a mismatch
against single-door records would be expected
and meaningless. The check that DOES apply: each
retrained agent's training summary is recorded
and reported, and E0 (W1) is evaluated for every
retrained agent as the sanity endpoint (§3).
An agent with no recorded training summary, or
with no E0 evaluation, is not carried into
Step 3 analysis.

## 10. Out of scope (explicitly forbidden)

No fine-tuning of existing single-door weights
(from-scratch retraining only). No schedule
variation (timing stays uniform across doors —
that is P-Db2). No hyperparameter changes
mid-run. No edits to `world.py`, the doors/sched/
luck variants, or any frozen file. No edits to
`nn/agent_nn.py`, `nn/agent_wrapped.py`,
`nn/agent_mem.py`, `nn/agent_controls.py`, or
`nn/agent_generalization.py` (their code is
reused unchanged; the only new training code is
additive `nn/train_pda.py`, §1). No claims
beyond the worlds, arms, and seeds evaluated.
No JIT work and no benchmark spike under this
work order (owner ruled proceed, no spike).
No other tracks. No AI-approach work (J1/J2/J3
belong to the reciter track, not this session).
No Step 1 retraining, no Step 2 evaluation, and
no Step 3 report (`nn/decisions_pda.md`) in
Step 0 — those start only after instructor
verification of this note.

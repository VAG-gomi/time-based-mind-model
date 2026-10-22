# WO-TBM-FT1 — Step 0a design note (written before any fine-tuning)

**Status: EXPLORATORY.** No two-conclusion rule, no
confirmatory claims. Confirmatory follow-ups return as
proposals with their own preregistration.

**Issuance (recorded here so it cannot be misread
later).** The issued work order authorises Step 0
ONLY (design note + world variant + fine-tune loop +
validity checks + sanity fine-tunes). No spike/proceed
ruling was given at issuance. The compute gate in §8
fires (182 arm-runs > 150), so Step 0 ends with a
return to the owner for the spike/proceed ruling
before any Step 1 fine-tuning starts. P-Db2's PROCEED
ruling does NOT extend to FT1: its scope was the
P-Db2 programme only. Step 0 does not authorise
fine-tuning on its own authority, and instructor
verification of this note does not substitute for
the owner's gate ruling.

**Question.** P-Db2 showed that schedule-varied
retraining from scratch produces fluent rival use
but no novel-door crossing except one agent
(memory/3, 20/20 in F3). FT1 asks, by a different
mechanism: if the SUPERIOR agents (the P-Db2 kept
set, §1) receive a SMALL amount of additional
training on the one door they never trained on —
(5,8) — what happens? Do they acquire (5,8)? Do
they forget the other doors? And decisively: does
learning one novel door unlock novel doors in
general, or only that door? This is FINE-TUNING
(continued training from existing weights). It was
explicitly forbidden in P-Da and P-Db2 precisely
because it is a different experiment; it is
authorised here, if at all, only by this work
order's own issuance, never by P-Da/P-Db2
authority.

## 1. Agents: the P-Db2 kept set (22 agents, verified)

Kept = perfect multi-door record in P-Db2 (every
episode reached across F1, F2, and F4). Verified
from `nn/results_pdb2.jsonl` by the Step 0 author
before writing this note: all 22 agents below are
perfect in F1/F2/F4, and all 8 set-aside agents
fall short somewhere. Any mismatch found here
would have been a design-note deviation; there
was none.

- wrapped (20/10): seeds 0, 1, 2, 3, 4.
- plain (20/10): seeds 0, 2, 3.
- memory (20/10): seeds 0, 2, 3, 4.
- oracle_ind (30/14): seeds 100, 101, 102, 103.
- oracle_new (30/14): seeds 100, 101, 102, 103.
- phase_only_new (30/14): seeds 102, 104.

Set-aside (NOT part of FT1): plain/1, plain/4,
memory/1, oracle_ind/104, oracle_new/104,
phase_only_new/100, /101, /103. A set-aside
fine-tuning probe is a separate proposal, not a
silent extension of this one. (For the record:
plain/1 is 0/20 in F1; oracle_ind/104 is 20/30
in F2; oracle_new/104 is 14/30 in F2;
phase_only_new/101 is 0/30 in F1 despite 30/30
in F0 — F0 success does not imply multi-door
success.)

## 2. Checkpoints: realised deterministically, persisted by snapshot

**Checkpoint definition.** A checkpoint = (arm,
seed, protocol, P-Db2 training world F1)
realised by deterministic re-training with
frozen code (P-Db2 §1 convention, via
`nn/train_pdb2.py` TRAINERS, 2000 episodes).
Original P-Db2 weights, records, and files are
never modified, overwritten, or pooled with
fine-tuned records. Fine-tuned agents are NEW
agents for record purposes: their records live
in NEW files (`nn/training_ft1.jsonl`,
`nn/results_ft1.jsonl`), labelled FT1, with
checkpoint identity recorded per agent.

**Verification (fixed).** Every realised
checkpoint's training summary is verified
against its P-Db2 training record(s) before
any fine-tuning from it: reached, crushes,
mean_ticks, mean_ticks_all, mean_loss,
door_entry_counts, episodes_with_entry
(integers exact; floats to 1e-9). For wrapped,
all 5 P-Db2 eval-world copies must agree with
each other and with the realised summary (P-Db2
already showed them identical; the check
re-asserts it). Any mismatch is a deviation:
that agent is not fine-tuned, and the mismatch
is reported, never hidden by substituting a
weight or adjusting anything.

**Persistence across the gate — the reuse
decision (fixed explicitly, as the work order
requires).** Step 1 REUSES Step 0's realised
checkpoints; it does NOT re-realise them.
Re-realising would add 22 full trainings and
move the programme total from 182 to 204 (§8).
Reuse is exact here for two independent
reasons: (a) the realisation is deterministic
(the verification above proves it for each
agent: same code, same seed, same world gives
the same summary), and (b) the mechanism below
makes reuse a lossless load, not a
re-computation.

Mechanism (`nn/train_ft1.py`, new additive
code): Step 0 saves each sanity checkpoint as
a snapshot file under `nn/checkpoints_ft1/`
(`<arm>_<seed>.pkl`: net params, target params,
Adam moments m/v/t/lr, update counter, agent
RNG states, and for wrapped the ScheduleModel
object — all via pickle, exact float64). The
replay buffer is deliberately NOT snapshotted
(it is fresh at fine-tune start, §3). Snapshot
fidelity is audited at save time by a
save/load round-trip asserting bitwise
equality of net, target, Adam state, update
counter, and RNG streams (no training
involved, so the audit is not an arm-run).
Step 1 then loads the 6 sanity snapshots and
realises only the remaining 16 checkpoints
fresh (each verified against its P-Db2 record
as part of that single realisation, before
fine-tuning). Total full trainings in the
programme is therefore 22 (6 in Step 0 + 16
in Step 1), as counted in §8.

**Deliberate departure, disclosed.** This
snapshot mechanism departs from P-Db2's
no-weights convention (P-Db2 stored no
weights because re-training was cheap and
in-programme). FT1 checkpoints must cross a
gate: Step 0 stops for instructor verification
and the owner's ruling, and Step 1 runs later,
so in-memory weights cannot carry over. The
alternatives were re-realisation (204
arm-runs, §8) or an in-programme-only design
that the gate forbids. Snapshots are NEW
additive files, never P-Db2 files, and their
fidelity is audited as above. If the
instructor rules snapshots unacceptable, the
fallback is the 204-count re-realisation plan,
not silent re-use of sanity fine-tuned
weights: sanity fine-tuned weights are ALWAYS
discarded; only checkpoint snapshots are kept.

**Step 1 → Step 2 handoff (fixed now, so Step
2 needs no new decision).** Step 1 saves a
fine-tuned snapshot per agent by the same
mechanism (after the 200-episode fine-tune).
Step 2 loads fine-tuned snapshots for
evaluation; loads are not trainings and are
not counted. The fine-tune itself is counted
once, in Step 1.

## 3. Fine-tuning protocol (fixed in advance)

- **Dose: FIXED, one dose only — 200 episodes**
  (10% of the original 2000-episode
  training). No second dose, no dose escalation
  inside this work order. A dose-response curve
  is a follow-up proposal, not FT1.
- **Dose bound (stated now, to be stated in
  the report).** If acquisition after 200
  episodes is zero, that means "200 episodes
  insufficient," NOT "cannot learn (5,8)" — so
  a zero is not over-read in either direction.
- **Fine-tuning world: (5,8) ALONE, on the
  arm's training schedule** (20/10 or 30/14,
  offset 0 — the same world as P-Db2 F3,
  reused unchanged via
  `world_pdb2.f3_schedules`). Location is the
  ONLY new thing: same schedule, same code,
  same hyperparameters as P-Db2 training
  (learner MLP [n_inputs, 32, 5], Adam
  lr=1e-3, batch 32, target sync 200, gamma
  0.99, rewards step −0.01 / goal +1.0 /
  crush −1.0, cap 200 ticks), except for the
  fine-tuning-specific choices below.
- **Fine-tuning loop: NEW additive file
  `nn/train_ft1.py`** (`finetune_neural`),
  mirroring `train_pdb2.train_neural_sched`
  except for the choices below. The checkpoint
  NeuralQ object is continued, not replaced:
  constructing a fresh NeuralQ would reset
  weights, target, optimiser, and RNG, and
  would not be fine-tuning at all.
- **Replay buffer: FRESH at fine-tune start**
  (the checkpoint buffer is discarded, not
  carried, not snapshotted). Carried replay
  would mix trained-door experience into
  "brief (5,8) training" and confound the
  dose; the rejected alternative is carrying
  the checkpoint buffer.
- **Epsilon during fine-tuning: FLAT 0.05**
  (the end-of-training value) for all 200
  episodes. The annealed epsilon(ep) schedule
  is REJECTED: at fine-tune episodes 0–199 it
  would give eps ≈ 1.0 → 0.86, i.e. near-random
  exploration, which would confound the dose
  with an exploration schedule. The dose
  measures learning at deployment exploration
  level.
- **Target network and optimiser state:
  CARRIED with the weights** (target params,
  Adam moments m/v/t, update counter, all as
  they stand at checkpoint end; the update
  counter continues, so the next target sync
  falls at the next multiple of 200 updates).
  This is continued training by definition.
  Note on the Adam docstring ("reusing it
  across trainings carries momentum over,
  usually not what you want"): that warning
  is about separate trainings. FT1 is not a
  separate training; it is a continuation,
  so carrying momentum is the correct choice,
  stated here deliberately rather than by
  default.
- **RNG streams: the agent's own streams are
  CARRIED** (exploration draws via agent.rng,
  replay sampling via agent.nprng, continuing
  from checkpoint end). The episode
  start-tick stream is NOT carried — it cannot
  be, because it is local to
  train_neural_sched, not part of the agent
  object — so it is fixed anew as
  random.Random(seed + 2000). The +2000
  offset is deliberate: reusing seed + 1000
  would replay the checkpoint's first 200
  start ticks; +2000 is a distinct, fixed,
  deterministic stream.
- **Wrapped fine-tuning: the checkpoint
  encoder is CARRIED, ScheduleModel and all.**
  Reason: the fine-tuning schedule is
  numerically identical to the familiar
  schedule the model converged on during
  checkpoint training, so carried state is
  correct, not stale. A fresh ScheduleModel
  would inject blank-feature warm-up episodes
  into the dose that are an artifact of
  encoder reset, not of (5,8) learning.
  ScheduleModel period/open_len at fine-tune
  start and end are recorded in the fine-tune
  summary, so any drift during fine-tuning is
  visible. For all other arms the encoders are
  stateless (plain, oracle, phase) or
  per-episode only (memory's previous bit),
  so a fresh encoder of the same type is
  identical; `train_ft1.finetune_encoder()`
  embodies this rule.
- **Wrapped evaluation (Step 2 rule, fixed
  now).** The P-Db2 remedy (fresh training
  per world) does NOT transfer literally,
  because FT1 evaluates fine-tuned weights,
  not retrained ones. Instead: evaluation in
  each world uses a FRESH WrappedEncoder (new
  ScheduleModel) wrapped around the
  fine-tuned network weights — one fresh
  encoder per (agent, world), persisting
  across that world's eval episodes exactly
  as a training encoder persists across
  training episodes. Schedule knowledge for a
  world is therefore built from that world's
  own episodes only, and no cross-world
  encoder state is carried. Non-wrapped arms
  use fresh encoders per world as well
  (identical behaviour, stated for uniformity).
  Drift detection (same as P-Db2): wrapped
  eval logs f101/f102/f103 per step; an
  episode with any f103 == 0.0 step is
  flagged blank_feature and AUDITED —
  transient spawn warm-up (blank only on
  first steps, recovering at the next
  opening, normal behaviour after, as in
  P-Db2's 6 episodes) is not drift, but
  persistent whole-episode blank with
  schedule-blind behaviour IS drift, and
  drifted rows are reported, audited,
  EXCLUDED, never pooled. The audit, not the
  flag alone, decides.

## 4. Evaluation worlds (greedy only, all start phases)

Same protocol as doors / P-Db / Luck / P-Da /
P-Db2: greedy evaluation only, one
deterministic episode per start phase, all
phases (20 for 20/10 arms, 30 for 30/14 arms).
Records flushed per agent-world pair to
`nn/results_ft1.jsonl`; no duplicates, no
partials. F0–F4 worlds are REUSED from P-Db2
(no new code, no edits); G1 is the only new
world (Step 0b, §9).

- **F0 — W1 {(5,5)}: forgetting/sanity
  check (reported, not gated).** Does (5,8)
  training change the F0 collapse? Kept
  plain/memory/phase were 0/20–0/30 before
  fine-tuning (§5).
- **F1 — schedule-varied training world:
  forgetting check.** Do they still pass the
  multi-door world they were perfect in
  (kept: 540/540 before fine-tuning)?
- **F2 — rivals-only staggered: forgetting
  check, plus conflict analysis** (same
  generalised definition as P-Db2 §4:
  per-gap staging, arrival = first staging
  occupation, conflict = arrival door closed
  + another door open, outcomes
  waits/takes/other; preferred-door-closed
  counts as in P-Db2, preferred door = modal
  F1 first door of the P-Db2 agent — note:
  preferred doors are defined from P-Db2 F1,
  not recomputed from post-fine-tuning F1,
  so the forgetting contrast is
  definitionally stable).
- **F3 — W1n {(5,8)}: ACQUISITION check.**
  Do they now cross (5,8)? Same endpoints
  and classes as P-Db2 F3 (D = (4,5) before
  novel entry). Pre-fine-tuning baseline
  (kept set): 1/22 agents (memory/3 only),
  20/540 kept-set episodes (programme total
  20/750 cited only as context; Step 3
  contrasts use the kept-set denominator).
- **F4 — W5n-sched: novel among familiars.**
  Does (5,8) become a first door when
  familiars exist? Novel entries NOT
  classified D/R/W/F (P-Da/P-Db2 rule carried
  over); report first-door distribution +
  (4,5)-flag/ticks/revisits for novel-first
  entries only. Pre-fine-tuning baseline
  (kept set): novel-first 2/540 episodes,
  both memory/3 (programme total 2/750 as
  context).
- **G1 — W1n2 {(5,6)}: SECOND novel door
  only, on the training schedule. TRANSFER
  check.** Does (5,8) training unlock novel
  doors in general? (5,6) sits between
  trained (5,5) and (5,7) — interpolation,
  like (5,8), but a different row. Same
  endpoints and classes as F3 (D = (4,5)
  before novel entry, with the same
  floor-not-door caveat: in G1, (4,5) is
  ordinary floor, not a door staging cell).
  Pre-fine-tuning baseline for G1 is
  UNMEASURED (no agent has ever seen (5,6));
  the report must not present G1 as a
  "before/after" contrast — it is an
  after-only transfer test, and its baseline
  is the P-Db2 F3 pattern (novel doors
  unentered except memory/3), stated as an
  analogy, not a measurement.

## 5. Classification baselines (fixed)

W thresholds use each agent's own P-Db2 F0
median ticks over reached episodes (the
pre-fine-tuning baseline, from
`nn/results_pdb2.jsonl` F0 records).
Post-fine-tuning F0 medians are NOT used
(too many agents reach 0 in F0, making
medians undefined or unstable, and the point
is comparability with P-Db2 classes).
W rule (as in P-Db2 §4): not D, enters a
rival/novel door, AND (ticks > 2 × own F0
median) OR (revisits > 10).

Kept-agent F0 medians (median, n reached in
P-Db2 F0), verified from records:

- wrapped/0: 18.0 (n=6); wrapped/1: 21 (n=17);
  wrapped/2: 33.5 (n=20); wrapped/3: 18.0
  (n=8); wrapped/4: 27 (n=19).
- oracle_ind/100: 33.5 (n=30); /101: 42.5
  (n=30); /102: 24.5 (n=30); /103: 27 (n=29).
- oracle_new/100: 27 (n=1 — THIN, see below);
  /101: 20.5 (n=24); /102: 28 (n=21);
  /103: 18.5 (n=18).

Fallback (median undefined, reached 0 in
P-Db2 F0 — W by revisits > 10 alone, same
fallback rule as P-Db2 §5): all kept
plain/memory/phase agents — plain/0, plain/2,
plain/3, memory/0, memory/2, memory/3,
memory/4, phase_only_new/102,
phase_only_new/104 (9 agents).

Thin-median note (fixed rule unchanged):
oracle_new/100's F0 median rests on a single
reached episode (27 ticks), so its W
threshold (2 × 27 = 54) is thin. Step 3 notes
the n=1 basis wherever that agent's W class
is reported; the rule itself is not changed
for one agent, which would break
comparability.

## 6. Per-agent fine-tuning records

Same summary format as P-Db2 training
records (episodes, reached, crushes, mean
ticks, mean ticks over all episodes, mean
loss, per-door entry counts — here, entries
at (5,8) and any other gap entered during
fine-tuning (the fine-tuning world has only
(5,8), so any entry is a (5,8) entry; the
counts still prove the world was the one
intended) — plus schedule open fractions),
kept per arm/seed, in the NEW FT1 files.
Training records ALSO record the realised
P-Db2 checkpoint identity per agent (arm,
seed, protocol, training world F1, snapshot
file for the 6 sanity agents / realisation
record for the other 16), the fine-tune
choices (eps 0.05 flat, fresh buffer,
start-tick RNG seed+2000), and for wrapped
the ScheduleModel period/open_len at
fine-tune start and end — so any fine-tuned
result traces to its exact starting weights
and its exact fine-tune run.

## 7. Pre-stated questions (verbatim in scope)

(1) Acquisition: after 200 episodes on (5,8)
alone, how many of the 22 kept agents cross
(5,8) in F3 (vs 1/22 before), per agent and
per arm? (2) Forgetting: F1/F2/F4 reached
after fine-tuning vs P-Db2 (perfect before)
— how much multi-door performance is lost,
per agent and per arm? (3) F0: does (5,8)
training change the F0 collapse? (4) F4:
does (5,8) become a first door among
familiars (vs 2/540 kept-set episodes before,
both memory/3)? (5) Transfer: any G1 (5,6)
crossings? (A single G1 crossing is a finding
that must be reported prominently, not
buried, because it bears on the memorisation
verdict.) (6) memory/3: does fine-tuning
preserve its F3 perfection, and does its F0
change? (7) Routes: do newly-crossing agents
use the column route (like memory/3) or
different routes? (8) Dose bound: whatever
the acquisition count, the report states what
200 episodes can and cannot show (§3 dose
bound).

## 8. Compute estimate + JIT-trigger gate

Counting rule (fixed, as in P-Da/P-Db2): one
arm-run = one agent-training (fine-tuning
counts as a training) OR one agent-world
evaluation job. Checkpoint realisation IS an
agent-training under this rule (a full
2000-episode training run), and is counted
as such — this was the draft's omission,
corrected at issuance.

- Checkpoint realisations: 22 (one per kept
  agent: 6 in Step 0, 16 at Step 1 start, §2).
- Step 1 fine-tunes: 22.
- Step 2 evaluation jobs: 22 agents × 6
  worlds (F0–F4, G1) = 132.
- Step 1 + Step 2 + realisations total: 176
  arm-runs.
- Step 0 checks (separate from the 176, stated
  so the total is not understated): scripted
  G1 mechanics (fast, no training) + 6 sanity
  fine-tunes (one per arm, §9). Programme
  total including Step 0 sanity = **182
  arm-runs, which exceeds the 150 threshold
  — the arm-run trigger FIRES**.
- Re-realisation variant (for the record):
  if Step 1 re-realised all 22 checkpoints
  instead of reusing snapshots (§2), the
  total would be 182 + 22 = **204**. That
  variant is NOT the plan; it is stated so
  the 182 figure cannot be misread as
  including re-realisation.
- Evaluation episodes (Step 2): 20/10 kept
  agents 12 × 6 × 20 = 1440; 30/14 kept agents
  10 × 6 × 30 = 1800; total **3240 episodes**
  (plus fine-tuning episodes: 22 × 200 =
  4400; plus training episodes: 22 × 2000 =
  44,000 checkpoint-realisation episodes and
  6 × 200 = 1200 sanity fine-tune episodes,
  counted as checks/realisations, not as
  Step 1 records).
- Time estimate (method: spike/P-Db2 pace).
  Checkpoint realisations dominate: wrapped
  ~163 s × 5 + other arms ~100 s × 17 ≈
  42 min. Fine-tuning is 200 episodes vs
  2000, roughly one-tenth of a full training
  per agent (~10–16 s at spike pace, ~5–6 min
  total). Evaluation is negligible (greedy
  episodes, no training; P-Db2 eval pace was
  ~0.03 s/job). Total ≈ **48 min, well under
  one hour**. Time is NOT the binding
  constraint; the arm-run count is.
- **Gate outcome for Step 0.** The design
  note is written, the variant and checks are
  written, sanity fine-tunes are run, and
  Step 0 then STOPS for instructor
  verification AND for the owner's
  spike/proceed ruling. Fine-tuning (Step 1)
  starts only after BOTH: instructor
  verification of this note, and the owner's
  gate ruling (benchmark spike first, or
  proceed on the current substrate). Neither
  Step 0 work nor instructor verification
  alone authorises fine-tuning. P-Db2's
  PROCEED ruling does NOT extend to FT1.

## 9. Step 0b regression — scope fixed BEFORE running

`nn/world_ft1.py` and `nn/train_ft1.py` are
NEW files; `world.py`, `world_doors.py`,
`world_doors_sched.py`, `world_luck.py`,
`world_pda.py`, and `world_pdb2.py` are NEVER
edited, and no frozen file is modified. The
fine-tuning loop mirrors
`train_pdb2.train_neural_sched` except for
the §3 choices (continued agent object,
fresh buffer, flat eps 0.05, 200 episodes,
(5,8)-only world, start-tick RNG seed+2000);
everything else — learner, target sync,
rewards, crush handling via env_step, episode
cap — is identical.

Regression, fixed here before running:

(i) **Scripted mechanics check**
(`nn/test_world_ft1.py`):
- G1 identity: g1_schedules(train) ==
  {(5,6): (period, open_len, 0)} for both
  protocols (constructed from TRAIN tables,
  not re-typed).
- Per-door open/close table:
  Pdb2AbsentWorld G1 door_open((5,6), tick)
  == door_open_at(tick, period, open_len, 0)
  for every tick in the period, both
  protocols.
- Observe bit: G1 observe bit == training
  schedule bit at every tick in the period,
  both protocols (no phantom/clock
  distinction, as in F3).
- Wall blocking / crush: the P-Da script set
  (scripted, pseudo-random seeds 0/1/2 × 250
  steps, right-spam × 120, stay × 120) on G1
  across all start phases, checking per-step
  invariants (never in a wall cell; crushed
  iff the agent was in a gap whose door closed
  at that tick; a move into the gap is
  possible only while that gap is open).
  Expected case count is computed by the
  script and reported, not asserted in advance
  beyond the stated grid.
- NOT checked, by design: no trace comparison
  against F3 or PdaWorld W1n. G1's gap row
  (5,6) differs from F3's (5,8), so wall/gap
  geometry differs and traces must differ
  where the agent reaches rows 6/8; a
  trace-equality claim would be false.
  Geometry is covered by the wall-blocking
  invariant instead.
- Result (Step 0 run): **51,502 cases
  CHECKED** (2 identity + 100 tick checks +
  51,400 invariant steps).

(ii) **Agent-level sanity**
(`nn/sanity_ft1.py`): ONE kept agent per arm
(6 agents: wrapped/0, plain/0, memory/0,
20/10; oracle_ind/100, oracle_new/100,
phase_only_new/102, 30/14), each: (a)
checkpoint realised by deterministic
re-training and VERIFIED against its P-Db2
record(s) (§2); (b) snapshot saved +
  load-audit PASS; (c) fine-tuned 200
  episodes on (5,8) from the checkpoint.
  Scope rationale, stated before running:
  one seed per arm validates the
  realisation + fine-tune loop for every
  encoder type (including the stateful
  wrapped encoder and the
  period-parameterised oracle/phase
  encoders) without doing Step 1. Sanity
  criteria, fixed here (these are
  loop-validity criteria, NOT
  experiment-outcome gates): checkpoint
  verification exact (no mismatch); snapshot
  load-audit PASS; fine-tune completes 200
  episodes without crash; no NaN in
  mean_loss; per-door counts recorded.
  Reached count is NOT a criterion for the
  fine-tune: the fine-tuning world is novel,
  and zero reached after 200 episodes is a
  legitimate possible outcome, not a loop
  failure — requiring reached > 0 would
  prejudge acquisition, which is the
  experiment's question. If any sanity agent
  fails any criterion, it is reported and
  Step 1 does NOT proceed with that arm
  unexamined; the failure is not hidden by
  substituting a P-Db2 weight or by adjusting
  hyperparameters. Sanity outputs are check
  outputs only (`nn/sanity_ft1.log`,
  `nn/sanity_ft1.jsonl`,
  `nn/checkpoints_ft1/*.pkl`); they are NOT
  Step 1 records and their fine-tuned agents
  are NOT reused in Step 1 (Step 1 fine-tunes
  fresh from the checkpoint snapshots, §2).

(iii) **Fine-tune-loop mirror verification.**
Before the sanity run, `nn/train_ft1.py` is
verified against `nn/train_pdb2.py`'s
`train_neural_sched` by line-level
comparison: the only differences are the
continued agent object (vs fresh
construction), the fresh buffer, the flat
eps, the episode count, the world class
(Pdb2AbsentWorld with F3 schedules), and
the start-tick RNG offset; learner, target
sync, rewards, crush handling, episode cap,
and entry logging are identical. (This
verification is by construction in the file's
docstring and by the sanity run's checkpoint
verification; a line-diff claim is not itself
a substitute for the sanity results.)

## 10. Step 0c agent-check rule (for Steps 1–2, fixed now)

Fine-tuned agents are NEW agents (P-Db2
weights + 200 fine-tune episodes). No 0c
reproduction against P-Db2 summaries applies
to the fine-tuned weights themselves — they
differ by design. The check that DOES apply:
each fine-tuned agent's starting checkpoint
is recorded (arm, seed, P-Db2 training
identity, snapshot file), its fine-tuning
summary is recorded, and it is evaluated in
all six worlds. An agent with no recorded
checkpoint, no fine-tuning summary, or a
missing eval world is not carried into Step 3
analysis. **This check is NOT done in Step
0:** it can only be completed after Step 1
fine-tuning and Step 2 evaluation, and this
note does not claim it as done.

## 11. Out of scope (explicitly forbidden)

No fine-tuning of set-aside agents (kept 22
only). No dose other than 200 episodes (no
escalation, no dose-response here). No new
training worlds beyond the (5,8)-only
fine-tuning world (G1 is an EVAL world only).
No hyperparameter changes mid-run. No edits
to `world.py`, the doors/sched/luck/pda/pdb2
variants, or any frozen file. No pooling of
FT1 records with P-Db2 records. No
overwriting of P-Db2 checkpoints or records.
No claims beyond the agents, worlds, dose,
and seeds evaluated. No JIT work (ruled E,
trigger-gated; Step 0's compute gate handles
the trigger, and P-Db2's proceed ruling does
NOT extend here). No other tracks. No
AI-approach work (J1/J2/J3 belong to the
reciter track, not this session). No Step 1
fine-tuning, no Step 2 evaluation, and no
Step 3 report (`nn/decisions_ft1.md`) in
Step 0 — those start only after instructor
verification of this note AND the owner's
gate ruling.

## 12. Instructor verification (2026-10-22) — PASS

Independently re-derived before committing:
all 6 sanity checkpoints match their P-Db2
training records exactly (reached, crushes,
mean ticks, mean loss, door counts,
episodes-with-entry; all 5 wrapped copies
agree); G1 mechanics check re-run: 51,502
cases CHECKED (2 identity + 100 tick checks
+ 51,400 invariant steps, recomputed exactly);
sanity run-2 numbers match the log exactly
for all 6 agents; kept-set and fallback lists
re-derived from `results_pdb2.jsonl` (22/22
perfect, 8/8 set-aside imperfect, 9 fallback
agents at 0 in F0); compute gate 182 with
204 variant, eval episodes 3240, time ≈48
min — all as stated. No tracked file
modified; Step 0 files are additive only.

**Instructor ruling on §2 snapshots:
ACCEPTED.** The snapshots persist weights
whose summaries were verified exact against
P-Db2, the save/load round-trip is audited
bitwise, the replay buffer is deliberately
excluded (consistent with the fresh-buffer
rule), and the 204-count re-realisation
fallback is stated. Sanity fine-tuned weights
are always discarded; only checkpoint
snapshots are kept. Step 1 reuses the 6
sanity snapshots and realises the remaining
16 fresh, each verified against its P-Db2
record before fine-tuning.

**Owner ruling at the Step 0 gate
(2026-10-22): BENCHMARK SPIKE FIRST,
measurement only.** At the gate (182
arm-runs, trigger FIRES; projected total
≈48 min, under one hour), the owner chose
the spike before any Step 1 fine-tuning,
per the trigger framework. The spike
measures current-substrate FT1 costs only
(fine-tune pace per arm from the 6 loaded
sanity snapshots, eval-sample pace,
hotspots, projected Steps 1–2 time, and the
equivalence gap for a future candidate).
**The spike authorises NO Step 1
fine-tuning, NO substrate change, and NO
JIT/vectorized implementation; spike
fine-tunes are SPIKE-labelled measurement
outputs, never Step 1 records, and their
weights are discarded. A second owner ruling
is still required after the spike numbers
before Step 1 starts.**

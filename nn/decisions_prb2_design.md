# PRB2 noisy-eval probe — design note (Step 0 instructor-verified)

**Status: Step 0 instructor-verified 2026-10-23 (see §17).**
This is the Step 0 design note for WO-TBM-PRB2 (issued 2026-10-23,
Step 0 ONLY), written under the governing proposal
`~/workspace/your_files/PROPOSAL-TBM-PRB2.md` (owner version,
2026-10-23). The proposal governs the design; the work order
governs the scope and sequence. **Step 0 verification does NOT
authorize the probe run: the owner gate ruling is still required
(counting settled there; spike-first only if rep-level counting
is chosen). The probe report's verdict section remains for the
verifying session to write after the probe run; nothing in this
note pre-fills it.

## 1. Authorization — Step 0 only; the probe run is NOT authorized yet

This note specifies the probe. It does not authorize it. The
probe run — any noisy evaluation beyond the single Step 0
sanity retrain in §13 — is NOT authorized by the Step 0 work
order and is NOT authorized by this note. It becomes authorized
only after, in order: (a) instructor verification of this Step
0 (this note, the runner, and the sanity result), and (b) an
owner gate ruling at the Step 0 gate (which also settles the
job-counting question in §10; spike-first only if rep-level
counting is chosen). Until both exist, the runner's probe mode
must not be invoked. The runner enforces this weakly with an
accident guard (probe mode refuses to start without the
literal flag `--gate-ruled`); the guard is not a substitute
for the ruling, and passing the flag without a ruling would
itself be a scope violation.

## 2. Background and research question

MD1 (moving-door training) returned UNINTERPRETABLE
(instructor-verified, commit f601339): pooled trained greedy
eval 388/4800 (8.08%) fell below the frozen 20% bar, so no
conclusion about following was drawn. Underneath that verdict
sits a descriptive gap: MD1 training reached 21–29% per arm,
while frozen greedy eval on trained rows fell to 2–17% (plain
21.4% → 1.75%, wrapped 29.0% → 5.00%). PRB1 proved this shape
for one agent: memory/3 scored 0/20 greedy, 45/100 under 5%
noise, and 161/200 during training, on the same frozen
weights, with verdict NEITHER at its pre-stated bar.

Research question: does ε = 0.05 evaluation lift MD1
agents' crossing rates — on trained rows and on held-out
rows — compared to their greedy eval, same agents, same
worlds, only the noise differing? If yes, greedy eval
understates what the networks learned, and the FOLLOWING
question must be re-asked under noisy eval. If no, the
policies genuinely are weak. PRB2 does NOT change MD1's
verdict: UNINTERPRETABLE stands for the greedy, pre-stated
test, and PRB2 asks a new question under its own bars.

## 3. Agents and retraining protocol

No MD1 checkpoints exist: MD1 agents were held in memory and
discarded after evals. The probe therefore deterministically
RETRAINS whoever it tests, with the exact MD1 Step 1 protocol
(train_md1.TRAINERS, same code, same seeds, same row
multisets, same hyperparameters). Retraining reproduces MD1
agents exactly — proven for seed 0 (programme seed-0
trainings matched Step 0 sanity to the number, and the spike
cross-check matched too).

Agents (6, one seed per arm, seed 0 — the sanity/spike seed):
wrapped/0, plain/0, memory/0, oracle_ind/0, oracle_new/0,
phase_only_new/0. All six arms represented. 6 retrainings
cost ~28 minutes at MD1 spike paces. A full 30-agent version
is a follow-up if the effect is interesting, not part of
this probe.

Determinism gate (binding): each retrained summary is
checked against its MD1 training record BEFORE evaluating.
The comparison is exact on five fields: reached, crushes,
mean ticks over reached, row counts, row-reached. If a
retrain does not match exactly, that agent's probe results
are flagged, not silently pooled: its evals are skipped and
the failure is recorded as a deviation. Retrain weights are
held in memory and discarded after evals. No checkpoints are
written. Retrains are reproductions, not new training.

## 4. Worlds and repetitions

All 10 MD1 fixed worlds (R0–R9), grouped trained (8: R0, R1,
R3, R4, R5, R6, R7, R9) vs held-out (2: R2, R8) — the MD1
verdict worlds, single gap per world, fixed schedule
period 20, open_len 10. The M3 mixed world is NOT evaluated
in the probe: fixed worlds suffice for the question, and
M3's partial row coverage would only muddy the pooled
comparison.

Repetitions: 5 reps per (agent, world). Each rep covers all
20 start phases once, in increasing order 0–19. Per
(agent, world): 5 × 20 = 100 episodes. Total probe episodes:
6 agents × 10 worlds × 5 reps × 20 phases = 6000 episodes.
Eval is fast (MD1 spike pace ~0.08 s per 20-episode job);
eval cost is negligible next to retraining.

## 5. Noisy-eval RNG protocol (exact, reproducible)

Noise level: a single level, ε = 0.05, PRB1's protocol
exactly. A second level is a follow-up, not bundled in now.

Seed formula (fixed, distinct per (agent, world) and rep,
no wall-clock or process-order dependence). Arm indices:
wrapped 0, plain 1, memory 2, oracle_ind 3, oracle_new 4,
phase_only_new 5 (identical to world_md1.ARM_INDEX). World
indices: R0 0, R1 1, R2 2, R3 3, R4 4, R5 5, R6 6, R7 7,
R8 8, R9 9 (row order). job_index = arm_index × 10 +
world_index (0–59). Rep index r ∈ {0, 1, 2, 3, 4}. The rep
RNG seed is:

    seed = 800000 + 1000 × job_index + rep

Examples: wrapped/R0 rep 0 → 800000; memory/R2 rep 2 →
822002; phase_only_new/R9 rep 4 → 859004. The 800000 base
is deliberately distinct from PRB1's 900000 base, so no
stream is ever confused with a PRB1 stream.

Stream object and lifecycle: one stream per (agent, world,
rep): `rng = random.Random(seed)`, created once when the
rep starts. The stream runs CONTINUOUSLY across that rep's
episodes (never reset mid-rep, never re-seeded per episode,
never shared across reps or (agent, world) pairs). A fresh
stream is created for each rep from its formula seed.

Consumption pattern (exact — the only draws from the
stream). At each step of each episode in the rep, in
order:

1. Exactly one draw: `u = rng.random()` (uniform in [0, 1)).
2. If `u < 0.05`, exactly one second draw:
   `k = rng.randrange(5)`, and the action is
   `ACTION_NAMES[k]` where `ACTION_NAMES = ["up", "down",
   "left", "right", "stay"]` (the agent_nn.py action list,
   the same list MD1's records use). The step is flagged
   `epsilon_action = True`.
3. If `u >= 0.05`, the action is the greedy action
   (lowest-index argmax of the frozen network output) and
   NO second draw is made. The step is flagged
   `epsilon_action = False`.

No other draws consume from the stream: no shuffling, no
sampling, no episode-order randomisation, no tie-breaking
draws. The greedy action's tie-break (lowest index) is
deterministic and draws nothing. No numpy RNG draw appears
anywhere in the runner. Any probe code that adds a draw
anywhere else violates this protocol and must be reported
as a deviation.

## 6. Encoder rule (explicit)

A fresh encoder object is created per (agent, world, rep)
via md1_run.fresh_encoder. Within a rep, the encoder is
reset at every episode start exactly as in MD1's greedy
evaluation (`encoder.reset(world)` after
`world.reset(start_tick=start)`, phases in increasing
order). The encoder persists across a rep's 20 episodes.
Concretely:

- wrapped: a fresh WrappedEncoder (blank ScheduleModel)
  per rep; the ScheduleModel starts blank at each rep and
  converges during the rep's episodes, so blank-feature
  warm-up is a per-rep-start effect, as MD1's was a
  per-world-start effect. Blank episodes are audited the
  same way (per-step f103 known flag on wrapped steps).
- memory: the previous-bit history is seeded by `reset()`
  with the door bit at the reset tick; the history then
  shifts per step within the episode and never carries
  across an episode boundary.
- plain, oracle_ind, oracle_new, phase_only_new: door-bit
  or schedule features computed from the current
  observation/tick — stateless across episodes by
  construction.

## 7. Forward-check (binding)

The runner implements ε-greedy explicitly: greedy action
from the frozen network (`int(numpy.argmax(agent.net
.forward(x[None, :])[0]))`, the same expression as the
greedy branch of agent_nn.NeuralQ.act), random action from
the §5 rep stream. The runner never invokes the agent's
own epsilon action mechanism: no code path in the runner
calls the agent's action method at all, and the agent's
own RNG streams are never touched. This is verifiable by
searching the runner for any invocation of the agent
action method and for any numpy random routine: both
searches must return nothing (the runner's docstring
names neither pattern literally, so a clean search is
conclusive). The Step 0 forward-check evidence is
recorded in §13's runner-evidence note: an abstract-syntax
walk of the runner found no action-method call among its
attribute calls, and its only random draws are the rep
stream's `random()` / `randrange(5)` calls.

## 8. Greedy baselines republished (recomputed from MD1 records)

The comparison is same-agents, same-worlds, only the noise
differing. The 30-agent pooled numbers (388/4800 trained,
4/1200 held-out) are NOT the baselines. The baselines below
were recomputed for this note from `nn/results_md1.jsonl`
(seed 0 only) and match the proposal's §3.5 table exactly,
every number — per-arm, per-world, pooled, and control.
The probe run must cite these baselines, not re-derive
them at run time and not borrow the 30-agent pooled
numbers.

Per-agent seed-0 greedy (trained / held-out):

| agent | trained greedy | held-out greedy |
|---|---|---|
| wrapped/0 | 0/160 | 0/40 |
| plain/0 | 14/160 | 0/40 |
| memory/0 | 20/160 | 0/40 |
| oracle_ind/0 | 0/160 | 0/40 |
| oracle_new/0 | 27/160 | 2/40 |
| phase_only_new/0 | 0/160 | 0/40 |
| **pooled seed-0** | **61/960 = 6.35%** | **2/240 = 0.83%** |

Per-world seed-0 greedy (6 agents × 20 phases = 120 per
world): R0 21/120, R1 0/120, R3 4/120, R4 12/120, R5 3/120,
R6 13/120, R7 8/120, R9 0/120 (trained, summing to 61);
R2 2/120, R8 0/120 (held-out, summing to 2).

Control baseline: memory/0 R0 greedy 20/20 (a perfect world
in MD1). The control's noisy pooled (§9) is judged against
this 20/20.

## 9. Interpretation rule (fixed in advance)

Pooled trained = noisy reached across the 8 trained fixed
worlds (6 agents × 8 worlds × 5 reps × 20 phases = 4800
episodes). Pooled held-out likewise (6 × 2 × 5 × 20 = 1200
episodes). Greedy baselines are the §8 seed-0 numbers
(61/960 trained, 2/240 held-out). Control = memory/0 R0
noisy pooled (5 reps × 20 = 100 episodes; greedy baseline
20/20).

- LIFTED (trained): noisy trained pooled ≥ 2× greedy
  trained pooled AND noisy trained pooled ≥ 20% absolute.
  Both conditions required.
- NO LIFT: noisy trained pooled < 1.25× greedy trained
  pooled.
- NEITHER: all other complete outcomes (catch-all,
  mirroring PRB1's NEITHER).
- HELD-OUT LIFT (flag, descriptive, not the verdict):
  noisy held-out pooled ≥ 20% absolute (baseline 2/240 =
  0.83%).
- Control guard: if noisy control pooled < 75%, the probe
  is UNINTERPRETABLE (noise destroying a healthy policy
  means lift cannot be interpreted).
- INCOMPLETE: any agent, world, or rep missing → no
  verdict.

Precedence, applied in this order: INCOMPLETE →
UNINTERPRETABLE (control) → LIFTED → NO LIFT → NEITHER
(catch-all).

BINDING-CONDITION NOTE (disclosed, not hidden): with the
recomputed baseline (61/960 = 6.35%), the absolute
condition binds — 2× the baseline is 122/960 = 12.7%,
below 20%, so LIFTED effectively requires ≥ 192/960 noisy
trained reached. NO LIFT effectively means ≤ 76/960
(1.25 × 61 = 76.25, and reached counts are integers), and
NEITHER covers 77–191. The bands are gap-free and
non-overlapping. No bar may change after results exist.

## 10. Job counting and the trigger (stated transparently)

Two honest counts describe the same compute. Total probe
compute is ~28–30 minutes — nowhere near the 4-hour
threshold either way — so this is purely a question of
how jobs are defined:

- PROPOSED counting (following MD1's job unit): a job is
  one (agent, world) evaluation with all 5 reps inside
  it. 60 jobs + 6 retrainings = **66 arm-runs → the
  trigger does NOT fire.**
- Rep-level counting (PRB1's unit): a job is one (agent,
  world, rep). 300 jobs + 6 retrainings = **306 arm-runs
  → the trigger FIRES**, which would require a spike-first
  ruling by the book, even though the MD1 spike already
  measured exactly these paces (MD1 trainings and MD1
  evals).

The proposal recommends the (agent, world) counting. This
note records the recommendation and both counts, but the
counting decision is NOT settled by this note or by the
Step 0 issuance: it is deferred to the owner gate ruling
after Step 0 verification, as the work order's §3 states.
If the gate ruling chooses rep-level counting, a spike-first
ruling is required by the book before the run. Either way,
the episodes, the compute, and the bars are identical;
only the counting label changes.

## 11. Record formats (separate files, never pooled with MD1)

- `nn/results_prb2.jsonl`: probe eval episodes. Fields:
  the MD1 episode fields (kind, world, gap_row,
  start_phase, ticks, crushes, reached, return,
  first_door, door_entries, switches, stall, stall_kind,
  revisits, staging_order, arrival, first_entry_index,
  post_entry_ticks, post_entry_stall,
  post_entry_stall_kind, blank_feature, conflict (None),
  conflict_outcome (None), staging_wait,
  door_status_at_arrival, door_status_at_entry,
  schedules, steps, arm, seed, period, open_len) PLUS
  rep, rep_rng_seed, epsilon = 0.05. Per-step records
  carry `epsilon_action` (True exactly on rep-stream
  random steps) and, for wrapped arms only, f101/f102/
  f103 (log_features, as in MD1). Non-wrapped arms record
  blank_feature False by construction, not by
  measurement, as in MD1.
- `nn/training_prb2.jsonl`: retrain reproduction summaries
  (kind, arm, seed, episodes, reached, crushes,
  mean_ticks_reached, row_counts, row_reached,
  determinism_match vs the MD1 training record,
  wall_time_s). Labeled as reproductions; never pooled
  with `nn/training_md1.jsonl`.
- `nn/decisions_prb2.md`: the probe report. Its verdict
  section is left EMPTY for the verifying session to
  write (see §14) — the report must not pre-fill a
  verdict. Step 0 creates NO probe report and NO probe
  records: `nn/results_prb2.jsonl` and
  `nn/training_prb2.jsonl` do not exist after Step 0.

## 12. Resume-safety and determinism gate (probe mode)

A (agent, world) job is complete only when it holds 100
records with all (rep, phase) pairs exactly once. A
partial job is stripped whole and re-run. A torn trailing
line is truncated at load time and the repair is logged.
Training records are written once per (arm, seed) after
the determinism gate; on resume the agent is retrained
deterministically to run missing evals without writing a
duplicate training record (MD1 Step 1 pattern). A
determinism-gate failure flags the agent: its evals are
skipped, the failure is recorded as a deviation, and
nothing from that agent is pooled. Final accounting
reports training records, eval records, duplicates (target
0), jobs run, and deviations.

## 13. Step 0 determinism sanity (result)

Protocol: retrain ONE agent (wrapped/0) by the probe
protocol (exact MD1 Step 1 protocol via train_md1) and
verify its summary against the MD1 training record
exactly (reached, crushes, mean ticks over reached, row
counts, row-reached). Sanity weights are discarded;
sanity is NOT a probe result and creates NO probe
records. Sanity output: `nn/prb2_sanity.log` and
`nn/prb2_sanity.json`. Any mismatch is reported, not
hidden, and blocks the probe run until resolved.

SANITY RESULT (measured 2026-10-23, self checked by the
executor, NOT independently verified): retrained wrapped/0
in 272.5 s — reached 619/2000, crushes 116,
mean_ticks_reached 87.17609046849758, row counts exactly
250 per trained row, row_reached {0: 78, 1: 84, 3: 98,
4: 106, 5: 86, 6: 64, 7: 76, 9: 27}. Determinism gate vs
the MD1 training record: MATCH on all five fields
(reached, crushes, mean_ticks_reached, row_counts,
row_reached all True). The probe run is NOT blocked by
sanity. Sanity weights were discarded; no probe records
were created. Full record: `nn/prb2_sanity.json`, log:
`nn/prb2_sanity.log`.

Runner-evidence note (forward-check, §7): an
abstract-syntax walk of `nn/prb2_run.py` found no
action-method call among its attribute calls; its only
random draws are the rep stream's `random()` /
`randrange(5)` calls, and its greedy action is the
lowest-index argmax of the frozen network output. The
`--self-test` mode verifies the seed formula for all 60
jobs × 5 reps (300 rep seeds, all matching
800000 + 1000 × job_index + rep) and the arm/world
index mapping.

## 14. Verification-label rules (structural — bind the executor)

These rules (proposal §7) apply to this note, the probe
report, and every commit carrying them:

- This note and the probe report are titled DRAFT
  (UNVERIFIED) until the instructor ruling exists, and
  that label stays on every copy until the ruling
  arrives.
- Never write verified or PASS in a title, header,
  status section, or commit message before the ruling
  exists. A result the executor checked itself is
  described as self checked only, never as
  independently verified.
- Title, status section, and commit message must say
  the same thing; when the status changes, ALL of them
  change at the same time. Section numbers are checked
  for duplicates and gaps before committing, and
  renumbered when a section is inserted or moved.
- A file is claimed as committed only after the commit
  exists, and as pushed only after the remote matches
  the local commit.
- The verdict section is left for the verifying session
  to write; it is not filled in advance with an expected
  verdict. When the ruling arrives, its reference is
  recorded beside the new status, and the draft label is
  replaced with the ruled verdict in the title, the
  status section, and the commit message at the same
  time. Numbers and evidence are unchanged when only a
  status label changes.

## 15. What this probe does not do

- It does NOT change MD1's verdict. UNINTERPRETABLE
  stands for the greedy, pre-stated test.
- It does NOT authorize MD1 stage 2 (within-episode
  movement), dose-response, a second noise level, a full
  30-agent version, or any substrate change.
- It does NOT train new policies. Retrains are
  reproductions of MD1 agents, verified exact before
  evaluating, weights discarded after evals.
- It does NOT pool probe records with MD1 records,
  sanity records, or spike measurements.
- Step 0 does NOT create probe records, does NOT run
  noisy evals beyond the sanity retrain, and does NOT
  commit or push anything.

## 16. Gate sequence (standard, no exceptions)

1. Owner approval of the proposal (done) and issuance of
   the work order — Step 0 ONLY (done 2026-10-23).
2. Executor writes this design note + runner + sanity
   (Step 0, DRAFT (UNVERIFIED)).
3. Instructor verifies Step 0 against the issued order
   (this note against the proposal and the order, the
   runner against this note, sanity results against the
   MD1 records). The instructor's ruling is stated
   plainly; only then does the instructor commit and
   push, verifying remote HEAD against local HEAD.
4. Owner gate ruling (counting settled here; spike-first
   only if rep-level counting is chosen; otherwise
   proceed-or-not on the current substrate).
5. Only after the ruling: the probe runs, under this
   note exactly as written. The report returns for
   instructor verification, and no verdict is claimed
   before that verification.

## 17. Instructor verification of Step 0 (2026-10-23) — PASS

Recomputed independently from the files (not from the executor's
claims): design note sections 1–16, numbered with no duplicates
or gaps, titled DRAFT (UNVERIFIED) as delivered; authorization
statement in §1 in the note's own words (probe run NOT
authorized until instructor verification AND the owner gate
ruling, in that order); RNG formula, arm/world order, and
consumption pattern match the proposal exactly, and the
runner's `--mode self-test` reproduces all 300 rep seeds from
800000 + 1000 × job_index + rep; forward-check re-derived by
AST walk (no attribute name containing 'act', no agent.act
or .act( occurrence, no numpy random routine; only rep-stream
random()/randrange(5) draws); sanity retrained wrapped/0
matches the MD1 training record on all five fields
(reached 619, crushes 116, mean_ticks_reached
87.17609046849758, row counts 250 per trained row,
row_reached {0: 78, 1: 84, 3: 98, 4: 106, 5: 86, 6: 64, 7: 76,
9: 27}) and the sanity file's embedded md1_reference equals
the MD1 training record on the same five fields; sanity wall
272.5 s consistent with spike/sanity paces; baselines in §8
match the proposal's §3.5 table and the instructor's own
recomputations (per-arm, per-world, pooled 61/960 and 2/240,
control 20/20); bars, precedence, binding note (≥ 192/960
LIFTED, ≤ 76/960 NO LIFT, 77–191 NEITHER), and both job
counts (66 / 306) match the proposal with the counting
decision correctly deferred to the gate ruling; probe-mode
accident guard (`--gate-ruled` refusal) present in the
runner; record superset in §11 (full MD1 episode fields
plus rep/rep_rng_seed/epsilon/epsilon_action) accepted as a
documented implementation choice, not a deviation — it adds
evidence fields and removes none. Scope confirmed: git status
shows only the 4 Step 0 files plus the pre-existing
luck_notified.marker; no tracked file modified;
nn/results_prb2.jsonl and nn/training_prb2.jsonl do not
exist; no probe records created; no pre-claim of
verification found in any Step 0 file (the only
verified/PASS mentions are MD1's own verified status, the
label rule text, and the gate procedure). **Step 0 ruling:
PASS. The probe run remains NOT authorized pending the
owner gate ruling.**

## 18. Owner gate ruling (2026-10-23) — PROCEED, (agent, world) counting

Gate chain complete: proposal approved → work order issued Step 0
ONLY → Step 0 verified PASS (§17, commit ce09457) → this ruling.
**The probe run is AUTHORIZED, on the current hand-written numpy
substrate only, under this note exactly as written.** Job counting
is SETTLED by this ruling: a job is one (agent, world) evaluation
with all 5 reps inside it — 60 jobs + 6 retrainings = **66
arm-runs; the trigger does NOT fire; no spike is required**
(total probe compute ~28–30 minutes, far under the 4-hour
threshold; the MD1 spike already measured these paces). The
rep-level count (306) is NOT adopted. The runner's `--gate-ruled`
flag may now lawfully be passed, because this ruling exists.
NOT authorized: any substrate change, any bar/formula/row-set/
counting change (counting is frozen by this ruling), any pooling
with MD1/sanity/spike records, or any verdict claim before
instructor verification of the probe report. The probe report's
verdict section remains EMPTY for the verifying session to
write after verification.

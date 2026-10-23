# MD1 design note (WO-TBM-MD1, Step 0 — DRAFT for instructor verification)

**Status: DRAFT.** Not instructor-verified, not
committed, not pushed. Step 0 scope only: this
note, the code listed in §10, and the sanity
checks in §11. **Steps 1+ are NOT authorized by
this note.** Steps 1+ start only after BOTH (a)
instructor verification of this Step 0 (this note
against the issued work order, code against this
note, sanity results against the records) AND (b)
a separate owner gate ruling at the Step 0 gate.
Because the arm-run count (§9) is 366 > 150, the
compute trigger in ruling E fires, so a benchmark
spike (MEASUREMENT ONLY, spike records never
pooled as Step 1 records) comes first per ruling E
§5(1), and Steps 1+ additionally require the
owner's second ruling ("proceed on the current
substrate") after the spike. **No substrate
change is authorized by the issued work order.**
No threshold, row set, eval variant, or formula
in this note may be changed after results exist;
in particular nothing here may be adjusted after
seeing Step 1 results.

## 1. Objective and lineage

Determine whether an agent can learn to FOLLOW a
door — go to wherever the gap is — instead of
memorizing one address. Lineage: Luck → P-Da →
P-Db2 → FT1 → PRB1 → MD1. Every programme so far
trained doors at fixed rows; MD1 is the first to
vary the door row during training. The test
stands or falls on the held-out rows (§2): training
on 8 rows and succeeding on those 8 rows is P-Da
with more addresses, not following.

## 2. Worlds and rows (frozen at issuance)

Ten rows exist (0–9). The wall is column x=5;
the gap in an episode is the single cell (5, row).

- TRAINED_ROWS = (0, 1, 3, 4, 5, 6, 7, 9).
- HELDOUT_ROWS = (2, 8).
- The gap row varies BETWEEN episodes and is
  fixed WITHIN an episode. Within-episode door
  movement is OUT of scope (stage 2, separate
  proposal).
- Schedule: fixed 20/10 for ALL training and eval
  (period 20, open_len 10). The gap opens on the
  20/10 cycle regardless of row; the schedule does
  not vary by row.

Mechanics are inherited, not re-implemented:
every MD1 world is a single-gap MultiDoorWorld
(world_doors.py) with schedule 20/10. Wall
blocking, gap-entry blocking while closed, and the
crush rule (agent in the gap cell when that door
is closed → back to start, episode continues)
are MultiDoorWorld's verified mechanics. This
note's world module (world_md1.py) only wires
factories (gap sets, schedules) and the row-draw
protocols; it introduces no new step/crush/
blocking code. world.py and every existing
variant are NEVER edited.

Fixed worlds: one per row 0–9, record name
"R{row}" (R0..R9). Mixed world M3 is NOT a
separate world class: an M3 eval job is 20
fixed-world episodes, one per start phase 0..19
in order, each with its gap row drawn by the M3
eval RNG (§4). The runner draws the row per
episode explicitly and records it; no RNG hides
inside reset().

## 3. Training row protocol (exact)

Three independent random streams per training run
(all random.Random instances; distinct seeds, so
no stream perturbs another):

- Agent stream: Random(seed) — NeuralQ's own
  stream (epsilon draws in act(); replay sampling
  uses a separate np.random.RandomState(seed)).
- Start-tick stream: Random(seed + 1000) — the
  P-Da/P-Db2 convention. Episode e's start tick
  is start_rng.randrange(20), drawn in episode
  order.
- Row stream: Random(seed + 5000) — used ONCE,
  to shuffle the row multiset (below). It never
  touches the agent stream or the start-tick
  stream, so agent RNG streams stay comparable to
  the P-Da/P-Db2 conventions.

Balanced exposure, by construction (not by luck):
row_multiset(seed) = 250 copies of each trained
row, shuffled once with the row stream. Episode
e of that agent's training uses multiset[e].
Exactly 250 episodes per trained row, 2000 total,
no held-out row, deterministic given seed. Per-row
training counts are recorded per training and
Step 1 asserts 250/row from the records; any
imbalance is disclosed, not smoothed over.

"Row order shuffled by the training RNG" in the
work order is implemented as this shuffled
multiset. A per-episode uniform draw would NOT
guarantee 250/row and is NOT what this note
specifies.

## 4. M3 eval protocol (exact)

M3 row set, stated explicitly: TRAINED ROWS ONLY.
M3 follows the training protocol, and the training
protocol never uses held-out rows; held-out rows
therefore never appear in M3. (If M3 drew from all
10 rows it would contaminate the sanity check
with held-out exposure; it does not.)

Per (arm, seed): eval_rng =
random.Random(700_000 + 1000 * ARM_INDEX[arm] +
seed), with ARM_INDEX = {wrapped: 0, plain: 1,
memory: 2, oracle_ind: 3, oracle_new: 4,
phase_only_new: 5}. The 20 M3 rows are
[eval_rng.choice(TRAINED_ROWS) for the 20
episodes in increasing start-phase order 0..19]:
uniform with replacement, deterministic given
(arm, seed), independent of process order and
wall-clock time. Row coverage per agent is
partial by construction (20 draws over 8 rows),
so M3 is a sanity check that training worked in
its own regime — NEVER a per-row measurement,
and its pooled number is NEVER substituted for
the fixed-world per-row numbers.

## 5. Observation contract: per-arm semantics

Principle (WO §2.2): the door-state information
available to each arm FOLLOWS the gap row.
Wherever the gap currently is, the agent is told
whether the door is open or closed THERE. No arm
observes the gap row directly. Every arm's
door-derived features are computed from the
CURRENT gap's door bit and/or the fixed 20/10
schedule. The row varies only in (a) which cells
are walls vs gap (geometry the agent discovers
by movement and blocking), and (b) the agent's
own position. At a held-out row, an arm's door
features behave identically to trained rows (same
schedule, same bit semantics); only the geometry
is new. That is what makes held-out rows a
genuine generalization test.

One qualification, stated so the report cannot
overclaim: the position one-hot DOES distinguish
rows positionally (standing at (4,8) vs (4,5)
are different inputs). "Following" must therefore
be learned as a row-general policy (go to the
staging cell of the open gap, whatever row it is
on), not read off a row feature. The network
must generalize across position inputs; MD1
tests exactly that.

Per-arm semantics (all defined cleanly; see the
UNINTERPRETABLE rule below the table):

- plain (101 inputs, PlainEncoder, stateless):
  position one-hot (100) + door bit. The door
  bit is world.observe()'s bit, i.e. the CURRENT
  gap's door state (a single-gap MultiDoorWorld's
  door bit IS its gap's door state). Clean.
- memory (102 inputs, MemoryEncoder): plain +
  previous-tick door bit. Reset seeds prev_door
  with the reset-tick door bit ("no change
  observed yet"); each encode uses the stored
  prev_door, then updates it to the current
  observation's bit. Both bits are the CURRENT
  gap's door state at consecutive ticks. Clean.
- wrapped (104 inputs, WrappedEncoder +
  ScheduleModel): plain + phase/20, remaining/10,
  known flag, from a ScheduleModel fed (tick, door
  bit) exactly once per observation. The door bit
  is the current gap's bit; the schedule is fixed
  20/10 regardless of row, so the model's
  period/open_len estimates (20, 10) are valid
  for every row, and the fixed scales (20.0, 10.0)
  match the schedule as in B2. reset() calls
  model.new_episode() (episode boundary ONLY;
  the model is NOT cleared). Training: ONE
  encoder per run, persisting across all 2000
  episodes (as in train_neural). Eval: FRESH
  encoder per (agent, world), persisting across
  that world's 20 episodes (the FT1 Step 2 rule;
  see §7). The door-bit sequence across an
  episode boundary jumps from one episode's end
  phase to the next episode's start tick — exactly
  as in P-Da/P-Db2 (random start ticks per
  episode) — so varying rows add no new
  model-handling requirement. Clean.
- oracle_ind (104 inputs, OracleIndEncoder):
  plain + phase/50, remaining capped 5/5, known=1,
  computed from world.period / world.open_len
  READ FROM THE WORLD OBJECT at encode time
  (20/10 on every MD1 world) + tick + the gap's
  door bit. These are the TRUE phase/remaining
  of the CURRENT gap's door, not of any fixed
  row. Clean. (The encoder reads the schedule
  from the world, so no fixed-row assumption is
  baked in.)
- oracle_new (104 inputs, OracleNewEncoder(20,
  10)): plain + (tick % 20)/20, remaining
  (10 - tick % 20)/10 when the door is open else
  0.0, known=1. Constructed with the fixed
  20/10 constants; the door bit is the current
  gap's bit. This is the perfect-knowledge upper
  bound FOR THE GAP'S DOOR. Clean.
- phase_only_new (102 inputs,
  PhaseOnlyNewEncoder(20)): plain + clock
  (tick % 20)/20. Row-independent by
  construction. Clean.

UNINTERPRETABLE-arm rule (WO §2.2): if this note
could not define an arm's observation semantics
cleanly, that arm's MD1 results would be reported
as UNINTERPRETABLE for that arm, not silently
pooled. NO arm triggers this rule: all six arms
are defined cleanly above, so no arm is excluded
on observation-semantics grounds. (Programme-
level UNINTERPRETABLE in §8 is a different,
pooled-trained rule; it is not an arm-level
exclusion.)

## 6. Arm-by-protocol table and hyperparameters

ALL arms run the fixed 20/10 schedule with seeds
0–4 (the 20/10 seed convention). 30 fresh agents
trained from scratch; no existing agent is
fine-tuned in MD1.

| arm | n_inputs | encoder | protocol | seeds |
|---|---|---|---|---|
| wrapped | 104 | WrappedEncoder | 20/10 | 0–4 |
| plain | 101 | PlainEncoder | 20/10 | 0–4 |
| memory | 102 | MemoryEncoder | 20/10 | 0–4 |
| oracle_ind | 104 | OracleIndEncoder | 20/10 | 0–4 |
| oracle_new | 104 | OracleNewEncoder(20,10) | 20/10 | 0–4 |
| phase_only_new | 102 | PhaseOnlyNewEncoder(20) | 20/10 | 0–4 |

RE-PROTOCOLLING WARNING (binding on the Step 1
report): in P-Da/P-Db2/FT1 the oracle_ind,
oracle_new, and phase_only_new arms ran 30/14
with seeds 100–104. In MD1 they run 20/10 with
seeds 0–4. MD1 oracle/phase numbers must NOT be
presented as continuations of those P-Db2
results; the protocol changed, so the comparison
is not like-for-like. Wrapped/plain/memory keep
their P-Da/P-Db2 protocol (20/10, seeds 0–4);
their MD1 numbers remain comparable in protocol
terms (the training WORLD changed, which is the
point of MD1).

Hyperparameters (identical to P-Da/P-Db2 except
the moving-gap world): NeuralQ [n_inputs, 32, 5],
Adam lr=1e-3, replay buffer 10000, batch 32,
target sync 200, gamma 0.99, 2000 episodes,
episode cap 200 ticks, epsilon linear 1.0 → 0.05
over 1000 episodes then flat 0.05, rewards step
-0.01 / goal +1.0 / crush -1.0. Training RNG
conventions per §3 (agent Random(seed),
start-tick Random(seed + 1000), row
Random(seed + 5000)).

## 7. Evaluation protocol (Step 1; specified here, not run here)

Greedy only (epsilon 0), weights frozen. One
episode per start phase per (agent, world): 20
phases (period 20), 20 episodes per (agent,
world).

- FIXED worlds (10): R0..R9, one per row.
  Grouped in analysis as TRAINED (R0, R1, R3,
  R4, R5, R6, R7, R9) vs HELD-OUT (R2, R8,
  reported per row and pooled). The R5 world
  doubles as the continuity check: does the
  moving-door agent still cross the original
  familiar door?
- MIXED world (1): M3 per §4. Sanity check
  only; never substituted for fixed-world
  per-row numbers.

Encoder rule: FRESH encoder per (agent, world),
persisting across that world's 20 episodes
(FT1 Step 2 rule). For wrapped this means the
ScheduleModel starts blank at each world and
converges during the world's episodes; blank-
feature warm-up is therefore a per-world-start
effect. For all other arms the fresh instance is
stateless or per-episode-reset, so the rule is
uniform and auditable.

Record schema (md1_run.evaluate, mirroring
ft1_run.evaluate): kind, world, gap_row,
start_phase, ticks, crushes, reached, return
(accumulated env_step reward total), first_door,
door_entries, switches (always 0: one gap gives
at most one entry event), stall, stall_kind
(doors_run.stall_signature: no-move-run ≥ 10
ticks, two-cycle ≥ 10, dwell ≥ 15 in a 3×3 block
excluding staging cells and the goal),
revisits (segment revisits, doors_run.segments),
staging_order, arrival {tick, cell,
arrival_door, gap_open_at_arrival},
first_entry_index, post_entry_ticks,
post_entry_stall, post_entry_stall_kind,
blank_feature, conflict (always None: no rival
doors exist in MD1), conflict_outcome (always
None), staging_wait, door_status_at_arrival,
door_status_at_entry, schedules, steps (t, x, y,
action, door, crushed, plus f101/f102/f103 for
wrapped), arm, seed, period (20), open_len (10).
Step 1 records go to nn/results_md1.jsonl (never
pooled with earlier programmes); training
summaries go to nn/training_md1.jsonl with
row_counts, row_reached, door_entry_counts,
episodes_with_entry per trained row.

Blank-feature audit (lesson of P-Db2/FT1): any
blank or degenerate feature episodes are audited
(exclusion sensitivity computed), and reported
with an explicit pool-or-exclude ruling. No
silent pooling. Drift = persistent whole-episode
blank with schedule-blind behaviour (reported,
audited, EXCLUDED, never pooled); transient
spawn/world-start warm-up (blank only on first
steps, recovering, normal behaviour after) is
NOT drift — the audit, not the flag alone,
decides.

Restart/duplicate discipline (Step 1): resume-
safe runs, 0 duplicates, every restart disclosed
with its cause and handling. Eval jobs are
resume-safe at the job level (20 existing records
for (arm, seed, world) skip the job).

## 8. Interpretation rule (frozen at issuance — WO §8 verbatim)

Pooled trained = reached across the 8 trained
fixed worlds. Pooled held-out = reached across
the 2 held-out fixed worlds. The fixed-training
baseline on a novel door is ~3% (P-Db2 F3:
20/750).

- FOLLOWING: pooled held-out ≥ 50% of pooled
  trained AND pooled held-out ≥ 20% absolute.
  Both conditions required.
- BIGGER ADDRESS BOOK: pooled trained ≥ 50% but
  pooled held-out < 10% (near the
  fixed-training baseline band).
- NEITHER: all other complete outcomes
  (including the middle zone, pooled held-out
  10–20%). Neither supported nor weakened; the
  bar is not moved after results exist.
- INCOMPLETE: any eval world missing, or any
  held-out world missing. No verdict on
  incomplete data.
- UNINTERPRETABLE (training failure, not a
  verdict about following): pooled trained <
  20%. If the agent cannot cross rows it trained
  on, the design or the training is broken; no
  conclusion about following is drawn.

Precedence, applied in this order: INCOMPLETE →
UNINTERPRETABLE → FOLLOWING → BIGGER ADDRESS
BOOK → NEITHER (catch-all for every other
complete outcome). Route/staging analysis is
supporting, not gating: do agents that cross a
held-out row stage at (4, row) for the current
gap (the column route generalized)? Reported
descriptively either way.

## 9. Exact arm-run count and time estimate (gate count)

Counting rule (fixed, as in P-Da/P-Db2/FT1): one
arm-run = one agent-training OR one agent-world
evaluation job.

- Programme trainings: 30 (6 arms × 5 seeds).
- Sanity trainings (Step 0, §11): 6 (1 seed/arm).
- Fixed-world eval jobs: 300 (10 worlds × 30
  agents).
- Mixed-world eval jobs: 30 (M3 × 30 agents).
- TOTAL: 30 + 6 + 300 + 30 = **366 arm-runs**.
  366 > 150, so the compute trigger in ruling E
  FIRES. This 366 is the GATE COUNT (the work
  order's estimate, now counted exactly and
  confirmed: no other jobs exist in the design).

Time estimate (ESTIMATE, not a measurement —
the spike measures it later). Composition from
the P-Db2 spike paces (one timed training per
arm: wrapped 162.6 s, plain 109.2 s, memory
98.7 s, oracle_ind 96.5 s, oracle_new 98.0 s,
phase_only_new 104.7 s; sum 669.7 s):

- 30 programme trainings: 5×162.6 + 5×109.2 +
  5×98.7 + 5×96.5 + 5×98.0 + 5×104.7 = 3348.5 s
  ≈ 55.8 min.
- 6 sanity trainings (sunk in Step 0): 669.7 s
  ≈ 11.2 min.
- 330 eval jobs: at the P-Db2 eval pace (~0.03
  s/job) ≈ 10 s; at the FT1 spike sample pace
  (~0.07 s/job) ≈ 23 s. Bound: under 1 min.
- Steps 1–2 total ≈ 56 min (trainings + evals);
  whole programme including sunk Step 0 sanity
  ≈ 67 min + analysis ≈ **~1.2 hours**, inside
  the work order's rough ~1–1.5 hours.

Caveats, stated so the estimate is not
over-read: P-Db2 spike paces are n=1 per arm
(single-sample noise); oracle/phase paces were
measured at 30/14 and are transferred here to
20/10 (same network size and episode count, so
the transfer is reasonable but approximate).
A ±25% pace band puts Steps 1–2 trainings at
roughly 42–70 min on that basis. The spike
refines this; time is NOT the binding
constraint — the arm-run count (366 > 150) is.

SANITY-PACED PROJECTION (measured in Step 0,
§11 — MD1-specific, n=1 per arm, same caveat):
sanity wall times sum to 1680.8 s for 6
trainings (mean 280.1 s; range 271.6–286.4 s),
vs the 669.7 s P-Db2-pace sum. MD1 episodes run
longer (sanity mean ticks over reached 87–94,
vs P-Db2's ~41–44), so MD1 trainings cost more
train_steps per episode. At sanity paces, 30
programme trainings project to 5 × 1680.8 s =
8404 s ≈ 140 min ≈ 2.33 hours; Steps 1–2 total
≈ 2.35 hours; whole programme including sunk
Step 0 sanity ≈ 2.8 hours + analysis. That is
ABOVE the work order's rough ~1–1.5 hours —
the rough estimate is likely low, and this
note says so plainly rather than repeating it.
It is still BELOW the 4-hour compute threshold
in ruling E (even +25% pace noise, ≈ 2.9 hours
for Steps 1–2, stays under 4 hours), so no new
trigger question arises on current evidence.
The spike measures the true pace; neither
projection above is a measurement of Step 1.

## 10. Code (Step 0, additive only)

- nn/world_md1.py: row sets, gap_cell,
  make_fixed_world, row_multiset, m3_seed,
  m3_row_sequence, fixed_world_name. Mechanics
  inherited from MultiDoorWorld; no step/crush/
  blocking code.
- nn/train_md1.py: train_neural_md1 (mirror of
  train_neural_sched with per-episode gap row)
  + six per-arm trainer wrappers + TRAINERS.
- nn/md1_run.py: evaluate (greedy, 20 phases,
  fresh-encoder-per-job contract),
  evaluate_m3, fresh_encoder, run_eval_job,
  run_eval_job_m3 (job-level resume-safe).
- nn/analysis_md1.py: pooled counts, per-agent ×
  per-row table, staging/far-side/blank
  summaries, verdict() implementing §8 with
  precedence, main() report printer.
- nn/test_world_md1.py: scripted mechanics
  checks (§11).
- nn/sanity_md1.py: 6 sanity trainings + smoke
  eval (§11).

No existing world, trainer, or record file is
modified by any of these. No changes to
world.py, any doors/sched/luck/pda/pdb2/ft1
variant, or any frozen file.

## 11. Sanity criteria and results

Sanity criteria (fixed in advance):
- Mechanics checks (test_world_md1.py): ALL
  PASS, no failures tolerated. Case counts
  reported per check.
- Sanity trainings: 6 full trainings (1
  seed/arm, seed 0, 2000 episodes, exact Step 1
  protocol). Sane = training completes without
  crash/NaN AND row_counts exactly 250 per
  trained row for every arm AND reached counts
  in a sane band (well above 0 and below 2000;
  P-Db2 Step 0 sanity reached 1658–1797/2000
  for comparison — MD1 sanity need not match
  those numbers, since the world changed, but
  a 0/2000 or a degenerate count would FAIL
  sanity and be reported, not hidden).
- Smoke eval: wrapped/0 sanity agent on R5 and
  R8 (fresh encoder per world, 20 phases each).
  Outcomes are plumbing evidence only (the
  runner produces well-formed records); smoke
  reached counts are NOT Step 1 results and are
  never pooled.
- Sanity weights are DISCARDED (held in memory
  only, no checkpoint file written) and never
  pooled as Step 1 records.

Sanity results (measured 2026-10-22; full logs
nn/sanity_md1.log + nn/sanity_md1.jsonl):

- Mechanics checks (test_world_md1.py): ALL
  PASS, 518 cases total: row_multiset balanced
  250/row, no held-out, deterministic (7) +
  differs across seeds (1) + fixed-world
  gaps/walls/schedule rows 0..9 (10) + rejects
  row 10 (1) + door timeline 20/10 and observe
  follows gap (400) + blocking/entry/crush
  scripted rows 0/5/9 (3) + training row wiring
  10 episodes (10) + M3 rows trained-only,
  deterministic, seed formula (30) + M3
  sequences differ (30) + verdict precedence
  and bars incl. overlap/gap examples (6) +
  eval runner shape 20 records (20).
- Sanity trainings (seed 0, 2000 episodes,
  exact Step 1 protocol, weights discarded):
  wrapped/0 reached 619/2000, crushes 116,
  mean ticks over reached 87.2, wall 271.6 s;
  plain/0 439/2000, crushes 124, mean 90.4,
  wall 282.1 s; memory/0 444/2000, crushes 149,
  mean 94.0, wall 285.6 s; oracle_ind/0
  552/2000, crushes 108, mean 88.9, wall 278.4 s;
  oracle_new/0 602/2000, crushes 97, mean 89.9,
  wall 276.7 s; phase_only_new/0 528/2000,
  crushes 144, mean 90.2, wall 286.4 s. Row
  counts exactly 250 per trained row for every
  arm (balanced True, asserted by the script;
  any imbalance would have exited non-zero).
  Row-reached spread per arm covers all 8
  trained rows (lowest single-row counts:
  wrapped row 9: 27; plain row 9: 12; memory
  row 9: 14; oracle_ind row 9: 45; oracle_new
  row 9: 43; phase row 9: 24 — row 9 is lowest
  for every arm, descriptively, not a verdict).
- Criterion assessment (pre-stated §11
  criteria, not moved after seeing numbers):
  all 6 trainings completed without crash/NaN;
  row counts exactly 250; reached counts
  439–619, well above 0 and below 2000. PASS
  by the stated criterion. Reported honestly
  alongside: these reached counts are
  substantially BELOW P-Db2 Step 0 sanity
  (1658–1797/2000), and mean ticks (87–94)
  are roughly double P-Db2's (~41–44). That is
  consistent with a harder task (8 rows to
  generalize across, vs fixed gaps an agent
  could lock onto) and with the longer wall
  times (§9 sanity-paced projection); it is NOT
  a sanity failure under the pre-stated
  criterion, and the criterion is not adjusted
  to make the numbers look better or worse.
- Smoke eval (DISCARDED, plumbing evidence
  only — NOT Step 1 results): wrapped/0 sanity
  agent, fresh encoder per world, 20 phases
  each: R5 reached 0/20 with entered 20/20
  (enters every phase, reaches none — far-side
  failure in greedy eval, descriptively);
  R8 reached 0/20 with entered 0/20.
  Blank-feature episodes 1/20 per world,
  consistent with fresh-encoder world-start
  warm-up (the first episode's early steps,
  before the ScheduleModel converges). Smoke
  outcomes say the runner produces well-formed
  records and the encoder plumbing works; they
  say nothing about MD1's verdict, and no
  smoke number is pooled anywhere.

## 12. Step 1–3 outline (NOT authorized by this note)

After instructor verification AND the owner gate
ruling (spike first, then proceed on the current
substrate): Step 1 trains 30 programme agents
and evaluates 11 worlds each (330 eval jobs,
6600 episodes); Step 2 analyzes from the records
(per-agent × per-row reached, trained pooled vs
held-out pooled, route/staging analysis, far-side
behaviour, blank-feature audit, restart/duplicate
accounting); Step 3 writes nn/decisions_md1.md +
records + analysis output, committed and pushed
with remote HEAD verified. All under this note
exactly as written; no threshold, row set, eval
variant, or formula changes after issuance, and
in particular none after results exist.

## 13. Out of scope (restated — Step 0 does none of these)

Within-episode door movement (stage 2);
fixed-row-bit and per-row-bits observation
variants; fine-tuning of MD1 agents; any
substrate change (JIT/vectorization remains
under ruling E); any pooling of MD1 records with
earlier programmes; any programme training or
programme evaluation (Step 0 runs ONLY the 6
sanity trainings + mechanics + smoke eval);
any edit, commit, or push beyond this order's
scope without explicit owner confirmation.

## 13. Instructor verification (2026-10-22) — PASS

Independently checked before committing: mechanics checks re-run,
ALL PASS (518 cases, section sums recomputed); sanity records
re-derived exactly (6 records; reached 619/439/444/552/602/528;
row counts exactly 250 per trained row, no held-out rows; wall
times sum 1680.8 s); §8 rule carried verbatim with the issued
precedence and the verdict function implements it exactly
(overlap/gap examples resolve: trained 25% / held-out 20% →
FOLLOWING, trained 25% / held-out 5% → NEITHER); M3 runner
inspected (single-episode evaluator, trained rows only, fresh
encoder per job); gate count 366 recomputed exactly; both time
estimates recomputed (P-Db2-pace 55.8 min for 30 trainings;
sanity-paced 8404 s ≈ 2.33 h for Steps 1–2). No tracked file
modified; no programme training, evaluation, or spike ran.

Non-blocking note (does not affect this PASS): the issued work
order's §8 parenthetical reasoning — that FOLLOWING and BIGGER
ADDRESS BOOK "cannot both hold (both holding would require pooled
trained < 20%)" — is incorrect as reasoning. The two cannot both
hold on held-out fractions alone (FOLLOWING needs held-out ≥ 20%,
BIGGER needs held-out < 10%), regardless of trained. The bars,
precedence, and implementation are unambiguous and correct; only
that explanatory sentence is wrong. Correcting the sentence is an
owner decision, not an instructor edit.

## 14. Owner gate ruling (2026-10-22) — BENCHMARK SPIKE FIRST

At the Step 0 gate (366 arm-runs, trigger FIRES; sanity-paced
Steps 1–2 ≈ 2.33 h, under the 4-hour threshold), the owner chose
the benchmark spike before any programme training or evaluation,
per ruling E §5(1). The spike measures current-substrate MD1
costs only (timed trainings per arm at the Step 0 protocol, eval
sample pace, hotspots, projected Steps 1–2 time). **The spike
authorises NO programme training, NO programme evaluation, NO
substrate change, and NO JIT/vectorized implementation; spike
trainings are SPIKE-labelled measurement outputs, never Step 1
records, and their weights are discarded. A second owner ruling
is still required after the spike numbers before Steps 1+ start.**

## 15. Owner gate ruling (2026-10-23) — PROCEED on the current substrate

Gate chain complete: Step 0 verified (4041dec) → spike-first ruling
(§14, 1068cb3) → spike verified PASS (4414be2, Steps 1–2 ≈ 2.31 h,
under the 4-hour threshold) → this ruling. **Steps 1, 2, and 3 are
AUTHORIZED under this design note exactly as written, on the current
hand-written numpy substrate only.** NOT authorized: any substrate
change (ruling E §5(2) gate + separate owner ruling still required
for any future substrate), any threshold/row-set/eval-variant/formula
change, any pooling with earlier programmes, or any verdict claim
before instructor verification of the Steps 1–3 report. The ruling
is addressed to the main session (implementer) and takes effect
when the main session receives it; the Steps 1–3 report returns to
the instructor session for verification first. Ferry file:
`~/workspace/your_files/RULING-TBM-MD1-proceed.md`.

## 16. Execution note (2026-10-23) — instructor session proceeds NOW

After the §15 ruling was recorded for ferrying, the owner directed
in this session: "Then proceed NOW." Steps 1–3 are therefore
executed in THIS (instructor) session via a delegated agent, under
this design note exactly as written, with the report returning here
for instructor verification before any verdict is claimed. The
main-session ferrying path is superseded by this direction, so
that the programme is NOT also run in the main session — a second
run would duplicate programme records. Scope unchanged: current
hand-written numpy substrate only; no threshold/row-set/eval-variant/
formula change; no pooling with earlier programmes, sanity records,
or spike measurements.

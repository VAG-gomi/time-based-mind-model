# WO-TBM-PRB1 — Step 0 design note (DRAFT, for instructor verification)

**Status: DRAFT.** Step 0 only. This note fixes, in advance, every
protocol choice Step 1 will follow. **Step 1 probe evaluation is NOT
authorised by this note** — it starts only after instructor
verification of this note AND an owner ruling at the Step 0 gate.
All conclusions from this probe will be EXPLORATORY,
task-and-protocol bounded. No causal language is licensed by it.

**Governing documents.** Issued work order
`~/workspace/your_files/WO-TBM-PRB1.md` (ISSUED 2026-10-22, Step 0
only, including its Issuance amendments section — the four mandatory
Step 0 fixes are discharged in §§3–5 below). FT1 report
`nn/decisions_ft1.md` §7 (the memory/3 paradox, hypothesis INFERRED,
not a finding). Where this note and the work order disagree, the work
order governs and the disagreement must be reported — no such
disagreement was found in writing this note.

## 0. Question and hypothesis (INFERRED, not a finding)

FT1 left one paradox standing: memory/3 crossed (5,8) 161/200 times
during its fine-tuning (training at ε = 0.05) but scored 0/20, 0
entered, in greedy F3 evaluation (ε = 0), stuck at (4,5). The leading
hypothesis (FT1 report §7): a deterministic greedy trap — training
noise occasionally dislodges the agent from (4,5) and lets the learned
(5,8) route take over, while greedy eval has no dislodging mechanism.
This probe tests that hypothesis directly, with NO new training, by
re-evaluating already-saved fine-tuned snapshots under ε = 0.05
evaluation (weights frozen, no learning) and seeing whether the trap
opens. A positive result SUPPORTS the hypothesis; it never establishes
it, and never means "proven" (§5).

## 1. Agents and worlds (fixed — 4 combos, no others)

| # | Combo | Greedy baseline (FT1, cited, NOT re-run) | Fine-tune (FT1) |
|---|---|---|---|
| P (primary) | memory/3 in F3 | 0/20 reached, 0 entered | 161/200 reached, 167 (5,8) entries |
| C-a (positive control) | plain/0 in F3 | 20/20 reached, 20 entered | 188/200 reached, 189 (5,8) entries |
| C-b (second trap case) | oracle_new/101 in F3 | 0/30 reached, 0 entered | 179/200 reached, 181 (5,8) entries |
| C-c (second zero world) | memory/3 in F2 | 0/20 reached, 0 entered | (same agent as P; F2 is a different eval world) |

Baseline verification (re-derived from `nn/results_ft1.jsonl` and
`nn/training_ft1.jsonl`, commit 9d1d6a0, before writing this note):
memory/3 F3 0/20, 0 entered; plain/0 F3 20/20, mean ticks over reached
23.25; oracle_new/101 F3 0/30, 0 entered; memory/3 F2 0/20, 0 entered;
fine-tune summaries memory/3 161/200, plain/0 188/200, oracle_new/101
179/200. All four work-order baseline numbers check out exactly.

Why each control is included, and what it rules in or out:

- **C-a (plain/0 F3)** is the noise calibration. If ε = 0.05
  evaluation preserves plain/0's crossing, noise does not trivially
  destroy crossing, and a memory/3 unlock is meaningful. If plain/0
  collapses under ε = 0.05 (threshold in §5), noise scrambles
  everything and the primary result is UNINTERPRETABLE — that outcome
  is reported as such, not hidden, and no trap conclusion is drawn.
- **C-b (oracle_new/101 F3)** is the generality check. It is a second
  agent with the same trap shape (high fine-tune reaching, zero greedy
  crossing). If it unlocks under ε = 0.05 while memory/3 does not (or
  vice versa), the trap shape is agent-dependent, not universal. Its
  verdict is reported separately and does NOT determine the primary
  verdict.
- **C-c (memory/3 F2)** is the world-generality check for the same
  agent. F2 is rivals-only: there is no (5,5) to open, so waiting at
  (4,5) never pays. If noise unlocks F3 but not F2, the trap opening
  is world-specific (the (5,8) route exists to be dislodged onto);
  if it unlocks both, the dislodging is generic. Either way, C-c
  informs the mechanism reading; it does not determine the primary
  verdict.

**No wrapped agents.** All four combos use stateless or
per-episode-only encoders (memory previous-bit, plain door bit, oracle
period tables), so no encoder-state decisions beyond §3 are needed.

**Worlds (reused unchanged).** F3 = the P-Db2/FT1 F3 world: (5,8) alone
on the arm's training schedule (20/10 for memory/3 and plain/0, 30/14
for oracle_new/101). F2 = the P-Db2 F2 world: rivals-only staggered
schedules (20/10 for memory/3). Both are constructed by the existing
`world_pdb2.py` code, unchanged. No new world code is authorised.

## 2. Weights — FT1 fine-tuned snapshots, loaded frozen

Snapshots: `nn/checkpoints_ft1_finetuned/<arm>_<seed>.pkl`, committed
in FT1 Steps 1–3 (commit 9d1d6a0). The three files this probe needs
(`memory_3.pkl`, `plain_0.pkl`, `oracle_new_101.pkl`) exist in the
working tree. No retraining, no fine-tuning, no weight modification of
any kind. The probe is evaluation-only.

**Load check (Step 1, per combo, before any probe episode runs).**
For each snapshot, Step 1 verifies all of the following and logs the
result; a combo whose snapshot fails any check is NOT evaluated, and
its absence is reported (see §7):

1. The file unpickles without error.
2. The stored `arm` and `seed` match the combo (memory/3, plain/0,
   oracle_new/101 respectively).
3. The stored `n_inputs` matches the arm (memory 102, plain 101,
   oracle_new 104) and `hidden` is 32.
4. The file path and commit (9d1d6a0) are recorded as the snapshot
   identity in every probe record from that combo.

**If a snapshot fails to load, Step 1 reports it and does NOT
substitute any other weight** — no P-Db2 weight, no sanity weight, no
re-realised weight. If the primary combo's snapshot fails, the primary
question is reported as INCOMPLETE (not WEAKENED, not SUPPORTED).

## 3. Evaluation mode — ε = 0.05, weights frozen, no learning

At each step, with probability 0.05 the agent takes a uniform random
action; otherwise it takes the greedy action (argmax of the network
output; ties broken by lowest action index, matching FT1's greedy
evaluation). No replay updates, no target updates, no optimiser
steps, no weight changes of any kind. **This ε-greedy action choice is
the ONLY difference from FT1's greedy evaluation**; worlds, start
phases, episode cap (200 ticks), and record format are otherwise
identical.

**Implementation rule (to keep the RNG protocol in §4 exact).** Step
1 must NOT call the agent's `act(x, 0.05)` or any other routine whose
random draws come from a stream outside §4. The probe runner
implements ε-greedy explicitly: greedy action from the frozen network,
random action from the §4 rep stream. No other randomness (no numpy
RNG draws, no unseeded sources) may influence probe episodes.

**Encoder reset rule (explicit).** A fresh encoder object is created
per (agent, world, rep). Within a rep, the encoder is reset at every
episode start exactly as in FT1's greedy evaluation
(`encoder.reset(world)` after `world.reset(start_tick=start)`, phases
in increasing order — see §4). Concretely:

- **memory (memory/3):** the previous-bit history is seeded by
  `reset()` with the door bit at the reset tick (the
  `MemoryNEncoder.reset` semantics: history = [door bit at reset
  tick]); the history then shifts per step within the episode and
  never carries across an episode boundary.
- **plain (plain/0):** the door bit is the current observation each
  step — stateless across episodes by construction.
- **oracle_new (oracle_new/101):** period/remaining features are
  computed from the current tick and the fixed schedule constants —
  stateless across episodes by construction.

No wrapped encoder is involved, so no ScheduleModel state, no
blank-feature warm-up, and no drift audit is needed for this probe.
(If a wrapped agent were ever added, that would be a different probe
with its own design note.)

## 4. Stochastic-eval RNG protocol (exact, reproducible)

**Reps.** 5 reps per (agent, world) combo. Each rep covers all start
phases once, in increasing order (0–19 for 20/10 arms, 0–29 for 30/14
arms), one episode per phase. Total probe episodes: 20 + 20 + 30 + 20
phases per rep × 5 reps = 450 episodes.

**Seed formula (fixed, distinct per combo and rep, no wall-clock or
process-order dependence).** Combo indices: 0 = memory/3 F3,
1 = plain/0 F3, 2 = oracle_new/101 F3, 3 = memory/3 F2. Rep index
r ∈ {0, 1, 2, 3, 4}. The rep RNG seed is:

    seed = 900_000 + 1_000 × combo_index + r

Giving, explicitly:

| Combo | Rep seeds (r = 0..4) |
|---|---|
| memory/3 F3 | 900000, 900001, 900002, 900003, 900004 |
| plain/0 F3 | 901000, 901001, 901002, 901003, 901004 |
| oracle_new/101 F3 | 902000, 902001, 902002, 902003, 902004 |
| memory/3 F2 | 903000, 903001, 903002, 903003, 903004 |

**Stream object and lifecycle.** One stream per rep:
`rng = random.Random(seed)`, created once when the rep starts. The
stream runs CONTINUOUSLY across that rep's episodes (it is never
reset mid-rep, never re-seeded per episode, and never shared across
reps or combos). A fresh stream is created for each rep from its
formula seed.

**Consumption pattern (exact — the only draws from the stream).**
At each step of each episode in the rep, in order:

1. Exactly one draw: `u = rng.random()` (uniform in [0, 1)).
2. If `u < 0.05`, exactly one second draw: `k = rng.randrange(5)`,
   and the action is `ACTION_NAMES[k]` where
   `ACTION_NAMES = ["up", "down", "left", "right", "stay"]` (the
   `agent_nn.py` action list, the same list FT1's records use).
3. If `u >= 0.05`, the action is the greedy action and NO second draw
   is made.

No other draws consume from the stream: no shuffling, no sampling,
no episode-order randomisation, no tie-breaking draws. The greedy
action's tie-break (lowest index) is deterministic and draws nothing.
Any Step 1 code that adds a draw anywhere else violates this protocol
and must be reported as a deviation.

**Reporting.** Reps are reported individually (per-rep reached,
entered, mean ticks) AND as a distribution (per-rep reached counts
listed, with median and range). Results are NEVER pooled into a single
number without the spread. Greedy baselines are cited from FT1
records (`results_ft1.jsonl`, commit 9d1d6a0), NOT re-run.

## 5. Interpretation rule (fixed in advance)

**Precedence.** Check in this order: (1) snapshot completeness (§7) —
a missing primary combo means INCOMPLETE, not a verdict; (2) the
positive control (§5.4) — if UNINTERPRETABLE triggers, the primary
verdict is UNINTERPRETABLE regardless of memory/3's numbers;
(3) otherwise, the primary verdict from §5.1. The second-trap verdict
(§5.2) and the F2 verdict (§5.3) are reported separately and never
override the primary verdict.

### 5.1 Primary: memory/3 F3 (20 phases per rep)

- **SUPPORTED** (for the trap hypothesis): memory/3 F3 reaches
  **≥ 10/20 in ≥ 3 of 5 reps**. Reasoning: the greedy baseline is
  0/20, and stray random-action crossings — a few favourable random
  actions fighting a policy that pushes right at (4,5), chaining all
  the way to (4,8) and through (5,8) — would produce at most a few
  crossings per rep. Half the phases in a majority of reps is far
  above stray levels and far above the WEAKENING band. The
  majority-of-reps form (not a pooled total) means one lucky rep
  cannot carry the verdict. This threshold is a REPORTING CONVENTION,
  not a significance claim, and SUPPORT never means "proven".
- **WEAKENED**: memory/3 F3 reaches **≤ 2/20 in ≥ 4 of 5 reps**
  ("at or near 0": the 2/20 = 10% band admits stray crossings of 1–2
  per rep without calling them an unlock). That weakens the trap
  hypothesis; it does NOT prove the F3 loss is permanent (one ε
  level, one agent, one world tested).
- **NEITHER** (middle zone — a judgement call made explicit here so a
  middle result is not forced into WEAKENED): any outcome that meets
  neither the SUPPORTED nor the WEAKENED bar (e.g. 3–9/20 in most
  reps, or a mixed spread across reps). NEITHER means the probe does
  not discriminate: partial or inconsistent unlocking, below the
  pre-stated SUPPORT bar and above the WEAKENING band. It is reported
  with the per-rep numbers, not folded into WEAKENED.

### 5.2 Second trap case: oracle_new/101 F3 (30 phases per rep)

Same convention, scaled to 30 phases:

- **SUPPORTED-second**: **≥ 15/30 in ≥ 3 of 5 reps**.
- **WEAKENED-second**: **≤ 3/30 in ≥ 4 of 5 reps** (the same 10% band:
  3/30 = 10%).
- **NEITHER-second**: anything else.

Reported separately from the primary verdict. Agreement between the
two trap cases (both SUPPORTED, or both WEAKENED) suggests the trap
shape generalises across these two agents; disagreement means it is
agent-dependent. Neither agreement nor disagreement changes the
primary verdict.

### 5.3 Second zero world: memory/3 F2 (20 phases per rep)

- **UNLOCKED**: memory/3 F2 reaches **≥ 10/20 in ≥ 3 of 5 reps** —
  the same half/majority convention as §5.1, because reaching is the
  criterion here exactly as crossing is in F3. Reaching (not merely
  entering) is what counts: F2's question is whether noise unlocks a
  different zero world, and a world is unlocked only if the agent gets
  to the goal in it.
- **Partial movement (not an unlock)**: entries above 0 with reached
  below the UNLOCKED bar. Entered counts are reported beside the
  verdict, so a route-found-but-far-side-failed shape is visible and
  not mistaken for an unlock.
- **Still locked**: reached below the bar with no systematic entries
  (0 reached across reps = still locked, stated plainly).

### 5.4 Positive control and UNINTERPRETABLE: plain/0 F3

- **Collapse threshold (quantitative, fixed in advance):
  plain/0 F3 reaches ≤ 10/20 in ≥ 3 of 5 reps → UNINTERPRETABLE.**
  If this triggers, NO trap conclusion is drawn from this probe — not
  SUPPORTED, not WEAKENED, not NEITHER for the primary question. The
  outcome is reported as UNINTERPRETABLE with plain/0's per-rep
  numbers, and the work order is NOT worked around inside itself (no
  different ε, no re-run to rescue the primary result).
- **Reasoning (noise arithmetic, from the verified baseline).**
  plain/0's greedy F3 mean ticks over reached episodes is 23.25. Under
  ε = 0.05, a crossing-length episode expects about 0.05 × 23.25 ≈ 1.2
  random actions. A single random action delays the agent (a wrong
  step) or, rarely, crushes it (only if it is inside the door cell
  when the door closes) — and after a crush the episode continues
  from spawn. Against that, the episode cap is 200 ticks, about 8.6×
  the 23-tick route, so there is ample slack to recover from one or
  two mishaps. Ordinary noise should therefore cost at most a few
  episodes per rep (plausible per-rep reached roughly 17–20/20). A drop
  to half or worse (≤ 10/20) in a majority of reps would mean noise is
  not perturbing the crossing policy but destroying it — the regime in
  which a memory/3 unlock could not be read as trap-opening rather
  than noise-scrambling. The threshold is deliberately set at the same
  half/majority convention as SUPPORT, so the control and the primary
  are judged on one consistent bar.
- **Non-trigger zone.** plain/0 at 11–20/20 in ≥ 3 of 5 reps means the
  probe is interpretable. Moderate degradation inside that zone (e.g.
  12–16/20 in most reps) does NOT trigger UNINTERPRETABLE, but
  plain/0's per-rep counts must still be cited beside the primary
  verdict, so the primary result is read against the actual noise
  cost, not against an assumed clean control.

### 5.5 What no verdict means

SUPPORTED never means "proven" and never licenses a general law: one ε
level, two trap-case agents, one primary world, five reps each.
WEAKENED never means the F3 loss is permanent. NEITHER never means
the hypothesis is dead — it means this probe, at this ε, does not
discriminate. Any larger stochastic-eval study, dose-response
programme, or training redesign returns as a proposal with its own
scope, not as an extension of this probe.

## 6. Records (Step 1/2 only — Step 0 creates no probe records)

NEW files only: `nn/results_prb1.jsonl` (probe records) and
`nn/decisions_prb1.md` (report). Probe records are NEVER pooled with
FT1 records and NEVER presented as FT1 results.

**Per-episode record fields** (FT1 eval record fields, plus the probe
fields): all FT1 fields (`kind`, `world`, `start_phase`, `ticks`,
`crushes`, `reached`, `return`, `first_door`, `door_entries`,
`switches`, `stall`, `stall_kind`, `revisits`, `staging_order`,
`arrival`, `first_entry_index`, `post_entry_ticks`, `post_entry_stall`,
`post_entry_stall_kind`, `blank_feature`, `conflict`,
`conflict_outcome`, `staging_wait`, `door_status_at_arrival`,
`door_status_at_entry`, `schedules`, `steps`, `arm`, `seed`, `period`,
`open_len`), PLUS:

- `rep` (0–4), `rep_rng_seed` (the §4 formula seed for that rep),
  `epsilon` (0.05), `snapshot_identity` (arm, seed, snapshot file
  path, commit 9d1d6a0, and the FT1 fine-tune choices for that agent).
- Per-step records carry `epsilon_action` (true when that step's
  action came from the §4 random draw, false when greedy), alongside
  the FT1 step fields (`t`, `x`, `y`, `action`, `door`, `crushed`).

Records are flushed per (agent, world, rep) — a completed rep is never
re-run, and a relaunch resumes from uncompleted reps only.

## 7. Completeness check (Step 1, analogue of FT1's Step 0c)

Before analysis, Step 1 verifies per combo: snapshot load-checked (§2)
+ 5 reps present + every rep covering all start phases exactly once
(20 or 30 episodes per rep) + no duplicate (agent, world, rep,
start_phase) records. A combo failing this check is reported as
incomplete with its reason and excluded from its verdict (a missing
primary combo means INCOMPLETE for the primary question, §5
precedence). No incomplete combo is silently pooled, averaged over, or
replaced.

## 8. Compute estimate + JIT-trigger gate arithmetic (checkable)

Counting rule (fixed, as in P-Da/P-Db2/FT1): one arm-run = one
agent-training OR one agent-world evaluation job. One rep of one
(agent, world) combo counts as one evaluation job.

- Trainings: **0** (evaluation-only probe).
- Evaluation jobs: 4 combos × 5 reps = **20 arm-runs**.
- Total: **20 arm-runs. 20 < 150, so the arm-run trigger does NOT
  fire.**
- Time: greedy-eval pace was ~0.03 s/job in P-Db2 and ~0.07 s/job in
  the FT1 spike sample; stochastic evaluation is the same order (one
  extra RNG draw per step, no learning). 20 jobs → seconds to a
  minute, plus runner overhead. Total: minutes at most. No spike is
  needed on either trigger (compute or arm-runs).
- **If the owner expands the probe at issuance of Step 1** (more
  agents, worlds, or reps), the count is recomputed and the trigger
  question re-answered BEFORE running. An expanded probe that crosses
  150 arm-runs or 4 hours fires the trigger and returns to the owner
  for a spike/proceed ruling; it does not proceed on this note's
  no-trigger conclusion.

## 9. Out of scope (restated — Step 1 must not do any of these)

- No retraining, no fine-tuning, no weight changes of any kind
  (evaluation-only). No agents, worlds, or reps beyond the 4 combos ×
  5 reps fixed in §1/§4. No ε other than 0.05 (a different ε is a
  different probe). No wrapped agents. No pooling of probe records
  with FT1 records. No edits to `world.py`, the doors/sched/luck/pda/
  pdb2/ft1 variants, or any frozen file. No claims beyond the agents,
  worlds, reps, and ε evaluated. No JIT work (ruled E, trigger-gated;
  this probe does not fire the trigger, §8). No other tracks.

## 10. Step 0 gate (what happens next)

This note is DRAFT until instructor-verified. Step 1 starts only
after BOTH: (a) instructor verification of this note (thresholds,
formulas, and baselines checked against the work order and the FT1
records), and (b) an owner ruling at the Step 0 gate. If verification
finds an error in a threshold, formula, or baseline, the note is
corrected and re-verified BEFORE any probe episode runs — thresholds
and formulas are never adjusted after seeing probe results.

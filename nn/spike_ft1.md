# FT1 benchmark spike — report (MEASUREMENT ONLY, instructor-verified)

**Status: instructor-verified 2026-10-22. All paces, shares, and
projection arithmetic independently re-derived from the spike
records before committing.**
**Gate ruling:** owner chose "Benchmark spike first" at the FT1
Step 0 gate (design-note commit ffa5582; 182 arm-runs, trigger
FIRES). **This spike authorises NO Step 1 fine-tuning, NO
substrate change, NO JIT/vectorized work.** Spike outputs are
SPIKE-labelled check/measurement files only, never Step 1
records, never pooled with Step 1. Spike fine-tuned weights
were used in-process for the eval sample only and then
DISCARDED — nothing was saved as a checkpoint, and no spike
weight is reused anywhere. Step 1 starts only after instructor
verification of this spike AND a second owner ruling.

**Scope and method.** Mirror of the P-Db2 spike, adapted to
FT1: 6 timed fine-tunes (200 episodes on (5,8), one per arm)
from the Step 0 sanity checkpoint snapshots, timed with
`perf_counter` around the fine-tune call only, under the
design-note protocol exactly (flat eps 0.05, fresh replay
buffer, start-tick RNG seed+2000, wrapped encoder carried);
a 12-job greedy eval sample (6 agents × 2 worlds: F3, the
fine-tuning world, and F1, the multi-door training world —
one one-door world + one multi-door world, chosen to bound
eval pace across the world shapes Step 2 will use), fresh
encoder per (agent, world) per the design-note Step 2 rule,
timed per job; one cProfile of a wrapped/0 spike fine-tune.
No existing file was edited; no interruption occurred; no
load audit failed. Per-arm fine-tune pace is a single timed
run per arm (n=1), so the projection range below carries
that single-sample noise explicitly.

## §1. Load audit (determinism check) — 6/6 PASS

Each snapshot was loaded twice and
`train_ft1.assert_load_equal` passed (plus the wrapped
ScheduleModel period/open_len check) before any fine-tuning
from it. Snapshot files: `nn/checkpoints_ft1/wrapped_0.pkl`,
`plain_0.pkl`, `memory_0.pkl`, `oracle_ind_100.pkl`,
`oracle_new_100.pkl`, `phase_only_new_102.pkl`. No agent
was fine-tuned from an unaudited load.

Cross-check (consistency only, not a result): every spike
fine-tune reproduced its Step 0 sanity fine-tune exactly —
reached, crushes, and (5,8) entries identical for all 6
agents (e.g. wrapped/0 105/7/129, oracle_new/100 0/3/2).
Same snapshot + same protocol giving the same numbers is
the expected determinism behaviour; these are SPIKE
measurements, not Step 1 records and not acquisition
findings.

## §2. Fine-tune pace per arm (200 episodes, timed)

| arm (agent) | seconds | reached | crushes | (5,8) entries |
|---|---|---|---|---|
| wrapped (0) | 18.8 | 105 | 7 | 129 |
| plain (0) | 9.6 | 188 | 22 | 189 |
| memory (0) | 29.0 | 30 | 13 | 79 |
| oracle_ind (100) | 32.7 | 12 | 12 | 27 |
| oracle_new (100) | 32.7 | 0 | 3 | 2 |
| phase_only_new (102) | 20.5 | 103 | 12 | 113 |
| **sum / mean** | **143.3 / 23.9** | | | |

Fine-tune time tracks episode length, not just episode
count: the oracle arms (long episodes, low reached counts)
take ~32.7 s while plain/0 (short episodes) takes 9.6 s.
The design note's ~10–16 s per-agent guess therefore
understates the oracle/memory arms; the measured mean is
23.9 s. This correction is carried into §4 and does not
change the gate conclusion (time is not binding).

## §3. Eval-sample pace — negligible

12 jobs (6 agents × F3 + F1; 300 episodes total):
mean **0.068 s/job**, max **0.105 s/job**, sum 0.81 s.
Bounds for Step 2 (132 jobs / 3240 episodes): at the mean
≈ 8.9 s; at the max ≈ 13.9 s; conservative bound 0.11
s/job → ≈ 14.5 s. Eval is negligible in the projection,
as in P-Db2. Eval-sample outcomes are pace-sample data
only and are NOT reported as findings here.

## §4. Hotspots (cProfile, wrapped/0, 200 episodes) — shares only

Profile total 20.621 s (profiler accounting, NOT wall
time — the timed wrapped/0 run was 18.8 s wall; the
profile run is a separate measurement and its absolute
time is never used as pace). `train_step` cumulative
19.241 s = **93.3%** of profile time. Largest terms:
`numpy.ndarray.dot` tottime 7.103 s (34.4%);
`RandomState.choice` cumulative 4.565 s (22.1%);
forward 4.575 s, backward 4.011 s, optimizer step
2.450 s, stack 1.769 s. Dot + choice = 11.668 s
(**56.6%**). Environment/encoding terms are ~1–2%
each (env_step 0.307 s, world step 0.198 s, wrapped
encode 0.348 s); ScheduleModel observe does not appear
in the top 25 — NOT material, as in the P-Db2 spike.
**A tick-loop-only JIT would NOT address the train_step
bulk**, which is where 93% of the time sits.

## §5. Projection: Steps 1–2 time on the current substrate

Arithmetic in full (all figures seconds unless stated):

- Remaining checkpoint realisations (16: the 6 Step 0
  sanity checkpoints are already done and sunk, not
  counted here). A realisation IS a full P-Db2 training,
  so P-Db2 spike pace applies directly (cited, not
  re-measured): wrapped 162.6, plain 109.2, memory 98.7,
  oracle_ind 96.5, oracle_new 98.0, phase_only_new
  104.7. Remaining composition — wrapped 4 (seeds 1–4)
  × 162.6 = 650.4; plain 2 (2,3) × 109.2 = 218.4;
  memory 3 (2,3,4) × 98.7 = 296.1; oracle_ind 3
  (101,102,103) × 96.5 = 289.5; oracle_new 3 × 98.0
  = 294.0; phase 1 (104) × 104.7 = 104.7. **Total
  1853.1 s = 30.9 min.** Range: all-at-min (96.5)
  1544.0 s = 25.7 min; all-at-max (162.6) 2601.6 s
  = 43.4 min. All-22 realisation composition (for
  reference) = 2522.8 s = 42.0 min, matching the
  design note's ~42 min estimate exactly.
- Step 1 fine-tunes (22, at measured spike pace):
  composition by arm — wrapped 5 × 18.8 = 94.0;
  plain 3 × 9.6 = 28.8; memory 4 × 29.0 = 116.0;
  oracle_ind 4 × 32.7 = 130.8; oracle_new 4 × 32.7
  = 130.8; phase 2 × 20.5 = 41.0. **Total 541.4 s
  = 9.0 min.** Range: all-at-mean 525.5 s = 8.8 min;
  all-at-min 210.2 s = 3.5 min; all-at-max 719.6 s
  = 12.0 min.
- Step 2 eval (132 jobs): at the max-bound rate
  13.8 s (conservative 14.5 s at 0.11 s/job).

**Steps 1–2 total (composition): 1853.1 + 541.4 + 13.8
= 2408.4 s ≈ 40.1 min. Range: 29.4–55.6 min**
(low = 1544.0 + 210.2 + 8.9; high = 2601.6 + 719.6
+ 13.8). This sits UNDER the design note's ~48 min
estimate for the remaining work, and far under the
4-hour threshold (headroom ≈ 3.3 h on composition).
Whole-programme total including the 6 sunk Step 0
realisations ≈ 51.3 min — slightly above the design
note's ~48 min, entirely because fine-tunes measured
~9.0 min rather than the guessed ~5–6 min. **Time is
NOT the binding constraint; the arm-run count (182 >
150) is** — stated explicitly, as the gate requires.

## §6. Equivalence gap for a future §5(2) gate (stated, not executed)

A future substrate candidate in reference mode would
have to reproduce exactly, before any experiment runs
on it: agent RNG streams (exploration draws via
agent.rng, replay sampling via agent.nprng, continuing
from checkpoint end); the fine-tune start-tick stream
(seed+2000, distinct from checkpoint training's
seed+1000); per-episode fine-tune summaries (ticks,
reached, crushes, mean loss); checkpoint snapshot
bitwise equality (save/load round-trip, net/target/
Adam/RNG); greedy trajectories per (agent, world,
phase); per-door entry counts + open fractions;
P-Db2 headline endpoints (F0–F4 tables, conflict
counts, class counts); and the FT1 design-note fixed
choices (flat eps 0.05, fresh replay buffer, carried
Adam/target, carried wrapped ScheduleModel into
fine-tuning, fresh encoder per eval world).

**NO JIT/vectorized gain is measured by this spike.**
No candidate exists; any 20–30× figure remains NOT
ESTABLISHED. A speedup ratio vs any baseline is
incoherent as a target where the loop is shared;
absolute wall-clock per job is the honest target.
A tick-loop-only JIT would not address the bulk (§4:
train_step 93.3%). Option B-style vectorized lockstep
would change RNG streams and fail this gate by
construction; Option A-style JIT float behaviour
must be checked, not assumed.

## §7. Deviations — none

Nothing skipped, nothing not doable, no interruption,
no load-audit failure, no existing file edited. The
eval-sample world pair (F3 + F1) is the stated choice
above, not a deviation. Spike files are NEW only.

## Files (DRAFT, uncommitted; SPIKE-labelled)

- `nn/spike_ft1.md` — this report (DRAFT).
- `nn/spike_ft1_run.py` — spike runner (timed fine-tunes
  + eval sample + load audits; weights discarded).
- `nn/spike_ft1.log` — timed run log (all 6 agents,
  audits PASS, exit 0).
- `nn/spike_ft1.jsonl` — 6 SPIKE records (fine-tune
  seconds + eval-sample seconds; never Step 1 records).
- `nn/spike_ft1_profile.py` — cProfile runner.
- `nn/spike_ft1_profile.txt` — cProfile output
  (shares only; absolute profile time is not wall time).

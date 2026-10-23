# MD1 benchmark spike — MEASUREMENT ONLY (instructor-verified)

**Status: SPIKE, instructor-verified 2026-10-23. Paces, projection,
profile shares, and the determinism cross-check independently
re-derived from the spike records, log, profile, and sanity records
before committing. The spike agent itself went quiet after writing
complete files and did not deliver a completion report despite a
nudge; verification was therefore done directly from its files,
and the idle agent was closed. Ruling E §5(1); owner gate ruling
(design note §14, commit 1068cb3): BENCHMARK SPIKE FIRST. This report measures
current-substrate MD1 costs only. **It authorises NO programme
training, NO programme evaluation, NO substrate change, and NO
JIT/vectorized implementation. Spike trainings are SPIKE-labelled
measurement outputs, never Step 1 records, and their weights are
discarded. A second owner ruling is still required after the spike
numbers before Steps 1+ start.** No JIT/vectorized gain is measured
or claimed here (no candidate exists; any 20–30× figure remains NOT
ESTABLISHED).

## 0. Interruption — reported plainly

The first spike attempt was killed by a service restart while its
first training (wrapped/0) was still in progress. Nothing had been
flushed at that point — the runner writes one record per completed
training — so NO measurement was lost. The spike was relaunched
cleanly (the runner overwrites `spike_md1.jsonl` and `spike_md1.log`
on start, by design); everything below comes from the single
relaunched run. No partial spike is presented as complete.

## 1. Timed trainings (one per arm, seed 0, exact Step 0 protocol)

Timed with `time.perf_counter` (wall time). Moving-door world,
shuffled 250/row multiset, fixed 20/10, 2000 episodes, cap 200,
same hyperparameters as Step 0. Seed 0 per arm — the same seed as
the Step 0 sanity trainings, so the comparison in §2 is a
determinism cross-check, NOT a programme result.

| arm | wall time (s) | reached /2000 | crushes | mean ticks (reached) | rows balanced |
|---|---|---|---|---|---|
| wrapped/0 | 270.5 | 619 | 116 | 87.18 | 250/row ✓ |
| plain/0 | 282.1 | 439 | 124 | 90.41 | 250/row ✓ |
| memory/0 | 283.6 | 444 | 149 | 93.96 | 250/row ✓ |
| oracle_ind/0 | 275.1 | 552 | 108 | 88.89 | 250/row ✓ |
| oracle_new/0 | 268.7 | 602 | 97 | 89.88 | 250/row ✓ |
| phase_only_new/0 | 274.7 | 528 | 144 | 90.20 | 250/row ✓ |
| **sum (6 arms)** | **1654.7** | | | | |

MD1 trainings run ~2.5× the P-Db2 spike paces (P-Db2 sum 669.7 s):
MD1 episodes run longer (sanity mean ticks 87–94 vs P-Db2's ~41–44),
so each training costs more train_steps. That is a pace fact, not
a verdict about the programme.

## 2. Determinism cross-check (measurement-quality check, NOT a result)

Every spike training reproduced its Step 0 sanity training EXACTLY
(reached, crushes, mean ticks over reached, row counts, and
row-reached per row — all 6 arms, no exceptions):

| arm | spike wall (s) | sanity wall (s) | wall delta | exact match |
|---|---|---|---|---|
| wrapped/0 | 270.5 | 271.6 | −0.4% | YES |
| plain/0 | 282.1 | 282.1 | +0.0% | YES |
| memory/0 | 283.6 | 285.6 | −0.7% | YES |
| oracle_ind/0 | 275.1 | 278.4 | −1.2% | YES |
| oracle_new/0 | 268.7 | 276.7 | −2.9% | YES |
| phase_only_new/0 | 274.7 | 286.4 | −4.1% | YES |

Same seed + same protocol → same numbers, as expected. Observed
run-to-run pace variation is ≤ 4.1% (this bounds the range
construction in §5: the ±10% band below is generous against it,
and the design note's ±25% band is very generous).

## 3. Eval sample (PACE ONLY — outcomes are not findings)

Timed eval jobs at Step 0 protocol (greedy, weights frozen, fresh
encoder per job), on the spike wrapped/0 agent held in memory
(weights read as inputs, not records). Eval job outcome counts are
recorded in `spike_md1.jsonl` for plumbing audit but are PACE DATA
ONLY and are deliberately not quoted here as results — the sample
exists to time jobs, not to measure MD1 performance.

| job | wall time (s/job) |
|---|---|
| R5 (fixed world) | 0.080 |
| R8 (fixed world, held-out row) | 0.084 |
| M3 (mixed world) | 0.086 |
| **mean** | **0.083** |
| **max** | **0.086** |

330 programme eval jobs bound: **27.5 s at the mean pace,
28.4 s at the max pace**. Evaluation is negligible next to
training, as in the P-Db2 precedent.

## 4. Profile (wrapped/0 training — PROFILER ACCOUNTING, NOT wall time)

cProfile run of one wrapped/0 training at the exact Step 0
protocol, separate from the timed runs. Total profiler time
302.648 s (the timed wall run was 270.5 s; the difference is
profiler overhead, and neither number is the other). 61,134,378
function calls.

| component | time (s) | share of profiler total |
|---|---|---|
| `train_step` (cumulative) | 286.044 | 94.5% |
| `dot` (tottime) | 99.654 | 32.9% |
| `choice` / RandomState sampling (tottime) | 75.457 | 24.9% |
| `dot` + `choice` (tottime) | 175.111 | 57.9% |
| `sample` (cumulative, incl. choice) | 81.982 | 27.1% |
| `forward` (cumulative) | 62.797 | 20.7% |
| `backward` (cumulative) | 57.050 | 18.8% |
| optimizer `step` (cumulative) | 34.611 | 11.4% |
| `env_step` (cumulative) | 3.989 | 1.3% |
| wrapped `encode` (cumulative) | 3.978 | 1.3% |
| world `step` (cumulative) | 2.469 | 0.8% |
| wrapper `observe` (cumulative) | 1.446 | 0.5% |

ScheduleModel `observe` does not appear in the top 30 — NOT
material. Environment and encoding sit at ~1% each. **A
tick-loop-only JIT would NOT address the train_step bulk
(94.5%)**, the same shape as the P-Db2 and FT1 spikes. No
candidate exists, so no JIT gain is measured or claimed.

## 5. Projection (composition from MEASURED spike paces)

Steps 1–2 by composition (no other jobs exist in the design):

- 30 programme trainings: 5 × 1654.7 s = **8273.5 s ≈ 2.30 h**.
- 330 eval jobs: **27.5 s at the mean pace** (28.4 s at max).
- **Steps 1–2 total: ≈ 8301 s ≈ 2.31 h.**
- Whole programme including sunk Step 0 sanity (1680.8 s):
  ≈ 9982 s ≈ **2.77 h + analysis**.

Range, and how it was built: single-sample paces carry noise.
Observed run-to-run variation (spike vs sanity wall times) is
≤ 4.1%, so a ±10% band is already generous, and the design
note's ±25% band is very generous. Both stay under threshold:

| band | Steps 1–2 range |
|---|---|
| ±10% | 2.08 – 2.54 h |
| ±25% (design-note band) | 1.73 – 2.88 h |

Comparison against the design note's two estimates: the
P-Db2-pace estimate (~56 min, 0.93 h) is refuted as a pace
predictor for MD1 — MD1 trainings cost ~2.5× P-Db2 paces,
because MD1 episodes run longer. The sanity-paced estimate
(~2.33 h) is CONFIRMED as a pace predictor: the measured
composition (2.31 h) lands on it almost exactly. The work
order's rough ~1–1.5 h is likewise low and is superseded by
this measurement.

Threshold comparison: Steps 1–2 at 2.31 h sit **1.69 h under
the 4-hour threshold**; even the top of the ±25% band (2.88 h)
stays under it. **Time is NOT the binding constraint; the
arm-run count (366 > 150) is**, as the design note states.

## 6. Equivalence gap (for a future ruling E §5(2) gate)

A candidate substrate would have to reproduce EXACTLY, in
reference mode, before any experiment runs on it:

- The 6 MD1 sanity training summaries (`sanity_md1.jsonl`):
  reached, crushes, mean ticks over reached, row counts (250
  per trained row), and row-reached per row, for all 6 arms
  at seed 0 — the determinism cross-check in §2 shows these
  are exactly reproducible on the current substrate.
- All 518 mechanics checks (`test_world_md1.py`): ALL PASS,
  including the row-multiset balance, M3 seed formula and
  trained-only rows, and the §8 verdict precedence cases.
- The row-multiset determinism (same seed → same shuffled
  multiset) and the M3 row sequences (same (arm, seed) →
  same 20 rows).
- A frozen programme record set (Step 1 trainings + evals)
  — **no such record set exists yet**, because Step 1 has
  not run. The current gap is therefore sanity + mechanics
  only; it widens to include a frozen programme record set
  the moment Step 1 produces one, and that set must be
  reproduced exactly before any re-baselining.

Until a candidate passes that gate, old- and new-substrate
records are never pooled, and no speedup ratio is claimed.

## 7. Files (all SPIKE-labelled, DRAFT, uncommitted)

- `nn/spike_md1.md` (this report)
- `nn/spike_md1_run.py` (timed trainings + eval sample)
- `nn/spike_md1.log` (run log)
- `nn/spike_md1.jsonl` (9 spike measurements: 6 trainings +
  3 eval-job summaries)
- `nn/spike_md1_profile.py` (cProfile run)
- `nn/spike_md1_profile.txt` (profiler output)

No tracked file was modified. `nn/training_md1.jsonl` and
`nn/results_md1.jsonl` were NOT created. No checkpoint file
was written; all spike weights were discarded on exit.

## 8. Instructor verification (2026-10-23) — PASS

Recomputed exactly from the spike files: 6 timed trainings sum
1654.7 s; determinism cross-check exact on all 6 arms (reached,
crushes, mean_ticks_reached, row_counts, row_reached — all equal
to the Step 0 sanity records); eval sample mean 0.083 s/job, max
0.086 s/job, 330-job bounds 27.5 s / 28.4 s; profile shares
recomputed from `spike_md1_profile.txt` (train_step 94.5%,
dot 32.9%, choice 24.9%, wrapper observe 1.446 s, not in the top
30); projection 5 × 1654.7 s = 8273.5 s ≈ 2.30 h for 30 trainings,
Steps 1–2 ≈ 8301 s ≈ 2.31 h, whole programme ≈ 9982 s ≈ 2.77 h;
±10% band 2.08–2.54 h and ±25% band 1.73–2.88 h recomputed; 4-hour
headroom 1.69 h. First-attempt restart disclosed in §0 is
consistent with the single relaunched run in the log (9 spike
records, no duplicates, no programme files created).

# Stage 1 results (v3.1 pilot + 20/10 regression)

Owner approval: 2026-10-17 (Stage 1 only; Stage 2 NOT approved).
Frozen code commit: 888355c. Prereg approval commit: 7c4f8e8.
Runs: 2026-10-17. 45 records total: 25 reused (v2 pilot, c70cabb) + 20 new.

## Reuse gate

`wrapped_ind` seed 100 re-ran at the frozen code and reproduced the
v2 pilot record **bit-for-bit** (full record identical: hyperparams,
greedy 0.7526666666666666 / 0 crushes, eps005, train). Code paths of
the five v2 arms verified unchanged between 023317f and 888355c
(additive-only diff). The 25 v2 records (seeds 100-104,
wrapped_ind/memory10/memory20/memory30/oracle_new, 30/14) are reused.

New records: `nn/results_stage1.jsonl` (10: phase_only_new,
oracle_ind x seeds 100-104, 30/14, --fast-threshold 0.744),
`nn/results_regression.jsonl` (10: wrapped_ind x seeds 100-109,
20/10, --fast-threshold 0.750). New records carry `fast_threshold`
in hyperparams; v2 records do not; training code identical.

## Gate verdicts (pre-registered 2026-10-17, before the runs)

1. **Success rule (>=3/5 seeds with greedy success >=0.9, every
   pilot arm): PASS.** All 7 arms: 5/5.
2. **Regression (>=4/10 clean-and-fast at fast=0.750 AND >=8/10
   zero-crush, wrapped_ind 20/10 seeds 100-109): PASS.**
   Clean-and-fast 6/10 (seeds 100,102,103,104,105,107);
   zero-crush 10/10.
3. **oracle_ind as upper bound (median greedy >= wrapped_ind
   median - 0.02 AND >=4/5 zero-crush): PASS.**
   wrapped_ind median 0.753; oracle_ind median 0.737 >= 0.733;
   oracle_ind zero-crush 5/5.

All three gates pass. Proceeding to Stage 2 requires a separate
owner approval.

**Correction (2026-10-17, owner review):** the clean-and-fast
column above was recomputed at the single 0.744 bar. The first
version of this table had evaluated the five reused v2 arms at
their original 0.75 bar; at 0.744, `oracle_new` is 1/5 (seed 103:
0.749, no crushes), `oracle_ind` 2/5 (was 1/5 at 0.75). Gate
verdicts are unaffected (they never used clean-and-fast on the
reused arms).

## Descriptives (30/14 pilot, fast=0.744)

| Arm          | Clean-and-fast | Median return | Zero-crush | Success>=0.9 |
|--------------|----------------|---------------|------------|--------------|
| wrapped_ind  | 3/5            | 0.753         | 5/5        | 5/5          |
| memory10     | 0/5            | 0.724         | 5/5        | 5/5          |
| memory20     | 0/5            | 0.726         | 5/5        | 5/5          |
| memory30     | 0/5            | 0.740         | 5/5        | 5/5          |
| oracle_new   | 1/5            | 0.655         | 4/5        | 5/5          |
| phase_only_new | 2/5          | 0.740         | 4/5        | 5/5          |
| oracle_ind   | 2/5            | 0.737         | 5/5        | 5/5          |

Notes:
- phase_only_new seed 102 had one crush (0.033/ep); all other pilot
  seeds except oracle_new's known misses were crush-free.
- oracle_ind seed 101 returned 0.556 with 0 crushes (safe but very
  slow), the same "excessive caution" failure shape as the v2
  wrapped_ind misses; it dragged the oracle_ind median to 0.737,
  only 0.004 above the gate bar. The arm passes the gate, but the
  margin is thin and seed 101 deserves diagnosis attention before
  Stage 2.
- No arm tripped the success-based failure rule; no halt was
  triggered. No missing records.

## Regression (20/10, fast=0.750)

wrapped_ind seeds 100-109: clean-and-fast 6/10, zero-crush 10/10,
median return 0.764. Against the verified 40-seed wrapped
reference (26/40 cf, 38/40 zc): 6/10 >= 4/10 and 10/10 >= 8/10.
Regression check passes; the schedule-independent encoding did not
break 20/10 behavior. Misses: seeds 101 (0.745), 106 (0.748),
108 (0.692), 109 (0.674) — all crush-free, too slow.

# Decisions: Trust Replay Layer 1.5 rematch (WO-TBM-L15-run)

**Design:** frozen in `nn/prereg_trust15.md` at `7eaca1e` (on
`origin/main` before any run). Rematch run 2026-10-20; records
in `nn/results_trust15.jsonl` (6 records: uniform + trust15 ×
seeds 200–202, 1000 episodes, hidden=32, lr=1e-3, 20/10);
runner `nn/run_trust15.py`. Bare wall-clock, arms interleaved
per seed, single uninterrupted session (no restarts, no
material change in machine conditions mid-run). Endpoints and
the success criterion are applied verbatim from the prereg;
no post-hoc endpoints, no reinterpretation.

## Per-seed endpoints

| seed | arm | wall-clock | final-100 success | first ep rolling-100 ≥ 0.8 |
|---|---|---|---|---|
| 200 | uniform | 82.8s | 0.99 | 431 |
| 200 | trust15 | 103.5s | 1.00 | 522 |
| 201 | uniform | 81.9s | 1.00 | 457 |
| 201 | trust15 | 92.6s | 1.00 | 392 |
| 202 | uniform | 73.3s | 1.00 | 341 |
| 202 | trust15 | 104.7s | 1.00 | 529 |

Means: wall-clock uniform 79.35s, trust15 100.24s (ratio
1.263). Final-100 success: uniform 0.997, trust15 1.000.

## The frozen criterion, verbatim

- **(a) Speed parity — FAIL.** Mean wall-clock/seed of
  trust-1.5 (100.24s) is NOT ≤ uniform's (79.35s). Trust-1.5
  is ~3.2x faster than the naive Layer-1 implementation on the
  same seeds (re-run means: trust 315.8s → 100.2s) but remains
  ~26% slower than uniform in bare wall-clock.
- **(b) Success — PASS.** Mean final-100 success 1.000 ≥
  0.997 − 0.05; per-seed differences +0.01 / 0.00 / 0.00 — no
  seed worse by more than 0.10. (Contrast with Layer 1's
  naive trust: 0.95 / 0.33 / 0.43.)
- **(c) Episodes-to-0.8 — FAIL.** Trust-1.5 no worse than
  uniform on 1 of 3 seeds (seed 201: 392 ≤ 457; seeds 200 and
  202 worse: 522 > 431, 529 > 341). The criterion requires
  ≥ 2 of 3.

## Verdict

**(a) and (c) fail → per the frozen prereg, the branch CLOSES
against a correct implementation.** The Layer-1 negative
stands as a property of the design on this task and protocol,
not of its implementation. No further trust-replay variants
are proposed on this evidence.

## Bounded interpretation

What this licenses: on the 20/10 protocol, seeds 200–202,
1000 episodes — a correct, fast implementation of
trust-weighted replay (segment tree + IS correction) (i)
repairs the learning damage seen with the naive
implementation (final-100 success 1.00 on all seeds, criterion
(b) passed), and (ii) cuts the naive implementation's
wall-clock by ~3.2x, yet (iii) still does not reach
wall-clock parity with uniform replay and does not reach the
rolling-0.8 criterion sooner on 2 of 3 seeds. Under the
preregistered criterion, that is a closing result.

What it does not license: any claim about trust-weighted or
prioritized replay in general, other tasks, other budgets, or
other implementations (vectorized/JIT variants were out of
scope by owner ruling and remain untested); any causal claim
about why parity was missed — the Step-0 Amdahl projection,
computed from cProfiled runs, said the non-buffer floor
clears parity, and the bare measurement disagrees; that
discrepancy is recorded here as an observation about the
projection method, not as a reinterpretation of the verdict.

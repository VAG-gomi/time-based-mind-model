# Decisions: variance-null test (WO-TBM-PC5-run)

**Design:** frozen in `nn/prereg_trust15_var.md` at `d060f50`
(on `origin/main` before any run). Executed 2026-10-20:
uniform vs trust15, code as frozen at `7eaca1e`, seeds
210–229 paired, 1000 episodes, 20/10. Records:
`nn/results_trust15_var.jsonl` (40/40, no duplicates, no
partials; two runtime restarts during collection, only
in-flight records lost). Analysis:
`nn/analyze_trust15_var.py`, run once on the complete data.
The decision rule below is the prereg's, applied verbatim.
The trust branch remains CLOSED; this test's outcome does
not change that status either way (per the prereg).

## Per-pair endpoints (first episode reaching rolling-100 ≥ 0.8)

| seed | uniform | trust15 | D (trust15 − uniform) |
|---|---|---|---|
| 210 | 392 | 510 | +118 |
| 211 | 476 | 519 | +43 |
| 212 | 508 | 564 | +56 |
| 213 | 392 | 648 | +256 |
| 214 | 406 | 658 | +252 |
| 215 | 440 | 582 | +142 |
| 216 | 404 | 520 | +116 |
| 217 | 487 | 644 | +157 |
| 218 | 464 | 399 | −65 |
| 219 | 423 | 493 | +70 |
| 220 | 420 | 566 | +146 |
| 221 | 419 | 408 | −11 |
| 222 | 431 | 459 | +28 |
| 223 | 355 | 465 | +110 |
| 224 | 545 | 494 | −51 |
| 225 | 446 | **never (censored → 1000)** | +554 (lower bound) |
| 226 | 379 | 590 | +211 |
| 227 | 346 | 512 | +166 |
| 228 | 441 | 348 | −93 |
| 229 | 421 | 475 | +54 |

Censoring: uniform 0, trust15 1, one-sided-censored pairs 1
(< 5 — the pre-stated censoring-limited label does NOT
attach).

## Frozen-rule evaluation

- Sign split: **+16 / −4 / 0 ties**.
- Median paired difference: **+113 episodes**.
- Wilcoxon signed-rank: statistic 21.0, **p = 0.0009**
  (two-sided, α = 0.05) → rejects, and the median is
  positive.

## Conclusion (frozen mapping, exactly two outcomes)

**NULL REFUTED.** A systematic episodes-to-0.8 disadvantage
of trust-1.5 exists on this task and protocol; the n=3
rematch pattern (+91 / −65 / +188) was not mere seed scatter.
Per the prereg, **P-C2 and P-C4 gain priority** as the
diagnostic proposals for the mechanism.

## Secondaries (descriptive only)

- Variance ratio of first80 (trust15 / uniform): **7.61**
  (substituted values; the censored 1000 inflates trust15's
  variance — read with the censoring count above).
- Final-100 success: uniform 0.998, trust15 0.955.
- Median first80: uniform 422.0, trust15 515.5.

## Bounded interpretation

What this licenses: on the 20/10 protocol, seeds 210–229,
trust-1.5 reaches the rolling-0.8 criterion systematically
later than uniform (median +113 episodes; 16 of 20 pairs
positive). The variance null — the diagnosis's
best-supported reading of the n=3 pattern — is refuted by
this preregistered test at its stated sensitivity.

What it does not license: any identification of the
mechanism (the prereg bounds conclusion (ii) to the
existence of a systematic cost). For the record, the
existing diagnostic evidence is unchanged by this test:
P-C2 implicated the β anneal diagnostically (same-seed,
hypothesis-generating); P-C4's manipulation check failed,
leaving insertion priority indeterminate. Any design change
remains a proposal for owner ruling. No claim extends
beyond this task, protocol, implementation pair, and seed
range; the trust branch's closed status is unaffected.

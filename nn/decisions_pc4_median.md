# PC4 diagnostic: median-insertion ablation (candidate 4) — DIAGNOSTIC, non-confirmatory

**Status:** hypothesis-generating only (same seeds as the
rematch). The trust branch remains CLOSED. Variant committed
before any run at `aeabaf1` (`nn/trust_replay_median.py`,
`nn/layer15_median_train.py`, `nn/run_pc4_median.py`):
insertion leaf = median of leaf values (trust^α) over filled
slots — equivalently median raw trust, the transform being
monotone; empty buffer inserts at 1.0. Everything else is the
frozen design, anneal intact. Records:
`nn/results_pc4_median.jsonl`.

## Manipulation check — FAILED (the ablation tested nothing about the mechanism)

Mean sampled age (env steps), batch-weighted:

| | seed 200 | seed 201 | seed 202 | mean |
|---|---|---|---|---|
| uniform | 4872 | 4870 | 4871 | 4871 |
| trust15 (frozen, max insertion) | 4540 | 4565 | 4552 | 4552 |
| trust15 (median insertion) | 4645 | 4646 | 4637 | 4643 |

The frozen design's age gap to uniform is 319 steps.
Median insertion narrowed it to 228 steps — **~29% closed,
not closed** (median-insertion samples are still ~4.7%
younger than uniform's). Per the work order's rule, a failed
manipulation check means the ablation did not isolate the
variable it aimed at, so **no confirm/refute verdict on
candidate 4 is taken from these runs; candidate 4 remains
INDETERMINATE.**

Why the treatment did not move the mediator (INFERRED):
insertion priority governs only an item's priority *before
its first sampling*; once sampled, its trust is set by its
TD error, and the sample → update → resample loop keeps the
sampled-age distribution young regardless of the insertion
rule. The insertion rule is evidently not the lever that
sets sampled age in this design.

## Endpoint table (recorded; NO mechanism reading taken)

First episode reaching rolling-100 ≥ 0.8:

| seed | uniform | trust15 (frozen) | trust15 (median insertion) |
|---|---|---|---|
| 200 | 431 | 522 (+91) | 473 (+42) |
| 201 | 457 | 392 (−65) | 515 (+58) |
| 202 | 341 | 529 (+188) | 468 (+127) |

Final-100 success under median insertion: 1.00 on all seeds.
These numbers are recorded for completeness; because the
manipulation check failed, they discriminate nothing about
candidate 4 and no reading is drawn from them.

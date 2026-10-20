# PC2 diagnostic: β ≡ 1.0 ablation (candidate 2) — DIAGNOSTIC, non-confirmatory

**Status:** hypothesis-generating only (same seeds as the
rematch). The trust branch remains CLOSED; nothing here is a
fix. Variant committed before any run at `b3e3e8d`
(`nn/layer15_beta1_train.py`, `nn/run_pc2_beta1.py`): the
frozen `Trust15NeuralQ` reused by import, the ONLY difference
from the frozen trainer being `beta = 1.0` every episode (no
0.4 → 1.0 anneal). Records: `nn/results_pc2_beta1.jsonl`.

## Manipulation check — PASS

Early-phase (episodes 0–340) mean IS effective sample size:
**16.1 / 16.1 / 15.9** (seeds 200/201/202) — at the late-run
level (≈ 17) from the start, against ≈ 26.0 early under the
anneal (diagnosis traces). The manipulation did what it was
supposed to do: full-strength correction from episode 0.

## Endpoint: first episode reaching rolling-100 ≥ 0.8

| seed | uniform | trust15 (annealed) | trust15 (β ≡ 1.0) |
|---|---|---|---|
| 200 | 431 | 522 (+91) | **447 (+16)** |
| 201 | 457 | 392 (−65) | **448 (−9)** |
| 202 | 341 | 529 (+188) | **376 (+35)** |

Final-100 success under β ≡ 1.0: 1.00 on all three seeds.

## Verdict (per the diagnosis proposal, verbatim)

**CONFIRM — the β anneal is implicated.** The lag on seeds
200 and 202 shrank materially under full early correction
(+91 → +16; +188 → +35, reductions of ~82% and ~81%).

Diagnostic caveats, stated in the record: (1) same seeds as
the rematch — this is hypothesis-generating, not
confirmatory; any confirmatory claim needs fresh seeds under
a prereg freeze. (2) The pattern is symmetric compression,
not one-directional repair: seed 201's annealed *lead* also
shrank (−65 → −9). The strongest reading this diagnostic
supports is that the weak early correction amplified
seed-level divergence in both directions — which implicates
the anneal in the criterion-(c) scatter without establishing
it as a one-directional delay mechanism. (3) If β is ever to
be changed in the design, that returns as a proposal for
owner ruling, not a patch.

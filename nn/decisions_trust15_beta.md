# Confirmatory β-anneal test — result (WO-TBM-PC6-run)

Prereg: `nn/prereg_trust15_beta.md`, frozen at `c2320cb`
before any run under it. Arms: uniform (frozen path);
trust15-annealed (code as frozen at `7eaca1e`); trust15-β1
(variant exactly as committed at `b3e3e8d`). Seeds 500–519,
three arms per seed, 1000 episodes, hidden=32, lr=1e-3,
20/10. Records: `nn/results_trust15_beta.jsonl` (60/60;
collection spanned one runtime restart; skip-existing
preserved every completed seed). The frozen analysis
(`analyze_trust15_beta.py`, validated on synthetic cases
before the run) was run exactly once, on the complete file.

## Per-seed endpoints (first80; "never" = censored)

| seed | uniform | annealed | β1 | D1 = ann−β1 | D2 = β1−uni |
|---|---|---|---|---|---|
| 500 | 469 | 627 | 393 | +234 | −76 |
| 501 | 449 | 575 | 434 | +141 | −15 |
| 502 | 475 | 458 | 411 | +47 | −64 |
| 503 | 452 | 541 | 418 | +123 | −34 |
| 504 | 471 | 645 | 469 | +176 | −2 |
| 505 | 470 | 646 | 409 | +237 | −61 |
| 506 | 571 | 487 | 407 | +80 | −164 |
| 507 | 398 | 416 | 441 | −25 | +43 |
| 508 | 434 | 478 | 419 | +59 | −15 |
| 509 | 522 | 510 | 444 | +66 | −78 |
| 510 | 424 | 547 | 363 | +184 | −61 |
| 511 | 430 | 398 | 417 | −19 | −13 |
| 512 | 467 | 392 | 313 | +79 | −154 |
| 513 | 385 | 390 | 318 | +72 | −67 |
| 514 | 482 | 699 | 560 | +139 | +78 |
| 515 | 350 | 411 | 352 | +59 | +2 |
| 516 | 365 | 495 | 477 | +18 | +112 |
| 517 | 478 | 605 | 518 | +87 | +40 |
| 518 | 474 | 517 | 376 | +141 | −98 |
| 519 | 344 | 469 | 361 | +108 | +17 |

Per-arm median first80: uniform 459.5, annealed 502.5,
β1 414.0. **No run was censored** — every arm on every seed
reached rolling-100 ≥ 0.8; one-sided censored pairs are 0
on both contrasts, so neither contrast carries the
censoring-limited label.

## Frozen-rule evaluations

**Primary contrast D1 (annealed − β1):** sign split
**+18 / −2 / 0**; median **+83.5 episodes**; Wilcoxon
signed-rank statistic 5.0, **p = 1.9e-05** (two-sided).
The test rejects at α = 0.05 and the median is positive.

→ **ANNEAL IMPLICATED (confirmatory).** Per the frozen
rule this is one of exactly two conclusions, and it is the
one the data select: the β anneal (0.4 → 1.0) causes a
systematic episodes-to-0.8 cost on this design — the
annealed regime crosses the criterion a median of 83.5
episodes later than the identical trainer with β fixed
at 1.0, on identical seeds.

**Secondary contrast D2 (β1 − uniform):** sign split
+6 / −14 / 0; median **−24.5 episodes**; Wilcoxon statistic
58.0, **p = 0.0826** (two-sided) — no rejection. Per the
frozen reading, limited to detection: **a residual
systematic disadvantage for β1 vs uniform is NOT
DETECTED.** This is not equivalence and is not stated as
such: at this design's sensitivity (planning SD 142.8;
power 0.30 at +50 episodes) a residual disadvantage of
≲ +50 episodes would likely go undetected. The negative
median is reported as part of the table; the frozen
reading licenses no claim that β1 is faster than uniform.

## Bounded interpretation

What this establishes, on this task (20/10 protocol), this
implementation trio, seeds 500–519: the systematic
disadvantage PC5 established confirmatorily is
attributable to the β anneal. The mechanism chain now
reads: PC5 showed the cost is systematic (median +113
episodes, annealed vs uniform, seeds 210–229); this test
shows the anneal contrast alone reproduces a cost of the
same order within-seed (+83.5 median), and no residual
disadvantage for the β1 regime was detected against
uniform at this design's sensitivity. That is the
strongest statement the design licenses; the medians come
from different seed sets and do not decompose exactly, and
the D2 non-detection is bounded as stated above.

What this does not touch: the rematch's criterion-(a)
speed failure is machinery cost (diagnosis Part A) and no
outcome here speaks to it; β1's wall-clock was not an
endpoint of this test. Nothing here licenses a general or
causal claim beyond the anneal contrast on this protocol.

**Branch status: unchanged — the trust branch remains
CLOSED.** Per the status bounds fixed before this test, an
implicated verdict means "trust without the anneal"
returns as a **design proposal for owner ruling**. It is
not a patch, and no design change is made or authorized by
this document.

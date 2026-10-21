# β1 learning-parity evaluation — result (WO-TBM-parity-run)

Prereg: `nn/prereg_trust15_parity.md`, frozen at `638956f`
before any run under it. Arms: uniform (frozen path);
trust15-β1 (variant exactly as committed at `b3e3e8d`).
Seeds 520–549, both arms per seed, 1000 episodes,
hidden=32, lr=1e-3, 20/10. Records:
`nn/results_trust15_parity.jsonl` (60/60, collected in a
single uninterrupted session). The frozen analysis
(`analyze_trust15_parity.py`, validated on five synthetic
cases before the run — including the frozen requirement
that computed CI endpoints equal the specified order
statistics) was run exactly once, on the complete file.

## Per-seed endpoints (first80; "never" = censored)

| seed | uniform | β1 | D = β1−uni |
|---|---|---|---|
| 520 | 497 | 433 | −64 |
| 521 | 455 | 482 | +27 |
| 522 | 458 | 380 | −78 |
| 523 | 381 | 393 | +12 |
| 524 | 490 | 476 | −14 |
| 525 | 521 | 408 | −113 |
| 526 | 581 | 500 | −81 |
| 527 | 399 | 428 | +29 |
| 528 | 348 | 329 | −19 |
| 529 | 511 | 326 | −185 |
| 530 | 421 | 527 | +106 |
| 531 | 393 | 425 | +32 |
| 532 | 414 | 355 | −59 |
| 533 | 435 | 427 | −8 |
| 534 | 466 | 376 | −90 |
| 535 | 480 | 364 | −116 |
| 536 | 408 | 370 | −38 |
| 537 | 476 | 395 | −81 |
| 538 | 425 | 496 | +71 |
| 539 | 469 | 354 | −115 |
| 540 | 407 | 371 | −36 |
| 541 | 448 | 405 | −43 |
| 542 | 459 | 440 | −19 |
| 543 | 419 | 531 | +112 |
| 544 | 488 | 583 | +95 |
| 545 | 546 | 483 | −63 |
| 546 | 404 | 371 | −33 |
| 547 | 448 | 422 | −26 |
| 548 | 428 | 456 | +28 |
| 549 | 499 | 434 | −65 |

Per-arm median first80: uniform 451.5, β1 423.5. Sign
split on D: +9 / −21 / 0. **No run was censored** —
every arm on every seed reached rolling-100 ≥ 0.8;
one-sided censored pairs 0, so the contrast does not
carry the censoring-limited label.

## Frozen-rule evaluation

Median of D: **−34.5 episodes**. 95% CI by exact
binomial order statistics (n = 30: ranks 10/21):
**[−64, −8]**. Margin: +50 episodes.

The CI upper endpoint (−8) is **< +50**.

→ **PARITY ESTABLISHED.** Per the frozen rule this is
one of exactly two conclusions, and it is the one the
data select: the adopted β1 design's median
episodes-to-0.8 disadvantage vs uniform is smaller than
the +50-episode margin — affirmatively, not by absence
of detection. This is the verdict PC6's detection-only
reading could not give: the ≲ +50 residual PC6 could not
exclude is now excluded at the margin, on fresh seeds,
under the frozen construction.

Descriptive note (not a further conclusion): the CI lies
entirely below zero — β1 crossed earlier than uniform on
the median seed (per-arm medians 423.5 vs 451.5, sign
split 21/30 in β1's favor). The frozen design tested
non-inferiority, not superiority, and no superiority
claim is made or licensed by this verdict; the tested
claim is parity within +50, and that claim is met.

## Bounded interpretation

What this establishes, on this task (20/10 protocol),
this implementation pair, seeds 520–549: the β1 design
learns at parity with uniform by the frozen
non-inferiority criterion. Combined with PC6, the trust
learning story is now complete at confirmatory grade on
both halves: the anneal caused the systematic
disadvantage, and removing it leaves no disadvantage
beyond the +50 margin — indeed none detected in either
direction the design could have shown.

What this does not touch: the rematch's criterion-(a)
speed failure is machinery cost and no outcome here
speaks to it; β1's wall-clock was not an endpoint of
this test. Nothing here licenses a general claim beyond
this task and protocol. **The trust branch's closed
status is unaffected** — this evaluation judged the
adopted design on its own criterion; it is not a
reopening. The JIT/vectorization identity decision
remains separate, to be ruled after this evaluation.
Candidate 4 (insertion recency) remains INDETERMINATE
and is untouched by this evaluation either way.

# Preregistration: confirmatory β-anneal test (WO-TBM-PC6)

**Status: FROZEN at the commit carrying this file. The run is
NOT authorized by this document — it requires a separate
work order after freeze verification. The trust branch
remains CLOSED; this test can identify or exonerate a
mechanism, it does not reopen the branch and authorizes no
design change.**

## Question and chain

PC5 (`nn/decisions_trust15_var.md`, `6ffac88`) established
confirmatorily that trust-1.5's episodes-to-0.8 disadvantage
is systematic (median +113 episodes vs uniform, 16/20 pairs,
Wilcoxon p = 0.0009, seeds 210–229). It did not identify the
mechanism. PC2 implicated one candidate diagnostically:
fixing β ≡ 1.0 shrank the lag on seeds 200–202 materially
(+91 → +16, +188 → +35) — same-seed, hypothesis-generating
evidence only. **This test is the confirmatory version of
the β question: does the β anneal (0.4 → 1.0) cause the
systematic disadvantage?**

## Arms (three; the contrast is within-seed)

- **uniform** — `agent_nn.train_neural` (frozen path).
- **trust15-annealed** — `layer15_trust_train.train_trust15`,
  code exactly as frozen at `7eaca1e`.
- **trust15-β1** — the PC2 variant **exactly as committed at
  `b3e3e8d`**: `nn/layer15_beta1_train.py` (its train
  function reuses the frozen `Trust15NeuralQ`; the only
  difference from the frozen trainer is `beta = 1.0` every
  episode). Verified at this freeze: `b3e3e8d` is the only
  commit touching that file and the diff to HEAD is empty.
  **If any change to the variant proves necessary, the work
  stops and reports instead — the variant is used as
  committed or not at all.**

Rationale for the design: the mechanism contrast
(annealed vs β1) must be **within-seed** — the same seed
under both β regimes — because episodes-to-0.8 varies by
hundreds of episodes across seeds. PC5's annealed numbers
live on seeds 210–229 with no β1 runs there and cannot be
reused for this contrast; PC2's β1 numbers live on the
hypothesis-generating seeds 200–202, where a confirmatory
claim would be double-dipping. Hence: three arms, fresh
seeds, all contrasts paired within seed.

## Seeds (fresh; provenance verified)

**Seeds 500–519** (20 seeds × 3 arms = 60 runs). Provenance,
verified by scanning every record file in `nn/` (260
distinct seeds in use, spanning 0–499): Experiment A used
200–299; the zero-crush confirmatory 400–499; replication
and controls 10–49; pilots and Stage 1 0–9 and 100–109; the
trust line 200–202 (rematch, diagnosis, PC2, PC4) and
210–229 (PC5). **No tracked record uses any seed ≥ 500.**
All three arms are equally naive to 500–519, and every run
is an independent training from its seed.

## Protocol

Identical to the rematch and PC5: 1000 episodes, hidden=32,
lr=1e-3, PlainEncoder, 20/10. Endpoint per run: **first80**,
the first episode index (0-based) at which rolling-100
success (window episodes ep−99..ep) reaches ≥ 0.8 — the
definition used throughout this chain.

Run mechanics (for the run work order): arms interleaved per
seed (uniform, annealed, β1); records in the rematch format
to `nn/results_trust15_beta.jsonl`, per-seed flush,
skip-existing on relaunch; analysis by
`nn/analyze_trust15_beta.py` (to be written under the run
work order, implementing exactly the rules below), run once
on the complete 60 records; **no inferential analysis on
partial data**.

## Censoring (carried over from PC5 verbatim)

An arm that never reaches rolling-100 ≥ 0.8 within its 1000
episodes is censored and assigned **1000** (a lower bound).
Each pair on each contrast carries a censoring indicator
(none / one side / both); both-censored pairs have D = 0
under substitution and are counted and reported. If ≥ 5 of
20 pairs on a contrast are censored on exactly one side,
that contrast is labeled **censoring-limited** (pre-stated;
not a third conclusion).

## Primary contrast — the anneal (frozen, exactly two conclusions)

Per-seed paired difference **D1 = first80(annealed) −
first80(β1)** (substituted values). Statistic: **Wilcoxon
signed-rank, two-sided α = 0.05** (scipy, default zero
handling — the same family as PC5, for chain comparability),
with the sign split and the median of D1 reported.

- **ANNEAL IMPLICATED (confirmatory)** iff the test rejects
  at α = 0.05 **and** median(D1) > 0.
- **ANNEAL EXONERATED as the primary cause** in all other
  cases — including, named here in advance and not a third
  reading, rejection with a negative median (the anneal
  regime faster than β1), recorded with its direction
  stated plainly.

The meaning of "exonerated as the primary cause" is bounded
by the sensitivity statement below: it asserts that no
anneal contribution of the size this design is powered for
was detected, not that the anneal contributes exactly
nothing.

## Secondary contrast — the residual (pre-stated limited reading)

Per-seed paired difference **D2 = first80(β1) −
first80(uniform)**, same statistic (Wilcoxon, two-sided
α = 0.05), reported with its sign split, median, and p. The
pre-stated reading is limited to **detection**: a residual
systematic disadvantage for β1 vs uniform is *detected* iff
the test rejects and median(D2) > 0. **A non-detection is
reported together with the sensitivity bound and is NOT
stated as equivalence** — β1 could still carry a residual
disadvantage below this design's sensitivity. This contrast
carries no conclusion label beyond that reading.

## Sensitivity (honest power arithmetic)

Planning SD: **142.8 episodes** — the sample SD of PC5's 20
paired differences (annealed − uniform; source named).
The primary contrast's true SD is unknown a priori: it is
plausibly *smaller*, because the annealed and β1 arms share
the trust machinery and differ only in the β regime (the
only direct evidence, PC2's three same-seed differences
+75 / −56 / +153, has SD ≈ 105 at n = 3 — weak evidence,
stated as such). Planning at 142.8 is therefore
conservative. Wilcoxon power at SD 142.8, n = 20 (Monte
Carlo, 20,000 replicates, normal shift alternatives, seed
20261020):

| anneal contribution (median D1) | power |
|---|---|
| +50 episodes | 0.30 |
| +75 episodes | 0.59 |
| +100 episodes | 0.83 |

**What a non-rejection cannot exclude, in so many words:**
an anneal contribution of +50 to +75 episodes — the scale
PC2's diagnostic point estimate suggests (mean annealed−β1
difference +57 at n = 3) — sits in this design's low-power
zone. If the primary test does not reject, the result
bounds the anneal's contribution to "not detected at this
sensitivity; a contribution of ≲ +75 episodes remains
compatible with the data," and the exoneration conclusion
must be read with exactly that bound attached.

## Interpretation bounds

Conclusions are bounded to this task (20/10 protocol), this
implementation trio, and seeds 500–519. Mechanism language
is limited to the anneal contrast actually tested: an
"implicated" verdict establishes that the β regime
contributes to the disadvantage on this design, not that
β ≡ 1.0 is the right design (that returns as a design
proposal for owner ruling, if anything does). This test
says nothing about the rematch's criterion-(a) speed
failure, which is machinery cost (diagnosis Part A) and is
untouched by any outcome here. The trust branch's closed
status is unaffected either way.

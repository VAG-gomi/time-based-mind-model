# Preregistration: β1 learning-parity evaluation
# (FREEZE-READY TEXT — for owner approval, then freeze commit)

**Status: FREEZE-READY TEXT, NOT YET FROZEN.** This text
becomes the frozen prereg only when it is committed to
`nn/prereg_trust15_parity.md`, pushed to `origin/main`, and
verified (`git ls-remote origin main` = local HEAD) before
any run under it. No run is authorized by this document
itself.

**Correction note (vs the earlier draft):** the draft's
power table assumed a normal-approximation sign-test CI
(upper order statistics 14/20/26). This text specifies
the **exact binomial order-statistic CI** (upper order
statistics 15/21/27) and retables power under it. The
draft's table is superseded; do not use it.

## Question

PC6 established two things about the adopted β1 design
(β ≡ 1.0, PC2 variant at `b3e3e8d`): the anneal causes the
systematic disadvantage (D1 confirmatory), and no residual
disadvantage for β1 vs uniform was *detected* (D2 median
−24.5, p = 0.0826). Detection is not parity: PC6's design
could not exclude a residual of ≲ +50 episodes. **This
evaluation is designed to answer the parity question
decisively — to affirm parity if it holds, not merely to
fail to detect a difference.**

## Arms (two)

- **uniform** — frozen path.
- **trust15-β1** — the PC2 variant **exactly as committed
  at `b3e3e8d`** (verified: `b3e3e8d` is the only commit
  touching `nn/layer15_beta1_train.py`, diff to HEAD
  empty). **If any change to the variant proves necessary,
  the work stops and reports instead.**
- The annealed arm is NOT included: the anneal question is
  closed (PC6, confirmatory); this evaluation judges the
  adopted design against uniform only.

## Seeds (fresh; provenance verified)

**Seeds 520–549** (30 seeds × 2 arms = 60 runs).
Provenance, verified by scanning every record file in
`nn/` (280 distinct seeds in use, spanning 0–519):
Experiment A 200–299; zero-crush 400–499; replication /
controls 10–49; pilots / Stage 1 0–9, 100–109; trust line
200–202, 210–229, and 500–519 (PC6). **No tracked record
uses any seed ≥ 520.** Both arms are equally naive to
520–549.

## Protocol

Identical to PC5/PC6: 1000 episodes, hidden=32, lr=1e-3,
20/10. Endpoint per run: **first80**, the first episode
index (0-based) at which rolling-100 success reaches
≥ 0.8 — the definition used throughout this chain.

Run mechanics (for the run work order): arms interleaved
per seed (uniform, β1); records in the rematch format to
`nn/results_trust15_parity.jsonl`, per-seed flush,
skip-existing on relaunch; analysis by
`nn/analyze_trust15_parity.py`, implementing exactly the
rules below; **the analysis script must be validated on
synthetic cases BEFORE the run, including a case
verifying that the computed CI endpoints equal the
specified order statistics**; the analysis runs once on
the complete 60 records; **no inferential analysis on
partial data**.

## Censoring (carried over from PC5/PC6)

An arm that never reaches rolling-100 ≥ 0.8 within its
1000 episodes is censored and assigned **1000** (a lower
bound). Each pair carries a censoring indicator (none /
one side / both); both-censored pairs have D = 0 under
substitution and are counted and reported. If ≥ 5 of 30
pairs are censored on exactly one side, the contrast is
labeled **censoring-limited** (pre-stated; not a third
conclusion).

## Primary decision rule — non-inferiority

Per-seed paired difference **D = first80(β1) −
first80(uniform)** (substituted values). Estimand: the
**median of D**.

**Confidence interval — exact binomial order statistics
(this construction is normative; no substitute):**
sort the n differences ascending, 1-indexed:
X_(1) ≤ X_(2) ≤ … ≤ X_(n). The two-sided 95% CI is
[X_(l), X_(u)], where l is the 2.5% quantile of
Bin(n, 0.5) (the smallest k with CDF(k) ≥ 0.025) and
u = n − l + 1. **For n = 30: l = 10, u = 21 — the CI is
the 10th and 21st smallest differences, and the decision
uses the upper endpoint X_(21).** No normal approximation
is used anywhere in this interval.

**Margin: +50 episodes** — β1 is at parity with uniform
if its median disadvantage is smaller than 50 episodes,
i.e. less than half the annealed design's confirmed cost
(PC5 median +113) and below what PC6 could exclude.

- **PARITY ESTABLISHED** iff the 95% CI upper endpoint is
  **< +50 episodes**.
- **PARITY NOT ESTABLISHED** otherwise, with the CI
  reported as what remains compatible. Named here in
  advance and not a third reading: if the CI lies
  entirely above 0 (lower endpoint > 0), a systematic
  disadvantage for β1 is *detected* — reported as a
  sub-case of NOT ESTABLISHED, with its direction stated.

Exactly two conclusions. This structure is what makes the
evaluation decisive where PC6's detection-only reading
could not be: a PARITY ESTABLISHED verdict is an
affirmative claim bounded by the margin, not an absence
of detection.

## Sensitivity (exact-binomial power arithmetic)

Planning SD: **70.2 episodes** — the sample SD of PC6's
own 20 D2 paired differences (β1 − uniform; source
named, value independently recomputed from the PC6
records). This planning SD comes from the *same contrast*
being tested — stronger footing than PC6's, whose
planning SD came from a different contrast. Power below
is computed exactly under normal shift alternatives
(P(obs < 50) from the normal CDF; claim probability from
the binomial tail at the specified order statistic):

| pairs | P(claim parity \| true median 0) | P(false claim \| true median +50) |
|---|---|---|
| 20 | 0.67 | 0.021 |
| **30 (recommended)** | **0.84** | **0.021** |
| 40 | 0.93 | 0.019 |

**Recommended: 30 pairs** (60 runs, ≈ 90 minutes at PC6's
pace): 0.84 probability of establishing parity if β1 is
truly at parity, and only ≈ 0.02 probability of claiming
parity if β1 truly carries a +50-episode disadvantage.
The 40-pair alternative (seeds 520–559, ≈ 2 hours) buys
0.93 power at the same false-claim rate; the owner may
substitute it at approval time without changing any other
rule (with u recomputed for n = 40: l = 14, u = 27).

**What a NOT ESTABLISHED verdict means, in so many
words:** the data did not affirm parity within +50
episodes. It does not by itself establish a disadvantage
unless the CI lies entirely above 0 (the named sub-case).
The CI bounds, not the label, carry the information.

## Interpretation bounds

Conclusions are bounded to this task (20/10 protocol),
this implementation pair, and seeds 520–549. This
evaluation judges learning parity only: it says nothing
about the criterion-(a) speed failure (machinery cost,
untouched), and it does not reopen the trust branch —
β1 is the adopted design, and this test judges it on its
own criterion. The JIT/vectorization identity decision
remains separate, to be ruled after this evaluation, and
no outcome here speaks to it. Candidate 4 (insertion
recency) remains INDETERMINATE and is untouched by this
evaluation either way.

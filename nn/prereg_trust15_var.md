# Preregistration: variance-null test for trust-1.5 (WO-TBM-PC5)

**Status: FROZEN at the commit carrying this file. The run is
NOT authorized by this document — it requires a separate work
order after freeze verification. This test governs a
diagnostic question about the rematch's criterion-(c) failure;
it does not reopen the trust branch, which remains CLOSED
under `nn/prereg_trust15.md` / `nn/decisions_trust15.md`
regardless of this test's outcome.**

## Question

The L15 diagnosis (`nn/diagnosis_trust15.md`, `3d8e854`)
found the rematch's episodes-to-0.8 differences (+91 / −65 /
+188 on seeds 200–202) consistent with the **variance null**
(candidate 5): no systematic disadvantage, only seed scatter
at n=3. This test is the discriminating test proposed there
(P-C5): is the disadvantage systematic, or was the n=3 pattern
scatter?

## Arms and protocol

- Arms: `uniform` (`agent_nn.train_neural`) vs `trust15`
  (`layer15_trust_train.train_trust15`), code exactly as
  frozen at `7eaca1e`. No modifications; the P-C2/P-C4 toggles
  are NOT part of this test.
- Protocol identical to the rematch: 1000 episodes,
  hidden=32, lr=1e-3, PlainEncoder, 20/10. Both arms run
  fresh, interleaved per seed; records (same format as
  `nn/results_trust15.jsonl`) to
  `nn/results_trust15_var.jsonl`, per-seed flush,
  skip-existing on relaunch. Wall-clock is recorded in the
  records but is **not** an endpoint of this test.
- **Seeds 210–229, paired (20 pairs).**

### Seed provenance (verified from tracked records)

Seeds 210–229 were used by Experiment A: `wrapped_ind` and
`phase_only_new` (seeds 200–299) and `oracle_ind` and
`memory30` (seeds 200–239) — all 20 seeds, per
`nn/results_expa_1.jsonl` / `nn/results_expa_2.jsonl`. They
have **never** been used by `uniform`, `trust`, or `trust15`
in any tracked record (the trust line has used only seeds
200–202). The comparison is unaffected by the Experiment A
use, for three reasons stated in advance: (1) the estimand
is the **within-pair difference** — any seed-specific
difficulty enters both arms of a pair identically; (2) every
run is an independent training initialized from its seed —
no state carries between experiments, so prior use by other
arms cannot advantage either arm here; (3) this range was
proposed in the diagnosis (`3d8e854`) before any uniform or
trust-1.5 run existed on it — it was not selected on
outcomes for these arms.

## Primary endpoint

Per pair (seed), the difference **D = first80(trust15) −
first80(uniform)**, where first80 is the first episode index
(0-based) at which rolling-100 success (window: episodes
ep−99..ep) reaches ≥ 0.8 — the exact definition used for the
rematch endpoints.

## Censoring rule (fixed in advance)

- An arm that never reaches rolling-100 ≥ 0.8 within its 1000
  episodes is **censored** and assigned the value **1000**
  (one past the final episode index 999) — a lower bound on
  its true crossing episode.
- Each pair carries a censoring indicator: none /
  uniform-only / trust15-only / both. Both-censored pairs
  have D = 0 under substitution and are counted and reported.
- The decision statistics use the substituted values over all
  20 pairs; the Wilcoxon test drops zero differences in the
  standard way, so tied/both-censored pairs reduce its
  effective n (reported with the result).
- Pre-stated caution: if ≥ 5 of 20 pairs are censored on
  exactly one side, substituted differences are lower bounds
  in the censored direction and the result is labeled
  **censoring-limited**. This label is fixed here in advance;
  it does not create a third conclusion.

## Decision rule (frozen; exactly two conclusions)

- **Primary statistic:** Wilcoxon signed-rank test on the 20
  paired differences D (two-sided, α = 0.05), computed once
  on the complete data by `nn/analyze_trust15_var.py` (to be
  written under the run work order, implementing exactly
  this rule). Reported with it: the sign split of D and the
  median of D. No inferential analysis on partial data.
- **Conclusion (ii) — NULL REFUTED:** the Wilcoxon test
  rejects at α = 0.05 **and** the median of D is positive.
  Reading: a systematic episodes-to-0.8 disadvantage exists;
  P-C2/P-C4 gain priority.
- **Conclusion (i) — NULL STANDS:** all other cases,
  including non-rejection and — named here in advance, not a
  third reading — rejection with a negative median, which is
  recorded as "null stands (no systematic disadvantage); a
  systematic advantage was observed." Reading of (i): the
  n=3 pattern was scatter at the sensitivity stated below;
  P-C2/P-C4 lose priority.

## Sensitivity (honest power arithmetic)

Paired-difference SD is unknown; planning value 127 episodes
(the sample SD of the three rematch diffs +91/−65/+188 —
itself estimated from n=3 and uncertain; power scales
inversely with it). Wilcoxon power (Monte Carlo, 20,000
replicates, normal shift alternatives, seed 20261020):

| systematic disadvantage | Wilcoxon power |
|---|---|
| +50 episodes | 0.37 |
| +75 episodes | 0.69 |
| +100 episodes | 0.90 |
| +125 episodes | 0.98 |

The sign test is weaker corroboration: at n=20 it rejects
only at ≥ 15/20 same-sign (exact two-sided α = 0.0414) and
has power 0.62 at P(trust15 later) = 0.75, 0.80 at 0.80.

**What 20 pairs cannot do:** exclude a moderate systematic
disadvantage. A non-rejection (conclusion (i)) means exactly:
no systematic disadvantage of ≳ +100 episodes was detected,
and effects around +50 episodes remain compatible with the
data. The conclusions will be worded with this bound.

## Secondary endpoints (descriptive only — no tests, no
decision weight)

- Time-to-criterion variance ratio between arms (variance of
  first80 across each arm's 20 seeds, substituted values;
  censoring counts reported alongside).
- Final-100 success, per arm (mean over seeds).
- Per-arm median first80.

## Interpretation bounds

Any conclusion is bounded to this task (20/10 protocol), this
implementation pair, and these seeds. No causal language
beyond the design: conclusion (ii) establishes that a
systematic cost exists, not which mechanism produces it;
conclusion (i) establishes that the n=3 pattern does not
demonstrate one, at the stated sensitivity. Either way, the
trust branch's closed status is unaffected — this test
informs only the priority of the P-C2/P-C4 diagnostic
proposals.

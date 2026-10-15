# Pre-registration: paired-seed replication (DRAFT for owner review)

**Status: DRAFT. Nothing is frozen and nothing has run. This note
needs the owner's explicit approval before any replication begins.**

## Question

Does the wrapped arm's safety/consistency advantage over the plain
and one-step-memory arms replicate at larger seed counts?

## What is frozen

- Code: the exact commit approved below (no architecture, hyperparameter,
  or reward changes).
- Arms: plain, memory, wrapped (the pre-registered comparison).
  The clock control is NOT replicated: it served its one-off purpose
  (0/10, unstable) and the compute is better spent on the core
  comparison. (Owner may reinstate it.)
- Protocol: identical to B2. 2000 training episodes, hidden=32,
  lr=1e-3, gamma=0.99, same rewards, episode cap 200, period=20,
  open_len=10. Evaluation: 20 start phases, greedy 1 episode/phase;
  eps=0.05, 10 episodes/phase.

## Seeds

- 40 fresh seeds: **10–49**, paired across arms (seed s gives every
  arm the same training start-phase stream via seed+1000, as in B2).
- Fresh (not reusing 0–9) so the replication is independent of the
  pilot that motivated it.

## Primary endpoint

- Paired difference in **clean-and-fast proportion**
  (wrapped minus plain; wrapped minus memory), with 95% confidence
  intervals. Clean-and-fast keeps its B2 definition
  (greedy: 0 crushes, success ≥ 0.9, return ≥ 0.75).
- Reported alongside: raw counts per arm, paired per-seed return
  differences (median + bootstrap 95% CI), zero-crush and success
  proportions separately.

## The 7/10 threshold

- **Retired as a binary win/lose gate for the replication.** It was
  calibrated for n=10 as a pilot decision rule. The replication
  reports estimates and intervals instead of a pass/fail verdict.
- The clean-and-fast *definition* is unchanged. Nothing is redefined
  after seeing data.
- Wording for the record: "The original 7/10 rule remains part of
  the historical B2 result, but the larger replication estimates
  effect size and uncertainty rather than using an arbitrary binary
  winner threshold." Do not describe the replication as the wrapper
  "passing" or "winning" if a confidence interval crosses zero.

## Confidence intervals

- Individual proportions: Clopper-Pearson exact binomial 95% CI.
- Paired binary comparison (clean-and-fast vs not, same seeds
  across arms): McNemar's exact test with an exact paired 95%
  confidence interval for the difference in proportions.
- Median return differences: bootstrap 95% CI with 10,000
  resamples, resampling **seeds** (not episodes) with replacement.

## Commit semantics

- `nn/summary_b2.json`'s `metadata.git_commit` records the commit
  whose `results_a5.jsonl` was read when the JSON was generated
  (data provenance). A file cannot contain its own commit hash, so
  this is necessarily the parent of the commit that contains the
  JSON.
- The **frozen replication commit** (approval item 1 below) is the
  commit the owner approves to run from. It must contain the final
  `summary_b2.json` and the approved `prereg_replication.md`.
  Record both hashes; they differ by construction.

## Inclusion / exclusion

- All 40 seeds per arm that complete training are included. No
  exclusions for poor performance.
- If a seed crashes or fails to save, it is reported as a missing
  record with the error log — never silently dropped or replaced.
- Missing-data rule for the paired primary analysis: a seed missing
  in either compared arm is reported in the missingness table and
  **excluded from the paired primary comparison**, while all
  available arm-level results remain included in descriptive
  analyses. The number and identity of incomplete pairs is reported.
  (Alternative considered and rejected: predeclaring missing results
  as failures. The exclusion rule is chosen because a crash is a
  missing measurement, not evidence of poor policy performance.)

## What this will not do

- No hyperparameter tuning, no architecture changes, no per-seed
  adjustments. Any tuning uses separate seeds 100+ per the standing
  rule, and is a different experiment.
- No claim beyond the protocol: one fixed schedule (20/10), one
  architecture, one budget.

## Stop / decision rule

After the replication, the owner decides based on the intervals —
not on whether a threshold was crossed. If the wrapped advantage
replicates (intervals exclude zero with margin), proceed to matched
controls (Priority 2) and unseen schedules (Priority 3). If not,
the B2 result stands as a pilot-scale suggestive finding.

## Approvals needed

1. [x] Code commit frozen for the replication: `7affcfb84f41024900927e1625678e5052a5fc75`
       (must contain the final summary_b2.json and this approved note;
       see "Commit semantics" above for how it relates to the JSON's
       internal git_commit field)
2. [x] Seed range 10–49, arms plain/memory/wrapped confirmed
3. [x] 7/10 retired as gate; estimates + intervals reported instead
       (wording above approved)
4. [x] Authorized to run (~120 seeds × ~1.8 min ≈ 4 hours)

Approved by owner 2026-10-15. Replication output goes to
`nn/results_replication.jsonl` / `nn/output_replication.txt`
(separate from the pilot's `results_a5.jsonl`), seeds 10–49.

# Report for Claude — Experiment A: status + analysis-code review pack

Prepared by Muse, 2026-10-18 ~03:50 −06, at the owner's request.
Purpose: handoff for independent review. Experiment A is mid-run;
the analysis code is written and validated **before** any real
results exist.

## 1. Design (frozen, owner-approved)

Source: `nn/prereg_generalization.md` v3.2 (commit `92cc70d`).

- Task: unseen schedule 30/14. Fast bar 0.744
  (`--fast-threshold 0.744`).
- Arms/seeds (280 total):
  - `wrapped_ind` × 100, seeds 200–299 (ScheduleModel + schedule-
    independent encoding)
  - `phase_only_new` × 100, seeds 200–299 (raw clock
    `(tick%30)/30`; **primary comparison**)
  - `oracle_ind` × 40, seeds 200–239 (true phase/remaining in the
    wrapper's encoding; inference test)
  - `memory30` × 40, seeds 200–239 (strongest history baseline)
- Frozen experimental code: `888355c`. Owner approval recorded in
  the prereg (`92cc70d`, committed 2026-10-17 23:22 −06) **before
  any Experiment A results existed**.
- Experiment B is **not** approved (v3.3 draft only).

## 2. Pre-registered analyses (what the code must do)

- **Primary:** clean-and-fast (greedy 0 crushes, success ≥ 0.9,
  return ≥ 0.744), paired `wrapped_ind` vs `phase_only_new`,
  exact McNemar, α = 0.05. Single primary — no Holm.
- **Secondaries:** `wrapped_ind` vs `oracle_ind`, `wrapped_ind`
  vs `memory30`; Holm over the 2 comparisons; equivalence margin
  ±0.25 on the paired rate difference, with "indistinguishable"
  as third outcome.
- **Median-return differences:** seed-bootstrap 95% CIs, 10,000
  resamples, resampling seeds (not episodes).
- **Missing-data rule:** a seed missing in either compared arm is
  identified and that incomplete pair excluded from the paired
  analysis.

## 3. Run status (as of 03:50 −06)

- **137/280 records**: `wrapped_ind` 67/100, `phase_only_new`
  70/100 (both processes still on their first arm).
- Two runtime restarts (~00:41, ~02:31 −06) killed both
  processes. Resume via the runner's skip-existing-records logic;
  seed ranges verified contiguous with no partial writes after
  each restart. **Zero records lost.**
- Pace ~3.6 min/seed with 2 processes sharing CPU; ETA ~4 more
  hours.
- Repo state: local HEAD = remote `main` = `5f434b4` (verified
  via `git ls-remote`).

## 4. Analysis code — for your review

- `nn/analyze_expa.py` (commit `5f434b4`): implements exactly §2,
  deterministically. Two details worth your eyes:
  - The 0.744 bar is computed from raw record metrics and
    **cross-checked** against `run_arms._clean_fast(rec, 0.744)`
    on every record — never from the runner's module-global
    default (still 0.75). Any drift fails loudly.
  - `paired_diff_ci` and the bootstrap are the same
    implementations as `analyze_replication.py` (precedent).
- `nn/validate_analyze_expa.py`: **12/12 checks pass on
  synthetic data only** — McNemar vs scipy's independent
  `binomtest` (300 tables), Holm vs a separately written
  textbook Holm (500 sets), bootstrap coverage sanity,
  the exact 0.744 boundary, and all four failure modes
  (missing / duplicate / wrong period / unexpected arm) —
  plus a full synthetic 280-record end-to-end run that
  exercised the Holm stop rule.
- This is analysis code only; the experimental code frozen at
  `888355c` is untouched (same precedent as the replication).

## 5. Open review questions

1. **Equivalence-rule reading.** My code reports DIFFERS only
   when the paired-diff CI lies *entirely outside* ±0.25;
   EQUIVALENT only when entirely inside; otherwise
   INDISTINGUISHABLE. In the synthetic end-to-end test, a CI of
   [+0.030, +0.439] (excludes zero, mostly inside the margin)
   came out INDISTINGUISHABLE. The prereg says
   "indistinguishable" when the interval "is neither inside the
   margin nor decisively excludes zero." Is my reading right, or
   should any zero-excluding CI count as differing? This must be
   settled **before** the real results land — changing it after
   would be post-hoc tuning.
2. Does the implementation match the prereg text anywhere else
   you can see?
3. Missing-data rule as coded (exclude incomplete pairs from
   paired tests; descriptives use all available records) — OK?

## 6. Interim tallies (partial — 137/280; descriptive only, no
conclusions)

| Arm | n | Clean-and-fast (0.744) | Median return | Zero-crush |
|-----|---|------------------------|---------------|------------|
| wrapped_ind | 67 | 16 | 0.714 | 63/67 |
| phase_only_new | 70 | 17 | 0.735 | 48/70 |

The pilot's shape (safe-but-slow, some collapsed seeds) is
visible, but 137 seeds is not the analysis. The pre-registered
McNemar result on all 280 is the answer.

## 7. What happens next

When both processes finish: merge the two files, run the
unmodified `analyze_expa.py`, commit + push results, and deliver
the report with the raw files attached. No post-hoc changes to
the analysis.

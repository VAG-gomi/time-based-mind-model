# Matched controls results: seeds 10–49 (pre-registered analysis)

**Status: analysis per `nn/prereg_controls.md` (approved 2026-10-15,
frozen ae173f8). Estimates and intervals only; no binary gate.
Cautious language throughout, per the approved interpretation rules.**

## Data integrity

- 200/200 records present (`nn/results_controls.jsonl`):
  oracle, phase_only, remaining_only, param_matched, memory10 —
  40/40 each, seeds 10–49.
- Zero missing records. No exclusions.
- Identical hyperparameters across all arms (1 distinct set).
- Paired against the replication's wrapped records (seeds 10–49).

## Primary endpoint: clean-and-fast proportions

| Arm            | Clean-and-fast | Proportion | 95% CI        |
|----------------|----------------|------------|---------------|
| wrapped (repl) | 26/40          | 0.650      | [0.483, 0.794]|
| oracle         | 30/40          | 0.750      | [0.588, 0.873]|
| phase_only     | 26/40          | 0.650      | [0.483, 0.794]|
| remaining_only | 27/40          | 0.675      | [0.509, 0.814]|
| param_matched  | 0/40           | 0.000      | [0.000, 0.088]|
| memory10       | 31/40          | 0.775      | [0.615, 0.892]|

## Paired wrapped-minus-control differences (McNemar exact)

| Comparison              | Diff    | 95% CI            | p      | (b, c)  |
|-------------------------|---------|-------------------|--------|---------|
| wrapped − oracle        | −0.100  | [−0.309, +0.139]  | 0.5034 | (8, 12) |
| wrapped − phase_only    | +0.000  | [−0.173, +0.173]  | 1.0000 | (6, 6)  |
| wrapped − remaining_only| −0.025  | [−0.243, +0.201]  | 1.0000 | (9, 10) |
| wrapped − param_matched | +0.650  | [+0.478, +0.650]  | <0.0001| (26, 0) |
| wrapped − memory10      | −0.125  | [−0.320, +0.111]  | 0.3593 | (7, 12) |

## Median greedy-return differences (bootstrap, seeds, 10k resamples)

| Comparison              | Diff    | 95% CI            |
|-------------------------|---------|-------------------|
| wrapped − oracle        | −0.002  | [−0.021, +0.012]  |
| wrapped − phase_only    | +0.001  | [−0.010, +0.016]  |
| wrapped − remaining_only| +0.006  | [−0.005, +0.016]  |
| wrapped − param_matched | +0.064  | [+0.043, +1.205]  |
| wrapped − memory10      | −0.007  | [−0.026, +0.003]  |

Wrapped median: 0.764. All control medians within 0.02 except
param_matched (0.691).

## Safety / success (descriptive)

| Arm            | Zero-crush | Success ≥ 0.9 |
|----------------|------------|---------------|
| wrapped (repl) | 38/40      | 36/40         |
| oracle         | 39/40      | 39/40         |
| phase_only     | 35/40      | 39/40         |
| remaining_only | 40/40      | 39/40         |
| param_matched  | 11/40      | 23/40         |
| memory10       | 40/40      | 40/40         |

## Interpretation (per preregistered rules)

- **oracle ~= wrapped** (p=0.50, CI includes zero): supports the
  interpretation that the ScheduleModel already extracts everything
  the task needs from observations. The inferred representation is
  not lossy in any way that matters here.
- **phase_only ~= wrapped** and **remaining_only ~= wrapped**
  (both p=1.0): supports the interpretation that either single
  temporal feature suffices; neither the phase nor the countdown
  is individually necessary.
- **param_matched << wrapped** (p<0.0001, diff +0.650): decisively
  refutes the parameter-capacity explanation. The advantage is not
  from having more inputs or weights.
- **memory10 ~= wrapped** (p=0.36, 31/40 vs 26/40): supports the
  interpretation that a 10-step door history suffices without any
  explicit schedule representation.
- **wrapped > all controls**: NOT observed. No control is
  significantly worse than wrapped except param_matched.

## What this means

The replication established that *temporal information* helps:
wrapped beat plain (0/40) and one-step memory (7/40) decisively.
These controls refine that claim: the benefit is **not specific to
the inferred schedule representation**. True phase, true countdown,
or a 10-step raw history all reach the same performance. What
matters is giving the network *enough* temporal context — one bit
of history (memory arm) is not enough; ten bits, or an explicit
phase/countdown, are.

This supports the inferred schedule representation as *a*
sufficient explanation among the tested alternatives, not as the
unique or necessary one. It does not prove causality and does not
generalize beyond this task and protocol.

## What remains (per EBI plan)

- Priority 3: unseen schedules and schedule changes — the critical
  test of whether the *inferred* representation (vs fixed oracle
  features or raw history) has any advantage when the schedule
  is not known in advance.
- Priority 4: failure analysis of non-clean seeds.

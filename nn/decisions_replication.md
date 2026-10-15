# Replication results: paired seeds 10–49 (pre-registered analysis)

**Status: analysis per `nn/prereg_replication.md` (approved 2026-10-15).
No binary win/lose gate is applied. Estimates and intervals only.**

## Data integrity

- 120/120 records present (`nn/results_replication.jsonl`):
  plain 40/40, memory 40/40, wrapped 40/40, seeds 10–49.
- Zero missing records. No exclusions.
- Identical hyperparameters across all arms (1 distinct set).
- Code frozen at `7affcfb` (approval recorded at `220c655`;
  code identical).

## Primary endpoint: paired clean-and-fast differences

Clean-and-fast = greedy: 0 crushes, success ≥ 0.9, return ≥ 0.75.

| Arm     | Clean-and-fast | Proportion | Clopper-Pearson 95% CI |
|---------|----------------|------------|------------------------|
| plain   | 0/40           | 0.000      | [0.000, 0.088]         |
| memory  | 7/40           | 0.175      | [0.073, 0.328]         |
| wrapped | 26/40          | 0.650      | [0.483, 0.794]         |

Paired differences (McNemar exact, same seeds):

| Comparison        | Difference | 95% paired CI       | McNemar p | Discordants (b, c) |
|-------------------|------------|---------------------|-----------|--------------------|
| wrapped − plain   | +0.650     | [+0.478, +0.650]    | < 0.0001  | (26, 0)            |
| wrapped − memory  | +0.475     | [+0.253, +0.563]    | 0.0001    | (21, 2)            |

Both intervals exclude zero. The wrapped advantage replicates and
is statistically significant against both baselines.

## Secondary: safety and success separately

| Arm     | Zero-crush (greedy) | Success ≥ 0.9 (greedy) |
|---------|---------------------|------------------------|
| plain   | 10/40               | 29/40                  |
| memory  | 26/40               | 34/40                  |
| wrapped | 38/40               | 36/40                  |

## Secondary: median return differences (bootstrap)

Resampling seeds with replacement, 10,000 resamples:

| Arm     | Median greedy return |
|---------|----------------------|
| plain   | 0.711                |
| memory  | 0.727                |
| wrapped | 0.764                |

| Comparison        | Median diff | 95% bootstrap CI   |
|-------------------|-------------|--------------------|
| wrapped − plain   | +0.046      | [+0.035, +0.092]   |
| wrapped − memory  | +0.036      | [+0.012, +0.053]   |

Both intervals exclude zero.

## Schedule model diagnostics (wrapped, training)

- Frac ticks with features known: median 1.000 (min 0.997).
- First converged episode: 0 on all 40 seeds.
- Contradictions: 0 on all 40 seeds.

## Interpretation (per preregistration)

The historical B2 pilot (seeds 0–9) found wrapped 5/10 clean-and-fast,
below the pre-registered 7/10 threshold — an honest negative under
that binary rule. The replication (seeds 10–49, n=40/arm) estimates
the effect with uncertainty: wrapped 65% (95% CI 48–79%) vs plain 0%
(0–9%) and memory 17.5% (7–33%). The paired differences (+0.65 vs
plain, +0.48 vs memory) are significant with intervals well clear of
zero.

Per the pre-registered wording: "The original 7/10 rule remains part
of the historical B2 result, but the larger replication estimates
effect size and uncertainty rather than using an arbitrary binary
winner threshold." This replication is therefore reported as a
replicated, statistically significant safety/consistency advantage
for explicit schedule features — not as the wrapper "passing" a
threshold.

## What remains (per EBI plan)

- Priority 2: matched controls (oracle phase, ablations,
  parameter-matched, stronger memory).
- Priority 3: unseen schedules and schedule changes.
- Priority 4: failure analysis of the 14 non-clean wrapped seeds.

# Experiment A — decisions

Frozen design: prereg v3.2 (commit 888355c, approved 2026-10-17 23:22 −0600,
before any run). Analysis: nn/analyze_expa.py at commit 6c1f9e0
(two-flag equivalence reading, owner-approved 2026-10-18, before
results). Script run unmodified, once, on the full 280 records.
No inferential analysis was run on partial data.

Run record: 4 runtime restarts during the night; each kill lost only
in-flight seeds. All 280 records are complete, contiguous by seed, and
30/14 protocol. Fresh seeds 200–299 (primary pair), 200–239
(secondaries); pilot seeds excluded per prereg.

## Results (script output, deterministic)

- records: 280/280 present, no missing seeds.
- clean-and-fast (bar 0.744): wrapped_ind 26/100 (CI 0.177–0.357),
  phase_only_new 26/100 (0.177–0.357), oracle_ind 19/40 (0.315–0.639),
  memory30 0/40 (0.000–0.088).
- PRIMARY wrapped_ind vs phase_only_new: discordant 16/16, diff
  +0.000, 95% CI [−0.116, +0.116], p=1.0000, no rejection.
- SECONDARY wrapped_ind vs oracle_ind: discordant 6/15, diff −0.225,
  CI [−0.407, +0.023], p=0.0784, Holm no rejection →
  **indistinguishable** (not detected, not within margin).
- SECONDARY wrapped_ind vs memory30: discordant 10/0, diff +0.250,
  CI [+0.096, +0.250], p=0.0020, Holm REJECT →
  **small difference, within margin** (detected, within ±0.25).
- EXPLORATORY (post-interim, non-confirmatory): zero-crush
  wrapped_ind vs phase_only_new: wrapped-only 27, phase-only 2,
  raw p<0.0001. success≥0.9 shows no difference (4/7, p=0.55).
- Descriptive: zero-crush wrapped_ind 95/100, phase_only_new 70/100,
  oracle_ind 38/40, memory30 40/40. Median greedy return:
  wrapped 0.720, phase 0.735, oracle 0.741, memory30 0.714.
  Median paired diff wrapped−phase: −0.012, CI [−0.022, −0.005].

## Interpretation (prereg-bounded, no post-hoc tests)

1. The pre-registered primary finds **no difference** between the
   wrapper and the raw clock on the mixed safety+speed measure
   (identical 26/100, p=1.0). This is a failure to detect, not
   evidence of equivalence — the prereg specified no margin for the
   primary.
2. Against oracle_ind, the wrapper is **indistinguishable** (diff
   −0.225, p=0.0784, CI mostly negative but crossing zero).
   Against memory30, the wrapper shows a **small difference within
   the margin** (+0.250, Holm reject, CI inside ±0.25).
3. The exploratory zero-crush test suggests the wrapper is
   substantially safer than the raw clock (27 vs 2 discordant
   seeds). This was added after interim tallies and is
   **non-confirmatory**; it cannot be cited as a confirmed finding.
   The primary, which mixes safety with speed, cannot see it.
4. Interpretation stays task-and-protocol bounded: 30/14 schedule,
   fixed hyperparams, hand-written NN substrate. No claim about the
   time-based model in general, and no causal language — the
   strongest explanation among the tested alternatives is that the
   inferred schedule representation adds safety the raw clock lacks
   on this task, while neither tested temporal form reaches the
   oracle's level at the 0.744 bar.

Raw records: nn/results_expa_1.jsonl, nn/results_expa_2.jsonl.
Script output: nn/output_expa.txt.

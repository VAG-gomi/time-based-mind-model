# Confirmatory zero-crush experiment — decisions

Frozen design: prereg v0.3 (`nn/prereg_zerocrush.md`, owner rulings
2026-10-19: two-sided McNemar α=0.05; 100 fresh seeds/arm, range
400–499; no confirmatory secondaries). Analysis: frozen
`nn/analyze_zerocrush.py` at commit `daa2cff`, validated 18/18 on
synthetic data, run **exactly once** on the full 200 records.
No inferential analysis was run on partial data.

Run record: 4 runtime restarts during the run; each kill lost only
in-flight seeds. All 200 records are complete, contiguous by seed
(400–499 per arm), and 30/14 protocol. Seed range disjoint from
all prior runs (disjointness in prereg §4).

## Results (frozen script output, deterministic)

- records: 200 present, no missing seeds, no exclusions.
- zero-crush (descriptive, Clopper–Pearson 95% CI):
  wrapped_ind 96/100 = 0.960 [0.901, 0.989];
  phase_only_new 67/100 = 0.670 [0.569, 0.761].
- PRIMARY (two-sided exact McNemar): paired seeds 100/100;
  discordant wrapped_ind-only 32, phase_only_new-only 3 (n_d=35);
  direction +, **p = 4.2e-07** → **REJECT null**.

## Interpretation (prereg §7 bounds)

**Positive.** Per prereg §7 (two-sided p < 0.05, direction
favoring the wrapper), this licenses: on the 30/14 task and fixed
protocol, the wrapper is safer than the raw clock. The exploratory
signal from Experiment A (27 vs 2) is **confirmed on fresh seeds**
(32 vs 3) — the confirmatory design upgrades it from a
post-interim observation to a pre-registered finding.

It does **not** license:
- superiority of the time-based model in general — this is one
  task, one schedule, one fixed protocol;
- any claim about speed — Experiment A's primary already showed
  no clean-and-fast difference (26/100 each); safety is the only
  confirmed edge;
- causal language about *why* the wrapper is safer. Task-and-
  protocol bounded, strongest explanation among the tested
  alternatives: the inferred schedule representation adds safety
  the raw clock lacks on this task.

Raw records: `nn/results_zerocrush.jsonl`.
Script output: `nn/output_zerocrush.txt`.

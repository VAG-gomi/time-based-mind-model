# Preregistration DRAFT v0.1: confirmatory zero-crush experiment

**Status: DRAFT v0.3 — not approved, no runs, no code freeze, no
owner approval sought. This document is returned for owner review,
nothing more. Nothing here authorizes execution.**
**v0.3 changes (owner rulings 2026-10-19):** two-sided primary
(§5, power sketch recomputed); 100 seeds/arm, range 400–499,
disjointness verified (§4); no confirmatory secondaries,
descriptives for the two primary arms only (§2).
**v0.2 change:** explicit record citations for the exploratory
attribution (see Background §) — no design change.

## Background (from tracked records)

- Experiment A (280 seeds, frozen analysis `6c1f9e0`,
  `nn/decisions_expa.md`): primary `wrapped_ind` vs
  `phase_only_new` on clean-and-fast — **no difference** (26/100
  each, p=1.0000).
- Exploratory, post-interim, non-confirmatory (same records):
  zero-crush `wrapped_ind` vs `phase_only_new`, 27 vs 2 discordant
  pairs, raw p<0.0001. Descriptive: zero-crush 95/100 vs 70/100.
  `decisions_expa.md`: "The primary, which mixes safety with speed,
  cannot see it."
  **Attribution record (auditable):** the 27-vs-2 result belongs to
  the `wrapped_ind` vs `phase_only_new` pair — `nn/output_expa.txt`
  line 21 ("EXPLORATORY primary pair wrapped_ind vs phase_only_new
  [zero-crush]: wrapped_ind-only 27, phase_only_new-only 2,
  p=0.0000 (raw, no Holm)") and `nn/decisions_expa.md` line 29.
  There is no `clock` arm in Experiment A (arms: `wrapped_ind`,
  `phase_only_new`, `oracle_ind`, `memory30`); "raw clock" in
  `decisions_expa.md` line 48 is informal shorthand for
  `phase_only_new`, the raw-clock arm — not a separate pair.

## 1. Hypothesis

**Directional, bounded:** on the fixed 30/14 task and protocol
(frozen code, 2000 episodes, hidden=32, lr=1e-3, hand-written NN
substrate), the `wrapped_ind` arm produces a higher zero-crush
rate than the `phase_only_new` arm. No causal language: a positive
result would license "the strongest explanation among the tested
alternatives is that the inferred schedule representation adds
safety the raw clock lacks on this task" — nothing about the
time-based model in general, nothing about speed or overall
superiority.

## 2. Arms and comparisons

**Primary (single): `wrapped_ind` vs `phase_only_new` on
zero-crush.** This is the preregistered pair from Experiment A —
the comparison the experiment was designed and powered around —
with a new endpoint. This is not silently adopting the exploratory
comparison: the pair was fixed before the exploratory signal
existed; only the endpoint changes, and the endpoint change is
itself the hypothesis (Experiment A's primary mixed safety with
speed and could not see a pure safety signal). The exploratory
contrast therefore becomes a pre-registered directional hypothesis
tested on independent data (see §4).

**No confirmatory secondaries** (owner ruling). The safety
question is exactly one comparison. Candidate secondaries
(`oracle_ind`, `memory30`) sit at the zero-crush ceiling in
Experiment A (38/40 and 40/40) and would add no information on
this endpoint. Per-arm zero-crush rates with Clopper–Pearson 95%
CIs are reported descriptively for the two primary arms only.
A single comparison means no multiplicity correction is needed.

## 3. Endpoint definition

**Zero-crush (binary, per seed):** 1 if the greedy-evaluation
`crush_rate == 0.0`, else 0 — mechanically, zero crushes across
all 30 greedy episodes (all 30 start phases × 1 rep, eps=0), from
the raw record field `greedy.crush_rate` (`nn/run_arms.py`:
`crush_rate = total crushes / n`). This is the exact flag the
exploratory test used (`analyze_expa.py`, exploratory section).

**How it differs from clean-and-fast:** clean-and-fast =
zero-crush AND success ≥ 0.9 AND mean_return ≥ 0.744. Zero-crush
isolates safety; clean-and-fast mixes safety with speed. The
draft treats zero-crush as the safety endpoint because the
exploratory signal lives entirely in the safety component.
**Useful property:** zero-crush is independent of the 0.744 bar —
no threshold choice is baked into the endpoint.

## 4. Seeds

**Fresh seeds, n=100 per arm (200 seeds total), paired by seed.**
Seed range fixed at freeze: **400–499**, disjoint from all prior
runs. Disjointness verified 2026-10-19 against every seed range in
tracked records: `results_a5.jsonl` 0–9; `results_replication.jsonl`
and `results_controls.jsonl` 10–49; `results_pilot.jsonl` and
`results_stage1.jsonl` 100–104; `results_regression.jsonl` 100–109;
`results_expa_1.jsonl` / `results_expa_2.jsonl` 200–299;
`results_trust_rerun.jsonl` 200–202. Range 400–499 overlaps none.

**Why not reuse Experiment A's seeds 200–299:** the exploratory
signal was computed on those seeds. Re-testing the same contrast
on the same seeds is circular — a "confirmation" there would
re-describe the exploratory finding, not test the hypothesis on
independent data. The confirmatory label requires data the
hypothesis has not seen. (A descriptive re-statement on old seeds
adds nothing inferential and is not proposed.)

**Why 100:** owner ruling; the power sketch below shows what
100 buys under two-sided testing.

## 5. Statistical plan

**Primary test:** paired exact McNemar on the binary zero-crush
flags, **two-sided** at α=0.05 (owner ruling). The hypothesis is
directional (wrapper safer); a rejection with the observed
direction favoring the wrapper confirms it. A rejection in the
opposite direction would be reported as an unexpected reversed
effect, not as confirmation. Direction (sign of b−c) is reported
with the p-value.

**Power sketch** (two-sided exact McNemar, α=0.05; effect-size
basis is the `phase_only_new` pair's own exploratory numbers —
27/29 discordant, within-discordant P(wrapped-only) = 0.931 —
per `nn/output_expa.txt` line 21, cited in Background; n_d fixed
at its expectation — a sketch, not a guarantee):

| N seeds/arm | expected n_discord | power at 0.931 (observed) | power at 0.80 | power at 0.70 |
|---|---|---|---|---|
| 60 | 17 | 0.995 | 0.758 | 0.389 |
| 80 | 23 | 0.999 | 0.840 | 0.440 |
| **100 (adopted)** | **29** | **1.000** | **0.892** | **0.479** |

At N=100, power is 1.000 at the observed effect; 0.892 at 0.80 —
just under 0.9, reported plainly. Power falls off for effects at
0.70 and below: the design is powered for effects at or near the
exploratory size, not for small effects.

**Missing-data rule** (mirrors Experiment A): a seed with a
missing or incomplete record is excluded from the paired test —
unscored ≠ zero, unscored ≠ crush-free. Descriptives use all
available records; any exclusions are reported with counts. The
number of discordant pairs n_d is reported; if n_d is tiny the
test has no power — that is reported as-is, not rescued by
post-hoc analysis.

## 6. Frozen-analysis rule

- **New script `nn/analyze_zerocrush.py`** — not an extension of
  `nn/analyze_expa.py`. That file is the frozen record of
  Experiment A's analysis at `6c1f9e0`; extending it would blur
  what was frozen when. The new script implements exactly §5
  (paired one-sided exact McNemar, Clopper–Pearson descriptives,
  missing-data rule, deterministic output).
- **Validation bar:** `nn/validate_analyze_zerocrush.py` with
  synthetic-data checks mirroring `validate_analyze_expa.py`
  (McNemar vs scipy, one-sided boundary cases, missing-data
  behavior, determinism) — passing before any real records exist.
- **Timestamped gate:** prereg + frozen analysis code committed
  and pushed to `origin/main` **before** any run produces
  results — the same gate that governed Experiment A
  (prereg 888355c before runs; analysis 6c1f9e0 before results).

## 7. Interpretation bounds

- **Positive (two-sided p < 0.05, direction favoring the
  wrapper):** licenses "on the 30/14 task and fixed protocol, the
  wrapper is safer than the raw clock" — task-and-protocol
  bounded, strongest-explanation-among-tested-alternatives
  posture. It would **not** license: superiority of the time-based
  model in general; any claim about speed (Experiment A's primary
  already showed no clean-and-fast difference); causal language
  about *why* the wrapper is safer.
- **Negative (p ≥ 0.05):** failure to confirm — not proof of no
  safety difference. The exploratory signal may have been noise,
  seed-specific, or smaller than the sketch assumed.
- **Inconclusive:** e.g. far fewer discordant pairs than the
  sketch assumed, leaving the test underpowered — reported as
  underpowered, not reinterpreted.
- No post-hoc endpoints, no re-barring, no subgroup tests. The
  binary flag is computed from raw records; the bar plays no
  role.

## Open questions for owner review

All three v0.2 open questions are resolved by the 2026-10-19
rulings (two-sided; 100 seeds; no `oracle_ind` secondary).
Remaining: none — the draft is ready for the freeze work order.

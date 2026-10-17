# Preregistration DRAFT v3.1: generalization — unseen schedules and
schedule changes (EBI priority 3)

**Status: DRAFT v3.1. Do not run the main experiments until
owner-approved. Requires new approval and new code freeze.**
**Revises v3 per owner review (2026-10-17).**

## Pilot log (v2 pilot — already ran)

**Correction to prior messages:** a 25-seed feasibility pilot ran
under v2 (approved 2026-10-16 14:49, code frozen 023317f) before
v3 was written. The "nothing has run" statement in the v3 delivery
message was wrong — it meant the main experiments, but should
have said so. This log corrects the record.

**What ran:** seeds 100–104, arms wrapped_ind/memory10/memory20/
memory30/oracle_new, 30/14, 2000 episodes, default 0.75 bar
(`nn/results_pilot.jsonl`, 25 records).

**Results (greedy; 0.75 bar):**

| Arm         | Clean-and-fast | Median return | Zero-crush | Success |
|-------------|----------------|---------------|------------|---------|
| wrapped_ind | 3/5            | 0.753         | 5/5        | 5/5     |
| memory10    | 0/5            | 0.724         | 5/5        | 5/5     |
| memory20    | 0/5            | 0.726         | 5/5        | 5/5     |
| memory30    | 0/5            | 0.740         | 5/5        | 5/5     |
| oracle_new  | 0/5            | 0.655         | 4/5        | 5/5     |

**Halt:** v2's rule (0/5 clean-and-fast AND median < naive 0.736)
tripped for memory10, memory20, and oracle_new. Per the rule, the
pilot halted — full runs were not launched.

**Notes:**
- v3's success-based rule (<3/5 seeds with success≥0.9) would have
  passed every arm (all 5/5). The owner proposed this rule from
  reasoning about the 20/10 plain arm **before** knowing the pilot
  results; both facts belong in the record.
- `wrapped_ind` 3/5 with 5/5 zero-crush fits the structural story:
  it has the clock plus inference.
- History arms at 0/5 clean-and-fast but 5/5 zero-crush and 5/5
  success fits the structural ceiling (~0.743 < 0.744 bar):
  safe but not fast enough.
- **`oracle_new` is not an upper bound.** Median 0.655, 4/5
  zero-crush, slow runs. It uses different scales (phase/30,
  remaining/14) than wrapped_ind (remaining capped 5/5, phase/50),
  so any wrapper-vs-oracle gap confounds encoding with inference.
  v3.1 adds `oracle_ind` (same encoding, true info) as the clean
  upper bound; `oracle_new` keeps its "upper bound" label
  **suspended** pending diagnosis.
- Stale label: the pilot output header said "20 start phases"
  (fixed in code; now prints the actual period).

**v3 was written after this pilot.** The revisions in v3/v3.1
(including the success-based rule) were proposed from the
owner's independent reasoning, but the pilot data was visible
before v3.1 was finalized. This document records that sequence
honestly.

## Yardsticks (recomputed with parametrized repo code)

`nn/compute_yardsticks.py`, committed at f2b6082:

| Schedule | Naive | Time-aware | Delay-line | Headroom | fast = script output |
|----------|-------|------------|------------|----------|----------------------|
| 20/10    | 0.737 | 0.787      | 0.725      | 0.050    | **0.750**            |
| 30/14    | 0.736 | 0.769      | 0.675      | 0.033    | **0.744**            |

**The fast threshold is exactly the script's output: 0.744 on
30/14** (not 0.745). Every occurrence in this document and the
code uses 0.744. (0.744 = 0.736 + 0.26×0.033 = 0.74458,
truncated to the script's print precision.)

## Primary endpoint

**Clean-and-fast**: greedy 0 crushes, success ≥ 0.9, return ≥
**0.744** on 30/14.

## Experiment A: unseen schedule (30/14)

### Construction problem (as v3)

History-only arms are structurally disadvantaged (ceiling
~0.743 < 0.744 bar). Beating them is expected by construction.

### Arms

- `wrapped_ind`: ScheduleModel + schedule-independent encoding.
  104 inputs.
- `phase_only_new`: (tick%30)/30, no inference. **Primary
  comparison.**
- `oracle_ind`: true phase/remaining with wrapped_ind's encoding
  (phase/50, remaining capped 5/5). **Clean upper bound and
  clean test of inference** — same encoding as wrapped_ind,
  perfect info vs inferred info.
- `memory10` / `memory20` / `memory30`: secondary, with the
  structural-disadvantage interpretation pre-registered.
- `oracle_new`: secondary, "upper bound" label **suspended**
  pending diagnosis of the pilot anomaly.

### Primary comparison

`wrapped_ind` vs `phase_only_new` (paired, McNemar exact).

### Secondary

vs `oracle_ind` (the inference test), vs memories, vs
`oracle_new` (diagnostic). Holm across 6 comparisons.
Equivalence margin **±0.25**; "indistinguishable" as third
outcome.

## Experiment B: mid-run schedule change (20/10 → 30/14)

### Primary endpoint (single)

**Mean return over post-change episodes 1–300** (paired
wrapped_ind vs phase_only_new-stale-variant and vs histories,
Holm-corrected). Return already includes the −1 per crush, so
crushes are descriptive, not a second primary.

### Arms

wrapped_ind, phase_only_new (adapts to 30-clock),
phase_only_stale (20-clock, sanity check), memory10/20/30,
oracle_stale (sanity check), oracle_ind (upper bound, switches
to true 30/14 info), oracle_new (diagnostic).

### Buffer handling

300 episodes; buffer-clear variant via **checkpoint forking**
(train to switch once, fork two continuations). Greedy eval
after window. Episode-length report (fraction ≥ 31 ticks).

## Scaling

As v3. `oracle_ind` uses wrapped_ind's exact scales.

### Regression check

`wrapped_ind` on 20/10, seeds 100–109, vs verified `wrapped`
40-seed result (26/40 cf, 38/40 zc). Pass: ≥4/10 clean-and-fast
(fast=0.750 on 20/10) AND ≥8/10 zero-crush.

## Feasibility pilot (v3.1 — new pilot after code freeze)

Seeds 100–104, all Experiment A arms **including oracle_ind
and phase_only_new**, 30/14, fast=0.744 (via --fast-threshold).

**Failure rule:** fewer than 3/5 seeds with success ≥ 0.9.
If any arm fails: halt, diagnose.

Note: this is a **second** pilot (the v2 pilot is logged
above). It validates the new arms and the 0.744 bar.

## Stage 2 gate (pre-registered before the pilot runs)

Proceed to Stage 2 (full Experiments A and B) **only if all three
hold:**

1. **Success rule:** every pilot arm has at least 3 of 5 seeds with
   greedy success ≥ 0.9.
2. **Regression:** `wrapped_ind` on 20/10 (seeds 100–109) reaches at
   least 4/10 clean-and-fast (fast = 0.750) and at least 8/10
   zero-crush, against the verified `wrapped` 40-seed reference
   (26/40 clean-and-fast, 38/40 zero-crush).
3. **`oracle_ind` as upper bound:** `oracle_ind`'s median greedy
   return ≥ `wrapped_ind`'s median greedy return − 0.02, AND at
   least 4/5 `oracle_ind` seeds have zero crushes.

**If condition 3 fails: diagnose first. Do not relabel the arm and
do not proceed to Stage 2.**

This gate was written before the pilot ran (owner order,
2026-10-17); the "behaves as an upper bound" judgment is not made
after seeing the data.

## Seeds, protocol, analysis, interpretation

As v3 (one primary per experiment, Holm, ±0.25, intervals
only, cautious language, missing-data rule).

## Code

- nn/agent_generalization.py: + OracleIndEncoder (structural
  check PASS).
- nn/run_arms.py: + oracle_ind arm; --fast-threshold; fixed
  "20 start phases" label (now prints actual period).
- nn/compute_yardsticks.py: committed.

## Compute estimate

- Pilot v3.1: 35 seeds ≈ 1.3 h. Regression: 10 seeds ≈ 0.4 h.
- Exp A: 280 seeds (7 arms) ≈ 8 h (4 h × 2).
- Exp B: 8 arms × 40, forked ≈ 9 h × 2 processes.
- Total ≈ 14 h with 2 processes.

Approved: owner, 2026-10-17 (Stage 1 only)  Date: 2026-10-17
Frozen code commit: 888355c

## Stage 1 approval record (owner, 2026-10-17)

**Approved scope: Stage 1 only.** The v3.1 feasibility pilot and the
20/10 regression check. Stage 2 (full Experiments A and B) is NOT
approved by this decision.

**Exact arms and seeds:**

- Feasibility pilot, 30/14, fast = 0.744 (`--fast-threshold 0.744`):
  `wrapped_ind`, `memory10`, `memory20`, `memory30`, `oracle_new`,
  `phase_only_new`, `oracle_ind` × seeds 100–104
  (7 arms × 5 seeds = 35 seeds).
- Regression check, 20/10, fast = 0.750 (`--fast-threshold 0.750`):
  `wrapped_ind` × seeds 100–109 (10 seeds).

**Frozen code commit: 888355c.** This approval commit changes only
`nn/prereg_generalization.md`; all executable code is byte-identical
to 888355c (verified by diff before commit).

**Reuse option (pre-authorized):** the five v2-pilot arms
(`wrapped_ind`/`memory10`/`memory20`/`memory30`/`oracle_new`, seeds
100–104, `nn/results_pilot.jsonl`, committed at c70cabb) may be
reused instead of re-run, provided the reuse gate passes first:
re-run `wrapped_ind` seed 100 at the frozen code and require
bit-for-bit reproduction of the recorded greedy evaluation
(mean_return 0.7526666666666666, crush_rate 0.0). If it fails, do a
full re-run instead. Note: new records carry `fast_threshold` in
their hyperparams while the v2 records do not; this metadata
difference does not affect training (the threshold only enters the
clean-and-fast summary classification).

**Stage 2 gate:** proceed to Stage 2 only if all three gate
conditions in "Stage 2 gate" above hold. If condition 3 fails,
diagnose first; do not relabel the arm and do not proceed.

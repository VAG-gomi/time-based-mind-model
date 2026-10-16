# Preregistration DRAFT v2: generalization — unseen schedules and
schedule changes (EBI priority 3)

**Status: DRAFT v2. Do not run until owner-approved.**
**Revises v1 per owner review (2026-10-16): fixed fast threshold,
buffer handling, schedule-independent encoding, primary
comparisons, Holm correction, equivalence margin, feasibility
pilot.**

## Research questions

1. Does the wrapped arm's advantage survive on a schedule it was not
   designed around — specifically one where a 10-step history cannot
   cover the open interval?
2. When the schedule changes mid-run, does the inferred schedule
   representation (with contradiction detection) adapt better than
   fixed oracle features or raw history?

## Yardsticks on 30/14 (owner-simulated, all 30 start phases)

- Naive: 0.736 return, crushed in 1/30 phases.
- Time-aware (ceiling): 0.769.
- Delay-line: 0.675.
- Headroom (time-aware − naive): 0.033 (was 0.05 at 20/10).

## Primary endpoint

**Clean-and-fast**, same structure as before: greedy 0 crushes,
success ≥ 0.9, return ≥ **fast**, where

  fast = naive + 0.26 × (time-aware − naive)

- On 30/14: 0.736 + 0.26 × 0.033 = **0.745**.
- On 20/10: 0.737 + 0.26 × 0.050 = **0.750** (exactly; confirms
  the 0.26 fraction reproduces the original bar).

The 0.26 fraction keeps the definition constant across schedules:
the bar sits 26% of the way from naive to the time-aware ceiling.
Headroom on 30/14 is small (0.033), so expect low power — stated
here, not discovered later.

## Experiment A: unseen schedule (30/14)

Train from scratch on period 30, open_len 14.

### Arms (hidden=32, lr=1e-3, 2000 episodes, same rewards/gamma)

- `wrapped_ind`: ScheduleModel wrapper with **schedule-independent
  encoding** (see Scaling). 104 inputs.
- `memory10`: last 10 door bits. 111 inputs. (Cannot cover the
  14-tick open interval by construction.)
- `memory20`: last 20 door bits. 121 inputs.
- `memory30`: last 30 door bits. 131 inputs. Covers the full period.
- `oracle_new`: TRUE phase (tick%30)/30, TRUE remaining/14,
  known=1. Upper bound.

### Primary comparison

`wrapped_ind` vs `memory30` (paired clean-and-fast difference).
Beating `memory10` is expected by construction (it cannot see the
full open window) and does not count as evidence for the
representation.

### Secondary comparisons

`wrapped_ind` vs `memory20`, vs `memory10`, vs `oracle_new`
(paired, same endpoint). Zero-crush, success, and median-return
descriptives as before.

### Multiple comparisons

Holm correction applied across the 4 paired primary/secondary
clean-and-fast comparisons in Experiment A. Reported: raw p,
Holm-adjusted p, and the paired 95% CI (CIs are not
multiplicity-adjusted; the correction applies to the p-values).

### Equivalence

"≈" is replaced by a pre-registered equivalence margin:
|paired clean-and-fast difference| < **0.15** with the 95% CI
fully inside [−0.15, +0.15] supports the interpretation of
practical equivalence. At n=40 the CIs span roughly ±0.3, so
equivalence will often be undecidable — that outcome is
reported as-is.

## Experiment B: mid-run schedule change (20/10 → 30/14)

Train on 20/10 for 2000 episodes, switch to World(30, 14),
continue the **same** agent+encoder.

### Buffer handling (pre-registered)

The replay buffer holds 10,000 transitions, mostly from the old
schedule after the switch. To measure adaptation rather than
buffer dynamics:

- Run **300 post-change episodes** (not 100): ~9,000+ new
  transitions, enough to substantially refill the buffer.
- **Documented variant**: clear the replay buffer for ALL arms at
  the switch (same handling, no arm advantaged). Both variants
  are pre-registered; the primary analysis uses the no-clear
  variant, the clear variant is reported secondarily.
- After the 300-episode window, run a **greedy evaluation** on
  30/14 (all 30 start phases), since the windows themselves use
  ε=0.05 training episodes.

### Arms

- `wrapped_ind`, `memory10`, `memory20`, `memory30` (as in A).
- `oracle_stale`: 20/10-based features throughout (wrong fixed
  knowledge after the switch).
- `oracle_new`: 30/14-based features from the switch onward
  (upper bound).

### Metrics

- Per-window mean crushes and mean return: episodes 1–50,
  51–150, 151–300 post-change (wider windows for the longer
  horizon), per arm, 95% bootstrap CIs (resampling seeds).
- Greedy clean-and-fast on 30/14 after the window (primary
  endpoint definition, fast=0.745).
- For `wrapped_ind`: contradiction count, first episode
  post-change with stable 30/14 estimates.
- **Episode-length report**: fraction of post-change episodes
  with ≥ 31 ticks (minimum to observe two 30/14 openings).
  Pre-registered because the model cannot learn the new period
  in shorter episodes.

### Primary comparison

Greedy clean-and-fast post-change: `wrapped_ind` vs `memory30`
(paired). Secondary: vs `memory20`, vs `oracle_stale`
(Holm-corrected with the primary).

## Scaling (both experiments)

No arm may use the true schedule's constants as feature scales —
except `oracle_new`, the explicit upper bound.

- `wrapped_ind`: **schedule-independent encoding**:
  - remaining_capped = min(open_ticks_remaining, 5) / 5.0.
    "2 ticks left" (minimum safe crossing margin) is 0.4 —
    a stable, schedule-free threshold. Capping at 5 keeps the
    feature in [0,1] on any schedule.
  - phase = ticks_since_opening / 50.0. Fixed generic divisor,
    larger than any tested period, not tuned to any schedule.
  - known flag as before (1 iff both features available).
  - This replaces the v1 live-estimate scaling, which made
    "2 ticks left" worth 0.2/0.14/0.02 depending on open_len —
    hiding the number that matters.
- `oracle_new`: permitted the true constants (upper bound).
- `oracle_stale`: 20.0/10.0 throughout.
- Histories: no scaling.

### Regression check (pre-registered)

`wrapped_ind` is a new arm. Before the big runs, verify it on
20/10 against the verified `wrapped` arm: seeds 100–104,
paired clean-and-fast. Expect equivalence (|diff| < 0.15);
if not equivalent, halt and diagnose — do not proceed.

## Feasibility pilot (pre-registered, before the full runs)

Seeds **100–104** on the final code, all Experiment A arms,
30/14, 2000 episodes. Confirms every arm can learn on 30/14
within the episode budget (longer episodes make learning harder;
this is not guaranteed). If any arm fails to learn (0/5
clean-and-fast AND median return below naive), halt and
diagnose — do not launch the 40-seed runs.

The pilot also reports the fraction of episodes ≥ 31 ticks
(informative for Experiment B's episode-length assumption).

## Seeds and protocol

- Pilot: seeds 100–104 (5 arms × 5 seeds = 25 seeds).
- Regression: seeds 100–104, wrapped_ind vs wrapped on 20/10.
- Experiment A: 5 arms × 40 seeds (10–49) = 200 seeds.
- Experiment B: 6 arms × 40 seeds (10–49) = 240 seeds ×
  2300 episodes each.
- Evaluation: greedy + eps=0.05 over all 30 start phases (30/14).
- Output: nn/results_generalization.jsonl,
  nn/output_generalization.txt.
- **Two parallel processes** to halve wall time (arms split
  across processes; seeds within an arm are independent).

## Analysis plan (intervals only, as before)

As in the replication: Clopper-Pearson, McNemar exact paired
CIs, 10,000-seed bootstrap (seed=0) for median diffs.
Experiment B adds the windowed adaptation metrics above.
One primary endpoint per experiment; Holm correction as
specified; equivalence margin ±0.15.

## Interpretation rules (pre-registered, cautious language)

- Exp A primary (`wrapped_ind` vs `memory30`, CI excludes zero):
  supports the interpretation that the inferred schedule
  representation adds value beyond a long history when the
  history must span the full period.
- Exp A equivalence (|diff| < 0.15, CI inside margin): supports
  the interpretation that a long-enough raw history remains
  sufficient.
- Exp B: `wrapped_ind` recovering (crushes → 0) faster than
  `oracle_stale`: supports the interpretation that
  contradiction-detection + re-learning adapts better than
  stale fixed features.
- All statements task-and-protocol bounded; none proves
  causality or generalizes beyond the tested schedules.

## Missing-data rule

Same as replication: incomplete pairs excluded from paired
analysis; available records retained descriptively.

## Code to be written (after approval)

- nn/agent_generalization.py: WrappedIndEncoder, Memory20Encoder,
  Memory30Encoder, OracleNewEncoder, OracleStaleEncoder;
  structural checks.
- nn/run_arms.py: wire new arms; --period/--open_len support;
  Experiment B continuation + buffer-clear variant.
- nn/analyze_generalization.py: deterministic regeneration.

## Compute estimate

- Pilot + regression: 25 + 10 seeds ≈ 1 hour.
- Experiment A: 200 seeds ≈ 6 hours (3 h with 2 processes).
- Experiment B: 240 seeds × 2300 ep ≈ 7.5 hours (3.75 h × 2).
- Total ≈ 14–15 hours serial, ~7–8 hours with 2 processes.

Approved: owner (via chat, 2026-10-16, v2).  Date: 2026-10-16.
Frozen code commit: ___________ (to be filled when code is frozen)

# Preregistration DRAFT v3: generalization — unseen schedules and
schedule changes (EBI priority 3)

**Status: DRAFT v3. Do not run until owner-approved.**
**Revises v2 per owner review (2026-10-16): phase_only primary,
fixed regression/equivalence rules, B speed endpoint,
success-based pilot rule, recomputed yardsticks, checkpoint
forking.**

## Yardsticks (recomputed with parametrized repo code)

`nn/compute_yardsticks.py` rebinds the repo's yardstick policies
for arbitrary (period, open_len). Committed; outputs:

| Schedule | Naive | Time-aware | Delay-line | Headroom | fast (0.26) |
|----------|-------|------------|------------|----------|-------------|
| 20/10    | 0.737 | 0.787      | 0.725      | 0.050    | **0.750**   |
| 30/14    | 0.736 | 0.769      | 0.675      | 0.033    | **0.745**   |

The 0.745 threshold is now derived from repo code, not the owner's
sandbox simulation. (Owner's numbers — 0.736/0.769/0.675 — match
exactly.)

## Primary endpoint

**Clean-and-fast**: greedy 0 crushes, success ≥ 0.9, return ≥
**0.745** on 30/14 (0.750 on 20/10, by the same 0.26 fraction).

## Experiment A: unseen schedule (30/14)

### The construction problem (stated pre-registration)

History-only arms start each episode with no clock and no
persistent schedule knowledge. Owner simulation: the best a
history-only agent can do on 30/14 is ~0.743 (wait for a fresh
opening) or 0.736 (cross on a guess). The fast bar is 0.745.
**History arms are therefore structurally disadvantaged**:
beating them is expected by construction and does not count as
evidence for the representation. Only the wrapper has the clock
(tick) plus a persistent schedule model.

### Arms (hidden=32, lr=1e-3, 2000 episodes, same rewards/gamma)

- `wrapped_ind`: ScheduleModel + schedule-independent encoding
  (remaining capped 5/5, phase/50). 104 inputs.
- `phase_only_new`: clock only — (tick % 30)/30, no inference, no
  history, no model. 102 inputs. **Primary comparison.** This arm
  matched the wrapper at 20/10; the question is whether inference
  adds anything beyond the raw clock on a harder schedule.
- `memory10`: last 10 door bits. 111 inputs. (Secondary;
  structurally disadvantaged as above.)
- `memory20`: last 20 door bits. 121 inputs. (Secondary.)
- `memory30`: last 30 door bits. 131 inputs. (Secondary.)
- `oracle_new`: true phase/remaining for 30/14. Upper bound.
  (Secondary.)

### Primary comparison

`wrapped_ind` vs `phase_only_new` (paired clean-and-fast
difference, McNemar exact). If inference adds nothing beyond the
clock, these two tie. If the schedule model matters, wrapped
wins.

### Secondary

Paired vs memory10/20/30 and oracle_new, with the
structurally-disadvantaged interpretation pre-registered for the
history arms. Holm correction across the 5 paired comparisons.
Zero-crush, success, median-return descriptives.

### Equivalence

Margin widened to **±0.25** (v2's ±0.15 is undecidable at n=40;
the earlier wrapped-vs-oracle interval was 0.45 wide, ~20
discordant pairs). |diff| < 0.25 with 95% CI inside [−0.25,
+0.25] supports practical equivalence; otherwise report
"indistinguishable" (not "different," not "same").

## Experiment B: mid-run schedule change (20/10 → 30/14)

### Primary endpoint (aligned with the question)

The question is whether the wrapper **recovers faster**, not its
end state. Primary: **total crushes and mean return over
post-change episodes 1–300**, paired `wrapped_ind` vs
`phase_only_stale` and vs `memory10`/`memory20`/`memory30`
(Holm-corrected). `oracle_stale` is a **sanity check** (expected
to fail by construction — wrong fixed features), not a
competitor.

Secondary: greedy clean-and-fast on 30/14 after the window
(fast=0.745); windowed crushes/return (1–50, 51–150, 151–300);
contradiction/re-convergence descriptives; episode-length
report (fraction ≥ 31 ticks).

### Arms

- `wrapped_ind`, `memory10`, `memory20`, `memory30`
- `phase_only_new`: switches to (tick%30)/30 at the change.
  (The clock adapts trivially; the question is whether
  inference beats the raw clock.)
- `phase_only_stale`: stays on (tick%20)/20 after the change.
  (Sanity check: stale clock.)
- `oracle_stale`: 20/10 features throughout. (Sanity check.)
- `oracle_new`: 30/14 features from the switch. (Upper bound.)

### Buffer handling

- 300 post-change episodes primary; buffer-clear variant
  secondary (same handling for all arms).
- **Checkpoint forking**: train each (arm, seed) to the switch
  point once (2000 episodes on 20/10); save the agent+encoder
  state; fork two continuations (no-clear and clear-buffer).
  Each variant costs only the 300 post-change episodes, not a
  full retrain.

## Scaling

As v2: `wrapped_ind` uses remaining min(x,5)/5.0 and phase/50.0
(fixed generic scales). `oracle_new` alone may use true
constants. Histories and clocks need no scaling.

### Regression check (revised)

v2's 5-seed |diff|<0.15 rule cannot work at n=5. Revised:
`wrapped_ind` on 20/10, seeds 100–109 (10 seeds), compared
against the verified `wrapped` arm's 40-seed replication result
(26/40 clean-and-fast, 38/40 zero-crush). Pass if **≥4/10
clean-and-fast AND ≥8/10 zero-crush**. This checks the new
encoding doesn't break the known-good behavior; it is not an
equivalence test.

## Feasibility pilot (revised)

Seeds 100–104, all Experiment A arms, 30/14, 2000 episodes.

**Failure rule (success-based, per owner review):** an arm
fails if **fewer than 3 of 5 seeds reach success ≥ 0.9**.
(Rationale: the plain arm at 20/10 "learned" with 6/10
success≥0.9 but median return 0.712 < naive 0.737 — a
return-based rule trips on working arms when headroom is
small. Success measures whether the task is learned at all.)

If any arm fails by this rule: halt, diagnose, do not launch
40-seed runs.

## Seeds and protocol

- Pilot: seeds 100–104 (6 arms × 5 = 30 seeds).
- Regression: seeds 100–109, wrapped_ind on 20/10 (10 seeds).
- Experiment A: 6 arms × 40 seeds (10–49) = 240 seeds.
- Experiment B: 8 arms × 40 seeds (10–49) = 320 seeds ×
  2300 episodes, forked from checkpoints.
- Evaluation: greedy + eps=0.05 over 30 start phases (30/14).
- Two parallel processes.

## Analysis plan (intervals only)

As before: Clopper-Pearson, McNemar exact paired CIs,
10,000-seed bootstrap (seed=0). One primary endpoint per
experiment; Holm correction as specified; equivalence margin
±0.25 with "indistinguishable" as the third outcome.

## Interpretation rules (cautious)

- Exp A primary (wrapped_ind vs phase_only_new, CI excludes
  zero): supports inference adding value beyond the raw clock.
- Exp A equivalence/indistinguishable: supports the clock
  being sufficient; inference not needed.
- Exp B primary (faster recovery vs phase_only_stale and
  histories): supports contradiction-detection adapting better
  than fixed features or raw history.
- All task-and-protocol bounded; none proves causality.

## Missing-data rule

As before.

## Code to be written (after approval)

- `phase_only_new` / `phase_only_stale` encoders in
  nn/agent_generalization.py (parametrized by period).
- Checkpoint save/fork in run_arms.py for Experiment B.
- nn/analyze_generalization.py.

## Compute estimate (revised)

- Pilot: 30 seeds ≈ 1.2 h. Regression: 10 seeds ≈ 0.4 h.
- Exp A: 240 seeds ≈ 7 h (3.5 h × 2 processes).
- Exp B: 320 seeds, but forked: 320×2000 ep (11 h) +
  640×300 ep variants (6.5 h) ≈ 17.5 h serial, ~9 h × 2.
- Total ≈ 26 h serial, ~13 h with 2 processes.

Approved: _________________  Date: ___________
Frozen code commit: ___________

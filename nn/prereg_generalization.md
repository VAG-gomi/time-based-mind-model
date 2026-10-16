# Preregistration DRAFT: generalization — unseen schedules and
schedule changes (EBI priority 3)

**Status: DRAFT. Do not run until owner-approved.**

## Research questions

1. Does the wrapped arm's advantage survive on a schedule it was not
   designed around — specifically one where a 10-step history cannot
   cover the open interval?
2. When the schedule changes mid-run, does the inferred schedule
   representation (with contradiction detection) adapt better than
   fixed oracle features or raw history?

## Experiment A: unseen schedule (30/14)

Train from scratch on period 30, open_len 14. A 10-step history
cannot cover the 14-tick open interval, and the 10-tick window is
much shorter than the 30-tick period. The ScheduleModel, which
learns the period from observed openings, should have a structural
advantage here — if the representation matters beyond just "enough
temporal context."

### Arms (all: hidden=32, lr=1e-3, 2000 episodes, same rewards/gamma)

- `wrapped_live`: ScheduleModel wrapper with **live scaling** (see
  Scaling below). 104 inputs.
- `memory10`: last 10 door bits. 111 inputs.
- `memory30`: last 30 door bits. 131 inputs. Covers the full period;
  tests whether a long-enough history matches the wrapper.
- `oracle_new`: TRUE phase (tick%30)/30 and TRUE remaining/14 for the
  30/14 schedule, known=1. Upper bound: perfect knowledge of the new
  schedule.

### Seeds

10–49 (same seeds for implementation consistency; not paired with
20/10 results — different task).

## Experiment B: mid-run schedule change (20/10 → 30/14)

Train all arms on 20/10 for 2000 episodes (standard protocol), then
switch to a fresh World(30, 14) and continue training the **same**
agent+encoder for 100 further episodes. Measure adaptation.

### Arms

- `wrapped_live`: ScheduleModel persists across the switch. Its
  consistency check should fire (contradictions), drop stale
  events, and re-learn the 30/14 schedule.
- `memory10`, `memory30`: history buffers refill with new-schedule
  observations naturally.
- `oracle_stale`: continues providing 20/10-based phase/remaining
  features after the switch. Tests wrong fixed knowledge.
- `oracle_new`: switches to 30/14-based features at the change.
  Upper bound: perfect knowledge of the new schedule, instantly.

### Metrics (first 100 episodes post-change)

- Mean crushes per episode in windows: episodes 1–20, 21–50, 51–100
  post-change.
- Mean return in the same windows.
- For `wrapped_live`: contradiction count and re-convergence time
  (first episode post-change with stable 30/14 estimates).

## Scaling (both experiments)

No arm may use the true schedule's constants (30, 14) as feature
scales — except `oracle_new`, which is explicitly the
perfect-knowledge upper bound.

- `wrapped_live`: phase / period_hat and remaining / open_len_hat,
  where period_hat and open_len_hat are the ScheduleModel's live
  estimates. While the period is unknown (partial knowledge),
  phase is scaled by GENERIC_SCALE = 50.0 (fixed, documented,
  larger than any schedule in the test suite — not tuned to any
  specific schedule). The known flag is 1 iff both features are
  available, as before.
- `oracle_new`: the only arm permitted the true constants.
- `oracle_stale`: uses 20.0/10.0 (the old schedule) throughout.
- `memory10`/`memory30`: no scaling involved.

## Wrapper limits and assumptions (stated pre-registration)

1. **Episode length.** The ScheduleModel learns a period from clean
   opening-to-opening spans. With EPISODE_CAP=200, a period of 30
   requires 31 consecutive observed ticks — available. Probe
   result (owner): with 21-tick episodes the model never learns
   period 20 from scratch, because two openings rarely fall in one
   episode without a gap. The experiments use cap 200 throughout;
   results are conditional on episodes long enough to contain two
   openings.
2. **Exploration.** The model learns only from observed
   transitions. If the agent never visits the door region, no
   schedule is learned. Results are conditional on the epsilon-greedy
   exploration (1.0 → 0.05 over 1000 episodes) producing door
   observations.
3. **Contradiction recovery.** After a schedule change, the model
   drops all events on the first contradiction and re-learns from
   scratch. There is no blending of old and new estimates.
4. **Partial knowledge.** Before the period is learned, the wrapper
   provides remaining-time from open_len + a clean opening (no
   period needed). This path is preserved in `wrapped_live`.

## Seeds and protocol

- Experiment A: 5 arms? No — 4 arms (wrapped_live, memory10,
  memory30, oracle_new) × 40 seeds (10–49) = 160 seeds.
- Experiment B: 5 arms (wrapped_live, memory10, memory30,
  oracle_stale, oracle_new) × 40 seeds = 200 seeds, each with
  2000 + 100 episodes.
- Same evaluation: greedy + eps=0.05 over all 20 (Exp A: 30)
  start phases. Note: start phases must cover the new period
  (30 phases for 30/14).
- Output: nn/results_generalization.jsonl, nn/output_generalization.txt.

## Analysis plan (intervals only, as before)

Experiment A (same as replication):
1. Clean-and-fast proportions with Clopper-Pearson 95% CIs.
   (Thresholds: 0 crushes, success ≥ 0.9, return ≥ 0.75 —
   same definition; the return ceiling differs on 30/14 and
   will be reported.)
2. Paired wrapped_live-minus-control differences: McNemar exact
   p + paired 95% CI.
3. Median return diffs: 10,000-seed bootstrap 95% CI (seed=0).
4. Zero-crush / success descriptive.

Experiment B (adaptation):
1. Per-window (1–20, 21–50, 51–100) mean crushes and mean return,
   per arm, with 95% bootstrap CIs (resampling seeds).
2. Paired differences wrapped_live vs each other arm per window.
3. Contradiction/re-convergence descriptives for wrapped_live.

## Interpretation rules (pre-registered, cautious language)

- If `wrapped_live` > `memory10` on 30/14 (paired CI excludes
  zero): supports the interpretation that the inferred schedule
  representation adds value beyond a short history when the
  history cannot cover the open interval.
- If `wrapped_live` ~= `memory30`: supports the interpretation
  that a long-enough raw history remains sufficient; the
  representation is not necessary.
- If `oracle_new` >> all: perfect knowledge still dominates, as
  expected; the question is the ranking among the non-oracle arms.
- Experiment B: if `wrapped_live` recovers (crushes → 0) faster
  than `oracle_stale` and `memory10`: supports the interpretation
  that contradiction-detection + re-learning adapts better than
  stale fixed features or short history.
- If `wrapped_live` recovers no faster than `memory30`: the
  adaptation advantage, if any, does not require the explicit
  model.
- All statements are task-and-protocol bounded; none proves
  causality or generalizes beyond the tested schedules.

## Missing-data rule

Same as replication: incomplete pairs excluded from paired
analysis; available records retained descriptively.

## Code to be written (after approval)

- nn/agent_generalization.py: WrappedLiveEncoder, Memory30Encoder,
  OracleNewEncoder, OracleStaleEncoder; structural checks.
- nn/run_arms.py: wire new arms; support --period/--open_len and
  post-change continuation for Experiment B.
- nn/analyze_generalization.py: deterministic regeneration of all
  reported statistics.

## Compute estimate

- Experiment A: 160 seeds ≈ 5 hours.
- Experiment B: 200 seeds × 2100 episodes ≈ 6.5 hours.
- Total ≈ 11–12 hours.

Approved: _________________  Date: ___________
Frozen code commit: ___________

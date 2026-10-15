# B2 — decisions and deviations

## The wrapped arm (agent_wrapped.py)

Same learner, training loop, and hyperparameters as the plain arm:
`train_neural()` from agent_nn is reused unchanged; only the encoder
differs. 104 inputs = 101 (position one-hot + door bit) + 3 time
features from a ScheduleModel.

**Information contract (stated and tested).** The ScheduleModel sees
ONLY `(tick, door_bit)`, exactly once per world observation. It never
sees position, reward, actions, or episode counters. One model
instance is created per `train_wrapped()` call and persists across
ALL training episodes and BOTH evaluations (greedy + eps=0.05): the
eval policy is built on the same encoder, so schedule knowledge
carries from training into evaluation. Tested by a counting-spy
(`_SpyModel`): 10 encodes -> exactly 10 `model.observe` calls, each
with exactly `(tick, door)`.

**Encoding justification.**
- phase = ticks_since_opening / 20.0, remaining =
  open_ticks_remaining / 10.0. Fixed scales (the world's period /
  open length) keep time inputs in [0, 1], commensurate with the
  one-hots; "2 ticks left" -- the minimum safe crossing margin
  (enter t, exit t+1) -- is 0.2, a clean threshold for tanh units.
  Fixed, not live estimates, because live scaling would make the
  input distribution shift as the model refines -- a moving target.
- known flag = 1 iff both time features are available (full
  estimates OR partial knowledge). It disambiguates 0.0 = "just
  opened / 0 left" from 0.0 = "unknown".
- No "safe to cross" bit: that would hand the agent the decision
  under test. The agent must learn the 2-tick threshold from reward.

**Partial knowledge (wrapper.py, tested in check 6).** Claude's
improvement is implemented: `open_ticks_remaining()` and
`ticks_since_opening()` fall back to partial knowledge when the
period is unknown -- open length + a cleanly observed opening (no
gaps since) is enough for exact remaining/phase. Verified: after one
short episode (open_len=10 known, period None), a fresh opening in
episode 2 with the door open gives remaining=9, phase=1 -- exact vs
ground truth -- while a stale cross-boundary opening correctly
yields None. Also added `first_estimates_tick` (first tick with full
estimates) to support the "first converged episode" metric.

## Verification (all before the full run)

1. Encoder structural check (`python3 agent_wrapped.py`): PASS --
   104-dim layout, once-per-observation (tick, door) via spy, known
   flag 0->1, feature slots track the model exactly, blanking zeros
   the slots, clock 102-dim layout.
2. Mutation test: blanked time features (slots forced to 0; extra
   weights multiply zeros, so the net sees exactly the plain arm's
   information), seed 0, full 2000 episodes -> greedy success 1.00,
   return 0.711, crush 0.05. Plain-arm-like: PASS. The wrapper
   machinery itself contributes nothing; the features are the active
   ingredient.
3. Smoke run (wrapped, 60 episodes): pipeline end-to-end PASS; model
   converged episode 0 tick 40, frac_known 0.999, 0 contradictions.
4. Clock smoke (60 episodes): pipeline PASS.

## The clock control

`ClockEncoder` (102 inputs): 101 + raw `tick/200.0`. Raw tick, NOT
tick % 20 -- handing the phase directly would duplicate the
wrapper's phase feature and confound the control. The network must
learn the periodic mapping itself. This tests whether the schedule
model adds anything beyond knowing the time. Included because it
was cheap (one small encoder, same runner).

## Results

INTERIM (8/20 seeds, run still in progress):
- wrapped 0: 0.765, 0 crush, 1.00 -> clean
- wrapped 1: 0.773, 0 crush, 1.00 -> clean
- wrapped 2: 0.755, 0 crush, 1.00 -> clean
- wrapped 3: 0.717, 0.05 crush, 1.00 -> not clean (1 crush)
- wrapped 4: 0.726, 0.05 crush, 1.00 -> not clean (1 crush)
- wrapped 5: 0.743, 0 crush, 1.00 -> clean
- wrapped 6: 0.771, 0 crush, 1.00 -> clean
- wrapped 7: 0.766, 0 crush, 1.00 -> clean
Clean-and-fast so far: 6/8. Needs 7/10 to win; seeds 8-9 decide.

[final tables after the background run]

## Skipped / deviations

- Nothing skipped. world.py untouched. No per-seed or per-arm
  tuning. The 7/10 clean-and-fast bar is unchanged from R.

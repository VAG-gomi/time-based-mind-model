# Preregistration: Trust Replay Layer 1.5 rematch (WO-TBM-L15)

**Status: FROZEN at the commit carrying this file. The rematch run
is NOT authorized by this document — it requires a separate
work order after freeze verification.**

## Background

Layer 1 (committed record, `nn/trust_rerun_note.md`,
`nn/results_trust_rerun.jsonl`) was an honest negative:
trust-weighted replay ~3x slower wall-clock than the uniform
baseline, reaching the rolling-0.8 criterion later or never on
every seed, worse final success on 2 of 3 seeds. Two candidate
explanations were named: (i) the naive buffer's O(N) sampling
cost per train step, and (ii) the missing importance-sampling
correction. Layer 1.5 tests whether the negative survives a
*correct and fast* implementation of the same design.

## Step-0 profile (measured, quoted for the record)

cProfile, seed 200, 200 episodes, buffer at capacity
(`nn/profile_trust.py`):

| arm | total | buffer | network | env | glue |
|---|---|---|---|---|---|
| uniform | 33.94s | 8.77s (25.9%) | 19.00s | 0.94s | 5.23s |
| trust (naive) | 55.29s | 29.62s (53.6%) | 19.05s | 1.04s | 5.58s |

Network time is identical across arms (19.0s) — the entire
profiled deficit is buffer cost. **Amdahl ceiling: post-fix trust
time cannot go below the trust arm's non-buffer time, 25.67s,
vs uniform's 33.94s — the ceiling clears parity with ~24%
headroom.** (Profiled times: cProfile compresses the bare ~3x
gap to ~1.6x because profiling overhead falls on the many small
Python calls in both arms. The rematch measures bare wall-clock;
the projection is a ceiling, not a promise.)

## Hypothesis (bounded, no causal language)

With the buffer cost removed (segment tree) and the IS
correction added, trust-weighted replay reaches wall-clock
parity-or-better with the uniform baseline on this protocol
while giving up no more than a small, pre-fixed amount of
final-100 success. Nothing here tests whether trust weighting
*helps* learning in general; it tests whether the Layer-1
negative was an artifact of implementation cost and a missing
correction, on this task and protocol only.

## Arms and protocol (identical to the Layer-1 re-run)

- Arms: `uniform` (`agent_nn.train_neural`, unchanged) vs
  `trust15` (`layer15_trust_train.train_trust15`).
- Seeds 200, 201, 202; 1000 episodes each; hidden=32, lr=1e-3,
  PlainEncoder, 20/10 protocol, epsilon schedule from
  `agent_nn` — exactly the re-run design, so results are
  comparable to `results_trust_rerun.jsonl`.
- Both arms run fresh in the rematch (wall-clock must be
  measured under the same conditions); records to
  `nn/results_trust15.jsonl`, per-seed flush, skip-existing on
  relaunch — the proven pattern.
- Fix constants: α=0.6, ε=1e-6, β annealed linearly 0.4 → 1.0
  across episodes, capacity 10000, batch 32.
- Named deviations inherent to the fix (not improvisations):
  stratified sampling draws independently per stratum, so an
  index may repeat within a batch (the naive sampler drew
  without replacement); IS weighting scales per-sample
  gradients (that is the correction under test); the logged
  loss is the IS-weighted loss.

## Endpoints (fixed in advance)

Per seed, per arm, computed from the records exactly as in the
re-run runner:
1. **Wall-clock seconds per seed** (runner-measured).
2. **Final-100 success:** mean of per-episode `reached` over
   episodes 900–999.
3. **First episode reaching rolling-100 success ≥ 0.8**
   (rolling window 100, as in the re-run note); "never" counts
   as +∞ — a failure for that seed on this endpoint.

## Success criterion (fixed; all three required)

(a) **Speed:** mean wall-clock/seed of trust-1.5 ≤ uniform's.
(b) **Success:** mean final-100 success of trust-1.5 ≥
    uniform's − 0.05, AND no seed worse than uniform's
    same-seed value by more than 0.10.
(c) **Sample efficiency:** episodes-to-0.8 of trust-1.5 no
    worse than uniform's on ≥ 2 of 3 seeds (∞ ≤ ∞ counts as
    no-worse; ∞ vs finite is worse).

**All three hold → the branch reopens** for further owner-ruled
work. **Any failure → the branch closes against a correct
implementation**: the Layer-1 negative then stands as a property
of the design, not of its implementation, and no further
trust-replay variants are proposed on this evidence.

Analysis is arithmetic against these fixed criteria on the
records — n=3 seeds, no inferential test, no post-hoc endpoints,
no per-seed retuning. Correctness gates for the implementation
(`nn/validate_trust15.py`, 13/13) passed before this freeze.

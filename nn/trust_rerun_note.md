# Trust-weighted replay head-to-head — re-run note (2026-10-19)

## What this is

Replication of the original Layer-1 test, run under WO-TBM-P3-followup
(owner order 2026-10-19). Same design: uniform replay baseline
(`nn/agent_nn.py::train_neural`) vs trust-weighted replay
(`nn/layer1_trust_train.py::train_trust`), seeds 200–202, 1000
episodes each, hidden=32, lr=1e-3, PlainEncoder, 20/10 protocol.
All numbers below are computed from the tracked records in
`nn/results_trust_rerun.jsonl`; no session-memory numbers are used.

Metric definitions (documented here, recomputable from the raw
per-episode `reached` arrays in the records): rolling window = 100
episodes; "first episode ≥0.8" = first episode index at which the
rolling-100 success rate reaches 0.8 (None if never); final-100 =
success rate over episodes 900–999.

## Results (from tracked bytes)

| arm | seed | wall (s) | first ep rolling-100 ≥ 0.8 | final-100 success | total crushes |
|---|---|---|---|---|---|
| uniform | 200 | 110.5 | 431 | 0.99 | 170 |
| uniform | 201 | 109.9 | 457 | 1.00 | 139 |
| uniform | 202 | 102.9 | 341 | 1.00 | 141 |
| trust | 200 | 327.4 | 967 | 0.95 | 138 |
| trust | 201 | 334.4 | never | 0.33 | 164 |
| trust | 202 | 285.7 | 730 | 0.43 | 143 |

## Verdict

**The Layer-1 branch is closed as an honest negative.** Trust-weighted
replay was ~3x slower wall-clock per seed and reached criterion later
or never on every seed, with worse final success on 2 of 3 seeds
(0.33 and 0.43 vs 1.00). The probability-weighted sampling overhead
(TD-error trust updates every train step, no importance-sampling
correction) did not earn its keep. No Layer 2 will be built on this
evidence. The branch remains committed as `nn/trust_replay.py` +
`nn/layer1_trust_train.py` so the tried-and-closed design is not
re-attempted blindly.

Runner: `nn/rerun_trust_head2head.py` (skips already-recorded seeds;
per-seed flush, so kills lose only in-flight seeds). Records:
`nn/results_trust_rerun.jsonl` (6 records, one per arm×seed, with
raw per-episode reached/ticks/crushes arrays).

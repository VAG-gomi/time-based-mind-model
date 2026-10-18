# Preregistration DRAFT v3.2: generalization — unseen schedules and
schedule changes (EBI priority 3)

**Status: Experiment A APPROVED (owner, 2026-10-17) — frozen
code 888355c. Experiment B is DRAFT v3.3 — not approved, not run.**
**v3.3 revises B per Claude's review (2026-10-17); v3.2 revised
v3.1 per Claude's review (2026-10-17).**

## What changed in v3.3 (Experiment B only)

1. **The mean-gap endpoint is dropped.** On the closest existing
   comparison (`wrapped` vs `oracle`, 20/10, 40 paired seeds), the
   95% bootstrap CI for the *mean* paired gap is [−1.073, +0.091]
   — about 100× wider than the ±0.01 margin — because a few
   collapsed seeds (one paired gap of −12.07) dominate the mean.
   "Indistinguishable" would be near-certain, not a residual
   risk. Verified from `nn/results_replication.jsonl` and
   `nn/results_controls.jsonl` before writing this.
2. **Primary is now the paired *median* gap** with a bootstrap
   95% CI (10,000 resamples, resampling seeds). On the same 20/10
   proxy data the median-gap CI is [−0.021, +0.012] (±0.0165).
   Scaled by 1/√n: ±0.0135 at 60 pairs, ±0.0104 at 100.
3. **60 pairs at ±0.02 (chosen over 100 pairs at ±0.01).** At 100
   pairs the expected half-width (≈0.010) merely equals the
   ±0.01 margin, so concluding equivalence would stay a coin
   flip even if the true gap is zero — the v3.1 trap again. At
   60 pairs the expected ±0.0135 fits inside ±0.02 with room for
   the center to sit slightly off zero, so the test can actually
   conclude. ±0.02 ≈ 60% of the 30/14 headroom (0.033), ≈ two
   steps' return per episode — a weaker claim than ±0.01,
   honestly labeled as such. Caveat: the 20/10 proxy uses
   different encodings than B's same-encoding pair, so the
   variance transfer is approximate.
4. **Collapse rate as descriptive secondary:** P(greedy return <
   0.5) per arm with Clopper–Pearson 95% CIs. In the 20/10 proxy,
   4/40 `wrapped` seeds vs 1/40 `oracle` seeds collapsed — the
   inference cost may show up as collapse frequency rather than
   the median, and the median endpoint cannot separate that from
   noise.
5. **Attribution corrected throughout:** Claude's review actions
   were misattributed to "the owner" in v3.1/v3.2; fixed below.
6. **Fork validation test required** (see Code): a forked
   no-clear continuation must reproduce an uninterrupted run
   bit-for-bit before B runs — pickled RNG state is the known
   hazard.

## What changed in v3.2 (and why)

1. **Stage 1 is complete; all three pre-registered Stage 2 gates
   passed** (details in "Stage 1 results"). The gates were computed
   from the raw records and independently verified by Claude.
2. **Experiment A was underpowered for its primary comparison.**
   Simulation of the exact McNemar test (independent Bernoulli
   0.6/0.4 per pair, two-sided α=0.05, 20k trials): power ≈ 0.33
   at n=40, ≈ 0.77 at n=100. v3.2 reallocates the same 280-seed
   budget: 100 seeds for the primary pair (`wrapped_ind` vs
   `phase_only_new`), 40 each for `oracle_ind` and `memory30`;
   `memory10`, `memory20`, `oracle_new` are dropped from A.
3. **Experiment B's primary comparisons were decided by
   construction** (stale clock fails by design; history arms lack
   the clock; the new clock and `oracle_ind` are handed the new
   schedule). v3.2 makes the primary the paired mean-return gap
   `wrapped_ind` − `oracle_ind` over post-change episodes 1–300 —
   the cost of inferring the schedule — with equivalence margin
   ±0.01 (proposed and justified below). Stale/history arms stay
   as sanity/secondary comparisons.
4. **All pilot tables now use the single 0.744 bar.** This corrects
   an error in `decisions_stage1.md` v1, which scored the five
   reused v2 arms at their original 0.75 bar: at 0.744,
   `oracle_new` is 1/5 (seed 103: 0.749, no crushes), not 0/5;
   `oracle_ind` is 2/5 (was 1/5 at 0.75). Gate verdicts were
   unaffected. The v2 pilot log below keeps its original 0.75-bar
   values as the historical record.

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
  passed every arm (all 5/5). Claude proposed this rule from
  reasoning about the 20/10 plain arm **before** knowing a pilot
  had run; both facts belong in the record.
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
(including the success-based rule) were proposed from Claude's
independent reasoning, but the pilot data was visible
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

## Stage 1 results (2026-10-17)

Owner approval 2026-10-17 (Stage 1 only; record at end of this
document). Frozen code: 888355c.

**Reuse gate:** `wrapped_ind` seed 100 re-ran at the frozen code
and reproduced the v2 record **bit-for-bit** (full record
identical). The five v2 arms' code paths were verified unchanged
between 023317f and 888355c (additive-only diff), so the 25 v2
records were reused. 20 new seeds ran: `phase_only_new` +
`oracle_ind` × 100–104 on 30/14 (`--fast-threshold 0.744`,
`nn/results_stage1.jsonl`); `wrapped_ind` × 100–109 on 20/10
(`--fast-threshold 0.750`, `nn/results_regression.jsonl`).

**Gate verdicts (pre-registered before the runs; independently
verified by the owner from the raw records):**

1. **Success rule: PASS.** All 7 pilot arms 5/5 seeds with
   success ≥ 0.9.
2. **Regression: PASS.** `wrapped_ind` 20/10 seeds 100–109:
   6/10 clean-and-fast (≥4), 10/10 zero-crush (≥8).
3. **`oracle_ind` as upper bound: PASS (thin).** Median 0.737 ≥
   0.733 (`wrapped_ind` 0.753 − 0.02), 5/5 zero-crush. Margin
   above the bar: 0.004.

**Pilot table at the single 0.744 bar (greedy):**

| Arm            | Clean-and-fast | Median return | Zero-crush | Success |
|----------------|----------------|---------------|------------|---------|
| wrapped_ind    | 3/5            | 0.753         | 5/5        | 5/5     |
| memory10       | 0/5            | 0.724         | 5/5        | 5/5     |
| memory20       | 0/5            | 0.726         | 5/5        | 5/5     |
| memory30       | 0/5            | 0.740         | 5/5        | 5/5     |
| oracle_new     | 1/5            | 0.655         | 4/5        | 5/5     |
| phase_only_new | 2/5            | 0.740         | 4/5        | 5/5     |
| oracle_ind     | 2/5            | 0.737         | 5/5        | 5/5     |

Reading (5 seeds each — descriptive only): the wrapper, the plain
clock, and the oracle are not distinguishable yet (3/5, 2/5, 2/5);
history arms are 0/5 but perfectly safe, as the ceiling analysis
predicted. Treat `oracle_ind` as "not clearly worse", not as a
demonstrated upper bound.

### Seed-101 diagnostic (records only)

`oracle_ind` seed 101: greedy return 0.556, 44.4 ticks, 0 crushes;
eps005 eval 0.580, 42.0 ticks, 0 crushes; late-training (ε=0.05)
behavior 28.2 ticks, 0 crushes, 1.00 success. The *learned greedy
policy* waits roughly one extra 30-tick cycle while the
ε-greedy *behavior policy* during training did not — the gap sits
exactly where depressed crossing Q-values would put it. Training
uses ε-greedy decaying 1.0→0.05 over 1000 episodes then flat
0.05 (`nn/agent_nn.py`), so early high-ε crushes in crossing
states could have depressed Q(cross) and the caution was never
corrected. This is **consistent with** the ε-noise hypothesis but
**not established** by aggregate records (no Q-values or
trajectories are saved). Decisive test (not run): re-run seed 101
with ε_end=0.0 for the final 1000 episodes, or log Q(cross) vs
Q(wait) at opening states — if the slow policy disappears, the
hypothesis is supported. The same slow-safe shape appears in
other arms' misses (e.g. v2 `wrapped_ind` seeds 5/8/9), so it is
a policy-class phenomenon, not an `oracle_ind` defect.

## Experiment A (v3.2): unseen schedule (30/14)

### Power problem (as found by Claude)

For the primary comparison (`wrapped_ind` vs `phase_only_new`,
paired clean-and-fast), assume true rates 60% vs 40% with ~50%
discordant pairs (the independent-rates model gives 52%
discordant). Simulation of the exact two-sided McNemar test at
α=0.05 (20,000 trials): **power ≈ 0.33 at n=40, ≈ 0.77 at
n=100.** At 40 seeds the likely outcome is "indistinguishable"
regardless of the truth. (The 60/40 rates are a scenario, not an
estimate — 5 pilot seeds cannot estimate rates.)

### Arms and seeds (v3.2)

- **Primary pair (100 seeds each):** `wrapped_ind`,
  `phase_only_new`. Fresh seeds 200–299, paired by seed.
- **Secondary (40 seeds each):** `oracle_ind` (inference test),
  `memory30` (strongest history). Fresh seeds 200–239.
- **Dropped from A:** `memory10`, `memory20` (structurally
  disadvantaged; ceiling analysis already predicts the outcome),
  `oracle_new` ("upper bound" label suspended; keeps its
  diagnostic role in B only).
- Pilot seeds 100–104 remain pilot-only and are excluded from
  the main analysis (no double-dipping).
- Total: 280 seeds — the same budget as v3.1's A, reallocated
  to the primary pair.

### Endpoints and analysis

- **Primary:** clean-and-fast (greedy 0 crushes, success ≥ 0.9,
  return ≥ 0.744), paired `wrapped_ind` vs `phase_only_new`,
  exact McNemar, α=0.05 (single primary; no Holm).
- **Secondary:** vs `oracle_ind`, vs `memory30`; Holm over the 2
  comparisons. Equivalence margin **±0.25** on the paired rate
  difference; "indistinguishable" as the third outcome when the
  interval is neither inside the margin nor decisively excludes
  zero (as v3.1).
- Seed-bootstrap 95% CIs for median-return differences
  (resampling seeds, 10,000 resamples), as in the replication.

## Experiment B (v3.2): mid-run schedule change (20/10 → 30/14)

### Design problem (as found by Claude)

v3.1's primary comparisons were decided by construction: the
stale clock fails by design, history arms lack the clock, and the
new clock plus `oracle_ind` are handed the new schedule — the
wrapper could not win any meaningful comparison. What B can
really measure is the **cost of inferring the schedule**:
`wrapped_ind` (inferred) vs `oracle_ind` (same encoding, perfect
information, switches to true 30/14 info at the change).

### Primary endpoint (single, v3.3)

**Paired *median* gap `wrapped_ind` − `oracle_ind` in mean return
over post-change episodes 1–300**, paired by seed, with a
seed-bootstrap 95% CI (10,000 resamples, resampling seeds).
Return already includes the −1 per crush, so crushes are
descriptive, not a second primary. The median replaces the v3.2
mean because collapsed seeds dominate the mean (see v3.3
changelog).

**Equivalence margin: ±0.02 return units; 60 pairs.** Chosen over
100 pairs at ±0.01: from the 20/10 proxy the median-gap CI
scales to ±0.0135 at 60 pairs, which fits inside ±0.02 with room
for the center to sit slightly off zero — the test can actually
conclude equivalence if the inference cost is small. ±0.02 ≈
60% of the 30/14 headroom (0.033), ≈ two steps' return per
episode. Decision rule: CI inside [−0.02, +0.02] → equivalent
(inference cost negligible); CI entirely below −0.02 →
`oracle_ind` better by a meaningful margin (inference costs);
CI entirely above +0.02 → `wrapped_ind` better (unexpected;
report as such); otherwise → indistinguishable.

### Descriptive secondary: collapse rate

P(post-change mean return < 0.5) per arm, with Clopper–Pearson
95% CIs, reported descriptively — not a primary. Rationale: in
the 20/10 proxy, 4/40 `wrapped` vs 1/40 `oracle` seeds collapsed,
so the inference cost may appear as collapse frequency rather
than a median shift.

**Seed-count note:** the 60-pair choice rests on the 20/10 proxy
(`wrapped` vs `oracle`, different encodings); B's same-encoding
pair may differ. If the observed CI half-width exceeds 0.02,
the pre-registered outcome is "indistinguishable" — no
post-hoc widening of the margin.

### Arms and seeds (v3.2)

- **Primary pair (60 seeds each):** `wrapped_ind`, `oracle_ind`.
  Fresh seeds 300–359, paired by seed.
- **Sanity/secondary (40 seeds each, fresh seeds 300–339):**
  `phase_only_new` (adapts to the 30-clock),
  `phase_only_stale` (20-clock; expected to fail — sanity
  check), `memory10`/`memory20`/`memory30` (history baselines),
  `oracle_stale` (sanity check), `oracle_new` (diagnostic;
  "upper bound" label still suspended).
- These arms are not primary competitors; beating them is
  expected by construction and is reported descriptively.

### Buffer handling (as v3.1)

300 post-change episodes. **Primary: the no-clear continuation**
(the agent keeps its 20/10 replay buffer — the realistic
continual-learning case). The buffer-clear variant is secondary,
via **checkpoint forking**: train each arm/seed to the switch
once, save agent/encoder state, fork no-clear and clear-buffer
continuations so each variant costs only the 300 post-change
episodes. Greedy evaluation on 30/14 after the window. Report
the fraction of post-change episodes at least 31 ticks long (a
30-tick period cannot be learned from shorter episodes).

## Seeds, protocol, analysis, interpretation

- A: fresh seeds 200–299 (primary pair), 200–239 (secondaries).
  B: fresh seeds 300–359 (primary pair), 300–339 (others).
- Protocol otherwise as v3.1 (2000 pre-change episodes, hidden
  32, lr 0.001, γ 0.99, same rewards, cap 200).
- One primary endpoint per experiment; Holm for secondaries
  only; intervals only; cautious causal language (strongest
  explanation among the tested alternatives; no generalization
  beyond this task and protocol); the missing-data rule stands
  (incomplete pairs excluded from paired analyses, identified
  by seed).

## Code

- v3.1 code (frozen 888355c): `OracleIndEncoder`,
  `PhaseOnlyNewEncoder`, `--fast-threshold`, `oracle_ind` wiring,
  structural checks PASS.
- **Experiment B is not yet implemented.** The runner has no
  mid-run schedule switch and no checkpoint forking. Both must
  be implemented, structurally checked, and frozen — with the
  approval recorded here — before Stage 2 runs.
- **Fork validation test (required before any B run):** fork a
  no-clear continuation at the switch point for one seed and
  require its records to be **bit-for-bit identical** to an
  uninterrupted run of the same seed — the same bar as the
  Stage 1 reuse gate. Pickled RNG state is the known silent-
  difference hazard: the fork must restore the exact generator
  state, not a fresh seed. If the test fails, diagnose; do not
  run B.

## Compute estimate (v3.2)

Observed: ~150 s/seed for 2000-episode 30/14 seeds (Stage 1).

- Experiment A: 280 seeds ≈ 11.7 h single-process, **≈ 6 h with
  2 processes** (same total as v3.1's A — reallocated, not
  enlarged).
- Experiment B: ≈ 2600 episodes/seed (2000 + 2×300 forked) ≈
  190 s/seed. 60×2 + 40×7 = 400 seeds ≈ 21 h single-process,
  **≈ 10.5 h with 2 processes**.
- Total ≈ **16.5 h with 2 processes**. B's implementation and
  freeze come first; A can run while B code is finished.

## Stage 2 gate — record (all passed 2026-10-17)

Pre-registered before the Stage 1 runs; computed from raw
records and independently verified by the owner:

1. Success rule (≥3/5 seeds success ≥ 0.9, every pilot arm):
   **PASS** (7/7 arms 5/5).
2. Regression (≥4/10 clean-and-fast at 0.750 AND ≥8/10
   zero-crush, `wrapped_ind` 20/10 100–109): **PASS** (6/10,
   10/10).
3. `oracle_ind` as upper bound (median ≥ `wrapped_ind` median −
   0.02 AND ≥4/5 zero-crush): **PASS** (0.737 ≥ 0.733, 5/5;
   margin 0.004 — thin).

## Stage 1 approval record (owner, 2026-10-17) — historical

**Approved scope: Stage 1 only.** The v3.1 feasibility pilot and
the 20/10 regression check. Stage 2 (full Experiments A and B)
was NOT approved by that decision.

**Exact arms and seeds run:**
- Feasibility pilot, 30/14, fast = 0.744: `wrapped_ind`,
  `memory10`, `memory20`, `memory30`, `oracle_new` (reused v2
  records, seeds 100–104) + `phase_only_new`, `oracle_ind`
  (new, seeds 100–104).
- Regression, 20/10, fast = 0.750: `wrapped_ind` × 100–109.

**Frozen code commit: 888355c** (approval commit 7c4f8e8 touched
only this document).

## Approval — Experiment A (owner, 2026-10-17)

**Owner approves Experiment A as designed in v3.2.** Frozen code
commit: **888355c** (no new code needed — all A arms exist and
are frozen at 888355c). Run in 2 processes.

Approved scope: `wrapped_ind` × 100 + `phase_only_new` × 100
(seeds 200–299) + `oracle_ind` × 40 + `memory30` × 40 (seeds
200–239), 30/14, `--fast-threshold 0.744`. 280 seeds.

**Experiment B is NOT approved.** B is DRAFT v3.3 above; it
needs owner approval of the v3.3 design *and* a B code freeze
(including the fork validation test) before any B run.

## Approval (v3.2 design) — superseded

The v3.2 design approval was never recorded; the owner instead
approved Experiment A as designed in v3.2 (above) and ordered
the v3.3 revision of B.

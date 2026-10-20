# WO-TBM-L15-diag — Diagnosis of the trust-1.5 rematch failures

**Scope: diagnosis only.** The trust branch remains CLOSED
under `nn/prereg_trust15.md` / `nn/decisions_trust15.md`
(`b563dd4`). Nothing here reopens it, fixes anything, or
authorizes an experiment. Part C items are proposals with
exact scope, awaiting owner confirmation. Labels: MEASURED /
INFERRED / NOT ESTABLISHED.

## 0. Skipped / not doable

Nothing skipped. **One failure, recovered:** a runtime restart
killed the Part B diagnostic run after 3 of 6 runs; the 3,000
completed traces were intact, resume support was added to the
diagnostic script, and the remaining runs completed. All six
diagnostic trajectories **MATCH the committed rematch records
bit-for-bit** (ticks and reached, all 6 arm×seed pairs), so the
Part B traces describe the actual rematch runs, not
approximations.

---

## Part A — Speed attribution (failure of criterion (a))

### A.1 Bare decomposition of the gap (MEASURED, from rematch records)

Per-step wall rates are strikingly stable: uniform
0.799/0.808/0.799 ms/step; trust15 0.965/0.966/0.964 ms/step.
The mean gap of 20.89 s/seed decomposes exactly (an identity
given the records) into:

| component | s/seed (mean) | share |
|---|---|---|
| Rate effect: trust15's higher per-step cost | **16.93** | 81% |
| Volume effect: trust15 runs more env steps (longer episodes) | **3.96** | 19% |
| Total | **20.89** | 100% |

Per seed — rate / volume: 200: 17.77 / +2.87; 201: 15.14 /
−4.45 (trust15 took *fewer* steps and still lost on rate);
202: 17.88 / +13.45. The volume term is criterion-(c)
territory (learning speed) appearing inside wall-clock; the
rate term is the machinery question.

### A.2 What the rate delta is made of (profile + call counts)

Profile (Step-0 method, seed 200, 200 episodes; this run
reproduces the Step-0 table within ~1s per arm):

| arm | total | buffer | network | env | glue |
|---|---|---|---|---|---|
| uniform | 32.85s | 8.51 | 18.53 | 0.90 | 4.91 |
| trust-naive | 55.45s | 29.64 | 19.07 | 1.04 | 5.70 |
| trust15 | 41.33s | **16.16** | 18.35 | 0.96 | 5.85 |

Trust-1.5 machinery (16.16s of 41.33s): segment-tree sampling
6.32 (of which prefix search `_find` 5.31), priority updates
4.79, insertion path repair (`add`) 4.63, **IS weights only
0.43**. Network per-step cost is identical across arms
(0.479 vs 0.484 ms/step profiled). **The IS correction is not
the speed problem; the tree walks are.** (MEASURED, profiled.)

Numpy calls per env step (MEASURED, `count_numpy_calls.py`,
train_step driven directly on pre-filled buffers): uniform
**118.0**, trust15 **166.1** (+48.1, +41%). The excess is
concentrated exactly where the design puts it: `_find` 33/step,
`is_weights` 4, `update_trust` 4, plus ~8 extra inside
train_step. Implied bare cost ≈ 3.4 µs per extra call
(0.163 ms/step bare rate delta ÷ 48.1 calls) — the signature
of fixed dispatch overhead on 32-element arrays.

**Attribution of the rate effect to buffer machinery:
INFERRED** — it is the only per-step component that differs
between arms (profile), its call-count excess is measured,
and the profiled machinery excess over uniform's buffer
(0.200 ms/step) is the same order as the bare rate delta
(0.163 ms/step; cProfile overstates the delta ~23%, as it
inflates Python-heavy paths).

### A.3 Reconciliation of the Step-0 projection miss

At the profiled 200-episode scale: Step-0 predicted
trust-1.5 ≈ floor 25.67s vs uniform 33.94s (Δ = −8.27s, i.e.
faster). Actual: trust15 41.33 vs uniform 32.85 (Δ = +8.48s).
Swing = 16.75s, fully decomposed:

| term | s |
|---|---|
| Replacement machinery, priced at 0 by the bound, actually | **+16.16** |
| Floor drift (trust15 non-buffer 25.17 vs predicted floor 25.67) | −0.50 |
| Comparator drift (uniform total 32.85 here vs 33.94 in Step-0) | +1.09 |
| **Total swing** | **16.75** |

The floor side of the projection was accurate (within 0.5s).
**The miss is the Amdahl bound's defining assumption: it prices
the replacement machinery at zero.** It is a bound, not an
estimate — and the machinery costs 16.16s because at batch 32
every tree operation is a handful of numpy calls whose fixed
overhead dwarfs the arithmetic (A.2). A contributing
distortion (MEASURED): under cProfile, uniform's single
`rng.choice` per step is attributed 0.222 ms/step — its buffer
line (8.51s) looks comparable to the tree machinery, flattering
the headroom; bare, uniform sampling is ~5 numpy calls/step
against the tree path's ~37.

---

## Part B — Learning-speed traces (failure of criterion (c))

Source: `nn/traces_trust15_diag.jsonl` (6,000 episode records;
trajectories verified identical to the rematch, §0).
Instrumentation invalidates timing; no wall-clock is taken
from Part B.

**Where the lag opens:** not early. Rolling-100 diffs
(trust15 − uniform) at episode 99: +0.02 / −0.01 / −0.05
(seeds 200/201/202). The divergence opens **mid-training**:
at ep 399, −0.09 / **+0.25** / −0.56; at ep 499, −0.35 /
+0.03 / −0.36. Seed 202's uniform run surges early (0.60 by
ep 299) while its trust15 twin stalls at 0.38–0.39 through
ep 399 — then still reaches final-100 1.00.

**The single strongest observation:** the machinery traces
are nearly identical across seeds while outcomes flip sign.
Early-phase entropy 8.61 / 8.60 / 8.61; top-decile mass
0.335 / 0.336 / 0.337; ESS 26.0 / 25.9 / 26.0; sampled ages
within 1% of each other. **No measured trace variable
separates seed 201 (the win) from seeds 200/202 (the
losses).**

### Candidate verdicts

**1. Early TD-error noise-chasing — REFUTED as stated.**
Sampling *is* concentrated from the start (entropy 8.61 vs
uniform's 9.21; top-decile slots hold ~34% of mass vs 10%),
but: (i) the concentration is identical on seed 201, where
trust15 wins; (ii) the lag opens mid-training, not early.
A mechanism that is present in equal measure on the winning
seed and precedes a lag that appears 300 episodes later does
not explain the failure in its stated (early) form. The
concentration itself is MEASURED; its role in the (c) failure
is refuted as framed.

**2. Weak early β under-correction — INDETERMINATE.**
Measured: early-phase ESS ≈ 26.0 of 32 (β averaging 0.50 over
the phase) — correction is real but partial by design; late
ESS ≈ 17.3 at β ≈ 0.86. Against a direct causal reading: the
lag opens at episodes ~300–500, where β is already 0.58–0.70,
and the identical β schedule produces a win on seed 201. A
delayed-effect story (early under-corrected updates worsening
the mid-training position) cannot be excluded from these
traces. Verdict stands as indeterminate, not resolved by
narrative.

**3. Within-batch repeats — REFUTED (MEASURED).**
Mean repeats per batch of 32: **0.01** in every phase and
seed (effective batch ≈ 31.99). Stratified sampling over
10,000 slots essentially never repeats an index. This
mechanism contributed nothing.

**4. Max-priority insertion recency bias — INDETERMINATE
(bias measured, role not established).**
Measured: trust15's sampled transitions are younger —
mean sampled age ≈ 4,540–4,565 steps vs uniform's ≈ 4,870
(phase range: trust15 4,305–4,713 vs uniform 4,627–4,997),
a consistent ~7–9% tilt, as max-priority insertion predicts.
But the tilt is small, roughly constant across phases, and
present on the winning seed too; nothing in the traces ties
it to the mid-training divergence. Existence: MEASURED.
Causal role in (c): NOT ESTABLISHED.

**5. The variance null — INDETERMINATE; cannot be excluded,
and the traces are consistent with it.**
Episodes-to-0.8 paired diffs: +91 / −65 / +188 — sign-mixed,
mean +71 on a base of ~400 (huge relative scatter). Final
success is identical (1.00 both arms, all seeds): there is no
systematic learning deficit, only time-to-criterion scatter.
Every machinery trace is seed-indistinguishable (§B head)
while outcomes flip sign — precisely the fingerprint of
seed-level variance dominating a small-or-zero systematic
effect at n=3. At n=3 no test can reject this null, and this
diagnosis does not pretend otherwise. It remains a live,
serious candidate — arguably the best-supported reading of
Part B as a whole, labeled indeterminate because consistency
is not confirmation.

---

## Part C — Ablation proposals (DESIGNS ONLY — not run)

For each candidate surviving as indeterminate. Each requires
owner confirmation before any work; confirmatory use would
additionally need a prereg freeze.

**P-C2 (β schedule).** Toggle: trust15 with β fixed at 1.0 from
episode 0 vs the frozen anneal; all else identical to the
frozen design (seeds 200–202, 1000 eps, 20/10). Measurement:
episodes-to-0.8 per seed + ESS trace (manipulation check: ESS
should sit at its late-run level ~17 from the start). Confirm:
lag on seeds 200/202 disappears or shrinks materially under
full early correction → β anneal implicated. Refute: lag
pattern unchanged → β schedule exonerated. Scope: 1 arm × 3
seeds ≈ 5 min compute; one additive variant flag.

**P-C4 (insertion priority).** Toggle: new transitions enter
at the current *median* leaf trust instead of the maximum;
all else frozen. Measurement: sampled-age trace
(manipulation check: the ~7–9% age gap vs uniform should
close) + episodes-to-0.8. Confirm: age gap closes AND the
lag on 200/202 disappears → insertion bias implicated.
Refute: age gap closes, lag persists → recency exonerated.
Scope: 1 arm × 3 seeds ≈ 5 min; one additive buffer flag.

**P-C5 (variance null — the discriminating test).**
Toggle: none — scale. Frozen pair (uniform vs trust15),
fresh paired seeds 210–229 (20 pairs), same protocol.
Measurement: distribution of paired episodes-to-0.8
differences (sign split, median difference), plus
time-to-criterion variance ratio between arms. Confirm the
null: diffs centered near zero, sign split ≈ 50/50, no
median shift → the n=3 pattern was scatter. Refute the null:
median paired difference significantly positive → a real
systematic cost exists, and P-C2/P-C4 gain priority.
Scope: 40 runs ≈ 60–70 min compute; records + a frozen
analysis rule required before running.

---

*End of diagnosis. The branch stays closed; Part C awaits
owner rulings.*

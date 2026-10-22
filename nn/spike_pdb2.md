# P-Db2 benchmark spike — report (measurement only)

**Status: EXPLORATORY, instructor-verified (timed log, profile output, and projection arithmetic re-checked independently from the spike files).**
**Authority:** owner ruling at the P-Db2 Step 0 gate (2026-10-22,
"Benchmark spike first"), per
`~/workspace/your_files/DECISION-TBM-jit-vectorization.md`
§5(1) and §6–§7. The arm-run trigger fired (206 > 150);
the first step is this spike, measurement only.
**This spike authorises NO substrate change, NO Step 1
retraining, NO Step 2 evaluation, and NO JIT/vectorized
implementation.** The substrate change still requires a
second owner ruling after these numbers.

**All conclusions bounded. No causal claims. No substrate
recommendation is made here** — the measured facts and
the gate options are presented; the ruling is the owner's.

## 0. What was measured, and what was NOT measured — lead section

**Measured (this spike, current hand-written numpy substrate):**

- Timed P-Db2 trainings, 1 per arm (6 total, 2000 episodes
  each, `nn/train_pdb2.py` trainers, F1 schedule-varied
  training world, design-note hyperparameters): wall-clock
  per training, reached counts, total door entries.
- cProfile hotspot breakdown of ONE representative
  training (wrapped/0 — see §3 for why).
- Timed greedy eval sample: 1 F1 job per arm (6 jobs)
  plus 1 failure-heavy F3 job for wrapped/0 (7 jobs
  total, 170 episodes). NOT the full 150-job evaluation.
- Projected Steps 1–2 total time on the current
  substrate, with arithmetic (§4).

**NOT measured (by design, and must not be claimed):**

- **No JIT/vectorized gain was measured, because no
  candidate exists in this codebase to time.** The
  decision document's statement stands: any 20–30×
  figure is NOT ESTABLISHED. This spike measures the
  current substrate's cost and overhead structure, plus
  the equivalence gap a future candidate would have to
  close (§5). It does not, and cannot, report a speedup
  ratio for a candidate that was not built.
- No Step 1 training and no Step 2 evaluation was run.
  Spike trainings reproduce the Step 0 sanity agents
  deterministically (see §1) but are labelled SPIKE in
  every output and must NOT be reused, quoted, or
  pooled as Step 1 records or findings.
- Only 1 seed per arm was timed. The projection (§4)
  is a pace projection from 6 measured trainings, not
  a measurement of all 30 Step 1 agents.

**Headline projection (arithmetic in §4):** Steps 1–2 on
the current substrate project to **~1.84 hours on the
composition estimate** (25 wrapped trainings at wrapped
pace + 25 non-wrapped at non-wrapped mean pace + eval),
with a pace range of **1.34–2.26 hours**, and
**2.13 hours at Step 0 sanity pace**. All are under
the design note's corrected ~2.5 hour estimate and
under the 4-hour time threshold. The 4-hour time
threshold is NOT the binding constraint on measured
pace; the arm-run trigger (206 > 150) is what fired
the gate, and it fires on counts, not time.

## 1. Timed trainings (current substrate)

Run: `nn/spike_pdb2_run.py`, log `nn/spike_pdb2.log`.
One training per arm, 2000 episodes each, same
hyperparameters as design-note §1. Wall-clock is
`time.perf_counter` around the trainer call only
(script startup and inter-training gaps excluded —
see the sanity-pace cross-check in §4).

| Arm / seed | Wall (s) | Reached / 2000 | Total entries |
|---|---|---|---|
| wrapped/0 | 162.6 | 1661 | 2387 |
| plain/0 | 109.2 | 1704 | 2311 |
| memory/0 | 98.7 | 1755 | 2236 |
| oracle_ind/100 | 96.5 | 1763 | 2279 |
| oracle_new/100 | 98.0 | 1746 | 2326 |
| phase_only_new/100 | 104.7 | 1722 | 2273 |
| **Sum / mean** | **669.7 / 111.6** | — | — |
| Non-wrapped mean (5 arms) | 101.4 | — | — |
| Median (6 arms) | 101.7 | — | — |

**Deterministic reproduction check (observed, not a
Step 1 claim):** every spike training's reached count
and total entry count exactly matches the Step 0
sanity record for the same (arm, seed)
(`nn/sanity_pdb2.log`: 1661/1704/1755/1763/1746/1722
and 2387/2311/2236/2279/2326/2273). This confirms
that deterministic reproduction is checkable on the
current substrate — the same property the §5(2)
reproduction gate would demand of any future
candidate (§5).

**Restart (reported, not hidden):** a runtime restart
at ~15:15 UTC on 2026-10-22 killed the first spike
run after wrapped/0 finished (its timed training was
already logged). The runner was relaunched as a
detached, resume-safe run: wrapped/0 was retrained
UNTIMED (labelled `SPIKE RETRAIN-UNTIMED` in the
log) to recover its agent for the eval sample, and
no timed training was duplicated. The relaunch lost
no timed measurement.

## 2. Timed eval sample (current substrate)

Sample actually run (stated exactly): 1 F1 job per
arm (training world, the most representative job),
plus 1 F3 job for wrapped/0 as a failure-heavy
upper bound (F3 episodes mostly run to the 200-tick
cap, as in P-Da E3). 7 jobs, 170 episodes total.
F0, F2, and F4 jobs were NOT timed directly; the
bound below does not depend on them, because the
per-tick cost is the same greedy forward pass in
every world and F3 is the tick-heaviest case.

| Job | Episodes | Wall (s) | Total ticks | Reached |
|---|---|---|---|---|
| wrapped/0 F1 | 20 | 0.01 | 472 | 20 |
| plain/0 F1 | 20 | 0.01 | 461 | 20 |
| memory/0 F1 | 20 | 0.01 | 476 | 20 |
| oracle_ind/100 F1 | 30 | 0.01 | 704 | 30 |
| oracle_new/100 F1 | 30 | 0.02 | 909 | 30 |
| phase_only_new/100 F1 | 30 | 0.09 | 5153 | 5 |
| wrapped/0 F3 | 20 | 0.07 | 4000 | 0 |
| **Mean (7 jobs)** | — | **0.031** | — | — |
| **Max observed** | — | **0.09** | — | — |

Per-eval-job cost is **seconds-scale at most, and
typically hundredths of a second**: even at a
conservative 0.10 s/job bound, 150 jobs cost
15 s (0.25 min); at the measured mean, ~4.7 s.
Eval is < 0.5% of the projected Steps 1–2 total
in every variant below. The speedup target, if
any future candidate is built, is therefore
**training wall-clock per seed (~97–163 s), in
absolute terms** — not a ratio against uniform,
which the decision document (§2) rules incoherent
as a target because the shared loop lifts both
arms.

## 3. Profile: one wrapped/0 training (cProfile)

**Choice and reason:** wrapped/0, because it is the
most expensive encoder type in the programme
(stateful `ScheduleModel` + 104 inputs, and 25 of
the 50 Step 1 trainings are wrapped). Its profile
therefore bounds the shared-loop overhead and
tests whether the ScheduleModel is a material term.
Run: `nn/spike_pdb2_profile.py`, output
`nn/spike_pdb2_profile.txt`. Profile overhead
distorts wall time, so the profile run's time is
NOT used for the §4 projection.

- Total cProfile time: **124.240 s** for the same
  deterministic wrapped/0 training (reached 1661,
  entries 2387, identical to §1). Note: cProfile
  time (124.2 s) is BELOW the untimed wall time
  (162.6 s) for the same trajectory — cProfile
  time is a profiler accounting, not wall time,
  and is used here only for hotspot SHARES, not
  for absolute pace. The difference is reported,
  not smoothed over.
- Total function calls: **25,301,784** (25,166,321
  primitive) in 124.240 s.
- `train_step` (learning update): **117.783 s
  cumulative, 94.8% of profile time**,
  135,461 calls. The learning update dominates;
  env/encode/world do not.
- Within `train_step`, largest tottime terms
  (absolute, not ratios):
  - `numpy.ndarray.dot`: **40.469 s tottime**,
    1,200,696 calls (32.6% of profile time).
  - `RandomState.choice` (replay sampling):
    **31.876 s tottime**, 135,430 calls
    (25.7% of profile time); enclosing
    `agent_nn.sample`: 34.587 s cumulative.
  - `nnet.forward`: 25.301 s cumulative,
    329,488 calls.
  - `nnet.backward`: 22.969 s cumulative,
    135,430 calls.
  - `nnet.step` (optimizer): 13.897 s
    cumulative, 135,430 calls.
  - `numpy.stack` (batch construction):
    10.460 s cumulative, 270,860 calls.
  - `masked_mse_grad`: 4.945 s cumulative.
- Env/encode/world terms (absolute, small):
  - `agent_nn.env_step`: 1.796 s cumulative
    (1.4%).
  - `WrappedEncoder.encode`: 1.738 s cumulative
    (1.4%).
  - `SchedDoorWorld.step`: 1.163 s cumulative
    (0.9%).
  - `wrapper.observe` (ScheduleModel):
    0.591 s cumulative (0.5%).
  - `SchedDoorWorld.observe`: 0.230 s cumulative.
- **Structure, stated as measurement:** the cost
  is concentrated in the learning update — replay
  sampling (`choice`), network forward/backward,
  optimizer step, and batch construction — with
  `dot` + `choice` alone accounting for 72.345 s
  tottime (58.2% of profile time). The act path
  (encode + env_step, with world stepping and
  ScheduleModel observe nested inside those
  calls) totals ~3.5 s cumulative
  (1.738 + 1.796 s). The wrapped ScheduleModel
  (`wrapper.observe`, 0.591 s) is NOT a material
  term. A candidate that compiled only the
  env/encode tick loop would therefore NOT
  address the bulk of the measured cost; any
  future candidate's claim must be checked
  against this breakdown, not against a generic
  "loop is slow" statement. (This is a
  measurement of where time goes on the current
  substrate; it is not a recommendation for or
  against any option.)

## 4. Projection: Steps 1–2 total time on the current substrate

Arithmetic shown in full. Training episodes:
50 trainings × 2000 = 100,000. Eval jobs:
150. Eval episodes: 3750. Step 3 analysis is
additional and NOT included, as in the design
note.

**Training projection variants (eval excluded
first, then added):**

- Composition estimate (the central case, because
  Step 1 has 25 wrapped + 25 non-wrapped
  trainings, and wrapped pace is distinctly
  slower): 25 × 162.6 s (wrapped) = 4065.0 s,
  plus 25 × 101.4 s (non-wrapped mean) =
  2535.5 s. Total training = 6600.5 s =
  **110.0 min = 1.83 h**.
- All-at-mean: 50 × 111.6 s = 5580 s =
  93.0 min = 1.55 h.
- All-at-max (every training at wrapped pace):
  50 × 162.6 s = 8130 s = 135.5 min = 2.26 h.
- All-at-min (every training at oracle_ind pace):
  50 × 96.5 s = 4825 s = 80.4 min = 1.34 h.
- Sanity-pace cross-check: Step 0 sanity wall was
  ~15.3 min for 6 trainings = 153.0 s/training
  average (vs spike timed sum 669.7 s = 11.2 min
  for the same 6 trainings — the difference is
  script startup, inter-training gaps, and
  machine-load variance, which `perf_counter`
  around the trainer call excludes). At sanity
  pace: 50 × 153.0 s = 7650 s = 127.5 min =
  **2.13 h**.

**Eval projection:** at the conservative 0.10 s/job
bound, 150 × 0.10 s = 15 s = 0.25 min = 0.004 h.
At the measured mean (0.031 s/job), ~4.7 s.
Either way, eval adds < 0.01 h.

**Totals (training + eval bound):**

- Composition: **~1.84 h**.
- Pace range: **1.34 h (min) – 2.26 h (max-arm
  pace)**, with sanity pace **2.13 h** inside
  the range.

**Comparison:**

- Design note's corrected estimate (instructor
  amendment 73fd306): ~2.5 h. Measured
  projection is BELOW it on every variant
  (composition 1.84 h; even max-arm pace
  2.26 h < 2.5 h).
- 4-hour time threshold: **UNDER on every
  variant.** Headroom on the composition
  estimate: 4.00 − 1.84 = **2.16 h**; even at
  max-arm pace, headroom is 1.74 h.
- The arm-run trigger (206 > 150) is unaffected
  by this projection: it fires on counts, not
  time, and it already fired at Step 0.

**Caveats (bounded):** 1 seed per arm was timed,
not 30 distinct agents; per-seed pace may vary
around the measured means, and machine load
varies (sanity vs spike sums differ, as noted).
Wrapped retrainings are deterministic identical,
so wrapped pace should repeat in principle;
non-wrapped seeds are the larger variance
source, and the max-arm-pace variant bounds
that direction. Even that bound stays under
both the corrected estimate and the 4-hour
threshold.

## 5. Equivalence gap: what a future candidate must reproduce (§5(2))

No experiment runs on a new substrate until it
passes the reproduction gate: the candidate, in
a reference mode, must reproduce a frozen result
from the existing chain exactly — same seeds,
same trajectories — before any new result is
claimed on it. Failures are reported, not tuned
away silently. Old records are never pooled with
new-substrate records; any experiment on a new
substrate gets its own preregistration.

**Exact reproduction targets (each must match,
not approximately, unless a tolerance is stated
in advance by the owner at the second ruling):**

1. **Deterministic RNG streams:** training-start
   RNG `random.Random(seed + 1000)` with
   `start_tick = rng.randrange(period)` per
   episode; agent RNG `random.Random(seed)`;
   numpy RNG `RandomState(seed)`; replay
   sampling sequence (`RandomState.choice`
   calls in `agent_nn.sample`, 135,430 calls
   in the profiled wrapped/0 training).
   Any change to stream order or stream
   identity changes trajectories.
2. **Training summaries, per episode:**
   (ticks, reached, crushes, mean_loss) for
   all 2000 episodes, per (arm, seed) — as in
   `nn/training_pda.jsonl` (P-Da, 50 records)
   and `nn/sanity_pdb2.log` (P-Db2 sanity, 6
   trainings: reached 1661/1704/1755/1763/
   1746/1722, per-door counts and open
   fractions as logged).
3. **Greedy trajectories, per tick:**
   position/action/door-bit/crush, per
   start phase, all phases — as in
   `nn/results_pda.jsonl` (P-Da, 3750 episodes,
   150 jobs) and, for P-Db2, the Step 2
   records once they exist (F0–F4). First
   door, door_entries, switches, stall
   signatures, and staging/arrival must all
   reproduce, not just success counts.
4. **Per-door entry counts and open fractions:**
   training and eval entry counts per gap
   cell (P-Da: 141–1209 per door per training;
   P-Db2 sanity: per-door counts as in
   `nn/sanity_pdb2.log`), plus schedule open
   fractions (20/10: 0.50 all doors; 30/14:
   familiar ≈0.467, rivals ≈0.533).
5. **Headline endpoints:** P-Da per-arm × world
   tables (E0 190/750, E1 712/750, E2 663/750,
   E3 0/750, E4 712/750; E2 classes R 485 /
   D 164 / W 15 / F 86), and the frozen trust
   chain named in the decision document (PC6
   or parity records, commit `ba7730e`) if the
   owner names those as the gate set at the
   second ruling.

**Option-specific costs (from the decision
document, stated here so the gate is not
misread):**

- **Option B (vectorized lockstep):** RNG streams
  change — episodes can no longer share a
  single sequential stream — so **no existing
  record is reproducible on that substrate**.
  A lockstep candidate therefore fails the
  §5(2) gate by construction unless it
  provides a reference mode that preserves
  sequential streams (which defeats the
  lockstep itself). This is the decision
  document's stated price, not a new finding
  of this spike.
- **Option A (numba JIT of the tick loop):**
  float behaviour may differ from numpy and
  must be checked, not assumed; a compiler
  dependency is added; the tick loop stops
  being readable Python; compiled code must
  be rebuilt per platform. Additionally, per
  §3, the measured bulk is in `train_step`
  (replay sampling + network updates), not in
  env/encode — so a tick-loop-only JIT's
  claimed gain must be measured against the
  §3 breakdown, and a JIT that does not cover
  `train_step` cannot be assumed to close
  most of the measured cost.
- **Tolerance question for the second ruling:**
  the gate must state in advance whether
  reproduction is bit-for-bit or
  stated-tolerance, and for which fields.
  This spike does not set that tolerance.

## 6. Gate options (facts only, no ruling)

- **Proceed on the current substrate:** Steps 1–2
  project to ~1.84 h (range 1.34–2.26 h) on
  measured pace, under both the corrected
  ~2.5 h estimate and the 4 h threshold. The
  arm-run trigger still fired (206 > 150), but
  the time threshold does not bind on these
  numbers.
- **Substrate change (A, B, or C):** still needs
  a second owner ruling after this spike, and
  then must pass the §5(2) reproduction gate
  (§5 above) and a preregistered re-baselining
  before any experiment runs on it. No gain
  for any such candidate is measured here.
- **Defer/reject:** the decision document's
  options D/E remain available to the owner;
  this spike does not rule them out.

## 7. Files

- Spike runner (new, SPIKE-labelled):
  `nn/spike_pdb2_run.py`
- Spike log (timed runs, SPIKE-labelled):
  `nn/spike_pdb2.log`
- Profile runner (new, SPIKE-labelled):
  `nn/spike_pdb2_profile.py`
- Profile output (cProfile, wrapped/0):
  `nn/spike_pdb2_profile.txt`
- This draft report: `nn/spike_pdb2.md`
- Governing documents (already committed, not
  edited here): decision document
  `~/workspace/your_files/DECISION-TBM-jit-vectorization.md`
  (§5(1), §6–§7); P-Db2 design note
  `nn/decisions_pdb2_design.md` (§8 as amended
  by instructor commit 73fd306); issued work
  order `~/workspace/your_files/WO-TBM-P-Db2.md`.

**Skipped / not doable: nothing in the spike scope
was skipped.** Bounded as instructed: no candidate
built (by design), no full 150-job eval run (by
design, sample only), one arm profiled (wrapped/0,
reason in §3), F0/F2/F4 jobs not timed directly
(bound via the tick-heaviest F3 case, §2).
No existing file was edited; `git status` shows
only the 5 new spike files (plus the pre-existing
untracked `nn/luck_notified.marker`).

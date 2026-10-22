# WO-TBM-PRB1 — Steps 1–2 report (instructor-verified, EXPLORATORY)

**Status: instructor-verified 2026-10-22. Record counts, per-rep
results, verdict arithmetic, and the RNG replay independently
re-derived from the record files before committing; one mechanism
correction applied (see §0). All conclusions EXPLORATORY,
task-and-protocol bounded. No causal language is licensed by this probe.
The trap hypothesis is neither supported nor weakened by this probe at
its pre-stated bar — see §3. Nothing here says "proven".

## 0. Deviations, restarts, and anything skipped — read first

- **Nothing skipped, substituted, or adjusted.** All 4 combos × 5 reps
  ran (450 episodes). No restarts: the runner completed in a single
  run (~1.6 s). No snapshot load failure: all three load checks PASS
  (arm/seed match, n_inputs 102/101/104, hidden 32; loaded net verified
  bitwise equal to the checked bundle). No P-Db2 weight, sanity weight,
  or re-realised weight was used anywhere.
- **No threshold, formula, or baseline was adjusted after seeing
  results.** Every verdict below applies design note §5 exactly as
  written before the run. No ε other than 0.05 was used; no wrapped
  agents; no additional worlds, reps, or agents.
- **RNG implementation (design note §4, the explicit Step 1 check
  item).** The runner NEVER calls `agent.act(x, 0.05)` — in fact it
  never calls `agent.act` at all. The ε-greedy choice is implemented
  in the runner: greedy action = `np.argmax` of the frozen network
  output (identical to the greedy branch of `NeuralQ.act`, which
  draws nothing), random action = the rep stream's `randrange(5)`.
  The agent's own RNG streams are never touched, and no numpy RNG
  draw appears in the runner (verified by grep: the only draws are
  `rng.random()` and `rng.randrange(5)` on the rep stream). The
  protocol was verified by replay: a fresh `random.Random(seed)`
  per rep, replayed over the recorded step counts in increasing
  start-phase order, reproduces every step's `epsilon_action` flag
  exactly — **67,017 steps replayed, 0 mismatches** (§1).
- **No tracked file was modified.** New files: `prb1_run.py`,
  `prb1_run.log`, `results_prb1.jsonl` (450 records),
  `prb1_analysis.py`, `prb1_analysis_output.txt`, this report.
  Probe records are in `results_prb1.jsonl` ONLY, never pooled with
  FT1 records.
- **Instructor correction (mechanism note, before committing).**
  The draft's departure analysis counted the epsilon flag of the
  step AFTER leaving (4,5), at the FIRST arrival — giving 37 greedy
  / 8 epsilon and the claim that departures are usually greedy.
  Corrected: steps record the pre-action position, so the departure
  action is the one at the LAST (4,5) step, giving 45/45 epsilon.
  The "departures are usually greedy" claim is withdrawn; random
  actions DO directly step the agent off (4,5). The stall-geography
  line was likewise corrected from "at (4,5)" to the (4,5)/(4,6)
  wall-column area. The headline verdict (NEITHER) rests only on
  per-rep reached counts and is unaffected by either correction.

## 1. Verification (records first)

- Completeness (§7 analogue of FT1's Step 0c): **450/450 episodes,
  20/20 rep-jobs, 0 duplicates**, every rep holding all start phases
  exactly once (20 or 30 per rep). No combo is incomplete.
- Every record carries `rep`, `rep_rng_seed`, `epsilon` = 0.05, and
  `snapshot_identity` (arm, seed, snapshot file, commit 9d1d6a0).
- RNG replay check: 67,017 steps, 0 mismatches (§0).
- Epsilon-step rates per rep are 3.8–5.9% across all 20 reps,
  consistent with the 0.05 coin (per-rep rates in §2).
- Greedy baselines below are FT1 numbers, cited, NOT re-run.

## 2. Per-rep results (ε = 0.05 evaluation, weights frozen)

Greedy baselines (FT1, cited): memory/3 F3 0/20, 0 entered;
plain/0 F3 20/20, mean ticks over reached 23.25; oracle_new/101 F3
0/30, 0 entered; memory/3 F2 0/20, 0 entered. Fine-tune summaries
(FT1): memory/3 161/200, plain/0 188/200, oracle_new/101 179/200.

### Primary: memory/3 F3 (20 phases per rep)

| Rep (seed) | Reached | Entered | Mean ticks (reached) | Crushes | ε steps |
|---|---|---|---|---|---|
| 0 (900000) | 10/20 | 11 | 122.60 | 1 | 164/3226 (5.1%) |
| 1 (900001) | 9/20 | 9 | 101.89 | 0 | 166/3117 (5.3%) |
| 2 (900002) | 7/20 | 7 | 125.29 | 0 | 160/3477 (4.6%) |
| 3 (900003) | 11/20 | 11 | 109.55 | 0 | 163/3005 (5.4%) |
| 4 (900004) | 8/20 | 9 | 134.00 | 1 | 176/3472 (5.1%) |
| **Total** | **45/100** | **47** | **median 128** | **2** | — |

Per-rep reached: [10, 9, 7, 11, 8]; median 9; range 7–11.

### Positive control: plain/0 F3 (20 phases per rep)

| Rep (seed) | Reached | Entered | Mean ticks (reached) | Crushes |
|---|---|---|---|---|
| 0 (901000) | 20/20 | 20 | 23.65 | 0 |
| 1 (901001) | 20/20 | 20 | 23.65 | 0 |
| 2 (901002) | 20/20 | 20 | 25.25 | 1 |
| 3 (901003) | 20/20 | 20 | 24.85 | 0 |
| 4 (901004) | 20/20 | 20 | 23.45 | 0 |
| **Total** | **100/100** | **100** | **median 23** | **1** |

Per-rep reached: [20, 20, 20, 20, 20]. Noise cost: zero episodes
lost in every rep (one crush total across 100 episodes, in a
reached episode).

### Second trap case: oracle_new/101 F3 (30 phases per rep)

| Rep (seed) | Reached | Entered | Mean ticks (reached) | Crushes |
|---|---|---|---|---|
| 0 (902000) | 4/30 | 7 | 127.75 | 0 |
| 1 (902001) | 7/30 | 10 | 143.00 | 0 |
| 2 (902002) | 6/30 | 8 | 123.50 | 1 |
| 3 (902003) | 5/30 | 9 | 167.40 | 0 |
| 4 (902004) | 7/30 | 7 | 144.71 | 0 |
| **Total** | **29/150** | **41** | **median 147** | **1** |

Per-rep reached: [4, 7, 6, 5, 7]; median 6; range 4–7.

### Second zero world: memory/3 F2 (20 phases per rep)

| Rep (seed) | Reached | Entered | Ticks |
|---|---|---|---|---|
| 0 (903000) | 0/20 | 0 | all 200 |
| 1 (903001) | 0/20 | 0 | all 200 |
| 2 (903002) | 0/20 | 0 | all 200 |
| 3 (903003) | 0/20 | 0 | all 200 |
| 4 (903004) | 0/20 | 0 | all 200 |
| **Total** | **0/100** | **0** | — |

Per-rep reached: [0, 0, 0, 0, 0]; entered: [0, 0, 0, 0, 0].

## 3. Verdict — primary question (design note §5, precedence applied)

Precedence: (1) completeness — all combos complete, no INCOMPLETE;
(2) positive control — plain/0 F3 is 20/20 in all 5 reps, so the
UNINTERPRETABLE trigger (≤ 10/20 in ≥ 3 of 5 reps) does NOT fire;
the probe is interpretable, and plain/0's per-rep counts are cited
beside the primary verdict above, as required; (3) primary verdict:

- SUPPORTED requires ≥ 10/20 in ≥ 3 of 5 reps. memory/3 F3 meets
  ≥ 10/20 in only 2 reps (10 and 11). **Not met.**
- WEAKENED requires ≤ 2/20 in ≥ 4 of 5 reps. memory/3 F3 meets
  ≤ 2/20 in 0 reps. **Not met.**

**PRIMARY VERDICT: NEITHER.** The probe does not discriminate at
its pre-stated bar. NEITHER does not mean the hypothesis is dead,
and it does not mean nothing happened — §4 states what did happen,
plainly, below the bar:

- The unlocking is REAL but PARTIAL and SUB-BAR. Every rep is
  above zero (7–11/20, median 9), against a greedy baseline of
  0/20 — noise converts a deterministic zero into a consistent
  35–55% per rep. That is far above stray levels, and it is
  consistent across all five independent streams, not one lucky
  rep. But the pre-stated SUPPORT bar was half the phases in a
  majority of reps, and only 2 of 5 reps reach it. Reporting this
  as SUPPORTED would move the bar after seeing the results; the
  note forbids that, and this report does not do it.
- The unlocking is INCONSISTENT in the sense the note means:
  per-rep reached spreads 7–11 with no rep at the greedy zero and
  no rep at the control's 20/20. The trap is broken
  probabilistically, not cleanly opened.

## 4. What the controls show

**Positive control (plain/0 F3): clean.** 100/100 reached across
all reps, mean ticks 23–25 — essentially the greedy route (23.25)
plus noise. Noise does not trivially destroy crossing, so the
memory/3 unlocking in §3 cannot be dismissed as noise-scrambling
that happens to produce crossings: the same noise leaves a
healthy crossing policy untouched.

**Second trap case (oracle_new/101 F3): NEITHER-second.** Reps
[4, 7, 6, 5, 7] meet neither ≥ 15/30 in ≥ 3 reps (0 reps) nor
≤ 3/30 in ≥ 4 reps (0 reps). Its unlocking is real but smaller
than memory/3's: 29/150 (19%) vs 45/100 (45%), entered 41/150.
Reported separately, as required; it does NOT change the primary
verdict. Agreement in verdict (both NEITHER) with different
magnitudes: the trap shape is partially shared (noise unlocks
something in both) but agent-dependent in degree.

**Second zero world (memory/3 F2): still locked.** 0/100 reached,
0 entered in every rep — not UNLOCKED, and not even partial
movement (partial movement requires entries above 0; there are
none). Noise does not unlock a different zero world for the same
agent. The F3 unlocking is world-specific: it happens where the
(5,8) route exists to be found, not generically.

## 5. Mechanism reading (descriptive; interpretations labelled)

All reached episodes in F3, for both trap-case agents, cross
via (5,8) only and stage at (4,8) only (memory/3: 45/45;
oracle_new/101: 29/29) — the column route, as in FT1's
newly-crossing episodes. Reached episodes are LONG: median
128 ticks (memory/3) and 147 ticks (oracle_new/101), against
plain/0's 23. Unlocked crossings are wandering routes, not the
clean greedy route — consistent with a dislodged-then-recovered
trajectory, not with the policy simply working.

**The naive knock-off story is NOT supported in its simple
form.** It would predict that a random action steps the agent
off (4,5) and the route follows. What the records show for
memory/3 F3:

- An epsilon action at (4,5) occurs in ALL episodes — reached
  45/45 and unreached 55/55 (medians 3 and 6 per episode; the
  higher unreached count is an exposure artifact: unreached
  episodes run all 200 ticks, reached ones leave earlier).
  Presence of a random action at (4,5) therefore does NOT
  discriminate reached from unreached. Noise perturbs the
  deterministic loop in every episode; whether the perturbation
  chains into a (4,8) crossing is state- and timing-dependent.
- The departure action from (4,5) in reached episodes is ALWAYS a
  random action: 45/45 departures are epsilon actions, 0/45 greedy.
  (The draft's 37/8 figure counted the flag of the step AFTER the
  departure, taken from the new cell, at the FIRST arrival at
  (4,5) — corrected here and in the analysis script before
  committing. Steps record the position BEFORE the action, so the
  departure action is the one recorded at the LAST step whose
  position is (4,5).) Greedy actions at (4,5) never move the agent
  in these episodes — that is the trap — so a random move action
  is what steps it off the staging cell, directly. What varies is
  whether the subsequent greedy wandering chains into a (4,8)
  crossing: that part remains state- and timing-dependent
  (INFERRED reading, not a finding).

**The two trap cases have different stall geographies**
(descriptive, from records). memory/3's trap is in the (4,5)/(4,6)
wall-column area: its F3 unreached episodes are all no-move-run
stalls (55/55), with modal cell (4,5) in 45 and (4,6) in 10, and its
F2 episodes under noise leave (4,5) in all 100 episodes yet
enter nothing — wandering the wall column (final positions
(4,8) 42, (4,5) 35, (4,6) 20, plus 3 elsewhere) without ever
stepping into a door cell. oracle_new/101's trap is NOT at
(4,5): 120/121 of its unreached F3 episodes never occupy (4,5)
at all; its late-episode positions cluster at (4,8), (3,8), and
(0,7) — it stalls around the novel door's staging area and the
spawn region instead. The two agents share the coarse shape
(high fine-tune reaching, zero greedy crossing, partial
ε = 0.05 unlocking) but not the same trap location. Any
mechanism claim must therefore be agent-specific, not a single
shared trap story.

## 6. What this means, and what it does not

- The FT1 paradox is PARTIALLY resolved, descriptively:
  memory/3's greedy 0/20 was a deterministic-evaluation
  artifact in the sense that the same frozen weights, given
  5% action noise, cross in 45% of episodes. The weights DO
  encode a usable (5,8) route under noise. But the probe's
  pre-stated verdict is NEITHER, not SUPPORTED: the unlocking
  falls short of the bar the note set before running, and the
  bar was not moved.
- WEAKENED is also not the verdict: no rep is at or near zero.
  The hypothesis is neither supported nor weakened by this
  probe at its stated bar. A future probe with more reps, a
  different ε, or a different agent set is a different probe
  with its own design note — proposals only, not executed here.
- The F2 result bounds the mechanism: noise unlocks the
  (5,8) route where it exists; it does not unlock worlds
  generally, and it does not even produce door entries in F2.
- No claim is made beyond the agents, worlds, reps, and ε
  evaluated: memory/3, plain/0, oracle_new/101; F3 and F2;
  5 reps each; ε = 0.05 only.

## 7. Files

- `prb1_run.py` (runner; never calls `agent.act`, §0), `prb1_run.log`
- `results_prb1.jsonl` (450 records, 20 rep-jobs, 0 duplicates)
- `prb1_analysis.py`, `prb1_analysis_output.txt`
- This report. Design note: `decisions_prb1_design.md` (committed
  b67a9ef, gate ruling e2f97a5). Work order:
  `~/workspace/your_files/WO-TBM-PRB1.md`.

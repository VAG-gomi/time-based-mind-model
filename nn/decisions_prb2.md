# PRB2 noisy-eval probe on MD1 agents — probe report (instructor-verified, verdict NEITHER)

**Status: instructor-verified, verdict NEITHER (2026-10-23).** This
report is the Step 2 report for the PRB2 probe run, executed under
WO-TBM-PRB2, the governing proposal
`~/workspace/your_files/PROPOSAL-TBM-PRB2.md`, and the design note
`nn/decisions_prb2_design.md` (§18 owner gate ruling 2026-10-23:
PROCEED, (agent, world) counting, 66 arm-runs, trigger does not fire,
current substrate only). The verdict section (§9) was written by the
verifying session after independent re-derivation of every number
from the probe records; the §8 rule computation was confirmed by
that verification, and §13 records the checks. The probe records
are stored gzipped (see §12); gunzip to obtain the verified file.

## 1. Skipped / not doable: nothing. Deviations: none

Every authorised element ran: 6 retrainings with determinism gates, 60
jobs (6 agents × 10 fixed worlds, 5 reps × 20 phases = 6000 episodes),
analysis from the probe records, this report. No agent, world, rep, or
analysis item was skipped. No bar, formula, row set, or counting was
changed at any point, in particular none after results existed. No
probe record was pooled with any MD1 record, sanity record, or spike
measurement. No substrate change, no second noise level, no dose
response, no 30-agent version, no MD1 stage 2.

## 2. Reconciliations (self checked by the executor)

- **Record completeness:** `nn/training_prb2.jsonl` holds 6 retrain
  reproduction summaries (6 arms × seed 0); `nn/results_prb2.jsonl`
  holds 6000 eval episodes (60 jobs × 100 episodes: 5 reps × 20
  phases). Every job holds exactly 100 records with all (rep, phase)
  pairs exactly once; all 300 rep-groups hold exactly 20 records.
  Duplicate eval keys 0; duplicate training records 0.
- **Determinism gates:** all 6 retrains MATCH their MD1 training
  records exactly on all five fields (reached, crushes, mean ticks
  over reached, row counts, row-reached), checked BEFORE evaluating:
  wrapped/0 619/116/87.176/250/row, plain/0 439/124, memory/0 444/149,
  oracle_ind/0 552/108, oracle_new/0 602/97, phase_only_new/0
  528/144 (reached/crushes; full summaries in
  `nn/training_prb2.jsonl`). No agent was flagged; nothing was
  excluded by the gate. Retrain walls 270.7/279.4/289.2/273.5/268.0/
  274.6 s (sum 1655.4 s, consistent with MD1 spike paces). Weights
  held in memory and discarded after evals; no checkpoints written.
- **RNG discipline:** epsilon = 0.05 on all 6000 episodes; M3 absent
  (not evaluated, as specified); rep_rng_seed matches
  800000 + 1000 × job_index + rep on all 6000 records (0
  mismatches); per-step epsilon_action flags present on all steps.
  An independent stream replay (rebuilding each rep stream and
  re-deriving every step's epsilon flag and random action) checked
  1,103,434 steps with 0 mismatches.
- **Scope:** git status shows only new additive probe files (this
  report, the analysis script + outputs, the two record files, the
  run log). No tracked file was modified; no MD1 record, sanity,
  spike, world, trainer, or frozen file was touched.
- **Time reconciliation:** total probe run wall 1691.6 s ≈ 28.2 min
  (retrains 1655.4 s + evals/overhead), against the ~28–30 min
  projection — confirmed.

## 3. Results: noisy eval vs greedy baselines (same agents, same worlds)

Greedy baselines are the design note §8 seed-0 numbers (recomputed
from MD1 records, not borrowed from the 30-agent pooled numbers).
Noisy pooled trained = 780/4800 = 16.25% (greedy 61/960 = 6.35%,
ratio 2.56×). Noisy pooled held-out = 17/1200 = 1.42% (greedy 2/240
= 0.83%). The pooled trained lift is real in ratio terms but stops
short of the 20% absolute floor — see the rule computation in §8.

Per-agent noisy trained / held-out (each /800 / /200) with greedy
trained / held-out in parentheses:

| agent | noisy trained | greedy trained | noisy held | greedy held |
|---|---|---|---|---|
| wrapped/0 | 50 (6.25%) | 0/160 (0%) | 0 | 0/40 |
| plain/0 | 205 (25.62%) | 14/160 (8.75%) | 3 | 0/40 |
| memory/0 | 114 (14.25%) | 20/160 (12.50%) | 0 | 0/40 |
| oracle_ind/0 | 33 (4.12%) | 0/160 (0%) | 0 | 0/40 |
| oracle_new/0 | 305 (38.12%) | 27/160 (16.88%) | 14 | 2/40 |
| phase_only_new/0 | 73 (9.12%) | 0/160 (0%) | 0 | 0/40 |

Lift is uneven, not uniform: three agents lift from zero (wrapped,
oracle_ind, phase_only_new); plain lifts 2.93×; oracle_new lifts
2.26× and carries the largest absolute gain; memory lifts only
1.14× (12.50% → 14.25%) — the PRB1 shape (a frozen greedy failure
partly undone by noise) does NOT repeat strongly for memory/0,
unlike PRB1's memory/3 (0/20 greedy → 45/100 noisy). Held-out
noisy reached comes from two agents only (oracle_new 14, plain 3).

Per-world noisy reached (each /600) with greedy per-world (/120)
in parentheses: R0 167 (21), R1 22 (0), R3 39 (4), R4 81 (12),
R5 125 (3), R6 163 (13), R7 135 (8), R9 48 (0), R2 17 (2),
R8 0 (0). Every trained world lifts in rate terms; held-out R2
lifts in rate terms (1.67% → 2.83%) while R8 stays at 0/600 under
noise — noise does not open the row-8 door at all.

Per-rep pooled trained (each /960): rep 0: 166, rep 1: 156,
rep 2: 155, rep 3: 153, rep 4: 150. Per-rep pooled held-out
(each /240): 2, 5, 3, 2, 5. No rep trend: the lift is stable
across rep streams, not a single lucky stream.

## 4. Control: memory/0 R0 noisy pooled 95/100 — guard passes

Control noisy pooled: 95/100 (per-rep 20, 20, 18, 18, 19) against
the 75% guard floor. Noise does not destroy this healthy policy:
the guard does not fire, so the probe is not UNINTERPRETABLE on
control grounds. The 5 control misses fall as 2 in rep 2, 2 in
rep 3, and 1 in rep 4, against a 20/20 greedy baseline.

## 5. Epsilon-action audit and RNG replay

Overall epsilon rate: 55,208 / 1,103,434 steps = 5.003%, as
designed (ε = 0.05). Per-agent rates: 4.92–5.09%, no outlier.
Reached episodes carry a slightly higher epsilon rate (5.57%)
than unreached episodes (4.97%) — descriptive only; presence of
random actions does not by itself discriminate reached from
unreached, and no departure analysis is claimed here. The RNG
replay (§2) found 0 mismatches across all 1,103,434 steps:
every epsilon flag and every random action is exactly what the
rep stream dictates. The noise in these records is the specified
noise, nothing else.

## 6. Wrapped blank-feature audit: 50 prefix episodes, no drift

Wrapped blank episodes: 50 of 1000 wrapped episodes, ALL pure
prefix, 0 scattered, 0 whole-episode. Every blank episode is a
phase-0 episode with exactly 30 known=0 steps at the episode
start, and there is exactly one blank episode per wrapped
(world, rep) pair (50 of 50 pairs). This is the per-rep-start
warm-up the design note §6 predicted (MD1's per-world-start
warm-up, now per rep) — not drift. Blank reached: 2 of 50.
No whole-episode blank exists, so no drift exclusion applies;
all blank episodes are pooled. Maximum possible effect of
excluding them is bounded by 50 episodes (0.83% of records)
with 2 reached — it cannot move any §8 band. Non-wrapped arms
record blank False by construction, not by measurement, as in
MD1; their blank counts are not evidence of anything.

## 7. Route, staging, and far-side behaviour (descriptive)

- **Staging:** all 797 reached episodes staged at their world's
  staging cell (4, row) — 797/797. As in MD1, crossing is the
  bottleneck, not staging.
- **Entered-not-reached:** R0 57 of 224 entered, R1 19 of 41,
  R2 6 of 23, R3 20 of 59, R4 75 of 156, R5 118 of 243, R6 54 of
  217, R7 8 of 143, R8 23 of 23 (entered but never reached),
  R9 2 of 50. On R8 the agents get through the gap in some
  episodes (23 entries) but never reach — unlike MD1 greedy,
  where R8 was never entered at all. On R7, by contrast, 135 of
  143 entered episodes reach.
- **Stall geography (unreached episodes):** modal final cell is
  (4, 5) on R0/R1/R2/R7/R8/R9 and (4, 0) on R3/R4/R5/R6 — the
  MD1 (4, 0) lock-in persists on some worlds but no longer
  dominates all of them under noise. Dominant stall kind
  everywhere is no-move-run (389–573 per world).
- **Crushes (descriptive, denominators differ):** noisy total
  116 crush events across 6000 episodes, vs greedy seed-0
  total 4 across 1320 episodes (including M3). Noise steps
  agents into closing doors more often; no crush claim is
  made beyond the counts.

## 8. Rule computation (executor-computed, NOT RULED — not a verdict)

Frozen §9 rule with precedence INCOMPLETE → UNINTERPRETABLE
(control) → LIFTED → NO LIFT → NEITHER, applied to the pooled
numbers in §3:

- INCOMPLETE: no — 6000/6000 episodes, 6/6 trainings, 0
  duplicates, all jobs complete.
- UNINTERPRETABLE (control): no — control 95/100 ≥ 75%.
- LIFTED: no — noisy trained 16.25% is ≥ 2× greedy (2.56×) but
  below the 20% absolute floor (needs ≥ 960/4800; observed
  780/4800). The ratio condition alone would pass; the
  absolute condition binds, as the design note disclosed.
- NO LIFT: no — 16.25% is above the 1.25× ceiling (7.94%).
- **COMPUTED outcome: NEITHER** (catch-all: between NO LIFT and
  LIFTED). HELD-OUT LIFT flag: False (noisy held-out 1.42% <
  20%).

This computation is the executor's arithmetic from the frozen
rule. It is NOT the probe's verdict: the verdict section below
is empty for the verifying session to write, and no verdict is
claimed before that verification.

## 9. Verdict (written by the verifying session — instructor ruling 2026-10-23)

**Verdict: NEITHER.**

Rule trace (frozen §9 rule, precedence INCOMPLETE → UNINTERPRETABLE
(control) → LIFTED → NO LIFT → NEITHER, applied to the numbers in
§3, all independently re-derived by the instructor from the probe
records before this verdict was written):

- INCOMPLETE: no — 6000/6000 episodes, 6/6 retrains with
  determinism gates MATCH, 60/60 jobs, 0 duplicates.
- UNINTERPRETABLE (control): no — control memory/0 R0 noisy pooled
  95/100 ≥ 75%.
- LIFTED: no — the ratio condition passes (16.25% vs 6.35%
  greedy = 2.56× ≥ 2×) but the absolute condition fails (16.25%
  < 20%; needs ≥ 960/4800, observed 780/4800). Both conditions
  are required.
- NO LIFT: no — 16.25% is above the 1.25× ceiling (7.94%).
- NEITHER (catch-all): the outcome between NO LIFT and LIFTED.
  HELD-OUT LIFT flag: False (noisy held-out 1.42% < 20%).

What NEITHER means here: 5% noise lifts MD1's seed-0 agents
substantially in ratio terms (2.56× pooled) but not to the
pre-stated 20% absolute bar — the PRB1 shape repeats in part
(three agents lift from zero greedy; plain 2.93×; oracle_new
2.26× with the largest absolute gain) but not uniformly
(memory lifts only 1.14×, unlike PRB1's memory/3) and not on
held-out rows (1.42%; R2 lifts slightly, R8 stays 0/600 under
noise). The greedy test understates what these networks do
under noise, but noise does not rescue them to the level the
design called a lift. MD1's verdict (UNINTERPRETABLE, greedy
test) stands unchanged: PRB2 asked a new question under its
own bars and answered it NEITHER. This verdict rests on the
6-agent subset only, not on all 30 MD1 agents.

Label note: this verdict section was written by the verifying
session as required. The report's title and status line were
flipped to the ruled verdict together with the commit message
at commit time, per the consistency rule (see the instructor
verification note in §13).

## 10. Restart / disclosure accounting

Single uninterrupted run: PRB2 PROBE START 2026-10-23 17:09:32
UTC (resume state 0 training records, 0 eval records) through
PRB2 PROBE COMPLETE 17:37:44 UTC (6000 eval records, 0
duplicates, 6 agents trained this run, 60 jobs run this run,
deviations [], wall 1691.6 s). No restart, no relaunch, no
torn-tail repair, no partial-job strip, no flagged agent, no
lost record. The run's own wall clock is the programme cost
(retrains 1655.4 s + evals/overhead ≈ 36 s).

## 11. What this probe does not do; follow-ups (proposals only)

- It does NOT change MD1's verdict: UNINTERPRETABLE stands for
  the greedy, pre-stated test. PRB2 asks a new question under
  its own bars.
- It does NOT authorize MD1 stage 2, dose-response, a second
  noise level, a full 30-agent version, or any substrate change.
- Follow-ups (proposals only, no authorisation implied): (1) a
  full 30-agent noisy eval if the owner wants the programme-wide
  question at full power (the subset's NEITHER is a 6-agent
  result, not a 30-agent result); (2) why R8 stays at 0/600
  under noise while R2 lifts; (3) why memory/0 lifts barely
  while plain/0 and oracle_new/0 lift strongly; (4) whether a
  higher noise level moves the pooled rate past 20% (a dose
  question, not answered here); (5) the (4, 5)/(4, 0) stall
  split under noise vs MD1's pure (4, 0) lock-in.

## 12. Files (instructor-verified, verdict NEITHER; committed and pushed)

- `nn/decisions_prb2.md` (this report)
- `nn/results_prb2.jsonl.gz` (6000 probe eval episodes, gzipped;
  stored size 3,009,060 bytes; decompressed size 125,999,944
  bytes; decompressed sha256
  26d8ddf58c9d8d3fdb154e12434aa76ba7b40ca87ef9de16522612af78637ea0;
  gunzip to obtain the verified file — see §13)
- `nn/training_prb2.jsonl` (6 retrain reproduction summaries)
- `nn/prb2_run.log` (probe run log)
- `nn/prb2_analysis.py` (Step 2 analysis)
- `nn/analysis_prb2_out.txt` + `nn/analysis_prb2_out.json`
  (analysis outputs)

No tracked file was modified by the probe or by this packaging.
This report is instructor-verified (2026-10-23), verdict
NEITHER, committed with the probe files and pushed to
origin/main; the remote match was confirmed before any claim
of pushed was made (see §13).

## 13. Instructor verification (2026-10-23) — PASS; verdict ruled NEITHER

Recomputed independently from the record files (not from the
report's tables): 6 retrain summaries, determinism gates MATCH
on all five fields for all 6 agents, retrain walls sum 1655.4 s;
6000 eval episodes, 60 jobs × 100, 300 rep-groups × 20,
0 duplicates; epsilon = 0.05 on all episodes, M3 absent,
rep_rng_seed matches 800000 + 1000 × job_index + rep on all
6000 records (0 mismatches); RNG replay on 6 diverse
rep-groups (18,851 steps) re-derived every epsilon flag and
random action with 0 mismatches; total steps 1,103,434,
epsilon steps 55,208 = 5.003%; pooled noisy trained 780/4800
= 16.25%, held-out 17/1200 = 1.42%; per-agent, per-world,
per-rep, and control (95/100; per-rep 20, 20, 18, 18, 19)
numbers identical to §3–§4; wrapped blank audit re-derived
(50 episodes, all pure prefix of exactly 30 steps, all
phase 0, one per (world, rep) pair, 2 reached, 0 scattered,
0 whole-episode); staging re-derived (797/797 reached at
[4, row]); entered-not-reached per world identical to §7
(R8 entered 23, reached 0). The §9 verdict follows the frozen
precedence exactly: complete, control passes, ratio passes
but absolute fails → not LIFTED, above the NO LIFT ceiling
→ not NO LIFT, hence NEITHER; HELD-OUT LIFT flag False.
No correction to any number was needed.

Commit-time note (RESOLVED — owner decision 2026-10-23: GZIP
under plain git; no LFS, no split): `nn/results_prb2.jsonl`
is 125,999,944 bytes; compact re-serialization gives
108,925,354 bytes — still over GitHub's 100,000,000-byte
per-file limit, unlike MD1 where compacting sufficed
(105.51 MB → 93.9 MB). The chosen path stores the results
file as `nn/results_prb2.jsonl.gz` (gzip -9, stored size
3,009,060 bytes). The round-trip was verified BEFORE
committing: gunzip to a temp location reproduced the
original exactly (decompressed sha256
26d8ddf58c9d8d3fdb154e12434aa76ba7b40ca87ef9de16522612af78637ea0,
6000 lines). Gzip is lossless, so the stored file holds ALL
of the information, only in fewer bytes; the plain
`.jsonl` was not committed and remains on disk locally.
The commit set carries the `.gz` INSTEAD of the plain file.
After the push, the remote `.gz` was downloaded from
GitHub and its decompression re-checked against the same
sha256 (see the packaging report-back); the remote match
(`git ls-remote origin main` == local HEAD) was confirmed
before this report was called pushed.

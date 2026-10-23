# MD1 Steps 1–3 — moving-door training (instructor-verified)

**Status: instructor-verified 2026-10-23 (see §10).** This report
covers MD1 Steps 1–3, executed under WO-TBM-MD1 and the owner
rulings in `nn/decisions_md1_design.md` §14 (spike first), §15
(PROCEED on the current substrate), and §16 (this session proceeds
NOW; the main-session ferrying path is superseded). The §8 verdict
in §6 is the frozen rule applied to independently re-derived
numbers; it is a claim only because the instructor verified it
(§10).

## 0. Skipped / not doable: nothing. Deviations: one correction

Every authorised element ran: 30 programme trainings, 330 eval jobs
(6600 episodes), analysis from the records, this report. No arm,
seed, world, or analysis item was skipped. No threshold, row set,
eval variant, or formula was changed at any point, in particular
none after results existed. No MD1 record was pooled with any
earlier programme, sanity record, or spike measurement.

One correction to the executor's own analysis code (not to any
record): the first draft of `nn/step2_md1_analysis.py` bucketed
blank episodes with a known=0 prefix longer than 10 steps as
"mixed". That label was wrong: every one of those episodes is a
pure prefix (all known=0 steps precede all known=1 steps). The
script was corrected (prefix / scattered / whole-episode
classification, no length cutoff) and re-run; the corrected
numbers are the ones in §5 (73 pure-prefix, 0 scattered, 0
whole-episode). No record was touched by the correction.

## 1. Reconciliations (checked before any result is stated)

- **Record completeness:** `nn/training_md1.jsonl` holds 30
  training records (6 arms × seeds 0–4); `nn/results_md1.jsonl`
  holds 6600 eval episodes (330 jobs × 20 phases: 10 fixed worlds
  R0–R9 + M3 × 30 agents). No missing or partial job; duplicate
  eval episodes 0; duplicate training records 0. Every (arm, seed,
  world) job holds exactly 20 records.
- **Seed-0 determinism:** all 6 programme seed-0 trainings
  reproduce their Step 0 sanity trainings EXACTLY (reached,
  crushes, mean ticks over reached, row counts, row-reached).
  These are programme records freshly trained in this run, NOT
  reuses of sanity or spike measurements; the match is a
  determinism check, not pooling.
- **Training balance:** row counts exactly 250 per trained row
  for all 30 agents (asserted by the runner; any imbalance would
  have halted the run). No held-out row appears in any training
  record.
- **M3 containment:** M3 gap rows seen are exactly the trained
  rows {0, 1, 3, 4, 5, 6, 7, 9}; no held-out row appears in M3.
- **Time reconciliation:** total programme training wall time
  8329.4 s ≈ 2.31 h, against the spike projection of ≈ 2.31 h
  for Steps 1–2 — the projection is confirmed almost exactly
  (evals add ~1 min on top; the run's own wall clock spans the
  restart gap in §7 and is not the programme cost).
- **Scope reconciliation:** `git status` shows only new additive
  files (this report, the Step 1 runner + logs, the Step 2
  analysis + outputs, the two record files). No tracked file was
  modified; `world.py`, every doors/sched/luck/pda/pdb2/ft1
  variant, every frozen file, and the sanity/spike files are
  untouched. No substrate change, no JIT/vectorized code, no
  checkpoint file was written for MD1 (agents were held in memory
  per agent and discarded after that agent's evals).

## 2. Results: fixed worlds (greedy, weights frozen, 20 phases each)

Per fixed world, reached / 600 (30 agents × 20 phases):

| world | row | reached |
|---|---|---|
| R0 | 0 (trained) | 151 |
| R1 | 1 (trained) | 7 |
| R3 | 3 (trained) | 84 |
| R4 | 4 (trained) | 77 |
| R5 | 5 (trained) | 25 |
| R6 | 6 (trained) | 33 |
| R7 | 7 (trained) | 11 |
| R9 | 9 (trained) | 0 |
| R2 | 2 (HELD-OUT) | 4 |
| R8 | 8 (HELD-OUT) | 0 |

Pooled trained: **388/4800 = 8.08%**. Pooled held-out: **4/1200
= 0.33%**. M3 (sanity only, never a per-row measurement): 42/600
= 7.0%.

Per-arm pooled (descriptive, not gating):

| arm | trained | held-out | M3 |
|---|---|---|---|
| wrapped | 40/800 = 5.00% | 0/200 | 6/100 |
| plain | 14/800 = 1.75% | 0/200 | 2/100 |
| memory | 42/800 = 5.25% | 1/200 | 2/100 |
| oracle_ind | 79/800 = 9.88% | 1/200 | 6/100 |
| oracle_new | 73/800 = 9.12% | 2/200 | 5/100 |
| phase_only_new | 140/800 = 17.50% | 0/200 | 21/100 |

The clock-only arm (phase_only_new) is the best of a weak field
on trained rows and still reaches nothing held-out. The oracle
arms — perfect gap-door information by construction — reach
9–10% on trained rows and 1–2 episodes held-out. Perfect
information does not rescue greedy eval here, which points at
the policies themselves, not at the observation contract (see
§4).

## 3. Per-agent pattern: sparse, row-specific mastery

13 of 30 agents reach nothing on any fixed world. Only 7 agents
hold any 20/20 world (memory/0 R0; memory/1 R3; oracle_ind/1 R4;
oracle_new/3 R4 and R5; phase_only_new/2 R0; phase_only_new/3 R6
and R4; phase_only_new/4 R3 — 9 perfect worlds total). No agent
crosses 4 or more worlds at ≥10/20; the widest agent
(phase_only_new/3) reaches ≥10/20 on 3 worlds. Mastery, where it
exists, is one row at a time — the address-book shape at agent
level, not a row-general policy. The full per-agent × per-row
table is in `nn/analysis_md1_out.txt`.

Training reached per arm (epsilon-greedy, 2000 episodes × 5
seeds): wrapped 2904/10000 (29.0%), plain 2137 (21.4%), memory
2296 (23.0%), oracle_ind 2794 (27.9%), oracle_new 2919 (29.2%),
phase_only_new 2587 (25.9%). Greedy eval on trained rows falls
far below training reached for every arm — most extremely for
plain (21.4% → 1.75%) and wrapped (29.0% → 5.00%). The gap
between training performance and frozen greedy performance is
itself a finding (descriptive): the final networks, evaluated
greedily from uniform start phases, cross far less than the
policies exhibited during training. Why is NOT ESTABLISHED here
— it is a follow-up question (§8), not an explanation smuggled
into this report.

Training row-reached totals across all 30 agents (of 7500
episodes per row): row 0: 1778; row 1: 1696; row 3: 2223; row 4:
2409; row 5: 2359; row 6: 2322; row 7: 1944; row 9: 906. Row 9
was trained worst (12.1%) and evaluates at 0/600 — the row the
programme already knew was hardest (lowest in Step 0 sanity for
every arm) stays hardest.

## 4. Route, staging, and far-side behaviour (descriptive)

- **Staging:** every reached episode on every fixed world staged
  at that world's staging cell (4, row) — 388/388 trained reached
  and 4/4 held-out reached. The column route generalises to the
  episodes that cross; crossing is the bottleneck, not staging.
  The 4 held-out reached episodes (memory/1 R2 phase 14;
  oracle_ind/2 R2 phase 5; oracle_new/0 R2 phases 14, 15) all
  staged at (4, 2) and arrived with the gap open in 3 of 4 cases.
- **Far side:** entered-not-reached concentrates on R0 (67),
  R4 (21), R5 (20), R1 (19); on R6 and R7 every entered episode
  reached (33/33, 11/11). R8 and R9 were never entered at all
  (0 entries in 600 episodes each). Crossing is not the whole
  story on R0/R1/R4/R5, but on R8/R9 the agents never get
  through the gap in the first place.
- **Stall geography (unreached episodes):** the modal final cell
  on every non-R0 world is (4, 0) with exactly 113 episodes —
  and it is the SAME 113 (arm, seed, phase) triples on R1, R3,
  and R9 (verified identical sets). A fixed subset of episodes
  drives to (4, 0) and stalls there regardless of where the gap
  is: a row-0 staging lock-in surviving into eval. The dominant
  stall kind everywhere is no-move-run (404–563 per world).
- **R0's own shape:** R0 is the only world with substantial
  reaching (151/600) and also the world with the most
  entered-not-reached (67). Row 0 is both the most crossed and
  the most fumbled row.

## 5. Blank-feature audit and pool-or-exclude ruling

Blank is measured only for wrapped (f101–f103 logged only on
wrapped jobs; f103 is the ScheduleModel known flag: 1.0 iff the
time features are available). Non-wrapped arms report
blank=False by construction, not by measurement — their blank
counts are not evidence of anything.

- 73 wrapped blank episodes total (of 1100 wrapped eval
  episodes): 55 are every job's first episode (start phase 0),
  each with exactly 30 known=0 steps at the episode start — a
  uniform job-start warm-up signature. The other 18 are all
  wrapped/3 (R0 phases 1–17, M3 phase 1), with known=0 prefixes
  of exactly 20 − phase steps (19 down to 3) — an exact observed
  regularity whose mechanism is NOT ESTABLISHED (a ScheduleModel
  question, not a verdict input).
- **Drift: 0 episodes.** No episode is blank for its whole
  duration; every blank episode recovers (known=1) and proceeds.
  All 18 wrapped/3 extras reached; 5 of the 55 phase-0 warm-up
  episodes reached.
- **RULING: POOL ALL blank episodes.** The design note's drift
  definition (persistent whole-episode blank with schedule-blind
  behaviour, EXCLUDED) matches nothing in the data; everything
  observed is transient warm-up shape, which the note says is
  NOT drift. Exclusion sensitivity: pooled trained 367/4743
  (7.74%) and held-out 4/1190 (0.34%) with blanks excluded —
  the DRAFT verdict (§6) is unchanged either way.

## 6. DRAFT §8 verdict (UNVERIFIED): UNINTERPRETABLE

Frozen rule, precedence INCOMPLETE → UNINTERPRETABLE →
FOLLOWING → BIGGER ADDRESS BOOK → NEITHER, applied to pooled
fixed-world numbers only (M3 never enters the verdict):

- INCOMPLETE: no — all 10 fixed worlds present, both held-out
  worlds present.
- **UNINTERPRETABLE: YES — pooled trained 8.08% < 20%.**
  Precedence stops here; FOLLOWING and BIGGER ADDRESS BOOK are
  not reached.
- For the record (not the verdict path): pooled held-out 0.33%
  also fails FOLLOWING's ≥20% absolute bar by a wide margin,
  and the middle-zone reading (held-out 10–20%) does not apply.

What UNINTERPRETABLE means here, per the frozen rule: the
agents cannot cross rows they trained on at a level the design
calls interpretable, so the design or the training is broken in
a way this programme cannot separate — **no conclusion about
following is drawn.** It is not a verdict that agents memorise
addresses (that would be BIGGER ADDRESS BOOK, which the
precedence does not reach), and it is not evidence for or
against following. The §8 parenthetical-reasoning note from the
design note §13 (the "cannot both hold" sentence) is untouched
by this report; correcting it remains an owner decision.

## 7. Restart / duplicate accounting (disclosed plainly)

- **First run (2026-10-23 06:48:49–08:42:31 UTC):** 24 agents
  completed (wrapped, plain, memory, oracle_ind, oracle_new/0–3),
  then the runner process died SILENTLY — no traceback, no
  error line, no partial record. The log ends cleanly at agent
  24/30's completion; the next agent's training had just begun
  and wrote nothing. Cause is UNKNOWN (no evidence in the log
  distinguishes a runtime restart from any other silent kill;
  the pattern matches the spike's first attempt, but that match
  is not evidence of cause). Nothing was lost: training records
  are written only after a training completes, and eval jobs are
  job-atomic at 20 records.
- **Relaunch (2026-10-23 08:51:55–09:21:54 UTC):** the
  resume-safe runner skipped all 24 complete agents, trained
  oracle_new/4 and phase_only_new/0–4, and ran their 66 eval
  jobs. Final accounting from the files: 30 training records
  (0 duplicates), 6600 eval records (0 duplicates). No record
  was rewritten or re-counted across runs; the two runs' records
  are disjoint by (arm, seed) and jointly complete.

## 8. Follow-ups (proposals only — no authorisation implied)

1. Why greedy eval falls so far below training reached,
   especially for plain and wrapped (a final-network vs
   training-policy question; possibly a probe, not a fix).
2. The wrapped/3 R0 shrinking-prefix mechanism (20 − phase
   known=0 prefixes; ScheduleModel internals question).
3. Row 9: trained worst (12.1%) and evaluated 0/600 — is row 9
   intrinsically harder (edge geometry) or an artefact of the
   row order / multiset position?
4. The (4, 0) lock-in: 113 identical episodes stall at (4, 0)
   across worlds — a memorised staging address surviving
   moving-door training.
5. MD1 stage 2 (within-episode door movement) remains a separate
   proposal; nothing here authorises it.

## 9. Files (all DRAFT, uncommitted, unpushed)

- `nn/decisions_md1.md` (this report)
- `nn/results_md1.jsonl` (6600 programme eval episodes)
- `nn/training_md1.jsonl` (30 programme training records)
- `nn/step1_md1_run.py` (resume-safe programme runner)
- `nn/step1_md1.log` + `nn/step1_md1_stdout.log` (run logs)
- `nn/step2_md1_analysis.py` (Step 2 analysis)
- `nn/analysis_md1_out.txt` + `nn/analysis_md1_out.json`
  (analysis output) + `nn/step2_md1_analysis_stdout.log`

No tracked file was modified. No commit, no push, no
instructor-verification claim has been made. The report returns
to the instructor session for verification; no verdict is
claimed before that verification.

## 10. Instructor verification (2026-10-23) — PASS

Recomputed independently from the record files (not from the
report's tables): 30 training records, 0 duplicates, row counts
exactly 250 per trained row, no held-out row in any training
record; 6600 eval episodes, 330 jobs × 20, 0 duplicates;
per-world reached R0 151, R1 7, R3 84, R4 77, R5 25, R6 33,
R7 11, R9 0, R2 4, R8 0 (pooled trained 388/4800 = 8.08%,
pooled held-out 4/1200 = 0.33%); per-arm trained/held/M3
identical to §2; training reached per arm identical to §3
(wrapped 2904, plain 2137, memory 2296, oracle_ind 2794,
oracle_new 2919, phase_only_new 2587, each /10000); training
row-reached totals identical to §3 (row 9: 906/7500 = 12.1%);
all 6 programme seed-0 trainings exact against Step 0 sanity
(reached, crushes, mean ticks, row counts, row-reached);
training wall sum 8329.4 s ≈ 2.31 h, confirming the spike
projection; M3 gap rows trained-only. Blank audit re-derived
from per-step f103 flags: 73 wrapped blank episodes, all pure
prefix, 0 scattered, 0 whole-episode; 55 phase-0 prefixes of
exactly 30 steps; 18 wrapped/3 prefixes of exactly 20 − phase
steps; exclusion sensitivity trained 367/4743, held-out 4/1190
— verdict unchanged either way, so the POOL ALL ruling stands
under the design note's drift definition. Staging re-derived:
all 392 reached episodes arrive at [4, row]; held-out reached
detail identical to §4 (memory/1 R2 phase 14, oracle_ind/2 R2
phase 5, oracle_new/0 R2 phases 14 and 15; gap open at arrival
in 3 of 4). Entered-not-reached identical to §4 (R0 67, R4 21,
R5 20, R1 19, R6 0, R7 0, R8/R9 0 entries). (4, 0) lock-in
re-derived: identical 113 (arm, seed, phase) triples stall at
(4, 0) on every non-R0 world. Per-agent pattern identical to
§3 (13 agents reach nothing; 9 perfect worlds; widest agent
phase_only_new/3 at ≥10/20 on 3 worlds). Restart accounting
confirmed from the log: first run ends cleanly at agent 24/30
(oracle_new/3, 08:42:31 UTC) with no error line; relaunch
resume state 24 trainings + 5280 evals; final COMPLETE 30
trainings, 6600 evals, 0 duplicates. The §8 verdict follows
the frozen precedence exactly: all worlds present (not
INCOMPLETE), pooled trained 8.08% < 20% → UNINTERPRETABLE,
and precedence stops there. No correction to any number was
needed; the report's "404–563" no-move-run range was confirmed
as the unreached per-world range including R0 (404).

Record-file note (instructor action, values unchanged): the
first push of this commit was rejected by GitHub because
`nn/results_md1.jsonl` was 105.51 MB, over GitHub's 100 MB
per-file limit (previous programmes' results files were all
under 50 MB). The file was re-serialised with compact JSON
separators only (no spaces after `,`/`:`); the compact file is
93,930,144 bytes and was verified value-identical to the
original by line-by-line parsed comparison of all 6600 records,
then re-checked for record count, pooled trained/held-out, and
0 duplicates (all identical). No record value, field, or
ordering changed; the analysis outputs above were derived
from these same values.

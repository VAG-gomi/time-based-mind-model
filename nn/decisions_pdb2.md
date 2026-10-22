# WO-TBM-P-Db2 — report (EXPLORATORY, instructor-verified)

**Status: instructor-verified 2026-10-22. All headline numbers
independently re-derived from the record files before committing;
one minor secondary correction applied (§5 tick range).
**Design note:** `nn/decisions_pdb2_design.md` (commits 7bc6338 + 17ac809,
amendment 73fd306, owner second ruling d689f8b: PROCEED on the current
hand-written numpy substrate, no substrate change).
**Scope:** retraining from scratch in ONE schedule-varied world (F1: gaps
{(5,1),(5,3),(5,5),(5,7),(5,9)} with P-Db W5s per-door schedules), greedy
evaluation in F0–F4. Same 6 arms/seeds as doors/P-Db/Luck/P-Da.
Familiar-only observe contract throughout: `observe()` returns
(position, familiar door bit); rival schedules appear in NO agent's
features. In F2 the observe bit toggles on the familiar schedule even
though (5,5) is a wall cell there — a phantom familiar clock, NOT a
door state; F2 agents are never described as observing a familiar door.
**All conclusions EXPLORATORY, task-and-protocol bounded. No causal
language.**

## §0. Verification, restarts, deviations — read first

**Step 0c check (done after Steps 1–2, per design note §9): PASS.**
Eval records 3750/3750, jobs 150/150, no duplicates, no partials
(every job count equals its period: 20 or 30). Training records 50/50
(25 wrapped fresh-per-world + 25 non-wrapped base), all with per-door
entry counts and schedule open fractions, all 2000 episodes. Every
agent has a training summary and an F0 evaluation. Wrapped per-seed
training door counts are identical across all 5 fresh trainings
(deterministic retraining confirmed). Stored-vs-recomputed conflict
fields agree on every episode after one encoding normalization (see
deviation (b) below): **0 substantive mismatches.**

**Restart (1, reported, not hidden).** The first run was killed by a
runtime interruption during oracle_ind/103 training (1950 episodes and
38 training records on disk at interruption; ≤1 training lost — the
in-progress one, never flushed). The runner was relaunched with
`tee -a`; it skipped every complete job and retrained oracle_ind/103
deterministically. The resumed run exited 0. No duplicates, no partials
(the verification above is post-resume).

**Deviation / judgement call (a) — blank-feature episodes, KEPT in
primary tables, flagged for instructor ruling.** 6 wrapped episodes
(all wrapped/0, F1 and F4, phases 17/18/19) have f103 == 0.0 on their
first 1–3 steps only (12 blank steps total across 500 wrapped
episodes). Audit: blank steps are the spawn steps at absolute ticks
17–19; the ScheduleModel recovers estimates exactly at the next
familiar opening (tick 20) in all 6; staging arrival follows 7–8 ticks
later; all 6 reach via (5,5); the F1/F4 twins have byte-identical step
sequences. This is transient spawn warm-up, NOT the doors-remedy drift
signature (persistent estimate loss across whole episodes with
schedule-blind behaviour). Judgement: these are not drifted rows, so
they stay pooled. Exclusion sensitivity (for the record): excluding
them leaves wrapped F1 at 97/97 reached (was 100/100, mean ticks
24.14 → 24.27) and wrapped F4 at 97/97 (was 100/100, 24.03 → 24.15).
Success rates unchanged either way. If the instructor rules otherwise,
the exclusion costs nothing on any headline.

**Instructor ruling (2026-10-22): ACCEPTED.** The audit was
independently re-derived (6 episodes, 12 blank steps, recovery at
tick 20 in all 6, all reached via (5,5), F1/F4 twins identical,
exclusion sensitivity exactly as stated). Transient spawn warm-up,
not the doors-remedy drift signature; the episodes stay pooled.

**Deviation (b) — conflict-field encoding, analysis-side only.** The
runner stores conflict=None for arrival-None episodes; independent
recomputation returns False there. Same verdict, different sentinel.
Normalized in `pdb2_analysis.py`; 0 substantive mismatches remain.
No record was changed.

**Deviation (c) — analysis script fixes, no records changed.** Two
bugs found while running the analysis and fixed in
`pdb2_analysis.py`: a variable shadowing the loaded P-Db records, and
the (b) normalization above. The analysis output file was regenerated
from the unchanged record files after both fixes.

**W-threshold fallback list (design-note rule, fixed before running).**
14 agents reached 0 in F0, so their W class uses revisits > 10 alone:
all 5 memory agents, all 5 plain agents, phase_only_new/100, /102,
/103, /104. Every other agent's W threshold is its own F0 median ticks
over reached episodes (Section 3 of the analysis output).

## §1. Headline per-arm × world (ticks: mean over reached only)

| arm | F0 | F1 | F2 | F3 | F4 |
|---|---|---|---|---|---|
| wrapped (100/world) | 70 | 100 | 100 | 0 | 100 |
| plain (100/world) | 0 | 70 | 70 | 0 | 70 |
| memory (100/world) | 0 | 90 | 90 | 20 | 90 |
| oracle_ind (150/world) | 149 | 150 | 140 | 0 | 150 |
| oracle_new (150/world) | 94 | 150 | 134 | 0 | 150 |
| phase_only_new (150/world) | 30 | 74 | 74 | 0 | 74 |
| **total (750/world)** | **343** | **634** | **608** | **20** | **634** |

Entered counts: F0 359, F1 694, F2 668, F3 20, F4 694. Crushes: F0 91
(90 of them phase_only_new/100 alone), F1 6, F2 3, F3 0, F4 6.
Full per-arm ticks/crushes/first-door tables are in
`pdb2_analysis_output.txt` §2.

## §2. Q1 — F0 sanity (REPORTED, NOT GATED): 343/750

Retrained agents solve the single familiar door far worse than in P-Da
(P-Da E0: 190/750 — already low; P-Db2 F0 is higher overall only
because oracle_ind is near-perfect here). The arm split is the finding:

- **plain 0/100, memory 0/100 — total collapse.** Both arms DO arrive
  at familiar staging (plain 48/100 episodes, memory 69/100) but enter
  **0 times**: they stage at (4,5), then migrate to trained rows and
  freeze at wall cells — last position (4,9) in 56/100 plain and 69/100
  memory episodes. Descriptively: schedule-varied retraining taught
  these arms to abandon the familiar staging row for other rows, and in
  a world where those rows hold no door, that migration is fatal.
- wrapped 70/100 (77 staged, 70 entered), oracle_ind 149/150,
  oracle_new 94/150 (95 entered), phase_only_new 30/150 (45 entered,
  90 crushes — repeated crushing by phase_only_new/100).
- F0 is sanity, not a gate: the collapse is reported as a finding, not
  a failure of the programme. It bounds every later reading: F0 medians
  exist for only 16 of 30 agents (see §0 fallback list).

## §3. Q2 — F1, the training world: preferred-door lock-in, not choice

F1 first doors (750): (5,3) 296, (5,5) 126, (5,9) 113, (5,1) 86,
(5,7) 73, none 56. Every agent has a modal (preferred) door over its
entered F1 episodes — **30/30 agents**. Preferred-door distribution
across agents: (5,3) 15, (5,9) 6, (5,5) 4, (5,1) 3, (5,7) 2; one tie
(wrapped/2: (5,5) 10 vs (5,3) 10, earliest phase 0 → (5,5)). Extreme
cases: plain/1 20/20 (5,3), phase_only_new/101 30/30 (5,1),
phase_only_new/104 30/30 (5,1).

**P-Da E1 contrast (programme-level bound — training AND eval both
changed, so this is not an isolated timing effect).** P-Da E1 first
doors: (5,1) 201, (5,3) 72, (5,5) 153, (5,7) 82, (5,9) 204, none 38
(712 entered). P-Db2 F1: (5,3) jumps 72 → 296, (5,1) falls 201 → 86,
(5,9) falls 204 → 113, (5,5) 153 → 126. Schedule variation did not
produce door choice; it moved the lock-in to a different door, usually
(5,3), and (5,5) is the preferred door of only 4/30 agents.

**F1 conflicts (generalized, 170): waits 92, takes 28, other 50**
(9 no-entry + 41 entered-a-closed-non-arrival-door). Takes by agent
are concentrated (oracle_new/103 8, memory/2 6); wrapped takes 0 in F1.
**Preferred-door-closed (268): waits-at-preferred 106, takes-open
102, other 60.** Under the preferred-door definition, agents take an
open door about as often as they wait at their preferred door — but
that is a definitional consequence of the generalized pattern plus
preferred-door assignment, not evidence of deliberation: the
generalized takes (28) are a small minority of conflicts, and the
preferred-door "takes" include every episode whose first door differs
from the modal door, including route artifacts (see §4).

## §4. Q3 — F2, rivals-only (true choice test): waits dominate, takes 21

F2 first doors (750): (5,3) 298, (5,9) 152, (5,7) 130, (5,1) 88,
none 82. Reached 608/750 (vs P-Da E2 663/750). The F1→F2 reached counts
are IDENTICAL for wrapped (100), plain (70), memory (90) and
phase_only_new (74); only the oracle arms drop (oracle_ind 150→140,
oracle_new 150→134) — removing the familiar door costs nothing for
four arms and a little for the oracle arms.

**Classes (F → D → W → R): R 460, D 176, W 32, F 82.**
Reached by class: R 432, D 174, W 2, F 0. D decomposition: 70
pass-through (exactly 1 step at (4,5)) vs 106 waited (≥2 steps at
(4,5)). **F2 D does NOT mean "abandoned the familiar door"** — there
is no familiar door in F2; D records only that the trajectory occupied
(4,5) before entry, which the wall-column route does routinely. All 32
W episodes are phase_only_new (2 reached); W is a failure class here
by construction (ticks > 2× own F0 median or revisits > 10, and failed
episodes sit at the 200 cap).

**Generalized conflicts (222): waits 150, takes 21, other 51** (9
no-entry + 42 entered-a-closed-non-arrival-door).
**Preferred-door-closed (299, phantom clock for a (5,5) preferred
door): waits-at-preferred 128, takes-open 74, other 97.**

**P-Db contrast (stated explicitly, with the familiar-presence
difference).** P-Db's W5s subset, recomputed from
`results_doors_sched.jsonl`: 257 conflicts → 236 waits / 21 takes /
0 other — but W5s CONTAINED the familiar door (5,5) on its training
schedule, and its agents were single-door trained. F2 is a NEW world:
no (5,5), retrained agents, phantom clock. F2's take count is also 21,
but the composition differs completely: P-Db's takes were wrapped/2 (6,
clean/fast), oracle_new/101 (14, all far-side freezes) and memory/1
(1); **P-Db2's 21 F2 takes are all reached, none by wrapped**, by
plain/3 (3), memory/2 (6), memory/3 (1), memory/4 (2), oracle_ind/100
(1), oracle_ind/103 (2), oracle_new/100 (3), oracle_new/102 (3) —
and every one carries a staging wait of 12–27 ticks before entering a
door that was open at arrival. These are wall-walking arrivals at a
second door after a long staging interval, not quick diversions: the
arrival doors are (5,1) 12, (5,3) 6, (5,7) 3, and the entered doors are
(5,7)/(5,9)/(5,3) — never the arrival door. **Zero diversion-class
takes in the P-Db sense (a fast switch to an open rival) appear in
F2; the 21 takes are slow, reached, and spread across 8 agents.**

**P-Da E2 contrast (cleanest retrained rivals-only contrast).** P-Da
E2: 663/750 reached, 664 entered, classes R 485 / D 164 / W 15 / F 86.
P-Db2 F2: 608/750 reached, 668 entered, classes R 460 / D 176 /
W 32 / F 82. Same shape, slightly lower reach, more W (all
phase_only_new). Schedule-varied retraining did not change the
rivals-only outcome class structure.

## §5. Q4 — F3, held-out novel door (5,8): 20/750, ONE agent

**F3: 20 reached / 750, 20 entered / 750 — all 20 are memory/3, all 20
start phases, all class D, 0 crushes, 0 post-entry stalls, mean
post-entry ticks 4.0.** Every other agent: 0 entered, 0 reached.
memory/3's other worlds: F0 0/20, F1 20/20 ((5,3) 18, (5,9) 2), F2
20/20 (same), F4 20/20 ((5,3) 18, (5,8) 2). Its F3 ticks run 31→18
across phases 0–13, then 37 at phase 14 and 35→32 across phases
16–19 (phase 15: 18 ticks, in the first regime) — two timing regimes,
both reaching.

**P-Da E3 contrast (exact repeat of the world):** P-Da E3 was 0/750
reached, 18 entered, all wrapped/4, all crushed, none crossed. P-Db2
F3 changes the entered count from 18 crushed non-crossings to 20 clean
crossings — but by a single agent, not by any arm. **Held-out verdict:
still memorisation at arm level, with one per-agent exception.**
memory/3 crosses a novel door it was never trained on, cleanly, in
every phase — that is a genuine per-agent counter-example to "no
agent crosses a novel door", and it does not license an arm-level or
generalisation claim. The D label on all 20 is a route artifact: the
wall-column route to (4,8) passes (4,5); in F3 (4,5) is ordinary floor
west of a wall cell, and no familiar door exists to abandon.
**Interpolation bound (carried):** (5,8) sits between trained (5,7)
and (5,9) — a pass here is interpolation, not extrapolation, and in
this 10-row geometry every novel row is adjacent to a trained row, so
interpolation is the strongest novel-door test the grid allows.

## §6. Q5 — F4, novel among familiars: novel-first 2, novel-anywhere 15

F4 reached 634/750, entered 694 — per-arm identical to F1 except
memory ((5,9) 41→39, plus (5,8) 2). **Novel-first episodes: 2 — both
memory/3, phases 13 and 15, both reached in 18 ticks, revisits 0,
(4,5)-before-entry True** (the same agent as F3; F4 novel-first is
reported with the (4,5)-flag/ticks/revisits only and is NOT classified
D/R/W/F, per the design note). **Novel entries anywhere: 15** — the
other 13 are wrapped/0, phases 0–12, each entering (5,9) first and
(5,8) second (a return crossing after a familiar first crossing, then
a re-cross eastward to the goal; the entry counter records gap
changes, so a same-gap re-cross adds no third event). So the novel
door is USED in 15/694 entered F4 episodes, FIRST in only 2.
**P-Da E4 contrast:** P-Da E4 novel-first was 0/750. P-Db2 F4
novel-first is 2/750, both one agent; novel-anywhere use is 15/694
entered, dominated by one agent's second-crossing pattern. No
arm-level novel preference appears when familiars exist.

## §7. Q6 — Far side after entry: freezes concentrate in 3 agents

Two different far-side quantities must not be blurred (P-Da's report
used the second):

- **Stall flags** (any post-entry stall signature): F1 120/694, F2
  110/668, F4 120/694, F0 35/359, F3 0/20. Most flagged episodes still
  reach (e.g. wrapped/2's 16–26-tick post-entry no-move-runs before
  reaching); the flag alone is not a freeze.
- **Freezes** (post-entry stall AND not reached, i.e. frozen to the
  cap): **F1 60, F2 60, F4 60, F0 16, F3 0.** The F1/F2/F4 60 are the
  SAME three agents in every world: phase_only_new/101 (30),
  plain/1 (20), phase_only_new/100 (10). F0's 16 are
  phase_only_new/100 (15) + oracle_new/103 (1).

**P-Da comparison:** P-Da E2+E4 had 1376 entered, 59 stall flags, and
**1 freeze**. P-Db2 F2 has 668 entered, 110 stall flags, 60 freezes —
but the freeze count is an agent-degeneracy count (3 agents freezing
in every world they enter), not a world-spread far-side effect. The
like-for-like statement is: far-side freezing after a trained-door
entry is rare-to-absent for 27/30 agents and total for 3 degenerate
agents; after a NOVEL entry (F3, memory/3) it is 0/20 with 4-tick
post-entry crossings. P-Dc's far-side question is therefore answered
in split form: clean crossing for the one novel-crossing agent,
degenerate freezing for three familiar-entry agents, and no
novel-entry far-side freeze exists to observe beyond memory/3.

## §8. Q8 — Training breadth vs eval breadth (joint reading)

All 30 base trainings visited every door (per-door entry counts
across trainings: (5,1) 96–480, (5,3) 430–1295, (5,5) 130–799,
(5,7) 196–615, (5,9) 304–856; reached 1658–1797/2000). Open fractions
as designed (20/10: 0.50 every door; 30/14: familiar 0.467, rivals
0.533 — too near-equal to explain door preference by exposure alone,
and the preferred doors do not track the highest-fraction doors).
Joint reading per the design-note rule: **lock-in + broad visits =
"location/schedule variation was insufficient"** — this programme
rules out the "training never varied" explanation that P-Da left
open for location alone, and extends it to location+schedule: every
agent trained through hundreds of entries at every door, on staggered
schedules, and still evaluates with a modal door (30/30 agents) and
usually not the familiar one. The variation changed WHICH door is
preferred (F1 modal (5,3), P-Da E1 modal spread with (5,1)/(5,9)
leading), not whether preference exists.

## §9. Q7 — Wrapped vs plain vs oracle after retraining

- **wrapped (time features):** best single-door retention among the
  non-oracle arms (F0 70/100) and perfect in multi-door worlds
  (F1/F2/F4 100/100), but 0 takes in F1/F2 conflicts and 0 in F3. Time
  features preserve performance, not choice.
- **plain (no time features):** F0 0/100, F1/F2/F4 70/100, F3 0/100.
  The F0 collapse (§2) is the plain-arm signature: without time
  features, schedule-varied training produces a policy that cannot
  even stage-and-enter the one door it was originally trained on.
- **memory (one-step history):** F0 0/100 like plain, F1/F2/F4 90/100,
  and the ONLY F3 reaches (memory/3, 20/20) plus the only F4
  novel-first episodes. The arm-level F3 result is 20/100 — but it is
  one agent, and the other four memory agents are 0/20 in F3, so this
  is not a memory-arm effect that can be claimed; it is a per-agent
  exception inside the arm with the same F0 collapse as plain.
- **oracle_ind (perfect familiar-clock information):** F0 149/150,
  F1/F4 150/150, F2 140/150, F3 0/100. Perfect information buys
  near-perfect familiar-world performance and nothing on the novel
  door.
- **oracle_new:** F0 94/150, F1/F4 150/150, F2 134/150, F3 0/100.
- **phase_only_new (raw clock):** F0 30/150 with 90 crushes, F1/F2/F4
  74/150, F3 0/100, and all 32 F2 W episodes. The raw clock is the
  weakest schedule representation here, as in earlier programmes.

Time-awareness and perfect information change success levels, not the
choice structure: preferred-door lock-in holds in every arm, novel-door
first use is confined to one memory agent, and conflict takes remain a
minority in every arm (F2 takes by arm: plain 3, memory 9, oracle_ind
3, oracle_new 6, wrapped 0, phase 0).

## §10. Answers to the eight pre-stated questions (numbers only)

1. **F0 sanity?** 343/750 reached. plain 0/100, memory 0/100,
   wrapped 70/100, oracle_ind 149/150, oracle_new 94/150,
   phase_only_new 30/150. Reported, not gated.
2. **F1 distribute or lock in?** Lock in: 30/30 agents have a modal
   door; preferred distribution (5,3) 15, (5,9) 6, (5,5) 4, (5,1) 3,
   (5,7) 2; one tie. Programme-level P-Da E1 contrast: (5,3) 72→296,
   (5,1) 201→86, (5,9) 204→113, (5,5) 153→126.
3. **F2 waits vs takes?** Generalized 222 → waits 150, takes 21,
   other 51. Preferred-door-closed 299 → waits 128, takes 74, other
   97. P-Db W5s contrast: 257 → 236/21/0 with familiar present and
   single-door agents; P-Db2 F2's 21 takes are all reached, slow
   (12–27-tick staging waits), none by wrapped. P-Da E2 contrast:
   608/750 reached vs 663/750, classes R 460/D 176/W 32/F 82 vs
   R 485/D 164/W 15/F 86.
4. **F3 held-out change?** Yes, narrowly and per-agent: 20/750 reached
   vs P-Da E3 0/750 — all 20 memory/3, clean crossings, interpolation
   bound applies. Arm-level verdict remains memorisation.
5. **F4 novel use?** Novel-first 2/750 (memory/3 only) vs P-Da E4 0;
   novel-anywhere 15/694 entered (13 wrapped/0 second-crossings).
6. **Far side?** Freezes: F1/F2/F4 60 each (same 3 agents),
   F0 16, F3 0. P-Da E2+E4: 1 freeze in 1376 entered. F3 novel
   crossings: 0 freezes, 4-tick post-entry.
7. **Wrapped vs plain vs oracle?** See §9. Success differs by arm;
   choice structure (modal door, minority takes, novel-first
   confinement) does not.
8. **Training breadth vs eval breadth?** Every training visited every
   door (≥96 entries per door); eval still modal per agent. Joint
   reading: location/schedule variation insufficient, not absent.

## §11. Bounds and limits

- F1 vs P-Da E1 is a programme-level contrast (training world and eval
  world both changed); no isolated timing effect is claimed.
- F2's observe bit is a phantom familiar clock; oracle "perfect
  information" in F2 means perfect familiar-clock information, not
  rival schedules (never observed by any agent).
- F3 is an interpolation test ((5,8) between (5,7)/(5,9)); no
  extrapolation claim is made or licensed.
- F4 novel entries are not classified D/R/W/F; the (4,5)-flag on the
  two novel-first episodes is reported, not interpreted as diversion
  (a familiar door exists and is open in F4).
- F1 episodes are not classified D/R/W/F.
- The F3 result rests on one agent (memory/3). No arm-level
  generalisation claim follows from it, and no prediction about
  retraining other agents follows either.
- All counts are greedy-evaluation counts on the stated worlds, arms
  and seeds. No claim extends beyond them.

## Files (DRAFT, uncommitted)

- `nn/pdb2_run.py` — runner (additive; smoke-tested before launch,
  including a walking-dummy conflict-logic check).
- `nn/pdb2_run.log` — run log (two launches: first killed mid-training
  by a runtime interruption, second resumed to exit 0).
- `nn/results_pdb2.jsonl` — 3750 eval records, 150 jobs.
- `nn/training_pdb2.jsonl` — 50 training records.
- `nn/pdb2_analysis.py`, `nn/pdb2_analysis_output.txt` — analysis
  script and its output (regenerated after the two script fixes in
  §0(c); records unchanged).
- `nn/decisions_pdb2.md` — this report (DRAFT).

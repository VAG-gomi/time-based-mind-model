# WO-TBM-FT1 — Steps 1–3 report (instructor-verified, EXPLORATORY)

**Status: instructor-verified 2026-10-22. Headline numbers, classes,
conflicts, far-side counts, and the blank audit independently
re-derived from the record files before committing; two corrections
applied (see §0). All conclusions EXPLORATORY, task-and-protocol
bounded. No causal language is licensed by this experiment.

## 0. Deviations, restarts, and anything skipped — read first

- **One runtime restart interrupted the first run.** The first `ft1_run.py`
  run died after wrapped/3's P-Db2 checkpoint had been verified exact
  (1710/212) but before any fine-tune record or fine-tuned snapshot for
  wrapped/3 existed — i.e. at most one agent lost, as designed. The runner
  was relaunched resume-safe (log appended with a relaunch banner):
  wrapped/0,1,2 skipped as complete, wrapped/3 re-realised and verified
  exact AGAIN (1710/212, identical — determinism confirmed), and the run
  continued to completion. No record was duplicated, no agent was skipped,
  no training was restarted from scratch. This interruption is reported,
  not hidden.
- **Nothing else was skipped, substituted, or adjusted.** No set-aside
  agent was fine-tuned; no dose other than 200 episodes was used; no
  hyperparameter was changed mid-run; no P-Db2 weight was substituted for
  a failed fine-tune; no sanity or spike fine-tuned weight was reused —
  every Step 1 fine-tune ran FRESH from its checkpoint (6 Step 0 snapshots
  or 16 freshly realised checkpoints, each verified before use).
- **Three agents reached 0/200 in fine-tuning** (wrapped/3, oracle_ind/102,
  oracle_new/100). These are legitimate outcomes under the design-note
  validity criteria (completion without crash/NaN), NOT failures: all
  three were evaluated in all six worlds and are included in every count.
- **No agent was excluded by the Step 0c check.** All 22 agents have a
  recorded checkpoint, a fine-tune summary, and all six eval worlds.
- **Blank-feature episodes: 30, all audited as transient spawn warm-up,
  NOT drift, and pooled.** See §8. Exclusion sensitivity: headlines move
  by at most 3 episodes (F3 237/540 → 234/535; F4 304/540 → 301/535;
  all other worlds unchanged in count). Pooling stands.
  **Instructor ruling: ACCEPTED** — start-only blank with recovery
  at tick 30 and normal behaviour after (the tick-30 vs P-Db2's
  tick-20 difference is noted in §8 and follows from the
  fresh-encoder-per-world rule); sensitivity nil on every headline.
- **Instructor correction (§10, before committing).** The draft
  contrasted FT1's preferred-door-closed counts against P-Db2's
  30-agent programme totals and misquoted F1 waits as 102 (that is
  the programme takes count; programme waits are 106). Corrected to
  like-for-like kept-set baselines recomputed from P-Db2 records:
  F1 81 waits / 93 takes / 26 other, F2 102 waits / 71 takes /
  63 other. The qualitative conclusion (waiting essentially gone:
  81 → 1 and 102 → 0) is unchanged.

## 1. Verification (records first)

- Fine-tune summaries: **22/22**, each with checkpoint identity
  (6 `snapshot`, 16 `realised`; §9).
- Eval records: **3,240/3,240 episodes, 132/132 jobs, 0 duplicates**,
  every agent in all six worlds (20 phases for 20/10 arms, 30 for 30/14
  arms: 12×6×20 + 10×6×30 = 3,240).
- Step 0c per agent: all 22 OK (checkpoint + fine-tune + 6 worlds).
- Conflict recomputation from trajectories: **0 mismatches** in both F1
  and F2 (recorded conflict fields match recomputation exactly).
- Headline totals below were independently recomputed from
  `results_ft1.jsonl` (not taken from the runner's log summaries) and
  match `ft1_analysis_output.txt` exactly.
- No tracked file was modified. New files: `ft1_run.py`, `ft1_run.log`,
  `training_ft1.jsonl`, `results_ft1.jsonl`,
  `checkpoints_ft1_finetuned/` (22 snapshots), `ft1_analysis.py`,
  `ft1_analysis_output.txt`, this report.

## 2. Headline — per-world reached (kept set, 540 episodes per world)

| World | After FT1 | P-Db2 kept-set before | Change |
|---|---|---|---|
| F0 W1 {(5,5)} sanity | **38/540** | 253/540 | −215 |
| F1 training world | **76/540** | 540/540 | −464 |
| F2 rivals-only staggered | **38/540** | 540/540 | −502 |
| F3 {(5,8)} alone | **237/540** | 20/540 | +217 |
| F4 W5n-sched | **304/540** | 540/540 | −236 |
| G1 {(5,6)} alone, after-only | **0/540** | (no pre-measurement) | — |

The one-line story, descriptively: **200 episodes on (5,8) alone bought
partial (5,8) acquisition at the price of near-total multi-door
forgetting, and bought zero transfer to the second novel door (5,6).**
Acquisition and forgetting are both partial and agent-dependent — §3–§7
give the per-agent split, which is where the actual finding lives.

## 3. Per-arm × world (reached/total, entered, mean ticks over reached, crushes)

| Arm | F0 | F1 | F2 | F3 | F4 | G1 |
|---|---|---|---|---|---|---|
| wrapped (100/world) | 0, ent 0, —, cr 0 | 7, ent 7, 24.86, cr 0 | 7, ent 7, 24.86, cr 0 | 60, ent 60, 31.35, cr 1 | 67, ent 67, 30.51, cr 1 | 0, ent 1, —, cr 0 |
| plain (60/world) | 0, ent 0, —, cr 0 | 0, ent 0, —, cr 0 | 0, ent 0, —, cr 0 | 20, ent 40, 23.25, cr 0 | 20, ent 40, 23.25, cr 0 | 0, ent 0, —, cr 0 |
| memory (80/world) | 20, ent 39, 30.05, cr 2 | 20, ent 41, 30.05, cr 2 | 0, ent 20, —, cr 0 | 31, ent 50, 25.03, cr 0 | 49, ent 70, 27.00, cr 2 | 0, ent 4, —, cr 0 |
| oracle_ind (120/world) | 0, ent 0, —, cr 0 | 30, ent 60, 97.50, cr 0 | 30, ent 60, 97.50, cr 0 | 30, ent 30, 28.97, cr 1 | 60, ent 90, 63.23, cr 1 | 0, ent 0, —, cr 0 |
| oracle_new (120/world) | 0, ent 0, —, cr 0 | 0, ent 29, —, cr 180 | 0, ent 29, —, cr 180 | 54, ent 54, 25.61, cr 0 | 54, ent 83, 25.61, cr 156 | 0, ent 0, —, cr 0 |
| phase_only_new (60/world) | 18, ent 24, 47.44, cr 0 | 19, ent 25, 46.05, cr 0 | 1, ent 1, 21.00, cr 0 | 42, ent 42, 29.31, cr 0 | 54, ent 60, 33.83, cr 0 | 0, ent 0, —, cr 0 |
| **TOTAL (540/world)** | **38, ent 63** | **76, ent 162** | **38, ent 117** | **237, ent 276** | **304, ent 410** | **0, ent 5** |

First-door distributions (totals): F0 — (5,5) 63, none 477. F1 — none 378,
(5,5) 63, (5,3) 10, (5,7) 29, (5,9) 30, (5,1) 30. F2 — none 423, (5,3) 10,
(5,7) 47, (5,9) 30, (5,1) 30. F3 — (5,8) 276, none 264. F4 — (5,8) 248,
none 130, (5,5) 63, (5,3) 10, (5,7) 29, (5,9) 30, (5,1) 30. G1 — (5,6) 5,
none 535.

Notes on the table (descriptive): oracle_ind's F1/F2 numbers are identical
(30 reached, 60 entered, 97.50 ticks) because its only surviving agent,
oracle_ind/102, takes the same trained doors in both worlds. oracle_new's
180 crushes in each of F1 and F2, and 156 in F4, are entirely
oracle_new/100's crush loop (§6). memory's F0 and F1 reached counts are
identical (20) because the same agent (memory/3) reaches via (5,5) in
both. wrapped's F1 and F2 numbers are identical (7) because wrapped/2's
7 reaches use the same route in both worlds.

## 4. G1 TRANSFER — PROMINENT, AFTER-ONLY (no pre-measurement)

**G1 result: 0 crossings out of 540 episodes. No agent crossed (5,6) to
the goal. This is stated prominently because it bears directly on the
memorisation verdict: 200 episodes on (5,8) alone did NOT produce a
crossing at the second novel door (5,6), even though the same dose
produced 237/540 crossings at the trained novel door (5,8).**

G1 is after-only: there is no pre-fine-tuning G1 measurement, so this is
NOT a before/after contrast. The comparison licensed is analogy only:
P-Db2's F3 pattern (one novel door, no training) was 20/540 reached, all
by one agent; FT1's G1 pattern (one novel door, no training, after
(5,8) fine-tuning) is 0/540 reached.

What did happen in G1: **5 entries, 0 reaches, all 5 far-side frozen.**
The entries: wrapped/1 phase 18 (class W, no (4,5) before entry, staged
at (4,6), then 188 post-entry ticks stalled, 200 total) and memory/2
phases 10, 11, 12, 13 (class D — with the floor-not-door caveat: in G1,
(4,5) is a floor cell, not a door staging cell, so the D label records
(4,5) occupation before (5,6) entry and does NOT mean "abandoned the
familiar door"; each then 187–188 post-entry ticks stalled, 200 total).
So two agents can ENTER (5,6) after (5,8) training, but neither can
CONTINUE past it to the goal — the same enter-but-freeze shape seen in
plain/2's (5,8) episodes (§6). Entry without continuation is not a
crossing, and it is not transfer of the crossing skill.

## 5. F3 acquisition — per agent (reached/entered, of 20 or 30 phases)

| Agent | F3 reached | F3 entered | Fine-tune reached (of 200) |
|---|---|---|---|
| wrapped/0 | 20/20 | 20 | 105 |
| wrapped/1 | 20/20 | 20 | 158 |
| wrapped/2 | 0/20 | 0 | 97 |
| wrapped/3 | 0/20 | 0 | 0 |
| wrapped/4 | 20/20 | 20 | 127 |
| plain/0 | 20/20 | 20 | 188 |
| plain/2 | 0/20 | 20 | 22 |
| plain/3 | 0/20 | 0 | 20 |
| memory/0 | 0/20 | 19 | 30 |
| memory/2 | 20/20 | 20 | 24 |
| memory/3 | 0/20 | 0 | 161 |
| memory/4 | 11/20 | 11 | 180 |
| oracle_ind/100 | 0/30 | 0 | 12 |
| oracle_ind/101 | 30/30 | 30 | 92 |
| oracle_ind/102 | 0/30 | 0 | 0 |
| oracle_ind/103 | 0/30 | 0 | 6 |
| oracle_new/100 | 0/30 | 0 | 0 |
| oracle_new/101 | 0/30 | 0 | 179 |
| oracle_new/102 | 30/30 | 30 | 88 |
| oracle_new/103 | 24/30 | 24 | 182 |
| phase_only_new/102 | 30/30 | 30 | 103 |
| phase_only_new/104 | 12/30 | 12 | 178 |

Acquisition answer: **11/22 agents cross (5,8) at least once in F3
(vs 1/22 before); 237/540 episodes reached (vs 20/540 before).** Eight
agents are perfect in F3 (wrapped/0, wrapped/1, wrapped/4, plain/0,
memory/2, oracle_ind/101, oracle_new/102, phase_only_new/102); three
more are partial (memory/4 11/20, oracle_new/103 24/30,
phase_only_new/104 12/30); eleven agents never cross (including
memory/3 — see §7 — and three agents that never even enter).

F3 classes (priority F → D → W → R): F 264 (0 reached), D 144
(125 reached), W 15 (15 reached, all of them), R 117 (97 reached).
Entered-but-not-reached: 39 episodes, entirely memory/0 (19) and
plain/2 (20) — both enter (5,8) and then stall on the far side (§10).
Crushes in F3: 2 total (wrapped 1, oracle_ind 1) — acquisition here is
essentially crush-free, unlike oracle_new/100's crush loops in F1/F2/F4.

## 6. Forgetting — per agent (before → after, P-Db2 kept-set baselines)

F1 (was 540/540, now 76/540). Unchanged agents: memory/3 (20/20),
oracle_ind/102 (30/30). Partial survivors: wrapped/2 (7/20),
phase_only_new/102 (1/30), phase_only_new/104 (18/30). Every other
kept agent went to 0/20 or 0/30 — including all of plain, all of
oracle_new, oracle_ind/100, oracle_ind/101, oracle_ind/103,
memory/0, memory/2, memory/4, wrapped/0, wrapped/1, wrapped/3,
wrapped/4.

F2 (was 540/540, now 38/540). Survivors: oracle_ind/102 (30/30),
wrapped/2 (7/20), phase_only_new/102 (1/30). Every other kept agent
0 — including memory/3, whose F2 went 20/20 → 0/20 with 0 entries
(it routes to (4,5) and waits there; §7), and phase_only_new/104,
whose F2 went 30/30 → 0/30 while its F1 survived at 18/30.

F4 (was 540/540, now 304/540). Unchanged: memory/3 (20/20),
oracle_ind/101 (30/30), oracle_ind/102 (30/30), oracle_new/102 (30/30),
wrapped/0 (20/20), wrapped/1 (20/20), wrapped/4 (20/20), plain/0 (20/20),
phase_only_new/104 (30/30). Partial: memory/2 (18/20), memory/4 (11/20),
oracle_new/103 (24/30), phase_only_new/102 (24/30), wrapped/2 (7/20).
Zero: memory/0, plain/2, plain/3, oracle_ind/100, oracle_ind/103,
oracle_new/100, oracle_new/101, wrapped/3.

F0 (was 253/540, now 38/540). The collapse DEEPENED overall — but two
agents moved the other way, and both matter: memory/3 F0 0/20 → 20/20
and phase_only_new/104 F0 0/30 → 18/30. All of F0's 38 reaches are
exactly these two agents (memory 20/80 + phase 18/60; wrapped, plain,
oracle_ind, oracle_new all 0). So (5,8) training did not uniformly
worsen F0: it destroyed (5,5)-only performance in 20 of 22 agents
while repairing it in the two agents that had it worst before. No
mechanism for that split is established by this experiment; it is
reported as an agent-level fact, not explained.

Degenerate post-fine-tuning shapes (descriptive labels for §13):
- **Enter-but-freeze:** oracle_ind/103 enters in F1 (30/30 entered,
  0 reached), F2 (30/30, 0 reached) and F4 (30/30 entered, 0 reached),
  stalling on the far side in every entered episode; plain/2 enters
  (5,8) first in all 20 F4 episodes and all 20 F3 episodes, reaching
  none (far-side freeze in all 40); memory/0 enters in F1/F2/F4 and
  reaches none (far-side stall in 19 F1 + 19 F2 + 19 F4 episodes).
- **Crush loop:** oracle_new/100 — 180 crushes in F1, 180 in F2,
  156 in F4, 0 reached anywhere; in F4 phases 0–3 it alternates
  (5,7)/(5,8) entries 22 times per episode without reaching.
- **No acquisition, no forgetting:** oracle_ind/102 — fine-tune 0/200,
  yet F1/F2/F4 retained at 30/30 and F3 0/30. Fine-tuning left this
  agent essentially untouched in eval.
- **Total loss:** wrapped/3 — fine-tune 0/200, 0 reached in all six
  worlds after having been perfect in F1/F2/F4 before.

## 7. memory/3 — the paradox agent (Question 6)

memory/3 was P-Db2's singular agent: the ONLY agent to cross (5,8) in
F3 before fine-tuning (20/20, clean). After 200 episodes on (5,8)
alone, **memory/3 does NOT cross (5,8) in F3 at all: 0/20 reached,
0/20 entered, all 20 episodes 200 ticks, all class F.**

What it does instead, directly observed in its F3 trajectories: it
walks to (4,5) — the familiar staging cell — and then predominantly
issues 'right' against the (5,5) wall cell (3661/4000 steps; the rest
stay/down), staying at (4,5) for the rest of the episode (stall kind
'no-move-run', revisits 0, staging_order empty, arrival None). It never
occupies (4,8). Meanwhile its fine-tune training summary on the SAME
(5,8)-alone world is 161/200 reached with 167 (5,8) entries — so the
weights DO encode (5,8) crossings under training conditions but the
greedy eval policy does not use them.

The leading hypothesis (INFERRED, not a finding): a deterministic
greedy trap. Training runs at ε = 0.05, so ~5% random actions can
dislodge the agent from (4,5) and let the learned (5,8) route take
over; greedy eval has no such dislodging mechanism, so the agent pushes
right at (4,5) forever. This hypothesis is testable by a follow-up
probe (ε = 0.05 eval for memory/3 in F3) — proposed only, NOT run
here, because it is outside FT1's scope.

memory/3's other worlds after fine-tuning: F0 0/20 → **20/20**
(repaired), F1 20/20 (retained), F2 20/20 → **0/20** (lost, 0 entries —
same (4,5) trap: in rivals-only F2 there is no (5,5) to open, so
waiting at (4,5) never pays), F4 20/20 (retained), G1 0/20 (0 entries).
So memory/3 post-fine-tuning is a (5,5)-routing agent in eval: it kept
exactly the worlds where (5,5) opens and lost exactly the worlds where
it does not — except F3, where its own fine-tuning world should have
given it (5,8) and did not, in greedy eval.

Question 6 answer, plainly: **No — fine-tuning did NOT preserve
memory/3's F3 perfection; it removed it (20/20 → 0/20 entered 0) in
greedy eval, while its F0 improved (0/20 → 20/20).**

## 8. Blank-feature audit (wrapped only) — transient warm-up, NOT drift

Blank episodes: **30 of 600 wrapped episodes** — exactly one per
(wrapped agent, world): all five wrapped agents, phase 0, in all six
worlds. In each, f103 = 0.0 for the first 30 steps, with the first
f103 = 1.0 at tick 30; behaviour after tick 30 is normal (the F3 and
F4 phase-0 episodes of wrapped/0, wrapped/1 and wrapped/4 reach via
(5,8); the F0/F1/F2/G1 phase-0 episodes fail the same way their
agents' other phases fail, not in a schedule-blind way).

Audit decision: this is transient spawn warm-up (fresh encoder's
ScheduleModel starting blank at episode start), NOT drift as defined
in the design note (persistent whole-episode blank with
schedule-blind behaviour). The audit, not the flag alone, decides —
and here the flag is confined to episode start with recovery and
normal behaviour after, so all 30 episodes are POOLED. Exclusion
sensitivity (dropping all 30): F0 38/540 → 38/535, F1 76/540 → 76/535,
F2 38/540 → 38/535, F3 237/540 → 234/535, F4 304/540 → 301/535,
G1 0/540 → 0/535. No headline changes. (Note the counting difference
from P-Db2's audit: P-Db2's 6 blank episodes recovered at tick 20;
FT1's 30 recover at tick 30 with 30 blank steps. The recovery tick
differs; the warm-up shape — start-only blank, then normal — is the
same, and the pooled-vs-excluded comparison is nil either way.)

## 9. Fine-tune summaries (200 episodes on (5,8) alone, arm's training schedule)

| Agent | Reached | Crushes | (5,8) entries | Checkpoint source |
|---|---|---|---|---|
| wrapped/0 | 105 | 7 | 129 | snapshot (Step 0) |
| wrapped/1 | 158 | 15 | 164 | realised, verified exact |
| wrapped/2 | 97 | 7 | 120 | realised, verified exact |
| wrapped/3 | 0 | 1 | 3 | realised, verified exact |
| wrapped/4 | 127 | 10 | 132 | realised, verified exact |
| plain/0 | 188 | 22 | 189 | snapshot (Step 0) |
| plain/2 | 22 | 5 | 40 | realised, verified exact |
| plain/3 | 20 | 12 | 48 | realised, verified exact |
| memory/0 | 30 | 13 | 79 | snapshot (Step 0) |
| memory/2 | 24 | 6 | 29 | realised, verified exact |
| memory/3 | 161 | 17 | 167 | realised, verified exact |
| memory/4 | 180 | 7 | 183 | realised, verified exact |
| oracle_ind/100 | 12 | 12 | 27 | snapshot (Step 0) |
| oracle_ind/101 | 92 | 10 | 105 | realised, verified exact |
| oracle_ind/102 | 0 | 3 | 4 | realised, verified exact |
| oracle_ind/103 | 6 | 5 | 14 | realised, verified exact |
| oracle_new/100 | 0 | 3 | 2 | snapshot (Step 0) |
| oracle_new/101 | 179 | 7 | 181 | realised, verified exact |
| oracle_new/102 | 88 | 10 | 100 | realised, verified exact |
| oracle_new/103 | 182 | 14 | 185 | realised, verified exact |
| phase_only_new/102 | 103 | 12 | 113 | snapshot (Step 0) |
| phase_only_new/104 | 178 | 11 | 180 | realised, verified exact |

Wrapped ScheduleModel period/open_len: [20, 10] at fine-tune start and
[20, 10] at fine-tune end for all five wrapped agents (carried, not
reset). Fine-tune choices (all agents): flat ε = 0.05, fresh replay
buffer, start-tick RNG seed+2000, target network + Adam moments
carried, agent RNG streams carried, wrapped encoder carried.
Fine-tune summaries are recorded in P-Db2 format plus checkpoint
identity (`training_ft1.jsonl`).

Fine-tune reached does NOT predict eval acquisition cleanly: the two
highest fine-tune scorers after plain/0 are oracle_new/103 (182) and
memory/4 (180) — oracle_new/103 reaches 24/30 in F3, memory/4 only
11/20; memory/3 (161) reaches 0/20 in F3 eval; oracle_new/101 (179)
reaches 0/30 in F3 eval while its F4 also collapsed to 0/30.
Training-time reaching and greedy-eval crossing are
different measurements, and this experiment shows they can diverge
sharply. No claim about WHY they diverge, beyond the memory/3
trap hypothesis in §7, is made here.

## 10. Conflicts (F1 and F2, generalised + preferred-door-closed)

F1 recorded conflicts: 157 (39 waits-at-closed-door / 39 takes-open-door
/ 79 other), recomputation mismatches 0. Takes by agent: memory/0 14,
memory/3 11, oracle_new/100 4, phase_only_new/104 10. (Takes ≠
reaches: memory/0's 14 takes reach 0; memory/3's 11 takes are part of
its retained 20/20.) F1 preferred-door-closed (preferred = modal F1
first door of the P-Db2 agent, NOT recomputed from post-fine-tuning
F1): 224 total (1 waits-at-preferred / 54 takes-open / 169 other).

F2 recorded conflicts: 154 (31 waits / 17 takes / 106 other),
recomputation mismatches 0. Takes by agent: memory/0 13,
oracle_new/100 4 — none of them reach. F2 preferred-door-closed:
212 total (0 waits-at-preferred / 34 takes-open / 178 other).

Contrast with P-Db2, kept set only (like-for-like: FT1's agents are
the 22 kept agents, so the baseline below recomputes P-Db2's
preferred-door-closed counts for those 22 agents alone from
`results_pdb2.jsonl`, NOT the 30-agent programme totals of 106/102/60
and 128/74/97): F1 preferred-door-closed was 81 waits / 93 takes /
26 other (200); F2 was 102 waits / 71 takes / 63 other (236).
The waiting behaviour is essentially GONE after fine-tuning: F1 waits
81 → 1, F2 waits 102 → 0. In P-Db2, waiting at a closed preferred
door was the dominant conflict response; after (5,8) fine-tuning,
agents either take an open door (and usually fail to reach) or do
something else entirely (freeze, crush, wait at a non-preferred door).
This is the conflict-level signature of the forgetting in §6.

## 11. Far side (post-entry stalls)

F0: entered 63, post-entry stall 56, stalled-and-unreached 25
(memory/0 18, memory/2 1, phase_only_new/102 6). F1: entered 162,
stall 88, stalled-and-unreached 57 (memory/0 19, memory/2 2,
oracle_ind/103 30, phase_only_new/102 6). F2: entered 117, stall 50,
stalled-and-unreached 50 (memory/0 19, memory/2 1, oracle_ind/103 30).
F3: entered 276, stall 41, stalled-and-unreached 20 (ALL plain/2).
F4: entered 410, stall 133, stalled-and-unreached 81 (plain/2 20,
memory/0 19, memory/2 2, oracle_ind/103 30, oracle_new/100 4,
phase_only_new/102 6). G1: entered 5, stall 5, stalled-and-unreached
5 (wrapped/1 1, memory/2 4).

The far side is where most post-fine-tuning failure lives: entering a
door no longer implies reaching the goal. In F3, the only far-side
failures are plain/2's 20 (every other entered F3 episode reaches).
In F1/F2/F4, far-side stalls are concentrated in the same four
degenerate agents (memory/0, oracle_ind/103, plain/2, oracle_new/100)
plus small counts from memory/2 and phase_only_new/102.

## 12. Routes for newly-crossing agents (F3 + G1)

Every newly-crossing F3 episode stages at **(4,8)** before entry —
the column route. No newly-crossing episode stages anywhere else.
Within that: R-class episodes go directly to (4,8); D-class episodes
occupy (4,5) before (4,8) and then enter (5,8); W-class episodes wait
at (4,8) past their threshold. So the answer to Question 7 is: the
column route, uniformly at entry, with D-class agents additionally
passing through (4,5) en route — the (4,5) occupation is a waypoint,
not an alternative staging strategy. In F4, the same column shape
appears: novel-first episodes enter (5,8) first (248 episodes, 228
reached). The 4 novel-anywhere-but-not-novel-first episodes are
oracle_new/100 F4 phases 0–3, entering (5,7) first and then
alternating (5,7)/(5,8) 22 times per episode, all unreached (the
crush loop of §6, not a route). Separately, wrapped/0's F4 episodes
in phases 0–10 and 17–19 record entries [[5,8],[5,7]] — (5,8) first
(hence novel-first), with (5,7) touched second after crossing, all
reached.

## 13. The eight pre-stated questions, answered with numbers

(1) Acquisition: **11/22 agents cross (5,8) in F3 (vs 1/22 before);
237/540 episodes (vs 20/540).** Per agent: §5. Per arm (F3 reached):
wrapped 60/100, plain 20/60, memory 31/80, oracle_ind 30/120,
oracle_new 54/120, phase_only_new 42/60.
(2) Forgetting: F1 540/540 → 76/540; F2 540/540 → 38/540;
F4 540/540 → 304/540. Per agent: §6. Only memory/3 and
oracle_ind/102 retain F1; only oracle_ind/102, wrapped/2 and
phase_only_new/102 retain anything in F2.
(3) F0: the collapse DEEPENED overall (253/540 → 38/540) but with two
reversals: memory/3 0/20 → 20/20, phase_only_new/104 0/30 → 18/30.
All 38 F0 reaches are those two agents.
(4) F4: **(5,8) became a first door massively: 248/540 novel-first
episodes (vs 2/540 before, both memory/3), 228 of them reached.**
Novel-anywhere: 252 episodes with a (5,8) entry out of 410 F4
entries. The 20 novel-first episodes that did NOT reach are all
plain/2's (enter-but-freeze).
(5) Transfer: **0 G1 crossings (0/540 reached; 5 entered, all
far-side frozen).** See §4, prominently.
(6) memory/3: F3 perfection NOT preserved (20/20 → 0/20, 0 entered);
F0 improved (0/20 → 20/20). See §7.
(7) Routes: column route uniformly — every newly-crossing F3 episode
stages at (4,8); D-class agents additionally occupy (4,5) en route.
See §12.
(8) Dose bound: see §14.

## 14. Dose bound — what 200 episodes can and cannot show

200 episodes on (5,8) alone is a single fixed dose, not a dose-response
curve. What it shows: 200 episodes CAN produce partial (5,8)
acquisition in greedy eval (11/22 agents, 237/540 episodes) and CAN
produce near-total multi-door forgetting in the same agents. What it
cannot show: whether the 11 non-acquiring agents would acquire with
more episodes, whether the forgetting would reverse or worsen with
more episodes, or whether any agent's failure is permanent. **Zero
after 200 episodes means "200 insufficient" for that agent, NOT
"cannot learn."** In particular: memory/3's 0/20 F3 eval after a
161/200 fine-tune shows that even "acquired in training" does not
imply "crosses in greedy eval" — dose statements must name which
measurement they refer to. Any dose-response, set-aside-agent, or
second-novel-door follow-up is a separate proposal with its own
scope, not part of FT1.

## 15. Interpretation (EXPLORATORY, task-and-protocol bounded)

The strongest explanation among the tested alternatives, within this
protocol only: a single 200-episode dose on one novel door partially
overwrites multi-door behaviour rather than adding to it. Agents do
not, in this experiment, accumulate doors: the agents that gain
(5,8) most completely (wrapped/0, wrapped/1, wrapped/4, plain/0,
oracle_ind/101, oracle_new/102, phase_only_new/102, memory/2) lose
their multi-door worlds almost entirely (their F1/F2 go to 0, except
phase_only_new/102, which retains a single episode — phase 19 — in
each, with only F4 partially retained via (5,8) itself); the agents
that retain
multi-door behaviour (memory/3, oracle_ind/102) gain nothing at
(5,8) in eval. The exception that breaks any simple trade-off story
is oracle_new/101 (fine-tune 179/200, yet F3 0/30 AND F4 0/30 — it
gained neither the new door in eval nor kept the old ones) and
phase_only_new/104 (F0 repaired, F1 partial, F2 lost, F3 partial
12/30, F4 retained 30/30 — a mixed profile). These agent-level
splits are the finding; any single-sentence summary ("fine-tuning
works" or "fine-tuning destroys") would misdescribe them.

On the memorisation verdict: FT1 does not overturn P-Db2's verdict,
and it sharpens it. P-Db2 showed trained locations are memorised,
not generalised (novel door (5,8) uncrossed without training).
FT1 shows a novel door CAN be acquired by training on it (237/540)
— but the acquired door does NOT transfer to a second novel door
(G1 0/540 crossed, 5 entered-and-frozen), and acquisition comes with
forgetting of the previously trained doors. That pattern — location
acquired by direct training, no transfer to an untrained location,
prior locations lost — is consistent with location memorisation
under a new training regime, not with door-general competence. This
is stated as consistency, not proof: FT1 tested one dose, one novel
door, one transfer door, and greedy eval only.

## 16. Files (all under ~/workspace/time-mind-model/nn/, DRAFT, uncommitted)

- `decisions_ft1.md` (this report)
- `ft1_analysis.py`, `ft1_analysis_output.txt`
- `ft1_run.py`, `ft1_run.log`
- `training_ft1.jsonl` (22 fine-tune summaries with checkpoint identity)
- `results_ft1.jsonl` (3,240 eval episodes)
- `checkpoints_ft1_finetuned/` (22 fine-tuned snapshots)
- Governing: `~/workspace/your_files/WO-TBM-FT1.md`,
  `decisions_ft1_design.md` (incl. second ruling, commit 6c9ed08),
  Step 0 snapshots `checkpoints_ft1/`, P-Db2 records
  `results_pdb2.jsonl` / `training_pdb2.jsonl`.

# WO-TBM-P-Da — report (Steps 1–3)

**Status: EXPLORATORY, instructor-verified (headline and key secondaries re-derived independently from the record files; see §1).**
**Task-and-protocol bounded. All conclusions EXPLORATORY. No causal language.**

## 0. Skipped, not doable, deviations, restarts — lead section

**Nothing skipped. Nothing not doable. No deviation from the design note
(`nn/decisions_pda_design.md`) or the issued work order.**

- **Restart (reported, not hidden):** a runtime restart at 11:13 UTC on
  2026-10-22 killed the first run mid-training of wrapped/1 E4. At that
  point 9 jobs were complete (180 episodes, 9 training records); the
  in-progress training was unflushed, so at most one training was lost, as
  designed. The runner was relaunched at 11:14 UTC with `tee -a`; resume
  verified on relaunch (complete jobs skipped, wrapped/1 E4 re-trained
  deterministically, no duplicate rows, no partial job carried over). The
  resumed run completed with exit 0 (duration ~70.6 min for the resumed
  portion). No restart-from-scratch occurred at any point.
- **Drifted rows: none.** Wrapped blank-feature episodes = 0 / 500
  (42,738 steps with f keys). No rows were excluded from analysis and no
  exclusions were needed. The blank-feature exclusion rule was never
  triggered.
- **Analysis provenance:** `nn/pda_analysis.py` was tested on partial data
  during the run with output to `/tmp` only. The delivered
  `nn/pda_analysis_output.txt` was produced by a fresh run on the complete
  records only. All headline numbers below were re-derived from the record
  files, not from the runner log.
- **Reading notes (definitions followed exactly, interpretation bounded):**
  - `first_door` = first gap-cell occupation, **including entries that end
    in a crush**. It is not "first successful crossing." One episode
    (plain/1 ph0, §5) turns on exactly this distinction.
  - Class D in E2/E3 uses the design note's definition verbatim: the
    episode occupied (4,5) at a tick before its first-entry tick, then
    entered a rival (E2) or novel (E3) door. In E2 and E3, (5,5) is NOT a
    gap, so (4,5) is a staging cell by the always-include rule, not a
    door staging cell in-world. D in these worlds therefore does NOT mean
    "abandoned the familiar door" — the familiar door is absent. §4
    decomposes D into pass-through vs waited cases so the label is not
    over-read.
  - W thresholds use each retrained agent's OWN E0 median (old
    single-door medians not reused). Six agents have no E0 reached
    episode, so their E0 median is undefined; they use the revisits-only
    W fallback and are listed explicitly in §4.

## 1. Verification (from the record files, not the log)

| Check | Result |
|---|---|
| Eval records | 3750 (expected 3750) |
| Eval jobs (arm, seed, world) | 150 (expected 150), 0 with unexpected count |
| Duplicate (arm, seed, world, start_phase) keys | 0 |
| Episodes where len(steps) != ticks | 0 |
| Episodes where recomputed entries != recorded door_entries | 0 |
| Training records | 50 (expected 50: 25 wrapped + 25 non-wrapped) |
| Duplicate training keys | 0 |
| Agents missing E0 / missing training record | none / none |
| Wrapped seeds with non-identical per-door training counts across their 5 trainings | none (deterministic re-training confirmed) |

## 2. Training (Step 1) — all 30 agents, from scratch, in W5

Training world W5 {(5,1),(5,3),(5,5),(5,7),(5,9)}, all doors on the arm's
training schedule, timing uniform. 2000 episodes per training. Every
training reached 1582–1783 episodes (of 2000); no training failed.

**Training door visits were broad for every agent.** Across all 50
trainings, per-door entry counts range 141–1209; every training visited
every door (minimum single-door count: plain/4 at (5,1), 141 entries).
Dominant-door shares range 0.214 (wrapped/0, near-uniform) to 0.533
(plain/0 at (5,7)). Two skewed cases carried from Step 0 sanity:
plain/0 (5,7) 1209 of 2270; phase_only_new/100 (5,9) 630 of 2176. The
joint reading with E1 is in §6: broad training visits did not produce
broad eval door use.

## 3. Per-arm × world endpoints (Step 2, greedy only)

Ticks = mean over reached episodes. Crushes = total across the arm's
episodes in that world. n = 100 (20/10 arms) or 150 (30/14 arms).

| Arm | E0 W1 succ | E0 ticks | E0 crush | E1 W5 succ | E1 ticks | E1 crush | E2 W5r succ | E2 ticks | E2 crush | E3 W1n succ | E3 crush | E3 entered | E4 W5n succ | E4 ticks | E4 crush |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wrapped | 25/100 (0.25) | 20.84 | 0 | 100/100 (1.00) | 22.98 | 0 | 100/100 (1.00) | 26.21 | 0 | 0/100 | 82 | 18 | 100/100 (1.00) | 22.98 | 0 |
| plain | 16/100 (0.16) | 24.38 | 1 | 80/100 (0.80) | 23.64 | 3 | 71/100 (0.71) | 24.00 | 2 | 0/100 | 0 | 0 | 80/100 (0.80) | 23.64 | 3 |
| memory | 63/100 (0.63) | 35.35 | 1 | 82/100 (0.82) | 31.54 | 3 | 42/100 (0.42) | 30.07 | 2 | 0/100 | 0 | 0 | 82/100 (0.82) | 31.54 | 3 |
| oracle_ind | 11/150 (0.07) | 26.55 | 0 | 150/150 (1.00) | 25.08 | 0 | 150/150 (1.00) | 25.89 | 0 | 0/150 | 0 | 0 | 150/150 (1.00) | 25.08 | 0 |
| oracle_new | 37/150 (0.25) | 21.30 | 0 | 150/150 (1.00) | 22.57 | 0 | 150/150 (1.00) | 28.57 | 0 | 0/150 | 0 | 0 | 150/150 (1.00) | 22.81 | 0 |
| phase_only_new | 38/150 (0.25) | 29.21 | 0 | 150/150 (1.00) | 24.20 | 0 | 150/150 (1.00) | 28.05 | 0 | 0/150 | 0 | 0 | 150/150 (1.00) | 24.20 | 0 |
| **Total** | **190/750 (0.25)** | — | 2 | **712/750 (0.95)** | — | 6 | **663/750 (0.88)** | — | 4 | **0/750** | 82 | 18 | **712/750 (0.95)** | — | 6 |

E0 per-seed reached (denominator 20 for 20/10 arms, 30 for 30/14 arms):
wrapped 2/6/11/6/0; plain 0/8/8/0/0; memory 20/20/20/1/2;
oracle_ind 2/8/0/0/1; oracle_new 17/6/3/2/9;
phase_only_new 3/1/4/29/1.

E1 first-door totals per arm (entered episodes only; full per-agent table
in §6): wrapped (5,1) 29, (5,9) 14, (5,7) 16, (5,5) 25, (5,3) 16;
plain (5,7) 20, (5,5) 17, (5,9) 31, (5,1) 12, none 20;
memory (5,3) 35, (5,5) 41, (5,9) 6, none 18;
oracle_ind (5,7) 18, (5,5) 11, (5,3) 6, (5,1) 63, (5,9) 52;
oracle_new (5,1) 50, (5,9) 46, (5,5) 37, (5,3) 11, (5,7) 6;
phase_only_new (5,1) 47, (5,9) 55, (5,7) 22, (5,5) 22, (5,3) 4.

E2 first doors (664 entered): (5,1) 201, (5,3) 72, (5,7) 86, (5,9) 305,
none 86. No (5,5) first door is possible in E2 ((5,5) is a wall there).

## 4. D/R/W/F classes — E2 and E3 only (design note §4, F→D→W→R, first entry only)

W thresholds: each agent's own E0 median ticks over reached episodes,
doubled; revisits > 10 also qualifies as W. **Fallback agents (E0 median
undefined, revisits-only W): oracle_ind/102, oracle_ind/103, plain/0,
plain/3, plain/4, wrapped/4.** UNCLASSIFIED: 0 in both worlds.

**E2 (750 episodes): R 485, D 164, W 15, F 86.** Reached by class:
R 485/485, D 164/164, W 14/15. The single entered-but-not-reached
E2 episode is class W (plain/2 ph4, §7). Per arm: wrapped R 70, D 30; plain R 51, D 20, W 1,
F 28; memory R 40, D 1, W 1, F 58; oracle_ind D 15, R 131, W 4;
oracle_new R 94, D 52, W 4; phase_only_new R 99, D 46, W 5.
(E2 entered = 664; reached = 663; F = 86 entered nothing.)

D decomposition (read with the §0 caveat — (5,5) is absent in E2):
all 164 D episodes first-entered (5,9) (133) or (5,7) (31) — never
(5,1)/(5,3). 93 of 164 arrived at (4,5) as their first staging cell.
Ticks spent at (4,5) before first entry: exactly 1 tick (pass-through)
for 58; 2–18 ticks (waited) for the remaining 106. D is therefore a
mixed class in E2: pass-throughs and waits, not a pure diversion count.
All 164 D episodes reached the goal.

W episodes in E2 (15): plain/2 ph4 via (5,1), 200 ticks, not reached
(the §7 freeze); the other 14 reached, and 13 of them qualify on
revisits (> 10) rather than on the ticks threshold (notably oracle_ind/100
ph14–17 and oracle_new/103 ph7/11/12/13, all via (5,1)).

F episodes in E2 (86): memory/1 20, memory/2 20, memory/4 18, plain/3 20,
plain/2 8. The memory/1 and memory/2 failures are (4,5)-waiting freezes:
3350 of 4000 steps (memory/1) and 3765 of 4000 steps (memory/2) sit at
(4,5) — the familiar staging cell — in a world where (5,5) is a wall.
This is the Luck failure shape recurring after retraining, for these two
agents only.

**E3 (750 episodes): F 732, D 18, R 0, W 0.** Reached by class: none.
All 18 D episodes are wrapped/4 (phases 0, 3–19); every other agent is
F 100% in E3. The D episodes are NOT passes: see §7 — wrapped/4 never
crossed to the east side (max x = 5 in all 18 episodes).

## 5. E0 — reported, not gated (design note: E0 is a sanity read, not a gate)

Retrained agents largely do NOT solve the single-familiar-door world:
190/750 (25%) reached in E0, against 712/750 (95%) in E1 and 663/750
(88%) in E2. The exception is memory (63/100), driven by memory/0,1,2
at 20/20 each; memory/3 (1/20) and memory/4 (2/20) do not share it.
The other arms sit at 7–25%. One agent is exceptional in the other
direction: phase_only_new/103 reached 29/30 in E0 (its E1 first doors
are (5,1) 15 / (5,5) 14 — it retains a (5,5) route that most agents
lose).

E0 success is NOT simply "the subset whose E1 route goes through (5,5)":
of the 190 E0-reached episodes, 152 have an identical E1 trajectory
with E1 first door (5,5); the other 38 reach in E0 via (5,5) while
their E1 route takes a different door first (memory 22, phase_only_new
16 — e.g. memory/0 ph0–2: E0 via (5,5) in 49/48/47 ticks, E1 via (5,3)
in 46/45/44 ticks). The multi-door world changes the route itself:
a gap that is a wall in E0 is traversable in E1, so the same policy
walks a different first gap. Conversely, 38 further pairs have identical
E0/E1 trajectories but fail in both (e.g. memory/4 frozen at (4,0)) —
identical trajectories do not imply success.

Single exception, mechanism identified (not a runner bug): plain/1 ph0
reaches in E1 via first door (5,5) in 29 ticks but fails in E0 in 200
ticks. Its route enters (5,5) at tick 10 and is crushed (mistimed entry:
back to (0,0)); the recovery route walks to (4,9) and steps east — in
E1 (5,9) is an open gap, so it crosses and reaches; in E0 (5,9) is a
wall, so it bumps at (4,9) for the rest of the episode. First-door
records the crushed (5,5) entry, not the successful (5,9) crossing.

## 6. E1 lock-in, jointly with training visits (question 2)

Training visits were broad (§2: every door visited by every training,
141–1209 entries per door). Eval first doors are nevertheless narrow
per agent:

- Single-door agents in E1: plain/0 (5,7) 20/20; plain/4 (5,9) 20/20;
  memory/0 (5,3) 20/20; memory/1 (5,5) 20/20; memory/2 (5,5) 20/20.
- Two-door agents: plain/1 (5,9) 11 / (5,5) 9; plain/2 (5,1) 12 /
  (5,5) 8; phase_only_new/104 (5,9) 29 / (5,1) 1; phase_only_new/103
  (5,1) 15 / (5,5) 14.
- Distributed agents: all 5 wrapped (3–5 doors each), all oracle_new
  (4–5 doors), all oracle_ind (3–4 doors), phase_only_new/100/101/102
  (4–5 doors). No wrapped agent uses fewer than 3 doors in E1.
- Degenerate agents: plain/3 enters nothing in E1 (0/20; frozen at
  (4,0) — 3920 of 4000 steps there, no-move-run in all 20 episodes,
  despite a normal training record: 1783/2000 reached, broad door
  visits); memory/4 enters only (5,3) in phases 7–8 (2/20, the same
  two phases that reach in E0/E2/E4).

Joint reading: "training never varied" is refuted as the explanation
for narrow eval doors — variation was present in every training.
The observed pattern is instead: location variation in training was
insufficient, within this protocol, to produce door choice at eval —
each agent's greedy policy collapses to a small preferred set (1 door
for 5 agents, 2 doors for 4 agents), while wrapped/oracle/phase agents
spread across 3–5 doors without choosing a consistent winner (no arm
shows a (5,5) preference in E1: (5,5)-first shares are wrapped 25/100,
plain 17/80 entered, memory 41/82, oracle_ind 11/150, oracle_new 37/150,
phase_only_new 22/150). The (5,5) lock-in of the single-door Luck
agents is gone; what replaced it is per-agent preferred-door lock-in,
not door choice.

## 7. Far side after entry (P-Dc connection, question 6)

- **E2:** 664 entries, 663 reached. Post-entry stall kinds: none 639,
  no-move-run 25. Long freezes among not-reached episodes
  (post-entry ticks ≥ 50): exactly 1 — plain/2 ph4 via (5,1), 193
  post-entry ticks, no-move-run, never reached. This is the only
  far-side freeze pattern in E2.
- **E4:** 712 entries, 712 reached. Post-entry stall kinds: none 678,
  no-move-run 34. Long freezes among not-reached: 0. No far-side
  freeze in E4.
- **E3:** 18 entries, 0 reached, all post-entry no-move-run with
  post-entry ticks 152–169 — but these are NOT far-side freezes. The
  agent (wrapped/4) never crossed: max x = 5 in all 18 episodes, 0
  steps at x ≥ 6, 82 steps at (5,8) against 82 crushes total — every
  (5,8) occupation step ends in a crush. The pattern is a pre-crossing
  crush loop (enter (5,8) → crushed to (0,0) → walk back via
  (4,5)→(4,8) → enter again), not a freeze on the far side.
  Staging order in all 18 episodes is [(4,5), (4,8)].

Verdict on the P-Dc pattern: after familiar/rival entries (E2/E4),
crossing is essentially clean — 1 far-side freeze in 1376 entries.
After novel entries (E3), there is no far-side behaviour at all to
assess, because the only agent that enters never gets across.

## 8. Luck contrast — E2 vs Luck W5r (question 3, stated explicitly)

Same world W5r {(5,1),(5,3),(5,7),(5,9)}, same greedy protocol,
different agents: single-door-trained (Luck) vs multi-door-retrained
(P-Da E2).

| Arm | Luck W5r reached | Luck entered | P-Da E2 reached | P-Da E2 entered |
|---|---|---|---|---|
| wrapped | 0/100 | 0 | 100/100 | 100 |
| plain | 0/100 | 0 | 71/100 | 72 |
| memory | 0/100 | 1 | 42/100 | 42 |
| oracle_ind | 0/150 | 0 | 150/150 | 150 |
| oracle_new | 0/150 | 0 | 150/150 | 150 |
| phase_only_new | 0/150 | 0 | 150/150 | 150 |
| **Total** | **0/750** | **1** | **663/750** | **664** |

The Luck result (0/1500 across W1r+W5r, H_luck standing) is overturned
in the retrained direction for this world: retraining in W5 converts
rivals from impassable (0 passes) to routinely used (663 passes).
The two exceptions bound the claim: memory reaches only 42/100 in E2
(memory/1 and memory/2 score 0/20, frozen at (4,5) as in Luck), and
plain reaches 71/100 (plain/3 scores 0/20 everywhere, plain/2 loses
8 episodes to F and 1 to the §7 freeze). The contrast is descriptive
and task-bounded: it shows retraining changes rival use on this world
and protocol; it does not show a general door concept (see §9).

## 9. Held-out verdict — E3, with E2 as the control (question 4, stated explicitly)

**Verdict: memorisation of trained locations, not generalisation —
within this protocol, on this held-out door.**

- E3 (novel door (5,8) alone): 0/750 reached, 18/750 entered, all 18
  by one agent (wrapped/4), all 18 crushed, none crossed to the east
  side (§7). Every other agent: 0 entered, 0 reached.
- Control E2 (rivals without (5,5)): 663/750 reached, 664 entered.
  Agents that use rivals fluently in E2 (all oracle/phase agents at
  150/150, wrapped at 100/100) enter nothing in E3. The E3 failure
  therefore cannot be read as "agents cannot use unfamiliar doors" —
  they use trained rivals; they do not use the untrained door.
- **Interpolation bound (stated alongside, as required):** (5,8) sits
  between trained doors (5,7) and (5,9) — this is interpolation, not
  extrapolation. The test is lenient in that sense, and the failure
  is correspondingly strong within protocol: even a door adjacent to
  two trained doors is not used. No claim is made about far
  extrapolation (e.g. a door far from all trained doors), which was
  not tested.
- E3 classes: F 732, D 18 (all wrapped/4, §4), R 0, W 0. The D label
  here records the (4,5)→(4,8)→(5,8) route (§7); it is not a pass.

## 10. E4 — novel door among familiars (question 5, NOT classified D/R/W/F)

First-door novel entries: **0/750 for every agent.** No agent's first
entry in E4 is (5,8); E4 first-door distributions are identical to E1
for every agent except oracle_new/102 (see below). E1 vs E4 paired
trajectories are identical in 738/750 pairs.

The only (5,8) occupations in E4 (12 episodes, all oracle_new/102,
phases 6–17) are mid-route transits, not crossings: door sequence
(5,9) → (5,8) → (5,7) in each episode, all 12 reached, ticks 22–33,
revisits 0–1, (4,5) occupied in 0 of the 12 (staging order [(4,9)]).
The agent crosses at its usual (5,9), then walks north inside the
wall column through the newly-open (5,8) cell to (5,7) — in E1 the
same route is blocked at (5,8) (wall) and goes east instead. This is
a route artifact of a newly traversable cell on an existing path,
not a choice of the novel door: the agent never enters (5,8) first,
never stages at (4,8) first in E4, and never uses (5,8) as its
crossing door. The remaining 3 E4 switched episodes (plain/1 ph0,
plain/2 ph4, memory/3 ph2) switch among familiar doors only, matching
their E1 switches.

## 11. The seven pre-stated questions, answered with numbers

1. **Do retrained agents still solve W1 (E0 sanity)?** Mostly no:
   190/750 (25%). Memory yes-ish (63/100, but only seeds 0–2);
   wrapped 25/100, plain 16/100, oracle_ind 11/150, oracle_new 37/150,
   phase_only_new 38/150. E0 is reported, not gated; the low E0 rate
   is itself a finding (§5), not a failed gate.
2. **In W5 (E1), does (5,5) lock-in persist among familiar doors, or
   do they distribute/choose?** (5,5) lock-in is gone; per-agent
   preferred-door lock-in replaces it. (5,5)-first shares: 25/100
   wrapped, 17/80 plain entered, 41/82 memory entered, 11/150
   oracle_ind, 37/150 oracle_new, 22/150 phase. 5 agents use exactly
   1 door, 4 use exactly 2, the rest spread across 3–5 doors with no
   consistent winner (§6). (5,5)-first totals 153/712 entered
   episodes; (5,5) is the per-agent mode for only 3 of 30 agents
   (memory/1, memory/2, oracle_new/100). (5,1), the door closest to
   the start, accounts for 201/664 E2 first entries and is absent
   from 5 agents' E1 first doors entirely (plain/3, memory/0,
   memory/3, memory/4, oracle_ind/101) — no agent's pattern matches
   a nearest-door rule.
3. **In W5r (E2), do retrained agents use rivals — the Luck contrast
   (0/1500 → ?)?** Yes, overwhelmingly: 663/750 reached, 664 entered
   (§8). Bounded by memory (42/100) and plain (71/100).
4. **In W1n (E3), do they pass a door never seen in training?** No:
   0/750 reached; 18 entered, all by wrapped/4, all crushed, none
   crossed (§9). Memorisation, not generalisation, within protocol
   and the interpolation bound.
5. **In W5n (E4), do they use the novel door when familiars exist?**
   Never as first entry (0/750); only as mid-route transit by one
   agent (oracle_new/102, 12 episodes, §10).
6. **Far side: after unfamiliar/novel entries, freeze or clean
   crossing?** After familiar/rival entries, clean: 1 far-side
   freeze in 1376 E2+E4 entries (plain/2 ph4). After novel entries,
   the question does not arise: the only novel entries never cross
   (§7).
7. **Wrapped vs plain vs oracle: does time-awareness or perfect
   information matter after retraining?** On success rates, perfect
   information wins in E1/E2/E4 (oracle_ind, oracle_new,
   phase_only_new all 150/150 in E1, E2, E4; wrapped 100/100 in the
   same worlds; plain 80/71/80; memory 82/42/82). Time-awareness
   (wrapped) matches perfect information on success in every world
   except E0 (wrapped 25 vs oracle_new 37, phase 38 per 100/150
   denominators — not directly comparable counts). The differences
   that remain are agent-level, not arm-level: the degenerate
   policies (plain/3, memory/4), the (4,5)-waiting memory seeds in
   E2, and the single wrapped/4 E3 pattern cut across the
   time-awareness distinction. No arm shows generalisation in E3;
   no arm shows door choice in E1. Within this protocol, arm
   differences are about reliability on trained doors, not about
   door concepts.

## 12. Anomalies and agent-level findings (all included in totals, none excluded)

- **plain/3 — degenerate greedy policy:** 0/20 in every world
  including the training world E1, frozen at (4,0) (no-move-run, all
  20 E1 episodes), despite a normal training record (1783/2000
  reached, all doors visited). Training success did not transfer to
  greedy eval for this agent. Reported as a result, not a data error;
  no re-training or substitution was performed.
- **memory/4 — near-degenerate:** 2/20 in E0, E1, E2, E4 (phases 7–8
  only, via (5,3)); 0/20 in E3. Same two phases reach in every world
  where it reaches at all.
- **memory/1, memory/2 in E2:** 0/20 each, frozen at (4,5) (§4) —
  the Luck failure shape persisting after retraining for these two
  agents, while memory/0 and memory/3 score 20/20 in E2.
- **phase_only_new/103 in E0:** 29/30 — the only agent that retains
  a near-complete (5,5) route after multi-door retraining.
- **wrapped/4 in E3:** the only novel-door entries in the programme
  (§7, §9). Its E0 score is 0/20 (revisits-only W fallback applies).

## 13. What this does NOT show (bounds)

- No causal claims: retraining is associated with rival use in E2
  and with E3 failure; the experiment does not identify which
  training feature causes either.
- No generality beyond these worlds, arms, seeds, schedules
  (20/10, 30/14), and the greedy protocol. E3 tests one held-out
  door, (5,8), an interpolation case; E4 tests one novel door among
  five familiars.
- No claim about fine-tuning (forbidden in this work order and not
  done), schedule variation (P-Db2, separate, not done), or any
  agent beyond the 30 retrained here.
- D/R/W/F classes are behavioural descriptions under fixed
  definitions, not evidence of intent or decision.

## 14. Files

- Runner: `nn/pda_run.py`
- Training records: `nn/training_pda.jsonl` (50 records)
- Eval records: `nn/results_pda.jsonl` (3750 episodes, 150 jobs)
- Runner log: `nn/pda_run.log`
- Analysis script: `nn/pda_analysis.py`
- Analysis output: `nn/pda_analysis_output.txt`
- This draft report: `nn/decisions_pda.md`
- Governing documents (already committed, Step 0): work order
  `~/workspace/your_files/WO-TBM-P-Da.md`, design note
  `nn/decisions_pda_design.md` (commit 0679d19), variant/checks
  (commit 54337d1)

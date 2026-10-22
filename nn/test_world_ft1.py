#!/usr/bin/env python3
"""Scripted mechanics regression for world_ft1.py G1 (FT1 Step 0b).

Checks, fixed in the FT1 design note before running:
- G1 identity: g1_schedules(train) == {(5,6): (period, open_len, 0)}
  for both protocols (constructed from the TRAIN tables, not
  re-typed).
- Per-door open/close table: Pdb2AbsentWorld G1 door_open((5,6), t)
  == door_open_at(t, period, open_len, 0) for every tick in the
  period, both protocols.
- Observe bit: G1 observe bit == training schedule bit at every
  tick in the period, both protocols (no phantom/clock
  distinction, as in F3: the gap is on the training schedule).
- Per-step invariants on G1 across the P-Da script set and all
  start phases (never in a wall cell; crushed iff the agent was
  in a gap whose door closed at that tick; a move into the gap
  is possible only while that gap is open; grid-edge clipping
  as in SchedDoorWorld).

NOT checked, by design: no trace comparison against F3 or
against PdaWorld W1n. G1's gap row (5,6) differs from F3's
(5,8), so wall/gap geometry differs and traces must differ
where the agent reaches rows 6/8; a trace-equality claim
would be false. Geometry is covered by the wall-blocking
invariant above instead.
No training involved (pure mechanics).
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import ACTIONS, AGENT_START
from world_doors import is_wall_cell
from world_doors_sched import door_open_at
from world_pdb2 import TRAIN_20, TRAIN_30, Pdb2AbsentWorld, \
    familiar_schedule
from world_ft1 import NOVEL2, g1_schedules


def action_sequences():
    seqs = {}
    scripted = ([ACTIONS["right"]] * 4 + [ACTIONS["down"]] * 5
                + [ACTIONS["right"]] * 3 + [ACTIONS["down"]] * 4
                + [ACTIONS["left"]] * 2 + [ACTIONS["right"]] * 2
                + [ACTIONS["stay"]] * 5 + [ACTIONS["up"]] * 9
                + [ACTIONS["right"]] * 4)
    seqs["scripted"] = scripted
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        acts = list(ACTIONS.values())
        seqs["random%d" % seed] = [rng.choice(acts) for _ in range(250)]
    seqs["right_spam"] = [ACTIONS["right"]] * 120
    seqs["stay"] = [ACTIONS["stay"]] * 120
    return seqs


def main():
    cases = 0
    for train, period, open_len in ((TRAIN_20, 20, 10),
                                    (TRAIN_30, 30, 14)):
        # G1 identity (constructed, not re-typed).
        assert g1_schedules(train) == {NOVEL2: (period, open_len, 0)}, \
            (period, "G1 schedules mismatch")
        cases += 1
        g1 = g1_schedules(train)
        # Per-door table + observe bit, every tick in the period.
        for t in range(period):
            w = Pdb2AbsentWorld(g1, period, open_len)
            assert w.door_open(NOVEL2, t) == door_open_at(
                t, period, open_len, 0), (period, "G1 door table", t)
            cases += 1
            w.reset(start_tick=t)
            assert w.observe()[1] == door_open_at(
                t, period, open_len, 0), (period, "G1 observe", t)
            cases += 1
        # Per-step invariants, all start phases, P-Da script set.
        gaps = frozenset(g1)
        for name, actions in action_sequences().items():
            for start in range(period):
                w = Pdb2AbsentWorld(g1, period, open_len)
                w.reset(start_tick=start)
                for a in actions:
                    pre_tick = w.tick
                    pre_agent = w.agent
                    n_log = len(w.log)
                    w.step(a)
                    crushed = any(e[2] == "crushed"
                                  for e in w.log[n_log:])
                    expect_crush = (pre_agent in gaps
                                    and not door_open_at(
                                        pre_tick,
                                        *w.schedules[pre_agent]))
                    assert crushed == expect_crush, \
                        (period, "G1", name, start, pre_tick,
                         pre_agent, crushed, expect_crush)
                    if expect_crush:
                        assert w.agent == AGENT_START, \
                            (period, "G1", name, start)
                    assert not is_wall_cell(w.agent, gaps), \
                        (period, "G1", name, start, w.agent)
                    if w.agent in gaps and not expect_crush:
                        assert door_open_at(
                            pre_tick, *w.schedules[w.agent]), \
                            (period, "G1", name, start, w.agent)
                    cases += 1
    print("FT1 G1 scripted mechanics regression: %d cases CHECKED"
          % cases)


if __name__ == "__main__":
    main()

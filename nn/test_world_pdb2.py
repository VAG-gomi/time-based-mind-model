#!/usr/bin/env python3
"""Scripted mechanics regression for world_pdb2.py (Step 0b, part i).

Checks, fixed in the design note before running:
- TRAIN tables equal world_doors_sched SCHED_W5S constants exactly.
- Union coverage: at least one rival open at every tick, and during
  the familiar-closed interval at least one rival open, both protocols.
- Per-door open/close table for Pdb2AbsentWorld (F2, F3) against
  door_open_at, every door, every tick in the period, both protocols.
- Observe bit: F2 == familiar schedule bit, F3 == training schedule
  bit, F1 (SchedDoorWorld) == familiar bit, every tick in the period.
- Per-step invariants on Pdb2AbsentWorld F2/F3 across the P-Da script
  set and all start phases (never in a wall cell; crushed iff the agent
  was in a gap whose door closed at that tick; a move into a gap is
  possible only while that gap is open).
- Trace comparison F3 vs PdaWorld W1n (same schedule): position,
  observe bit, crushed flag identical (full log NOT asserted — log
  lineages differ between the sched and multi-door classes).
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
from world_doors_sched import (FAMILIAR, SCHED_W5S_20, SCHED_W5S_30,
                               SchedDoorWorld, door_open_at)
from world_pda import PdaWorld, W1N
from world_pdb2 import (TRAIN_20, TRAIN_30, Pdb2AbsentWorld,
                        familiar_schedule, f3_schedules,
                        rival_schedules)


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
    # TRAIN identity (imported, not re-typed).
    assert TRAIN_20 == SCHED_W5S_20, "TRAIN_20 != SCHED_W5S_20"
    assert TRAIN_30 == SCHED_W5S_30, "TRAIN_30 != SCHED_W5S_30"
    cases += 2
    for train, period, open_len in ((TRAIN_20, 20, 10),
                                    (TRAIN_30, 30, 14)):
        rivals = rival_schedules(train)
        # Union coverage, from the door_open_at table.
        for t in range(period):
            assert any(door_open_at(t, *rivals[g]) for g in rivals), \
                (period, t, "no rival open")
            fam_open = door_open_at(t, period, open_len, 0)
            if not fam_open:
                assert any(door_open_at(t, *rivals[g]) for g in rivals), \
                    (period, t, "no rival open while familiar closed")
            cases += 1
        # Per-door table + observe bit for F2 and F3.
        f2 = Pdb2AbsentWorld(rivals, period, open_len)
        f3 = Pdb2AbsentWorld(f3_schedules(train), period, open_len)
        for t in range(period):
            for cell, sched in rivals.items():
                assert f2.door_open(cell, t) == door_open_at(t, *sched), \
                    (period, "F2", cell, t)
                cases += 1
            f2.reset(start_tick=t)
            assert f2.observe()[1] == door_open_at(t, period, open_len, 0), \
                (period, "F2 observe", t)
            f3.reset(start_tick=t)
            assert f3.observe()[1] == door_open_at(t, period, open_len, 0), \
                (period, "F3 observe", t)
            f1 = SchedDoorWorld(train)
            f1.reset(start_tick=t)
            assert f1.observe()[1] == door_open_at(t, period, open_len, 0), \
                (period, "F1 observe", t)
            cases += 3
        # Per-step invariants on F2/F3, all start phases.
        for world_kind, make in (
                ("F2", lambda: Pdb2AbsentWorld(rivals, period, open_len)),
                ("F3", lambda: Pdb2AbsentWorld(f3_schedules(train),
                                               period, open_len))):
            gaps = make().gaps
            for name, actions in action_sequences().items():
                for start in range(period):
                    w = make()
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
                            (period, world_kind, name, start, pre_tick,
                             pre_agent, crushed, expect_crush)
                        if expect_crush:
                            assert w.agent == AGENT_START, \
                                (period, world_kind, name, start)
                        assert not is_wall_cell(w.agent, gaps), \
                            (period, world_kind, name, start, w.agent)
                        if w.agent in gaps and not expect_crush:
                            assert door_open_at(
                                pre_tick, *w.schedules[w.agent]), \
                                (period, world_kind, name, start,
                                 w.agent)
                        cases += 1
        # Trace comparison F3 vs PdaWorld W1n (same schedule).
        for name, actions in action_sequences().items():
            for start in range(period):
                new = Pdb2AbsentWorld(f3_schedules(train), period,
                                      open_len)
                new.reset(start_tick=start)
                old = PdaWorld(period, open_len, W1N)
                old.reset(start_tick=start)
                for a in actions:
                    _pos, new_bit = new.observe()
                    n_log = len(new.log)
                    new.step(a)
                    new_crushed = any(e[2] == "crushed"
                                      for e in new.log[n_log:])
                    _pos, old_bit = old.observe()
                    o_log = len(old.log)
                    old.step(a)
                    old_crushed = any(e[2] == "crushed"
                                      for e in old.log[o_log:])
                    assert (new.agent, bool(new_bit), new_crushed) == \
                        (old.agent, bool(old_bit), old_crushed), \
                        (period, "F3-vs-W1n", name, start)
                    cases += 1
    print("scripted mechanics regression: %d cases IDENTICAL/CHECKED"
          % cases)


if __name__ == "__main__":
    main()

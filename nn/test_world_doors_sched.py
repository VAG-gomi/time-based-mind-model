#!/usr/bin/env python3
"""Scripted mechanics regression for world_doors_sched.py.

With all schedules set equal, SchedDoorWorld must behave
identically to MultiDoorWorld: same positions, same crushes,
same observe() bits, under scripted + pseudo-random action
sequences, across all start phases, both gap sets, both
protocols. No training involved (pure mechanics).
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import ACTIONS
from world_doors import MultiDoorWorld
from world_doors_sched import (SchedDoorWorld, door_open_at,
                               equal_schedules)
from world import door_is_open


def run_trace(world, actions):
    """Per-tick (agent, observe-bit, crushed-this-step)."""
    out = []
    for a in actions:
        _pos, bit = world.observe()
        n_log = len(world.log)
        world.step(a)
        crushed = any(e[2] == "crushed" for e in world.log[n_log:])
        out.append((world.agent, bool(bit), crushed))
    return out


def action_sequences():
    seqs = {}
    # Scripted: approach the wall, push, hug east side, wander.
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
    # door_open_at with offset 0 must equal door_is_open.
    for period, open_len in ((20, 10), (30, 14)):
        for t in range(3 * period):
            assert door_open_at(t, period, open_len, 0) == \
                door_is_open(t, period, open_len), (t, period)
    print("door_open_at offset-0 == door_is_open: OK")

    cases = 0
    for period, open_len, gaps in (
            (20, 10, ((5, 5), (5, 1))),
            (20, 10, ((5, 1), (5, 3), (5, 5), (5, 7), (5, 9))),
            (30, 14, ((5, 5), (5, 1))),
            (30, 14, ((5, 1), (5, 3), (5, 5), (5, 7), (5, 9)))):
        for name, actions in action_sequences().items():
            for start in range(period):
                old = MultiDoorWorld(period, open_len, gaps)
                old.reset(start_tick=start)
                new = SchedDoorWorld(
                    equal_schedules(gaps, period, open_len))
                new.reset(start_tick=start)
                t_old = run_trace(old, actions)
                t_new = run_trace(new, actions)
                assert t_old == t_new, (period, gaps, name, start)
                cases += 1
    print("scripted mechanics regression: %d cases IDENTICAL" % cases)


if __name__ == "__main__":
    main()

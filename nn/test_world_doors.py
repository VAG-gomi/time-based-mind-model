#!/usr/bin/env python3
"""Equivalence check for world_doors.py (WO-TBM-doors Step 0b).

With gaps = {(5, 5)}, MultiDoorWorld must behave identically to
World: scripted-agent logs, random-agent logs (several seeds),
and a crush-scenario log must all match byte for byte.
Exits non-zero on any mismatch. Additive only.
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import World, ACTIONS, run_random_agent, run_scripted_agent
from world_doors import MultiDoorWorld, run_scripted_multi


def random_multi(seed, ticks, period=10, open_len=4):
    rng = random.Random(seed)
    world = MultiDoorWorld(period, open_len, ((5, 5),))
    names = list(ACTIONS.keys())
    for _ in range(ticks):
        world.step(ACTIONS[rng.choice(names)])
    return world


def crush_multi():
    """Mirror of world.py check (d1): staying in the door cell
    through the close is fatal."""
    w = MultiDoorWorld(10, 4, ((5, 5),))
    approach = [(1, 0), (2, 0), (3, 0), (4, 0),
                (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]
    for tx, ty in approach:
        x, y = w.agent
        w.step((tx - x, ty - y))
    w.step(ACTIONS["right"])      # tick 9: blocked (closed)
    w.step(ACTIONS["right"])      # tick 10: enters (5, 5)
    for _ in range(4):
        w.step(ACTIONS["stay"])   # ticks 11..14
    w.step(ACTIONS["right"])      # tick 15
    return w


def crush_single():
    w = World()
    approach = [(1, 0), (2, 0), (3, 0), (4, 0),
                (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]
    for tx, ty in approach:
        x, y = w.agent
        w.step((tx - x, ty - y))
    w.step(ACTIONS["right"])
    w.step(ACTIONS["right"])
    for _ in range(4):
        w.step(ACTIONS["stay"])
    w.step(ACTIONS["right"])
    return w


def main():
    ok = True
    # Scripted agent, 25 ticks.
    a = run_scripted_agent(25)
    b = run_scripted_multi(25)
    print("scripted logs identical: %s (%d events)"
          % (a.log == b.log, len(a.log)))
    ok = ok and a.log == b.log
    # Random agents, several seeds, 200 ticks.
    for seed in (42, 43, 7, 123):
        r1 = run_random_agent(seed, 200)
        r2 = random_multi(seed, 200)
        print("random seed %d logs identical: %s (%d events)"
              % (seed, r1.log == r2.log, len(r1.log)))
        ok = ok and r1.log == r2.log
    # Crush scenario.
    c1 = crush_single()
    c2 = crush_multi()
    print("crush logs identical: %s (%d events)"
          % (c1.log == c2.log, len(c1.log)))
    ok = ok and c1.log == c2.log
    print("EQUIVALENCE: %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

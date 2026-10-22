#!/usr/bin/env python3
"""Scripted mechanics regression for world_pda.py (Step 0b, part i).

PdaWorld must behave identically to MultiDoorWorld under
scripted + pseudo-random action sequences, across all
start phases, for the P-Da gap sets and the single-gap
control. No training involved (pure mechanics).
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
from world_pda import PdaWorld, W5, W1, W5R, W1N, W5N


def run_trace(world, actions):
    out = []
    for a in actions:
        _pos, bit = world.observe()
        n_log = len(world.log)
        world.step(a)
        crushed = any(e[2] == "crushed" for e in world.log[n_log:])
        out.append((world.agent, bool(bit), crushed,
                    list(world.log)))
    return out


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
    for period, open_len, gaps in (
            (20, 10, W5), (20, 10, W1), (20, 10, W5R),
            (20, 10, W1N), (20, 10, W5N),
            (30, 14, W5), (30, 14, W1), (30, 14, W5R),
            (30, 14, W1N), (30, 14, W5N)):
        for name, actions in action_sequences().items():
            for start in range(period):
                old = MultiDoorWorld(period, open_len, gaps)
                old.reset(start_tick=start)
                new = PdaWorld(period, open_len, gaps)
                new.reset(start_tick=start)
                assert run_trace(old, actions) == run_trace(new, actions), \
                    (period, gaps, name, start)
                cases += 1
    print("scripted mechanics regression: %d cases IDENTICAL" % cases)


if __name__ == "__main__":
    main()

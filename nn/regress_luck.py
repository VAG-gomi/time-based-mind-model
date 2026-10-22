#!/usr/bin/env python3
"""Agent-level regression for world_luck.py (Step 0b, part ii).

One seed per arm, fresh training, DIRECT W1 eval on LuckWorld
with gaps={(5,5)} (no preceding eval). Summaries must equal
the tracked single-door greedy records exactly.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_nn as ann
from doors_run import load_expected, matches, summarize, train
from world_luck import LuckWorld

REGRESSION_AGENTS = [("wrapped", 0, 20, 10),
                     ("plain", 0, 20, 10),
                     ("memory", 0, 20, 10),
                     ("oracle_ind", 100, 30, 14),
                     ("oracle_new", 100, 30, 14),
                     ("phase_only_new", 100, 30, 14)]


def main():
    expected = load_expected()
    ok_all = True
    for arm, seed, period, open_len in REGRESSION_AGENTS:
        ann.PERIOD = period
        ann.OPEN_LEN = open_len
        print("%s/%d: training..." % (arm, seed), flush=True)
        agent, encoder = train(arm, seed)
        # Direct W1 eval on LuckWorld (same evaluate path as the
        # luck runner, but with the familiar gap restored).
        from luck_run import evaluate_luck
        eps = evaluate_luck(agent, encoder, period, open_len,
                            ((5, 5),), "W1")
        got = summarize(eps)
        exp = expected.get((arm, seed, period, open_len))
        ok = matches(got, exp)
        ok_all = ok_all and ok
        print("%s/%d W1-on-LuckWorld: ticks=%s succ=%.2f -> %s"
              % (arm, seed, got["mean_ticks"], got["success"],
                 "MATCH" if ok else "MISMATCH"), flush=True)
    print("agent-level regression: %s"
          % ("ALL MATCH" if ok_all else "MISMATCH PRESENT"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic for the wrapped/0, wrapped/4 W1-vs-W1m mismatch.

Retrains each seed deterministically (same frozen code as
doors_run.py), then runs W1 eval followed by W1m eval with the
SAME persistent encoder (as doors_run does), logging the
ScheduleModel's state (period, open_len, last opening tick,
contradiction count) and the first-step time features of every
episode. Goal: show exactly what changed in the encoder
between the two evals. Diagnostic only; writes no records.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_nn as ann
from agent_nn import EPISODE_CAP, env_step
from agent_wrapped import train_wrapped
from world import World
from world_doors import MultiDoorWorld


def run_eval(agent, encoder, period, open_len, world_kind, label):
    print("== %s ==" % label)
    for start in range(period):
        w = (World(period=period, open_len=open_len) if world_kind == "W1"
             else MultiDoorWorld(period, open_len, ((5, 5),)))
        w.reset(start_tick=start)
        encoder.reset(w)
        m = encoder.model
        pre = (m.period, m.open_len, m.last_opening_tick,
               m.contradictions)
        x = encoder.encode(w)
        feats = (round(float(x[101]), 3), round(float(x[102]), 3),
                 float(x[103]))
        ticks = 0
        reached = False
        while ticks < EPISODE_CAP:
            a = agent.act(x, 0.0)
            r, reached, _c = env_step(w, a)
            ticks += 1
            if reached:
                break
            x = encoder.encode(w)
        m = encoder.model
        post = (m.period, m.open_len, m.last_opening_tick,
                m.contradictions)
        print("  ph %2d: ticks=%2d pre=%s feats0=%s post=%s"
              % (start, ticks, pre, feats, post))


def main():
    for seed in (0, 4):
        ann.PERIOD = 20
        ann.OPEN_LEN = 10
        print("##### wrapped/%d: training..." % seed)
        agent, _st, encoder = train_wrapped(seed, hidden=32,
                                            episodes=2000, lr=1e-3)
        m = encoder.model
        print("  end-of-training model: %r" % (m,))
        run_eval(agent, encoder, 20, 10, "W1", "wrapped/%d W1" % seed)
        run_eval(agent, encoder, 20, 10, "W1m", "wrapped/%d W1m" % seed)


if __name__ == "__main__":
    main()

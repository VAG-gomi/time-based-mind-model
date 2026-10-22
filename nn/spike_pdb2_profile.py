#!/usr/bin/env python3
"""SPIKE ONLY - cProfile of one wrapped/0 P-Db2 training.

Measurement only, NOT a Step 1 training. Wrapped/0 was chosen
because it is the most expensive encoder type in the programme
(stateful ScheduleModel + 104 inputs, and 25 of the 50 Step 1
trainings are wrapped), so its profile bounds the shared-loop
overhead and shows whether ScheduleModel is a material term.
Profile overhead distorts wall time; this run's wall time is
NOT used for the per-training timing projection.
"""

import cProfile
import os
import pstats
import sys
from io import StringIO

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_pdb2 import TRAINERS
from world_pdb2 import TRAIN_20


def main():
    pr = cProfile.Profile()
    pr.enable()
    agent, stats, enc, dec, ewe, frac = TRAINERS["wrapped"](
        0, TRAIN_20, hidden=32, episodes=2000, lr=1e-3)
    pr.disable()
    reached = sum(1 for s in stats if s[1])
    print("SPIKE PROFILE wrapped/0 episodes=%d reached=%d "
          "total_entries=%d" % (len(stats), reached, sum(dec.values())),
          flush=True)
    out = StringIO()
    ps = pstats.Stats(pr, stream=out)
    ps.sort_stats("cumulative")
    ps.print_stats(40)
    print(out.getvalue(), flush=True)


if __name__ == "__main__":
    main()

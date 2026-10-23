#!/usr/bin/env python3
"""spike_md1_profile.py -- MD1 spike profiler run (MEASUREMENT ONLY).

cProfile run of ONE wrapped/0 training at the exact
Step 0 protocol (train_md1.TRAINERS["wrapped"](0)).
This is SEPARATE from the timed spike runs: the
numbers in spike_md1_profile.txt are PROFILER
ACCOUNTING, not wall time, and must never be
reported as pace. Weights are discarded on exit;
no checkpoint is written; this is not a programme
training and never a Step 1 record.
"""

import cProfile
import io
import os
import pstats
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_md1 import TRAINERS

OUT_PATH = os.path.join(_HERE, "spike_md1_profile.txt")


def main():
    pr = cProfile.Profile()
    pr.enable()
    TRAINERS["wrapped"](0)
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf)
    ps.sort_stats("cumulative")
    ps.print_stats(30)
    with open(OUT_PATH, "w") as f:
        f.write("MD1 SPIKE PROFILE (profiler accounting, NOT wall "
                "time; wrapped/0 training, exact Step 0 protocol; "
                "weights discarded)\n\n")
        f.write(buf.getvalue())
    print("profile written to %s" % OUT_PATH)


if __name__ == "__main__":
    main()

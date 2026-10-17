#!/usr/bin/env python3
"""compute_yardsticks.py -- recompute hand-coded policy yardsticks.

Parametrizes the repo's yardstick policies (naive, time-aware,
delay-line) for an arbitrary (period, open_len) schedule and reports
mean return over all start phases. Used to derive the "fast"
threshold for unseen schedules.

Usage: python3 compute_yardsticks.py --period 30 --open-len 14
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Rebind schedule globals BEFORE importing policies that use them.
import agent_nn as _ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=20)
    ap.add_argument("--open-len", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    _ann.PERIOD = args.period
    _ann.OPEN_LEN = args.open_len

    # Re-import after rebinding so the policies see the new values.
    import importlib
    import run_arms as _ra
    importlib.reload(_ra)

    from world import World
    results = {}
    for name, policy in [("naive", _ra.FuncPolicy(_ann.naive_policy)),
                         ("time-aware",
                          _ra.FuncPolicy(_ann.time_aware_policy)),
                         ("delay-line", _ra.DelayLinePolicy())]:
        rows = []
        for start in range(args.period):
            for _ in range(args.repeats):
                rows.append(_ra.run_episode(policy, start, 0.0, None))
        mean_ret = sum(r[0] for r in rows) / len(rows)
        crush = sum(r[2] for r in rows) / len(rows)
        n_crush_phases = sum(1 for r in rows if r[2] > 0)
        results[name] = (mean_ret, crush, n_crush_phases)
        print("%s: return=%.3f crush/ep=%.3f phases-with-crush=%d/%d"
              % (name, mean_ret, crush, n_crush_phases, len(rows)))

    naive, ceiling = results["naive"][0], results["time-aware"][0]
    headroom = ceiling - naive
    fast = naive + 0.26 * headroom
    print("headroom=%.3f fast(0.26)=%.3f" % (headroom, fast))


if __name__ == "__main__":
    main()

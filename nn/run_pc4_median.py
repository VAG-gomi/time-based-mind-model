#!/usr/bin/env python3
"""run_pc4_median.py -- WO-TBM-PC4 runner (diagnostic).

Seeds 200-202, trust-1.5 with median-priority insertion, 1000
episodes, 20/10. Records per seed to nn/results_pc4_median.jsonl
(flush per seed, skip-existing): reached/ticks/crushes,
first80, final-100, and the sampled-age manipulation check
(batch-weighted mean sampled age overall and for episodes
0-340, plus the per-episode age trace).
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from layer15_median_train import train_trust15_median

OUT = os.path.join(_HERE, "results_pc4_median.jsonl")
SEEDS = (200, 201, 202)


def first80(reached):
    run = 0
    for i, r in enumerate(reached):
        run += r
        if i >= 100:
            run -= reached[i - 100]
        if i >= 99 and run / 100 >= 0.8:
            return i
    return None


def wmean(age_trace, lo, hi):
    num = den = 0.0
    for ep in range(lo, hi + 1):
        m, n = age_trace[ep]
        if m is not None:
            num += m * n
            den += n
    return (num / den) if den else None


def load_existing():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["seed"])
    return done


def main():
    done = load_existing()
    with open(OUT, "a") as f:
        for seed in SEEDS:
            if seed in done:
                print("seed %d already recorded, skipping" % seed,
                      flush=True)
                continue
            t0 = time.perf_counter()
            _a, stats, ages = train_trust15_median(seed, episodes=1000)
            wall = time.perf_counter() - t0
            reached = [bool(s[1]) for s in stats]
            rec = {
                "arm": "trust15_median", "seed": seed, "wall_s": wall,
                "episodes": 1000, "hidden": 32, "lr": 1e-3,
                "period": 20, "open_len": 10,
                "reached": reached,
                "ticks": [int(s[0]) for s in stats],
                "crushes": [int(s[2]) for s in stats],
                "first80": first80(reached),
                "final100": sum(reached[-100:]) / 100,
                "age_mean_all": wmean(ages, 0, 999),
                "age_mean_early": wmean(ages, 0, 340),
                "age_trace": [a[0] for a in ages],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("seed %d done: wall %.1fs first80 %s age_all %.0f"
                  % (seed, wall, rec["first80"], rec["age_mean_all"]),
                  flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

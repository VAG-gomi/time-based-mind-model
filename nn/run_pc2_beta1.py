#!/usr/bin/env python3
"""run_pc2_beta1.py -- WO-TBM-PC2 runner (diagnostic).

Seeds 200-202, trust15 with beta == 1.0, 1000 episodes, 20/10.
Records per seed to nn/results_pc2_beta1.jsonl (flush per seed,
skip-existing): reached/ticks/crushes, wall-clock (recorded,
not an endpoint), first80, final-100, and the ESS manipulation
check trace (early-phase mean ESS over episodes 0-340 plus the
full per-episode ESS list).
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from layer15_beta1_train import train_trust15_beta1

OUT = os.path.join(_HERE, "results_pc2_beta1.jsonl")
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
            _a, stats, ess = train_trust15_beta1(seed, episodes=1000)
            wall = time.perf_counter() - t0
            reached = [bool(s[1]) for s in stats]
            early = [e for e in ess[:341] if e is not None]
            rec = {
                "arm": "trust15_beta1", "seed": seed, "wall_s": wall,
                "episodes": 1000, "hidden": 32, "lr": 1e-3,
                "period": 20, "open_len": 10,
                "reached": reached,
                "ticks": [int(s[0]) for s in stats],
                "crushes": [int(s[2]) for s in stats],
                "first80": first80(reached),
                "final100": sum(reached[-100:]) / 100,
                "ess_early_mean": (sum(early) / len(early)) if early else None,
                "ess_trace": ess,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("seed %d done: wall %.1fs first80 %s ess_early %.1f"
                  % (seed, wall, rec["first80"], rec["ess_early_mean"]),
                  flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""run_trust15_beta.py -- WO-TBM-PC6-run: the confirmatory
beta-anneal test (nn/prereg_trust15_beta.md, frozen c2320cb).

Three arms per seed, interleaved (uniform, trust15-annealed,
trust15-beta1 as committed at b3e3e8d). Seeds 500-519; 1000
episodes, hidden=32, lr=1e-3, 20/10. Records (rematch format)
to nn/results_trust15_beta.jsonl; a seed's records are written
and flushed when the seed completes (per the prereg); any
(arm, seed) already present is skipped on relaunch. No
endpoints are computed here -- the frozen analysis
(analyze_trust15_beta.py) runs once, on the complete 60
records only.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent_nn import train_neural, PlainEncoder
from layer15_trust_train import train_trust15
from layer15_beta1_train import train_trust15_beta1

OUT = os.path.join(_HERE, "results_trust15_beta.jsonl")
SEEDS = tuple(range(500, 520))
EPISODES = 1000
ARMS = ("uniform", "trust15", "trust15_beta1")


def load_existing():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["arm"], rec["seed"]))
    return done


def run_one(arm, seed):
    enc = PlainEncoder()
    t0 = time.perf_counter()
    if arm == "uniform":
        _a, stats = train_neural(seed, hidden=32, episodes=EPISODES,
                                 lr=1e-3, encoder=enc)
    elif arm == "trust15":
        _a, stats = train_trust15(seed, hidden=32, episodes=EPISODES,
                                   lr=1e-3, encoder=enc)
    else:
        _a, stats, _ess = train_trust15_beta1(
            seed, hidden=32, episodes=EPISODES, lr=1e-3, encoder=enc)
    return {
        "arm": arm, "seed": seed, "wall_s": time.perf_counter() - t0,
        "episodes": EPISODES, "hidden": 32, "lr": 1e-3,
        "period": 20, "open_len": 10,
        "reached": [bool(s[1]) for s in stats],
        "ticks": [int(s[0]) for s in stats],
        "crushes": [int(s[2]) for s in stats],
    }


def main():
    done = load_existing()
    with open(OUT, "a") as f:
        for seed in SEEDS:
            seed_recs = []
            for arm in ARMS:
                if (arm, seed) in done:
                    continue
                seed_recs.append(run_one(arm, seed))
                done.add((arm, seed))
            for rec in seed_recs:
                f.write(json.dumps(rec) + "\n")
            if seed_recs:
                f.flush()
            print("records complete: %d/60" % len(done), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

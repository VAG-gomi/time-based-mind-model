#!/usr/bin/env python3
"""run_trust15_var.py -- WO-TBM-PC5-run: the preregistered
variance-null test (nn/prereg_trust15_var.md, frozen d060f50).

Arms uniform vs trust15, code exactly as frozen at 7eaca1e.
Seeds 210-229 paired; 1000 episodes, hidden=32, lr=1e-3, 20/10.
Arms interleaved per seed. Records (same format as the rematch)
to nn/results_trust15_var.jsonl, flushed per seed, completed
(arm, seed) pairs skipped on relaunch. No endpoints are
computed here -- the frozen analysis (analyze_trust15_var.py)
runs once, on the complete 40 records only.
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

OUT = os.path.join(_HERE, "results_trust15_var.jsonl")
SEEDS = tuple(range(210, 230))
EPISODES = 1000


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
    else:
        _a, stats = train_trust15(seed, hidden=32, episodes=EPISODES,
                                   lr=1e-3, encoder=enc)
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
            for arm in ("uniform", "trust15"):
                if (arm, seed) in done:
                    continue
                rec = run_one(arm, seed)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done.add((arm, seed))
                print("records complete: %d/40" % len(done), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""run_trust15.py -- WO-TBM-L15-run: the preregistered rematch.

Executes nn/prereg_trust15.md exactly: arms uniform vs trust-1.5,
seeds 200-202, 1000 episodes, hidden=32, lr=1e-3, 20/10 protocol
-- identical to the Layer-1 re-run (rerun_trust_head2head.py),
whose structure this runner mirrors with the trust15 train
function swapped in.

Bare wall-clock per seed (perf_counter; no profiler, no other
instrumentation). Arms interleaved per seed so machine conditions
are shared. Records flush per (arm, seed) to
nn/results_trust15.jsonl; completed pairs are skipped on
relaunch. Endpoints and the success criterion are NOT computed
here -- they are arithmetic on the records, applied verbatim
from the prereg in nn/decisions_trust15.md.
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

OUT = os.path.join(_HERE, "results_trust15.jsonl")
SEEDS = (200, 201, 202)
EPISODES = 1000


def load_existing():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done.add((rec["arm"], rec["seed"]))
    return done


def run_pair(arm, seed):
    enc = PlainEncoder()
    t0 = time.perf_counter()
    if arm == "uniform":
        _agent, stats = train_neural(seed, hidden=32, episodes=EPISODES,
                                     lr=1e-3, encoder=enc)
    else:
        _agent, stats = train_trust15(seed, hidden=32, episodes=EPISODES,
                                      lr=1e-3, encoder=enc)
    wall = time.perf_counter() - t0
    return {
        "arm": arm, "seed": seed, "wall_s": wall, "episodes": EPISODES,
        "hidden": 32, "lr": 1e-3, "period": 20, "open_len": 10,
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
                    print("%s seed %d already recorded, skipping"
                          % (arm, seed), flush=True)
                    continue
                rec = run_pair(arm, seed)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print("%s seed %d done: wall %.1fs"
                      % (arm, seed, rec["wall_s"]), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

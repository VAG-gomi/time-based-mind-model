#!/usr/bin/env python3
"""rerun_trust_head2head.py -- WO-TBM-P3-followup (2026-10-19).

Re-run of the Layer-1 head-to-head: uniform replay baseline
(agent_nn.train_neural) vs trust-weighted replay
(layer1_trust_train.train_trust), seeds 200-202, 1000 episodes each --
the same seeds and episode budget as the original Layer-1 test, so
this is a replication, not a new design.

The original test's head-to-head numbers lived only in chat history;
this re-run establishes the negative result as tracked bytes in
nn/results_trust_rerun.jsonl. No numbers from session memory are
used anywhere in the permanent record.

Protocol: hidden=32, lr=1e-3, PlainEncoder, PERIOD/OPEN_LEN from
agent_nn (20/10), epsilon schedule from agent_nn. Per-seed records
are flushed as each seed completes, so a killed process loses only
in-flight seeds. Metric definitions (window=100) are documented in
nn/trust_rerun_note.md; raw per-episode reached flags are stored so
any window can be recomputed.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent_nn import train_neural, PlainEncoder
from layer1_trust_train import train_trust

SEEDS = [200, 201, 202]
EPISODES = 1000
OUT = os.path.join(_HERE, "results_trust_rerun.jsonl")


def run_one(arm, seed):
    train = train_neural if arm == "uniform" else train_trust
    t0 = time.perf_counter()
    _, stats = train(seed, episodes=EPISODES, encoder=PlainEncoder())
    wall = time.perf_counter() - t0
    return {
        "kind": "trust_rerun",
        "arm": arm,
        "seed": seed,
        "episodes": EPISODES,
        "hidden": 32,
        "lr": 1e-3,
        "wall_seconds": wall,
        "reached": [1 if s[1] else 0 for s in stats],
        "ticks": [s[0] for s in stats],
        "crushes": [s[2] for s in stats],
    }


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["arm"], r["seed"]))
    with open(OUT, "a") as f:
        for arm in ("uniform", "trust"):
            for seed in SEEDS:
                if (arm, seed) in done:
                    print("%s seed %d: already recorded, skipping"
                          % (arm, seed), flush=True)
                    continue
                print("%s seed %d: training (%d episodes)..."
                      % (arm, seed, EPISODES), flush=True)
                rec = run_one(arm, seed)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print("%s seed %d: wall %.1fs" % (arm, seed,
                      rec["wall_seconds"]), flush=True)
    print("done: %s" % OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""sanity_md1.py -- MD1 Step 0 sanity trainings (WO-TBM-MD1, Step 0).

Runs 6 full sanity trainings (1 seed per arm,
seed 0, 2000 episodes each, exact Step 1 protocol
via train_md1.TRAINERS) and reports reached counts
plus per-row training counts. Weights are held
only in memory and DISCARDED on exit: nothing
from this script is a Step 1 record, and no
checkpoint file is written.

Smoke test (labelled SMOKE, discarded): the
wrapped/0 sanity agent is evaluated greedily on
fixed worlds R5 (trained) and R8 (held-out), 20
phases each, with a fresh encoder per world
(the Step 1 eval rule). Smoke outcomes are pace /
plumbing evidence only, never Step 1 results.

Writes sanity_md1.log (human-readable) and
sanity_md1.jsonl (one summary per arm). Exits
non-zero if any row count is not exactly 250.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_md1 import TRAINERS
from world_md1 import TRAINED_ROWS
from md1_run import evaluate, fresh_encoder

LOG_PATH = os.path.join(_HERE, "sanity_md1.log")
JSONL_PATH = os.path.join(_HERE, "sanity_md1.jsonl")


def main():
    log = open(LOG_PATH, "w")
    jsonl = open(JSONL_PATH, "w")

    def emit(line):
        log.write(line + "\n")
        log.flush()
        print(line)

    wrapped_agent = None
    for arm in ("wrapped", "plain", "memory", "oracle_ind",
                "oracle_new", "phase_only_new"):
        t0 = time.perf_counter()
        agent, stats, _enc, rc, _dec, _ewe, rr = TRAINERS[arm](0)
        dt = time.perf_counter() - t0
        reached = sum(1 for s in stats if s[1])
        crushes = sum(s[2] for s in stats)
        ticks_ok = [s[0] for s in stats if s[1]]
        mean_ticks = (sum(ticks_ok) / len(ticks_ok)) if ticks_ok else None
        balanced = all(rc[r] == 250 for r in TRAINED_ROWS)
        emit("sanity %s/0: reached %d/2000, crushes %d, "
             "mean_ticks_reached %s, row_counts_balanced %s, "
             "wall_time %.1fs"
             % (arm, reached, crushes, mean_ticks, balanced, dt))
        emit("  row_reached: %s" % ({r: rr[r] for r in TRAINED_ROWS},))
        if not balanced:
            raise SystemExit("row counts not balanced for %s/0: %s"
                             % (arm, rc))
        jsonl.write(json.dumps({
            "kind": "sanity_md1", "arm": arm, "seed": 0,
            "episodes": 2000, "reached": reached, "crushes": crushes,
            "mean_ticks_reached": mean_ticks,
            "row_counts": {str(r): rc[r] for r in TRAINED_ROWS},
            "row_reached": {str(r): rr[r] for r in TRAINED_ROWS},
            "wall_time_s": round(dt, 1)}) + "\n")
        jsonl.flush()
        if arm == "wrapped":
            wrapped_agent = agent

    # Smoke eval (discarded): wrapped/0 on R5 and R8, fresh encoder
    # per world, 20 phases each.
    for wname, row in (("R5", 5), ("R8", 8)):
        eps = evaluate(wrapped_agent, fresh_encoder("wrapped"),
                       wname, row, log_features=True)
        reached = sum(1 for e in eps if e["reached"])
        entered = sum(1 for e in eps if e["door_entries"])
        blank = sum(1 for e in eps if e["blank_feature"])
        emit("SMOKE (discarded) wrapped/0 %s: reached %d/20, entered "
             "%d/20, blank_feature episodes %d/20"
             % (wname, reached, entered, blank))

    emit("sanity_md1: ALL DONE (weights discarded, nothing saved)")
    log.close()
    jsonl.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""spike_md1_run.py -- MD1 benchmark spike (MEASUREMENT ONLY).

WO-TBM-MD1, ruling E section 5(1); owner gate ruling
(design note section 14, commit 1068cb3): BENCHMARK
SPIKE FIRST. This script measures current-substrate
MD1 costs only. It authorises NO programme training,
NO programme evaluation, NO substrate change.

What it does:
  1. Exactly ONE timed training per arm (6 total),
     seed 0, exact Step 0 protocol via
     train_md1.TRAINERS (moving-door world, shuffled
     250/row multiset, fixed 20/10, 2000 episodes,
     cap 200, same hyperparameters). Timed with
     time.perf_counter (wall time).
  2. A small timed eval sample on the spike
     wrapped/0 agent (held in memory): one fixed
     job on R5, one fixed job on R8, one M3 job,
     greedy, fresh encoder per job (Step 0 rule).
     Eval outcomes here are PACE data only, never
     programme results.
  3. Writes SPIKE-labelled measurements to
     spike_md1.jsonl and a human-readable
     spike_md1.log. It does NOT create
     nn/training_md1.jsonl or nn/results_md1.jsonl,
     writes NO checkpoint file, and discards all
     spike weights on exit. Spike trainings are
     NEVER Step 1 records and must not be pooled
     with sanity or programme records.

If interrupted, relaunching this script restarts
the spike cleanly (it overwrites spike_md1.jsonl
and spike_md1.log); spike measurements are
re-runnable, and a partial spike is never
presented as complete.
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
from md1_run import evaluate, evaluate_m3, fresh_encoder

LOG_PATH = os.path.join(_HERE, "spike_md1.log")
JSONL_PATH = os.path.join(_HERE, "spike_md1.jsonl")
ARMS = ("wrapped", "plain", "memory", "oracle_ind",
        "oracle_new", "phase_only_new")


def main():
    log = open(LOG_PATH, "w")
    jsonl = open(JSONL_PATH, "w")

    def emit(line):
        log.write(line + "\n")
        log.flush()
        print(line)

    emit("SPIKE MD1 (measurement only, ruling E 5(1)): "
         "6 timed seed-0 trainings + eval sample; "
         "weights discarded, never Step 1 records")
    wrapped_agent = None
    for arm in ARMS:
        t0 = time.perf_counter()
        agent, stats, _enc, rc, _dec, _ewe, rr = TRAINERS[arm](0)
        dt = time.perf_counter() - t0
        reached = sum(1 for s in stats if s[1])
        crushes = sum(s[2] for s in stats)
        ticks_ok = [s[0] for s in stats if s[1]]
        mean_ticks = (sum(ticks_ok) / len(ticks_ok)) if ticks_ok else None
        balanced = all(rc[r] == 250 for r in TRAINED_ROWS)
        emit("SPIKE training %s/0: reached %d/2000, crushes %d, "
             "mean_ticks_reached %s, row_counts_balanced %s, "
             "wall_time %.1fs"
             % (arm, reached, crushes, mean_ticks, balanced, dt))
        jsonl.write(json.dumps({
            "kind": "spike_md1_training", "arm": arm, "seed": 0,
            "episodes": 2000, "reached": reached, "crushes": crushes,
            "mean_ticks_reached": mean_ticks,
            "row_counts": {str(r): rc[r] for r in TRAINED_ROWS},
            "row_reached": {str(r): rr[r] for r in TRAINED_ROWS},
            "wall_time_s": round(dt, 1)}) + "\n")
        jsonl.flush()
        if arm == "wrapped":
            wrapped_agent = agent

    # Timed eval sample (PACE ONLY) on the spike wrapped/0
    # agent, fresh encoder per job, greedy, 20 phases.
    eval_jobs = []
    for label, fn in (
            ("R5", lambda enc: evaluate(wrapped_agent, enc, "R5", 5)),
            ("R8", lambda enc: evaluate(wrapped_agent, enc, "R8", 8)),
            ("M3", lambda enc: evaluate_m3(wrapped_agent, enc,
                                            "wrapped", 0))):
        t0 = time.perf_counter()
        eps = fn(fresh_encoder("wrapped"))
        dt = time.perf_counter() - t0
        reached = sum(1 for e in eps if e["reached"])
        entered = sum(1 for e in eps if e["door_entries"])
        emit("SPIKE eval sample wrapped/0 %s: %d episodes, "
             "reached %d, entered %d, wall_time %.3fs "
             "(PACE ONLY, not a programme result)"
             % (label, len(eps), reached, entered, dt))
        eval_jobs.append({"job": label, "episodes": len(eps),
                          "reached": reached, "entered": entered,
                          "wall_time_s": round(dt, 3)})
        jsonl.write(json.dumps({
            "kind": "spike_md1_eval", "arm": "wrapped", "seed": 0,
            "job": label, "episodes": len(eps), "reached": reached,
            "entered": entered, "wall_time_s": round(dt, 3)}) + "\n")
        jsonl.flush()

    emit("SPIKE MD1 RUN COMPLETE (weights discarded, "
         "nothing saved as a checkpoint)")
    log.close()
    jsonl.close()


if __name__ == "__main__":
    main()

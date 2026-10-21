#!/usr/bin/env python3
"""Agent-level regression for world_doors_sched.py (Step 0b, part ii).

One seed per arm, fresh training, DIRECT equal-schedule W2/W5
evals on SchedDoorWorld (no preceding eval -- the fresh-direct
procedure, matching the doors remedy for wrapped). Summaries must
equal the doors records exactly:
  - plain/memory/oracle_ind/oracle_new/phase_only_new:
    results_doors.jsonl W2/W5 (their encoders are stateless, so
    the main run's sequential procedure is equivalent).
  - wrapped/0: results_doors_remedy.jsonl W2/W5 (fresh-direct).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_nn as ann
from doors_run import summarize, train
from sched_run import evaluate_sched
from world_doors_sched import equal_schedules

REGRESSION_AGENTS = [("wrapped", 0, 20, 10),
                     ("plain", 0, 20, 10),
                     ("memory", 0, 20, 10),
                     ("oracle_ind", 100, 30, 14),
                     ("oracle_new", 100, 30, 14),
                     ("phase_only_new", 100, 30, 14)]
GAPS = {"W2": ((5, 5), (5, 1)),
        "W5": ((5, 1), (5, 3), (5, 5), (5, 7), (5, 9))}


def load_targets():
    # Main file first, then remedy rows overwrite (remedy is
    # canonical for wrapped).
    targets = {}
    for fn in ("results_doors.jsonl", "results_doors_remedy.jsonl"):
        eps = {}
        for line in open(os.path.join(_HERE, fn)):
            r = json.loads(line)
            if r.get("kind") == "episode":
                eps.setdefault((r["arm"], r["seed"], r["world"]),
                               []).append(r)
        targets.update(eps)
    return {k: summarize(v) for k, v in targets.items()}


def main():
    targets = load_targets()
    ok_all = True
    for arm, seed, period, open_len in REGRESSION_AGENTS:
        ann.PERIOD = period
        ann.OPEN_LEN = open_len
        worlds = list(GAPS.items())
        if arm != "wrapped":
            print("%s/%d: training..." % (arm, seed), flush=True)
            agent, encoder = train(arm, seed)
            jobs = [(agent, encoder, wname, gaps)
                    for wname, gaps in worlds]
        else:
            # Wrapped: fresh training per world (its encoder is
            # stateful; a sequential second eval would drift).
            jobs = []
            for wname, gaps in worlds:
                print("%s/%d %s: fresh training..."
                      % (arm, seed, wname), flush=True)
                agent, encoder = train(arm, seed)
                jobs.append((agent, encoder, wname, gaps))
        for agent, encoder, wname, gaps in jobs:
            eps = evaluate_sched(agent, encoder,
                                 equal_schedules(gaps, period, open_len),
                                 wname)
            got = summarize(eps)
            exp = targets.get((arm, seed, wname))
            ok = (exp is not None
                  and got["mean_ticks"] == exp["mean_ticks"]
                  and got["mean_return"] == exp["mean_return"]
                  and got["crush_rate"] == exp["crush_rate"]
                  and got["success"] == exp["success"])
            ok_all = ok_all and ok
            print("%s/%d %s-equal: ticks=%s succ=%.2f -> %s"
                  % (arm, seed, wname,
                     got["mean_ticks"], got["success"],
                     "MATCH" if ok else "MISMATCH"), flush=True)
    print("agent-level regression: %s"
          % ("ALL MATCH" if ok_all else "MISMATCH PRESENT"))


if __name__ == "__main__":
    main()

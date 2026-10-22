#!/usr/bin/env python3
"""Agent-level training sanity for world_pda.py (Step 0b, part ii).

ONE seed per arm (6 agents), each trained from scratch
in the W5 training world (its protocol schedule) using
train_pda.py, 2000 episodes, same hyperparameters as
the design note. Sanity criteria (fixed in the design
note, NOT experiment-outcome gates): each training
completes 2000 episodes without crash; stats format
matches train_neural (ticks, reached, crushes, mean
loss); no NaN in stats; at least one training episode
reached the goal; per-door entry counts recorded and
total entries > 0.

Sanity trainings are CHECK outputs only (sanity_pda.log);
they are NOT Step 1 training records and their agents
are NOT reused in Step 1 (Step 1 trains fresh).
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_pda import TRAINERS
from world_pda import W5

SANITY_AGENTS = [("wrapped", 0, 20, 10),
                 ("plain", 0, 20, 10),
                 ("memory", 0, 20, 10),
                 ("oracle_ind", 100, 30, 14),
                 ("oracle_new", 100, 30, 14),
                 ("phase_only_new", 100, 30, 14)]


def main():
    ok_all = True
    for arm, seed, period, open_len in SANITY_AGENTS:
        print("%s/%d %d/%d: training..." % (arm, seed, period,
                                            open_len), flush=True)
        trainer = TRAINERS[arm]
        _agent, stats, _enc, dec, ewe = trainer(
            seed, period, open_len, W5, hidden=32,
            episodes=2000, lr=1e-3)
        n = len(stats)
        reached = sum(1 for s in stats if s[1])
        total_entries = sum(dec.values())
        nan = any((s[0] is None or (isinstance(s[3], float)
                                   and math.isnan(s[3])))
                  for s in stats)
        ok = (n == 2000 and not nan and reached > 0
              and total_entries > 0)
        ok_all = ok_all and ok
        print("%s/%d: episodes=%d reached=%d total_entries=%d "
              "per_door=%s episodes_with_entry=%s -> %s"
              % (arm, seed, n, reached, total_entries, dec, ewe,
                 "SANE" if ok else "NOT SANE"), flush=True)
    print("agent-level training sanity: %s"
          % ("ALL SANE" if ok_all else "FAILURE PRESENT"))


if __name__ == "__main__":
    main()

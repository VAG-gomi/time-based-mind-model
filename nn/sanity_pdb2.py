#!/usr/bin/env python3
"""Agent-level training sanity for world_pdb2.py (Step 0b, part ii).

ONE seed per arm (6 agents), each trained from scratch
in the F1 schedule-varied training world (its protocol
schedule) using train_pdb2.py, 2000 episodes, same
hyperparameters as the design note. Sanity criteria
(fixed in the design note, NOT experiment-outcome
gates): each training completes 2000 episodes without
crash; stats format matches train_neural (ticks,
reached, crushes, mean loss); no NaN in stats; at
least one training episode reached the goal;
per-door entry counts recorded and total entries > 0;
schedule open fractions recorded alongside.

Sanity trainings are CHECK outputs only
(sanity_pdb2.log); they are NOT Step 1 training
records and their agents are NOT reused in Step 1
(Step 1 trains fresh).
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_pdb2 import TRAINERS
from world_pdb2 import TRAIN_20, TRAIN_30

SANITY_AGENTS = [("wrapped", 0, TRAIN_20),
                 ("plain", 0, TRAIN_20),
                 ("memory", 0, TRAIN_20),
                 ("oracle_ind", 100, TRAIN_30),
                 ("oracle_new", 100, TRAIN_30),
                 ("phase_only_new", 100, TRAIN_30)]


def main():
    ok_all = True
    for arm, seed, schedules in SANITY_AGENTS:
        print("%s/%d: training..." % (arm, seed), flush=True)
        trainer = TRAINERS[arm]
        _agent, stats, _enc, dec, ewe, frac = trainer(
            seed, schedules, hidden=32, episodes=2000, lr=1e-3)
        n = len(stats)
        reached = sum(1 for s in stats if s[1])
        total_entries = sum(dec.values())
        nan = any((s[0] is None or (isinstance(s[3], float)
                                   and math.isnan(s[3])))
                  for s in stats)
        ok = (n == 2000 and not nan and reached > 0
              and total_entries > 0 and len(frac) == len(dec))
        ok_all = ok_all and ok
        print("%s/%d: episodes=%d reached=%d total_entries=%d "
              "per_door=%s episodes_with_entry=%s "
              "open_fractions=%s -> %s"
              % (arm, seed, n, reached, total_entries, dec, ewe,
                 frac, "SANE" if ok else "NOT SANE"), flush=True)
    print("agent-level training sanity: %s"
          % ("ALL SANE" if ok_all else "FAILURE PRESENT"))


if __name__ == "__main__":
    main()

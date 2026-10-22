#!/usr/bin/env python3
"""world_pda.py -- P-Da multi-door worlds (WO-TBM-P-Da, Step 0).

Additive only: world.py, world_doors.py,
world_doors_sched.py, and world_luck.py are NEVER edited.
Mechanics are inherited from MultiDoorWorld deliberately:
the P-Da axis is location-only (same single schedule for
all doors, same wall/blocking/crush rules), so
reimplementing identical mechanics would add bug surface
for zero information. This file names and freezes the
P-Da worlds; the regression (scripted + agent-level
sanity) verifies the inheritance.

Gap sets:
- W5 (training world, also E1) = {(5,1),(5,3),(5,5),
  (5,7),(5,9)}: the doors W5 gaps.
- W1 (E0) = {(5,5)}: single familiar door.
- W5r (E2) = {(5,1),(5,3),(5,7),(5,9)}: rivals only,
  the Luck world.
- W1n (E3) = {(5,8)}: held-out novel door only.
- W5n (E4) = W5 + (5,8): novel door among familiars.

All doors in every world share the arm's training
schedule (20/10 or 30/14); observe() therefore returns
the same shared door bit regardless of which gap the
agent is near. Location variation is via wall geometry
only, not per-door timing (that is P-Db2).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world_doors import MultiDoorWorld

W5 = ((5, 1), (5, 3), (5, 5), (5, 7), (5, 9))
W1 = ((5, 5),)
W5R = ((5, 1), (5, 3), (5, 7), (5, 9))
W1N = ((5, 8),)
W5N = ((5, 1), (5, 3), (5, 5), (5, 7), (5, 8), (5, 9))

PDA_WORLDS = {"E0_W1": W1, "E1_W5": W5, "E2_W5r": W5R,
              "E3_W1n": W1N, "E4_W5n": W5N}


class PdaWorld(MultiDoorWorld):
    """MultiDoorWorld named for the P-Da experiment.

    No mechanics overridden: period/open_len/gaps pass
    straight through. Gap sets are frozen by the
    constants above, not by a runtime guard, so the
    regression can run the same class with any gap
    set for equivalence.
    """

    def __init__(self, period, open_len, gaps):
        super(PdaWorld, self).__init__(period, open_len, gaps)

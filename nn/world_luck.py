#!/usr/bin/env python3
"""world_luck.py -- familiar-removed multi-door worlds (WO-TBM-Luck).

Additive only: world.py, world_doors.py, and
world_doors_sched.py are NEVER edited. Mechanics are
inherited from MultiDoorWorld deliberately: the luck axis
is location-only (same single schedule for all doors, same
wall/blocking/crush rules), so reimplementing identical
mechanics would add bug surface for zero information. This
file names and freezes the luck worlds; the regression
(scripted + agent-level) verifies the inheritance.

Gap sets (familiar door (5,5) ABSENT by definition):
- W1r = {(5,1)}: single rival gap only.
- W5r = {(5,1), (5,3), (5,7), (5,9)}: rivals only.
The cell (4,5) is ordinary floor in these worlds (west of
a wall cell). Agents can occupy it freely; the D
classification in the design note turns on whether they do
so before their first door entry.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world_doors import MultiDoorWorld

W1R = ((5, 1),)
W5R = ((5, 1), (5, 3), (5, 7), (5, 9))
LUCK_WORLDS = {"W1r": W1R, "W5r": W5R}


class LuckWorld(MultiDoorWorld):
    """MultiDoorWorld named for the luck experiment.

    No mechanics overridden: period/open_len/gaps pass
    straight through. The familiar-removed property is
    guaranteed by the W1R/W5R constants (which exclude
    (5,5)), not by a runtime guard, so the regression can
    run the same class with gaps={(5,5)} for equivalence.
    """

    def __init__(self, period, open_len, gaps):
        super(LuckWorld, self).__init__(period, open_len, gaps)

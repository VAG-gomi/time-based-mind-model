#!/usr/bin/env python3
"""world_ft1.py -- FT1 G1 world (WO-TBM-FT1, Step 0).

Additive only: world.py, world_doors.py,
world_doors_sched.py, world_luck.py, world_pda.py,
and world_pdb2.py are NEVER edited. Mechanics are
inherited from Pdb2AbsentWorld deliberately: G1 is
F3's (5,8)-only construction with the gap row
changed to (5,6), and nothing else changes.

G1 = W1n2 {(5,6)} on the arm's training schedule
(period, open_len, offset 0). Observe bit =
training schedule bit (clock bit), exactly as in
F3: the gap is on the training schedule, so no
phantom/clock distinction arises. (5,6) sits
between trained (5,5) and (5,7): interpolation,
like (5,8), but a different row. G1 is an EVAL
world only, never a training world.

F0-F4 worlds are REUSED from P-Db2 (world_pdb2);
the fine-tuning world is F3's world
(world_pdb2.f3_schedules), reused unchanged.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world_pdb2 import familiar_schedule

# Second novel door: between trained (5,5) and (5,7).
NOVEL2 = (5, 6)


def g1_schedules(train):
    """G1 schedules: second novel door only, on the training schedule.

    train is a P-Db2 TRAIN table (TRAIN_20 / TRAIN_30);
    the schedule is the familiar (training) schedule,
    offset 0, exactly as f3_schedules builds F3.
    """
    period, open_len = familiar_schedule(train)
    return {NOVEL2: (period, open_len, 0)}

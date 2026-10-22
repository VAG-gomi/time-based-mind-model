#!/usr/bin/env python3
"""world_pdb2.py -- P-Db2 schedule-varied worlds (WO-TBM-P-Db2, Step 0).

Additive only: world.py, world_doors.py,
world_doors_sched.py, world_luck.py, and world_pda.py
are NEVER edited. Mechanics are inherited from
SchedDoorWorld deliberately: the P-Db2 axis is
schedule variation on top of the already-verified
sched mechanics.

Worlds (all via SchedDoorWorld except the two
familiar-absent worlds, which SchedDoorWorld cannot
represent because its __init__ requires the familiar
door in its schedule table):
- F1 (training world) = P-Db W5s schedules, imported
  unchanged from world_doors_sched (TRAIN_20 /
  TRAIN_30 below are the SCHED_W5S constants, not
  re-typed copies).
- F0 = {(5,5)} on the familiar schedule.
- F4 = TRAIN + (5,8) on the familiar schedule.
  Both use SchedDoorWorld directly.
- F2 = rivals only, W5s rival schedules, with a
  PHANTOM FAMILIAR CLOCK for the observe bit (see
  Pdb2AbsentWorld): the bit toggles on the familiar
  schedule even though (5,5) is a wall cell, never
  a gap. Blocking/crushing use rival schedules only.
- F3 = {(5,8)} on the training schedule, observe
  bit = training schedule bit (numerically identical
  to the familiar schedule bit, so no phantom/clock
  distinction arises).

Observe contract (fixed in the design note):
observe() returns the familiar door's bit in
familiar-present worlds, and the clock bit in
familiar-absent worlds. period/open_len exposed by
every world are the clock (familiar / training)
schedule. Rival schedules appear in NO agent's
features.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import AGENT_START, GOAL_POS, GRID_SIZE
from world_doors import is_wall_cell
from world_doors_sched import (FAMILIAR, SCHED_W5S_20, SCHED_W5S_30,
                               SchedDoorWorld, door_open_at)

# TRAIN tables are the W5s constants, imported — not re-derived,
# not re-typed. The Step 0 test asserts this identity.
TRAIN_20 = SCHED_W5S_20
TRAIN_30 = SCHED_W5S_30

NOVEL = (5, 8)


def familiar_schedule(train):
    """(period, open_len) of the familiar door in a TRAIN table."""
    period, open_len, offset = train[FAMILIAR]
    assert offset == 0, "familiar offset must be 0"
    return period, open_len


def rival_schedules(train):
    """Rival-only subset of a TRAIN table (F2 schedules)."""
    return {g: s for g, s in train.items() if g != FAMILIAR}


def f0_schedules(train):
    """F0 schedules: familiar door only."""
    return {FAMILIAR: train[FAMILIAR]}


def f3_schedules(train):
    """F3 schedules: novel door only, on the training schedule."""
    period, open_len = familiar_schedule(train)
    return {NOVEL: (period, open_len, 0)}


def f4_schedules(train):
    """F4 schedules: TRAIN + novel door on the familiar schedule."""
    period, open_len = familiar_schedule(train)
    out = dict(train)
    out[NOVEL] = (period, open_len, 0)
    return out


def open_fractions(schedules):
    """Schedule open fraction (open_len / period) per door."""
    return {g: open_len / float(period)
            for g, (period, open_len, _offset) in schedules.items()}


class Pdb2AbsentWorld(object):
    """SchedDoorWorld mirror for familiar-absent worlds (F2, F3).

    Mirrors SchedDoorWorld.step() line-for-line except
    that the observe bit comes from a separately stored
    clock schedule (familiar schedule in F2, training
    schedule in F3) instead of a familiar gap entry.
    period/open_len exposed are the clock schedule, so
    OracleIndEncoder reads the clock, exactly as it
    reads the familiar schedule in SchedDoorWorld.
    Wall blocking, per-door blocking, per-door crush,
    log tokens, and grid-edge clipping are identical
    to SchedDoorWorld.
    """

    def __init__(self, schedules, clock_period, clock_open_len):
        self.schedules = dict(schedules)
        self.gaps = frozenset(schedules)
        if FAMILIAR in self.gaps:
            raise ValueError("Pdb2AbsentWorld is for familiar-absent "
                             "worlds; familiar present: use "
                             "SchedDoorWorld")
        self.period = clock_period
        self.open_len = clock_open_len
        self.tick = 0
        self.log = []
        self.agent = AGENT_START
        self._door_prev = None
        self._was_on_goal = False

    def door_open(self, cell, tick=None):
        if tick is None:
            tick = self.tick
        period, open_len, offset = self.schedules[cell]
        return door_open_at(tick, period, open_len, offset)

    def clock_open(self, tick=None):
        if tick is None:
            tick = self.tick
        return door_open_at(tick, self.period, self.open_len, 0)

    def reset(self, start_tick=0):
        self.tick = start_tick
        self.agent = AGENT_START
        self.log = []
        self._door_prev = None
        self._was_on_goal = False

    def observe(self):
        return (self.agent, self.clock_open())

    def step(self, action):
        t = self.tick
        open_clock = self.clock_open(t)
        if open_clock != self._door_prev:
            self.log.append((t, "door",
                             "open" if open_clock else "closed"))
            self._door_prev = open_clock
        if self.agent in self.gaps and not self.door_open(self.agent,
                                                          t):
            self.log.append((t, "agent", "crushed"))
            self.agent = AGENT_START
            self._was_on_goal = False
            self.tick += 1
            return
        dx, dy = action
        nx = min(max(self.agent[0] + dx, 0), GRID_SIZE - 1)
        ny = min(max(self.agent[1] + dy, 0), GRID_SIZE - 1)
        target = (nx, ny)
        enters_gap_while_closed = (target in self.gaps
                                   and not self.door_open(target, t))
        blocked = is_wall_cell(target, self.gaps) or \
            enters_gap_while_closed
        if blocked and target != self.agent:
            self.log.append((t, "agent", "blocked at (%d, %d)" % (nx, ny)))
        else:
            self.agent = target
            self.log.append((t, "agent", "move to (%d, %d)" % (nx, ny)))
        on_goal = self.agent == GOAL_POS
        if on_goal and not self._was_on_goal:
            self.log.append((t, "goal", "reached"))
        self._was_on_goal = on_goal
        self.tick += 1

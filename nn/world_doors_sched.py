#!/usr/bin/env python3
"""world_doors_sched.py -- schedule-varied multi-door variant (WO-TBM-P-Db).

Additive only: world.py and world_doors.py are NEVER edited. This
module mirrors MultiDoorWorld.step() exactly, except each door gap
has its OWN schedule (period, open_len, phase offset).

Door-open rule (fixed in the Step 0a design note): a door with
(period, open_len, offset) is open at tick t iff
    (t - offset) % period < open_len.
Offset 0 reproduces door_is_open exactly.

Observe contract (fixed): observe() returns the FAMILIAR door's
state (gap (5,5)) as the door bit, and the world exposes
period/open_len = the familiar door's schedule (OracleIndEncoder
reads these at encode time). Rival schedules are NOT encoded in
any agent's features.

Crush rule, per door: the agent in gap cell (5,y) at a tick when
THAT door is closed is sent back to start, tick advances, log
token "crushed" (identical token so env_step detects it).
Wall cells x=5 non-gap always block; cannot move into gap (5,y)
while that door is closed; grid-edge clipping as in world.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import AGENT_START, GOAL_POS, GRID_SIZE
from world_doors import WALL_X, is_wall_cell

FAMILIAR = (5, 5)


def door_open_at(tick, period, open_len, offset):
    """True iff a door with this schedule is open at `tick`."""
    if period < 1:
        raise ValueError("period must be >= 1, got %r" % (period,))
    if open_len < 0 or open_len > period:
        raise ValueError("open_len must satisfy 0 <= open_len <= "
                         "period, got %r with period %r"
                         % (open_len, period))
    return (tick - offset) % period < open_len


# ---- Step 0a schedule tables (familiar keeps the training schedule;
# W2s rival is pure anti-phase; W5s rivals are staggered so that at
# any tick at least one rival is open) ----

# 20/10 protocol: familiar (5,5) open [0,10) mod 20.
SCHED_W2S_20 = {(5, 5): (20, 10, 0), (5, 1): (20, 10, 10)}
SCHED_W5S_20 = {(5, 5): (20, 10, 0), (5, 1): (20, 10, 10),
                (5, 3): (20, 10, 14), (5, 7): (20, 10, 18),
                (5, 9): (20, 10, 2)}

# 30/14 protocol: familiar (5,5) open [0,14) mod 30; anti-phase
# rival open_len 16 = the familiar-closed interval length.
SCHED_W2S_30 = {(5, 5): (30, 14, 0), (5, 1): (30, 16, 14)}
SCHED_W5S_30 = {(5, 5): (30, 14, 0), (5, 1): (30, 16, 14),
                (5, 3): (30, 16, 20), (5, 7): (30, 16, 26),
                (5, 9): (30, 16, 2)}

SCHEDS = {(20, "W2s"): SCHED_W2S_20, (20, "W5s"): SCHED_W5S_20,
          (30, "W2s"): SCHED_W2S_30, (30, "W5s"): SCHED_W5S_30}


def equal_schedules(gaps, period, open_len):
    """All doors on the familiar schedule (regression only)."""
    return {g: (period, open_len, 0) for g in gaps}


class SchedDoorWorld(object):
    """10x10 grid world with per-door schedules."""

    def __init__(self, schedules):
        if FAMILIAR not in schedules:
            raise ValueError("schedules must include familiar %r"
                             % (FAMILIAR,))
        self.schedules = dict(schedules)
        self.gaps = frozenset(schedules)
        period, open_len, offset = schedules[FAMILIAR]
        if offset != 0:
            raise ValueError("familiar offset must be 0, got %r"
                             % (offset,))
        self.period = period
        self.open_len = open_len
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

    def reset(self, start_tick=0):
        self.tick = start_tick
        self.agent = AGENT_START
        self.log = []
        self._door_prev = None
        self._was_on_goal = False

    def observe(self):
        return (self.agent, self.door_open(FAMILIAR))

    def step(self, action):
        t = self.tick
        open_fam = self.door_open(FAMILIAR, t)
        if open_fam != self._door_prev:
            self.log.append((t, "door",
                             "open" if open_fam else "closed"))
            self._door_prev = open_fam
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

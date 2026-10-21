#!/usr/bin/env python3
"""world_doors.py -- multi-door variant of world.py (WO-TBM-doors).

Additive only: world.py is NEVER edited. This module mirrors
World.step() exactly, except the wall at x=5 has a configurable
set of door gaps instead of the single gap at (5,5).

Rules (identical to world.py, generalized):
- Door schedule: one schedule for ALL doors (open iff
  tick % period < open_len). Doors with different schedules
  are out of scope for WO-TBM-doors.
- Wall: column x=5 for every y, EXCEPT the gap cells.
- Cannot move into a wall cell; cannot move INTO a gap
  cell while the door is closed (blocked, agent stays).
- CRUSH RULE: if the agent is in ANY gap cell when the
  door is closed, it is sent back to start, the tick
  advances, and the step returns with no move that tick.

With gaps = {(5, 5)} this class must behave identically
to World (verified by test_world_doors.py: scripted and
random-agent logs byte-identical).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import (AGENT_START, DOOR_CELL, GOAL_POS, GRID_SIZE,
                   door_is_open, ACTIONS)

WALL_X = 5


def is_wall_cell(cell, gaps):
    """True iff `cell` is a wall cell under `gaps`."""
    x, _y = cell
    return x == WALL_X and cell not in gaps


class MultiDoorWorld(object):
    """10x10 grid world with a configurable set of door gaps."""

    def __init__(self, period, open_len, gaps):
        self.period = period
        self.open_len = open_len
        self.gaps = frozenset(gaps)
        self.tick = 0
        self.log = []
        self.agent = AGENT_START
        self._door_prev = None
        self._was_on_goal = False

    def reset(self, start_tick=0):
        self.tick = start_tick
        self.agent = AGENT_START
        self.log = []
        self._door_prev = None
        self._was_on_goal = False

    def observe(self):
        return (self.agent,
                door_is_open(self.tick, self.period, self.open_len))

    def set_door(self, period, open_len):
        if period < 1:
            raise ValueError("period must be >= 1, got %r" % (period,))
        if open_len < 0 or open_len > period:
            raise ValueError("open_len must satisfy 0 <= open_len <= "
                             "period, got %r with period %r"
                             % (open_len, period))
        self.period = period
        self.open_len = open_len

    def step(self, action):
        t = self.tick
        open_now = door_is_open(t, self.period, self.open_len)
        if open_now != self._door_prev:
            self.log.append((t, "door", "open" if open_now else "closed"))
            self._door_prev = open_now
        if self.agent in self.gaps and not open_now:
            self.log.append((t, "agent", "crushed"))
            self.agent = AGENT_START
            self._was_on_goal = False
            self.tick += 1
            return
        dx, dy = action
        nx = min(max(self.agent[0] + dx, 0), GRID_SIZE - 1)
        ny = min(max(self.agent[1] + dy, 0), GRID_SIZE - 1)
        target = (nx, ny)
        enters_gap_while_closed = (target in self.gaps and not open_now)
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


def run_scripted_multi(ticks=25, world=None, gaps=((5, 5),)):
    """world.run_scripted_agent on a MultiDoorWorld."""
    if world is None:
        world = MultiDoorWorld(10, 4, gaps)
    waypoints = [(1, 0), (2, 0), (3, 0), (4, 0),
                 (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]
    phase = "approach"
    for _ in range(ticks):
        x, y = world.agent
        if phase == "approach":
            tx, ty = waypoints.pop(0)
            world.step((tx - x, ty - y))
            if world.agent == (4, 5):
                phase = "push"
        elif phase == "push":
            world.step(ACTIONS["right"])
            if world.agent == DOOR_CELL:
                phase = "escape"
        else:
            if world.agent == GOAL_POS:
                world.step(ACTIONS["stay"])
            elif y == 5 and x < 9:
                world.step(ACTIONS["right"])
            elif x == 9 and y < 9:
                world.step(ACTIONS["down"])
            else:
                world.step(ACTIONS["stay"])
    return world

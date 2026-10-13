#!/usr/bin/env python3
"""
world.py -- Step 1 of the Time-Based Mind Model (Agent Blueprint v0.1).

A 10x10 grid world with a tick counter and an event log.
The log is a list of (tick, node, event) tuples.
Nodes: "agent", "door", "goal".

The door follows a periodic schedule:
    door is OPEN  iff  tick % period < open_len
    door is CLOSED otherwise.
The schedule is configurable per world: World(period=PERIOD, open_len=OPEN_LEN)
stores them as self.period / self.open_len and step() uses those for the door
rule. The module-level PERIOD / OPEN_LEN are defaults only; door_is_open()
takes period and open_len as required arguments and can never silently fall
back to the module constants.

A wall covers column x=5 for every y in 0..9 EXCEPT the door cell (5, 5).
- The agent cannot move into a wall cell.
- The agent cannot move INTO the door cell while the door is closed;
  when the door is open it may enter.
- A blocked move leaves the agent where it was and logs
  (tick, "agent", "blocked at (x, y)"), where (x, y) is the cell it tried
  to enter.
- CRUSH RULE: if the agent is in the door cell (5, 5) when the door
  is closed, the agent is sent back to the start: the log records
  (tick, "agent", "crushed"), the agent is set to AGENT_START, the goal
  tracking state is cleared, the tick advances, and the step returns
  with no move that tick. The episode continues on the next step().
  Entering on the last open tick of a cycle is therefore fatal.

Documented fixed configuration (kept unchanged by every check):
    SEED      = 42   (fixed random seed for the random-agent run)
    GRID_SIZE = 10   (10x10 grid)
    PERIOD    = 10   (door cycle length, in ticks)
    OPEN_LEN  = 4    (ticks per cycle the door is open:
                      ticks where tick % 10 is 0, 1, 2 or 3)
    AGENT_START = (0, 0)
    GOAL_POS    = (9, 9)
    TICKS       = 50 (ticks in the check-(a) run -> 5 full door cycles)
    WALL_X      = 5  (wall column)
    DOOR_CELL   = (5, 5) (the single open cell in the wall)

Plain Python 3, standard library only (uses `random` and `hashlib`).
"""

import hashlib
import random

# ---------------- documented fixed configuration ----------------
SEED = 42
GRID_SIZE = 10
PERIOD = 10
OPEN_LEN = 4
AGENT_START = (0, 0)
GOAL_POS = (9, 9)
TICKS = 50
WALL_X = 5
DOOR_CELL = (5, 5)


def door_is_open(tick, period, open_len):
    """True iff the door is open at `tick` under the periodic rule.

    `period` and `open_len` are required arguments on purpose: there are no
    defaults, so this function can never silently fall back to the module
    constants PERIOD / OPEN_LEN.
    """
    return (tick % period) < open_len


def is_wall_cell(cell):
    """True iff `cell` is a wall cell (column x=5 except the door cell)."""
    x, _y = cell
    return x == WALL_X and cell != DOOR_CELL


class World(object):
    """10x10 grid world with a tick counter, a wall, and an event log."""

    def __init__(self, period=PERIOD, open_len=OPEN_LEN):
        self.period = period        # door cycle length, in ticks
        self.open_len = open_len    # ticks per cycle the door is open
        self.tick = 0
        self.log = []            # list of (tick, node, event) tuples
        self.agent = AGENT_START
        self._door_prev = None   # last door state; None until first tick
        self._was_on_goal = False

    def reset(self, start_tick=0):
        """Reset the world for a fresh run on the same door schedule.

        tick -> start_tick (0 by default), agent -> AGENT_START, the event
        log is cleared, and the door and goal tracking state is cleared,
        so a new run re-logs everything from scratch. self.period /
        self.open_len are untouched.
        """
        self.tick = start_tick
        self.agent = AGENT_START
        self.log = []
        self._door_prev = None
        self._was_on_goal = False

    def observe(self):
        """Read-only snapshot: (agent_position, door_open_now).

        door_open_now is the door state for the CURRENT tick -- the state
        the next step() will use. Returns only the 2-tuple (never the tick,
        period, or open length) and does not change the world in any way.
        """
        return (self.agent,
                door_is_open(self.tick, self.period, self.open_len))

    def set_door(self, period, open_len):
        """Change the door schedule from the current tick onward.

        Sets self.period / self.open_len without touching tick, the agent
        position, or the event log. The door-transition tracking
        (_door_prev) is intentionally left alone: the next step() logs a
        door event only if the new schedule's state differs from the last
        logged state.

        Raises ValueError if period < 1, open_len < 0, or open_len > period.
        (open_len == 0 and open_len == period are valid schedules.)
        """
        if period < 1:
            raise ValueError("period must be >= 1, got %r" % (period,))
        if open_len < 0 or open_len > period:
            raise ValueError("open_len must satisfy 0 <= open_len <= period, "
                             "got %r with period %r" % (open_len, period))
        self.period = period
        self.open_len = open_len

    def step(self, action):
        """Advance one tick: update door, check crush, move the agent,
        check the goal.

        Moves into wall cells are blocked. Moves INTO the door cell are
        blocked while the door is closed (entering only). A blocked move
        leaves the agent where it was and logs "blocked at (x, y)".

        Crush rule: if the agent is in the door cell (5, 5) when the door
        is closed this tick, the agent is crushed -- the log records
        (tick, "agent", "crushed"), the agent returns to AGENT_START, the
        goal tracking state is cleared, the tick advances, and the step
        returns with no move that tick. The episode continues.
        """
        t = self.tick

        # 1) door state this tick; log initial state and every transition
        open_now = door_is_open(t, self.period, self.open_len)
        if open_now != self._door_prev:
            self.log.append((t, "door", "open" if open_now else "closed"))
            self._door_prev = open_now

        # 2) crush check: caught in the door cell when the door is closed
        if self.agent == DOOR_CELL and not open_now:
            self.log.append((t, "agent", "crushed"))
            self.agent = AGENT_START
            self._was_on_goal = False
            self.tick += 1
            return

        # 3) move the agent (actions are (dx, dy) pairs; clipped to grid)
        dx, dy = action
        nx = min(max(self.agent[0] + dx, 0), GRID_SIZE - 1)
        ny = min(max(self.agent[1] + dy, 0), GRID_SIZE - 1)
        target = (nx, ny)
        enters_door_while_closed = (target == DOOR_CELL and not open_now)
        blocked = is_wall_cell(target) or enters_door_while_closed
        if blocked and target != self.agent:
            self.log.append((t, "agent", "blocked at (%d, %d)" % (nx, ny)))
        else:
            self.agent = target
            self.log.append((t, "agent", "move to (%d, %d)" % (nx, ny)))

        # 4) goal: log arrivals (transition into the goal cell only)
        on_goal = self.agent == GOAL_POS
        if on_goal and not self._was_on_goal:
            self.log.append((t, "goal", "reached"))
        self._was_on_goal = on_goal

        self.tick += 1


ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "stay": (0, 0),
}


def run_random_agent(seed=SEED, ticks=TICKS):
    """Run a uniform-random agent for `ticks` ticks with the given seed.

    Uses a LOCAL random.Random(seed) generator, so the run is repeatable
    regardless of the state of the global random generator. With seed=42
    the log matches the Step-1 run byte for byte.
    """
    rng = random.Random(seed)
    world = World()
    names = list(ACTIONS.keys())
    for _ in range(ticks):
        world.step(ACTIONS[rng.choice(names)])
    return world


def run_scripted_agent(ticks=25, world=None):
    """Scripted agent: walk (0,0) -> (4,5), push right through the door cell
    each tick until it enters (blocked attempts are logged by the World),
    then walk (5,5) -> (9,9). Deterministic; no randomness used.

    Runs on the world passed in, or on a new World() if none is given."""
    if world is None:
        world = World()
    waypoints = [(1, 0), (2, 0), (3, 0), (4, 0),
                 (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]
    phase = "approach"   # approach -> push -> escape
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
        else:  # escape: walk to the goal via row 5 then column 9
            if world.agent == GOAL_POS:
                world.step(ACTIONS["stay"])
            elif y == 5 and x < 9:
                world.step(ACTIONS["right"])
            elif x == 9 and y < 9:
                world.step(ACTIONS["down"])
            else:
                world.step(ACTIONS["stay"])
    return world


def verify_door_rule(log, period, open_len):
    """Mechanically check every door event in the log against the rule.

    `period` and `open_len` are required arguments (no defaults): every
    caller must pass them explicitly.

    Returns (all_ok, number_of_door_events_checked).
    """
    ok = True
    checked = 0
    for tick, node, event in log:
        if node != "door":
            continue
        checked += 1
        expected_open = door_is_open(tick, period, open_len)
        if event not in ("open", "closed"):
            print("INVALID door event %r at tick %d" % (event, tick))
            ok = False
        elif (event == "open") != expected_open:
            print("MISMATCH at tick %d: logged %r, rule says %s"
                  % (tick, event, "open" if expected_open else "closed"))
            ok = False
    return ok, checked


def verify_door_complete(log, period, open_len, ticks_run, start_tick=0):
    """Build the full expected door timeline and compare it to the log.

    Expected events for ticks start_tick .. start_tick + ticks_run - 1:
    an initial event at start_tick, then one event at every tick t after
    start_tick where door_is_open(t, period, open_len) differs from the
    previous tick. `actual` is the log's (tick, 'door', state) entries in
    order.

    Full list equality catches missing, extra, and wrong events by
    construction. Returns (ok, expected, actual).
    """
    expected = []
    prev_open = None
    for t in range(start_tick, start_tick + ticks_run):
        open_now = door_is_open(t, period, open_len)
        if t == start_tick or open_now != prev_open:
            expected.append((t, "door", "open" if open_now else "closed"))
        prev_open = open_now
    actual = [(tick, node, event) for tick, node, event in log
              if node == "door"]
    return (expected == actual), expected, actual


def verify_no_wall_landings(log):
    """Check that no 'move to' event lands on a wall cell.

    Entering the door cell (5, 5) is allowed. Returns (all_ok, moves_checked).
    """
    ok = True
    moves = 0
    for tick, node, event in log:
        if node == "agent" and event.startswith("move to ("):
            moves += 1
            x, y = (int(v) for v in event[len("move to ("):-1].split(","))
            if x == WALL_X and (x, y) != DOOR_CELL:
                print("WALL LANDING at tick %d: %r" % (tick, event))
                ok = False
    return ok, moves


def verify_scripted_run(log):
    """Check the scripted run: at least one 'blocked at (5, 5)' before the
    agent first enters (5, 5), and a goal-reached event.

    Returns (blocked_before_entry_ok, blocked_ticks, enter_tick, goal_ticks).
    """
    blocked_ticks = []
    enter_tick = None
    goal_ticks = []
    for tick, node, event in log:
        if node == "agent" and event == "blocked at (5, 5)":
            blocked_ticks.append(tick)
        elif node == "agent" and event == "move to (5, 5)":
            if enter_tick is None:
                enter_tick = tick
        elif node == "goal" and event == "reached":
            goal_ticks.append(tick)
    before = [t for t in blocked_ticks
              if enter_tick is not None and t < enter_tick]
    ok = enter_tick is not None and len(before) >= 1
    return ok, blocked_ticks, enter_tick, goal_ticks


def run_wall_push_agent():
    """Scripted agent for check (c): walk (0,0) -> (4,3) along open cells,
    then push right on two consecutive ticks into the wall at (5, 3).
    Deterministic; no randomness used."""
    world = World()
    waypoints = [(1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
    for tx, ty in waypoints:
        x, y = world.agent
        world.step((tx - x, ty - y))
    world.step(ACTIONS["right"])   # push into the wall: blocked at (5, 3)
    world.step(ACTIONS["right"])   # push again: blocked at (5, 3)
    return world


def verify_wall_push(log, world):
    """Check the wall-push run: the log shows "blocked at (5, 3)" exactly
    twice and the agent is still at (4, 3).

    Returns (all_ok, blocked_events).
    """
    blocked = [e for e in log
               if e[1] == "agent" and e[2] == "blocked at (5, 3)"]
    ok = len(blocked) == 2 and world.agent == (4, 3)
    return ok, blocked


def run_custom_door_agent(period=6, open_len=2, ticks=30):
    """Check (e): a World with a custom door schedule; the agent just stays
    for `ticks` ticks. Deterministic; no randomness used."""
    world = World(period=period, open_len=open_len)
    for _ in range(ticks):
        world.step(ACTIONS["stay"])
    return world


def verify_custom_door(log):
    """Check the custom-door run: door events are exactly open at ticks
    0, 6, 12, 18, 24 and closed at ticks 2, 8, 14, 20, 26.

    Returns (all_ok, open_ticks, closed_ticks).
    """
    opens = [e[0] for e in log if e[1] == "door" and e[2] == "open"]
    closes = [e[0] for e in log if e[1] == "door" and e[2] == "closed"]
    ok = opens == [0, 6, 12, 18, 24] and closes == [2, 8, 14, 20, 26]
    return ok, opens, closes


def main():
    all_ok = True

    # ---- check (a): existing random agent, 50 ticks, fixed seed ----
    print("##### check (a): random agent, %d ticks, seed %d #####"
          % (TICKS, SEED))
    world_a = run_random_agent()
    print("=== event log: (tick, node, event) ===")
    for entry in world_a.log:
        print(entry)
    print("=== door events vs rule (tick %% %d < %d) ===" % (PERIOD, OPEN_LEN))
    ok, n = verify_door_rule(world_a.log, PERIOD, OPEN_LEN)
    print("door events checked: %d" % n)
    print("DOOR RULE CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok
    ok, n = verify_no_wall_landings(world_a.log)
    print("agent 'move to' events checked: %d" % n)
    print("NO-WALL-LANDING CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check (b): scripted agent through the door to the goal ----
    print("##### check (b): scripted agent (approach, push, escape) #####")
    scripted = run_scripted_agent()
    print("=== event log: (tick, node, event) ===")
    for entry in scripted.log:
        print(entry)
    ok, blocked_ticks, enter_tick, goal_ticks = \
        verify_scripted_run(scripted.log)
    print("'blocked at (5, 5)' ticks: %s" % (blocked_ticks,))
    print("first entry into (5, 5) at tick: %s" % (enter_tick,))
    print("goal reached at ticks: %s" % (goal_ticks,))
    print("BLOCKED-BEFORE-ENTRY CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok
    goal_ok = len(goal_ticks) >= 1
    print("GOAL REACHED CHECK: %s" % ("PASS" if goal_ok else "FAIL"))
    all_ok = all_ok and goal_ok

    # ---- check (c): push into the wall at (5, 3) ----
    print("##### check (c): wall push at (5, 3) #####")
    world_c = run_wall_push_agent()
    print("=== event log: (tick, node, event) ===")
    for entry in world_c.log:
        print(entry)
    ok_c, blocked_c = verify_wall_push(world_c.log, world_c)
    print("'blocked at (5, 3)' events: %d" % len(blocked_c))
    print("final agent position: %s" % (world_c.agent,))
    print("WALL PUSH CHECK: %s" % ("PASS" if ok_c else "FAIL"))
    all_ok = all_ok and ok_c

    # ---- check (d): crush rule scenarios ----
    print("##### check (d): crush rule #####")
    approach = [(1, 0), (2, 0), (3, 0), (4, 0),
                (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]

    def approach_walk(world):
        for tx, ty in approach:
            x, y = world.agent
            world.step((tx - x, ty - y))

    # (d1) staying in the door cell through the close is fatal
    print("--- (d1) staying is fatal ---")
    w1 = World()
    approach_walk(w1)              # ticks 0..8, agent ends at (4, 5)
    w1.step(ACTIONS["right"])      # tick 9: blocked (door closed)
    w1.step(ACTIONS["right"])      # tick 10: enters (5, 5)
    for _ in range(4):
        w1.step(ACTIONS["stay"])   # ticks 11..14
    w1.step(ACTIONS["right"])      # tick 15
    print("=== event log: (tick, node, event) ===")
    for entry in w1.log:
        print(entry)
    log1 = w1.log
    try:
        idx_close = log1.index((14, "door", "closed"))
        crushed_follows = log1[idx_close + 1] == (14, "agent", "crushed")
    except (ValueError, IndexError):
        crushed_follows = False
    no_move_14 = not any(e[0] == 14 and e[2].startswith("move to")
                         for e in log1)
    move_15 = (15, "agent", "move to (1, 0)") in log1
    ok_d1 = crushed_follows and no_move_14 and move_15
    print("(14, 'door', 'closed') immediately followed by "
          "(14, 'agent', 'crushed'): %s" % crushed_follows)
    print("no 'move to' event at tick 14: %s" % no_move_14)
    print("tick 15 'move to (1, 0)' present: %s" % move_15)
    print("D1 STAYING-IS-FATAL CHECK: %s" % ("PASS" if ok_d1 else "FAIL"))
    all_ok = all_ok and ok_d1

    # (d2) late but safe entry
    print("--- (d2) late but safe entry ---")
    w2 = World()
    approach_walk(w2)              # ticks 0..8, agent ends at (4, 5)
    for _ in range(3):
        w2.step(ACTIONS["stay"])   # ticks 9..11
    w2.step(ACTIONS["right"])      # tick 12: enters (5, 5)
    w2.step(ACTIONS["right"])      # tick 13: leaves to (6, 5)
    print("=== event log: (tick, node, event) ===")
    for entry in w2.log:
        print(entry)
    log2 = w2.log
    no_crush_2 = not any(e[2] == "crushed" for e in log2)
    exit_13 = (13, "agent", "move to (6, 5)") in log2
    ok_d2 = no_crush_2 and exit_13
    print("no 'crushed' event in log: %s" % no_crush_2)
    print("tick 13 'move to (6, 5)' present: %s" % exit_13)
    print("D2 LATE-SAFE-ENTRY CHECK: %s" % ("PASS" if ok_d2 else "FAIL"))
    all_ok = all_ok and ok_d2

    # (d3) entering on the last open tick is fatal
    print("--- (d3) entering on the last open tick is fatal ---")
    w3 = World()
    approach_walk(w3)              # ticks 0..8, agent ends at (4, 5)
    for _ in range(4):
        w3.step(ACTIONS["stay"])   # ticks 9..12
    w3.step(ACTIONS["right"])      # tick 13: enters (5, 5), last open tick
    w3.step(ACTIONS["right"])      # tick 14: door closed -> crushed
    print("=== event log: (tick, node, event) ===")
    for entry in w3.log:
        print(entry)
    log3 = w3.log
    crushed_14 = (14, "agent", "crushed") in log3
    no_exit = not any(e[2] == "move to (6, 5)" for e in log3)
    back_at_start = w3.agent == AGENT_START
    ok_d3 = crushed_14 and no_exit and back_at_start
    print("(14, 'agent', 'crushed') present: %s" % crushed_14)
    print("no 'move to (6, 5)' in log: %s" % no_exit)
    print("agent back at (0, 0): %s" % back_at_start)
    print("D3 LAST-OPEN-TICK CHECK: %s" % ("PASS" if ok_d3 else "FAIL"))
    all_ok = all_ok and ok_d3
    ok_d = ok_d1 and ok_d2 and ok_d3
    print("DOOR CRUSH CHECK: %s" % ("PASS" if ok_d else "FAIL"))
    all_ok = all_ok and ok_d

    # ---- check (e): custom door schedule ----
    print("##### check (e): custom door (period=6, open_len=2), 30 ticks #####")
    world_e = run_custom_door_agent()
    print("=== event log: (tick, node, event) ===")
    for entry in world_e.log:
        print(entry)
    ok_e, opens_e, closes_e = verify_custom_door(world_e.log)
    print("door open ticks: %s" % (opens_e,))
    print("door closed ticks: %s" % (closes_e,))
    print("CUSTOM DOOR SCHEDULE CHECK: %s" % ("PASS" if ok_e else "FAIL"))
    all_ok = all_ok and ok_e
    ok_rule, n_rule = verify_door_rule(world_e.log,
                                       world_e.period, world_e.open_len)
    print("verify_door_rule(period=%d, open_len=%d): %s (%d events checked)"
          % (world_e.period, world_e.open_len,
             "PASS" if ok_rule else "FAIL", n_rule))
    all_ok = all_ok and ok_rule

    # ---- check (f): reset() ----
    print("##### check (f): reset() #####")
    world_f = World()
    run_scripted_agent(25, world=world_f)
    saved_f = list(world_f.log)
    print("saved log: %d events; first 3: %s; last 3: %s"
          % (len(saved_f), saved_f[:3], saved_f[-3:]))
    world_f.reset()
    f_tick = world_f.tick == 0
    f_agent = world_f.agent == AGENT_START
    f_log = world_f.log == []
    f_sched = (world_f.period == PERIOD and world_f.open_len == OPEN_LEN)
    print("after reset(): tick=%d (0: %s), agent=%s (%s: %s), log empty: %s, "
          "period=%d (%d: %s), open_len=%d (%d: %s)"
          % (world_f.tick, f_tick, world_f.agent, AGENT_START, f_agent,
             f_log, world_f.period, PERIOD, f_sched,
             world_f.open_len, OPEN_LEN, f_sched))
    run_scripted_agent(25, world=world_f)
    f_rerun = world_f.log == saved_f
    print("re-run on the same world: new log equals saved log: %s (%d events)"
          % (f_rerun, len(world_f.log)))
    # custom schedule survives reset
    world_f2 = World(period=6, open_len=2)
    rng_f2 = random.Random(7)
    names_f2 = list(ACTIONS.keys())
    for _ in range(10):
        world_f2.step(ACTIONS[rng_f2.choice(names_f2)])
    world_f2.reset()
    f_custom = (world_f2.period == 6 and world_f2.open_len == 2
                and world_f2.tick == 0 and world_f2.log == []
                and world_f2.agent == AGENT_START)
    print("custom world after reset(): period=%d, open_len=%d, tick=%d, "
          "log empty=%s, agent=%s"
          % (world_f2.period, world_f2.open_len, world_f2.tick,
             world_f2.log == [], world_f2.agent))
    ok_f = f_tick and f_agent and f_log and f_sched and f_rerun and f_custom
    print("RESET CHECK: %s" % ("PASS" if ok_f else "FAIL"))
    all_ok = all_ok and ok_f

    # ---- check (g): local random generator ----
    print("##### check (g): local random generator #####")
    log1 = run_random_agent(42).log
    random.seed(7)          # pollute the GLOBAL generator ...
    for _ in range(5):
        random.random()     # ... and draw from it five times
    log2 = run_random_agent(42).log
    log3 = run_random_agent(43).log

    def log_hash(log):
        return hashlib.sha256(repr(log).encode("utf-8")).hexdigest()

    h1, h2, h3 = log_hash(log1), log_hash(log2), log_hash(log3)
    print("log1 (seed 42): %d events, sha256 %s" % (len(log1), h1))
    print("  first 2: %s; last 2: %s" % (log1[:2], log1[-2:]))
    print("log2 (seed 42, global generator polluted in between): %d events, "
          "sha256 %s" % (len(log2), h2))
    print("log3 (seed 43): %d events, sha256 %s" % (len(log3), h3))
    print("  first 2: %s; last 2: %s" % (log3[:2], log3[-2:]))
    g_isolated = log1 == log2
    print("GENERATOR-ISOLATION CHECK: %s (log1 == log2 despite pollution)"
          % ("PASS" if g_isolated else "FAIL"))
    g_diff = log3 != log1
    print("SEED-DIFFERENCE CHECK: %s (log3 != log1)"
          % ("PASS" if g_diff else "FAIL"))
    ok_g = g_isolated and g_diff
    print("RANDOM GENERATOR CHECK: %s" % ("PASS" if ok_g else "FAIL"))
    all_ok = all_ok and ok_g

    # ---- check (h): complete door-transition verification ----
    print("##### check (h): complete door transitions #####")
    ok_h1, exp_h1, act_h1 = verify_door_complete(world_a.log, PERIOD,
                                                 OPEN_LEN, TICKS)
    print("check-(a) log (%d ticks): ok=%s (%d expected vs %d actual events)"
          % (TICKS, ok_h1, len(exp_h1), len(act_h1)))
    ok_h2, _exp_h2, _act_h2 = verify_door_complete(world_e.log, 6, 2, 30)
    print("check-(e) log (30 ticks, period=6, open_len=2): ok=%s" % ok_h2)
    world_h3 = World()
    for _ in range(3):
        world_h3.step(ACTIONS["stay"])
    ok_h3, exp_h3, act_h3 = verify_door_complete(world_h3.log, PERIOD,
                                                 OPEN_LEN, 3)
    print("3-tick run: ok=%s, expected=%s, actual=%s"
          % (ok_h3, exp_h3, act_h3))
    h3_exact = exp_h3 == [(0, "door", "open")]
    # tamper 1: delete one door event from a copy of the (a) log
    tampered_del = list(world_a.log)
    del tampered_del[next(i for i, e in enumerate(tampered_del)
                          if e[1] == "door")]
    ok_h4, _exp_h4, _act_h4 = verify_door_complete(tampered_del, PERIOD,
                                                   OPEN_LEN, TICKS)
    print("deleted one door event: ok=%s (must be False)" % ok_h4)
    # tamper 2: add a wrong door event (tick 25 is closed, no transition)
    tampered_add = list(world_a.log)
    tampered_add.append((25, "door", "open"))
    ok_h5, _exp_h5, _act_h5 = verify_door_complete(tampered_add, PERIOD,
                                                   OPEN_LEN, TICKS)
    print("added wrong door event (25, 'door', 'open'): ok=%s (must be False)"
          % ok_h5)
    ok_h = (ok_h1 and ok_h2 and ok_h3 and h3_exact
            and (not ok_h4) and (not ok_h5))
    print("COMPLETE-TRANSITIONS CHECK: %s" % ("PASS" if ok_h else "FAIL"))
    all_ok = all_ok and ok_h

    # ---- check (i): observe() ----
    print("##### check (i): observe() #####")
    world_i = World()
    obs0 = world_i.observe()
    print("fresh world observe(): %s" % (obs0,))
    tick_before_i = world_i.tick
    log_before_i = list(world_i.log)
    o1 = world_i.observe()
    o2 = world_i.observe()
    o3 = world_i.observe()
    i_unchanged = (world_i.tick == tick_before_i
                   and world_i.log == log_before_i
                   and o1 == o2 == o3 == obs0)
    print("observe() x3: tick and log unchanged: %s" % i_unchanged)
    # check-(b) approach walk, 9 steps (ticks 0..8); agent ends at (4, 5)
    for tx, ty in [(1, 0), (2, 0), (3, 0), (4, 0),
                   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]:
        x, y = world_i.agent
        world_i.step((tx - x, ty - y))
    obs9 = world_i.observe()
    print("after 9 steps (tick=%d, agent=%s): observe() = %s"
          % (world_i.tick, world_i.agent, obs9))
    world_i.step(ACTIONS["right"])   # tick 9: door closed -> blocked
    blocked9 = world_i.log[-1] == (9, "agent", "blocked at (5, 5)")
    print("step right at tick 9: blocked=%s, tick now %d, agent %s"
          % (blocked9, world_i.tick, world_i.agent))
    obs10 = world_i.observe()
    print("at tick 10: observe() = %s" % (obs10,))
    world_i.step(ACTIONS["right"])   # tick 10: door open -> enters
    at_door = world_i.agent == DOOR_CELL
    print("step right at tick 10: agent=%s" % (world_i.agent,))
    ok_i = (obs0 == (AGENT_START, True) and i_unchanged
            and obs9 == ((4, 5), False) and blocked9
            and world_i.tick == 11 and obs10 == ((4, 5), True) and at_door)
    print("OBSERVE CHECK: %s" % ("PASS" if ok_i else "FAIL"))
    all_ok = all_ok and ok_i

    # ---- check (j): set_door() mid-run schedule change ----
    print("##### check (j): set_door() #####")
    world_j = World()
    for _ in range(12):
        world_j.step(ACTIONS["stay"])   # ticks 0..11
    tick_before_j = world_j.tick
    agent_before_j = world_j.agent
    log_before_j = list(world_j.log)
    world_j.set_door(6, 2)
    ok_j_untouched = (world_j.tick == tick_before_j
                      and world_j.agent == agent_before_j
                      and world_j.log == log_before_j
                      and world_j.period == 6 and world_j.open_len == 2)
    print("set_door left tick/agent/log untouched: %s" % ok_j_untouched)
    for _ in range(18):
        world_j.step(ACTIONS["stay"])   # ticks 12..29
    expected_j = [(0, "door", "open"), (4, "door", "closed"),
                  (10, "door", "open"), (14, "door", "closed"),
                  (18, "door", "open"), (20, "door", "closed"),
                  (24, "door", "open"), (26, "door", "closed")]
    actual_j = [e for e in world_j.log if e[1] == "door"]
    print("door events in log: %s" % (actual_j,))
    ok_j_sched = actual_j == expected_j
    pre_j = [(0, "door", "open"), (4, "door", "closed"), (10, "door", "open")]
    ok_j_pre = all(e in world_j.log for e in pre_j)
    print("schedule matches expected exactly: %s; pre-change events still "
          "in log: %s" % (ok_j_sched, ok_j_pre))
    # ValueError cases on fresh worlds
    j_raises = True
    for p, o in [(0, 1), (5, 6), (5, -1)]:
        w = World()
        try:
            w.set_door(p, o)
            raised = False
        except ValueError:
            raised = True
        print("set_door(%d, %d) raised ValueError: %s" % (p, o, raised))
        j_raises = j_raises and raised
    # boundary-valid cases must NOT raise
    j_valid = True
    for p, o in [(6, 0), (6, 6)]:
        w = World()
        try:
            w.set_door(p, o)
            raised = False
        except ValueError:
            raised = True
        print("set_door(%d, %d) raised ValueError: %s (must be False)"
              % (p, o, raised))
        j_valid = j_valid and not raised
    ok_j = ok_j_sched and ok_j_pre and ok_j_untouched and j_raises and j_valid
    print("SET-DOOR CHECK: %s" % ("PASS" if ok_j else "FAIL"))
    all_ok = all_ok and ok_j

    # ---- check (k): start phase ----
    print("##### check (k): start phase #####")
    world_k = World()
    world_k.reset(start_tick=7)
    k_tick = world_k.tick == 7
    k_agent = world_k.agent == AGENT_START
    k_log = world_k.log == []
    k_obs = world_k.observe() == (AGENT_START, False)
    print("after reset(start_tick=7): tick=%d (7: %s), agent=%s ((0, 0): %s), "
          "log empty: %s, observe()=%s (((0, 0), False): %s)"
          % (world_k.tick, k_tick, world_k.agent, k_agent, k_log,
             world_k.observe(), k_obs))
    for _ in range(6):
        world_k.step(ACTIONS["stay"])   # ticks 7..12
    print("=== event log: (tick, node, event) ===")
    for entry in world_k.log:
        print(entry)
    door_k = [e for e in world_k.log if e[1] == "door"]
    k_sched = door_k == [(7, "door", "closed"), (10, "door", "open")]
    print("door events: %s (expected [(7, 'door', 'closed'), "
          "(10, 'door', 'open')]): %s" % (door_k, k_sched))
    ok_k1, _ek, _ak = verify_door_complete(world_k.log, 10, 4, 6,
                                           start_tick=7)
    print("verify_door_complete(..., start_tick=7): ok=%s" % ok_k1)
    tampered_k = list(world_k.log)
    del tampered_k[next(i for i, e in enumerate(tampered_k)
                        if e[1] == "door")]
    ok_k2, _ek2, _ak2 = verify_door_complete(tampered_k, 10, 4, 6,
                                             start_tick=7)
    print("first door event deleted: ok=%s (must be False)" % ok_k2)
    world_k2 = World()
    world_k2.reset()                    # no args: tick 0, as before
    for _ in range(3):
        world_k2.step(ACTIONS["stay"])
    world_k3 = World()
    for _ in range(3):
        world_k3.step(ACTIONS["stay"])
    k_rerun = world_k2.log == world_k3.log
    print("reset() then 3 stays == fresh World() 3 stays: %s" % k_rerun)
    ok_k = (k_tick and k_agent and k_log and k_obs and k_sched and ok_k1
            and (not ok_k2) and k_rerun)
    print("START PHASE CHECK: %s" % ("PASS" if ok_k else "FAIL"))
    all_ok = all_ok and ok_k

    print("=== OVERALL: %s ===" % ("PASS" if all_ok else "FAIL"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

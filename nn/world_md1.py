#!/usr/bin/env python3
"""world_md1.py -- MD1 moving-door worlds (WO-TBM-MD1, Step 0).

Additive only: world.py, world_doors.py,
world_doors_sched.py, world_luck.py, world_pda.py,
world_pdb2.py, and world_ft1.py are NEVER edited.
Mechanics are inherited from MultiDoorWorld
(world_doors.py) deliberately: the MD1 axis is the
gap ROW varying between episodes, on top of the
already-verified single-schedule multi-door
mechanics. No new step/crush/blocking code is
introduced here; this module only wires factories
(gap sets, schedules) and the row-draw protocols.

Worlds (all single-gap MultiDoorWorld instances,
fixed schedule period=20, open_len=10):
- FIXED world for row r: gaps = {(5, r)}.
  observe() returns the gap's door bit, because
  a single-gap MultiDoorWorld's door bit IS that
  gap's door state (shared schedule). period /
  open_len exposed are 20 / 10 for every world.
- MIXED world M3 is NOT a separate world class:
  an M3 eval job is 20 fixed-world episodes, one
  per start phase 0..19 in order, each with its
  gap row drawn by the M3 eval RNG (§M3 below).
  The runner draws the row per episode explicitly
  and records it; no RNG hides inside reset().

Row sets (WO §2.3, frozen at issuance):
  TRAINED_ROWS = (0, 1, 3, 4, 5, 6, 7, 9)
  HELDOUT_ROWS = (2, 8)
  ALL_ROWS     = 0..9

Training row protocol (design note §3):
  row_multiset(seed) = 250 copies of each trained
  row, shuffled once with random.Random(seed +
  5000). Episode e of that agent's training uses
  multiset[e]. Balance is by construction (exactly
  250/row), not by luck. The row stream is a
  SEPARATE Random instance from the agent stream
  (Random(seed)) and the start-tick stream
  (Random(seed + 1000)), with a distinct seed
  offset, so row draws never perturb learning
  dynamics or start phases relative to the
  P-Da/P-Db2 conventions.

M3 eval protocol (design note §4):
  M3 draws from TRAINED ROWS ONLY (it follows the
  training protocol; held-out rows never appear in
  M3). Per (arm, seed): eval_rng =
  random.Random(700_000 + 1000 * ARM_INDEX[arm]
  + seed); the 20 rows are [eval_rng.choice
  (TRAINED_ROWS) for the 20 episodes in
  increasing start-phase order 0..19]. Uniform
  with replacement; row coverage per agent is
  partial by construction, so M3 is a sanity
  check, never a per-row measurement.
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world_doors import MultiDoorWorld

PERIOD = 20
OPEN_LEN = 10
TRAINED_ROWS = (0, 1, 3, 4, 5, 6, 7, 9)
HELDOUT_ROWS = (2, 8)
ALL_ROWS = tuple(range(10))
EPISODES = 2000
PER_ROW = 250

# Seed offsets for the three independent training streams.
# Agent stream: Random(seed). Start-tick stream:
# Random(seed + 1000) (P-Da/P-Db2 convention).
# Row stream: Random(seed + ROW_SEED_OFFSET).
ROW_SEED_OFFSET = 5000

ARM_INDEX = {"wrapped": 0, "plain": 1, "memory": 2,
             "oracle_ind": 3, "oracle_new": 4,
             "phase_only_new": 5}
M3_SEED_BASE = 700000
M3_SEED_ARM_STEP = 1000


def gap_cell(row):
    """Door cell for a gap row: column x=5, row y=row."""
    return (5, row)


def fixed_world_name(row):
    """Record name for a fixed-gap world: R{row} (e.g. R5, R8)."""
    return "R%d" % (row,)


def make_fixed_world(row):
    """Single-gap MultiDoorWorld for gap row `row`, schedule 20/10.

    Raises ValueError for rows outside 0..9. The returned
    world carries .gap_row and .world_kind for records.
    """
    if row not in ALL_ROWS:
        raise ValueError("gap row must be 0..9, got %r" % (row,))
    w = MultiDoorWorld(PERIOD, OPEN_LEN, [gap_cell(row)])
    w.gap_row = row
    w.world_kind = "fixed"
    return w


def row_multiset(seed):
    """Balanced training row sequence for one agent's 2000 episodes.

    250 copies of each trained row, shuffled once with
    random.Random(seed + 5000). Episode e uses element e.
    Deterministic given seed; contains no held-out row.
    """
    rows = []
    for r in TRAINED_ROWS:
        rows.extend([r] * PER_ROW)
    rng = random.Random(seed + ROW_SEED_OFFSET)
    rng.shuffle(rows)
    assert len(rows) == EPISODES
    return rows


def m3_seed(arm, seed):
    """Seeded eval-RNG seed for one agent's M3 job."""
    return M3_SEED_BASE + M3_SEED_ARM_STEP * ARM_INDEX[arm] + seed


def m3_row_sequence(arm, seed, n=20):
    """M3 gap rows: 20 uniform draws from TRAINED_ROWS ONLY.

    Drawn in increasing start-phase order (episode i uses
    element i, paired with start phase i). Deterministic
    given (arm, seed); partial coverage by construction.
    """
    rng = random.Random(m3_seed(arm, seed))
    return [rng.choice(TRAINED_ROWS) for _ in range(n)]

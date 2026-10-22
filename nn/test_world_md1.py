#!/usr/bin/env python3
"""test_world_md1.py -- MD1 Step 0 scripted mechanics checks.

Recomputed from code, not from any training run.
Every check prints its case count; any failure
raises SystemExit(1). Checks:
  (1) row_multiset: 2000 entries, exactly 250 per
      trained row, no held-out row, deterministic
      across calls, distinct across seeds.
  (2) make_fixed_world(row) for rows 0..9: gaps ==
      {(5, row)}; wall cells are exactly x=5 except
      the gap; period 20, open_len 10.
  (3) Fixed-gap door timeline: observe bit over
      40 ticks equals tick % 20 < 10, for every row.
  (4) Observe bit follows the gap: a staying agent's
      observe bit equals door_is_open(tick, 20, 10)
      at every tick, for every row.
  (5) Blocking/crush scripted: approach (4, row),
      push right while closed -> blocked; enter
      while open; stay in the gap through the close
      -> crushed. Rows 0, 5, 9 (edges + familiar).
  (6) Training row wiring: a 10-episode
      train_neural_md1 run uses row_multiset(seed)
      [:10] as its episode rows, in order.
  (7) M3: 20 rows per (arm, seed), all in
      TRAINED_ROWS, deterministic, distinct across
      (arm, seed) pairs; seed formula spot values.
  (8) Verdict logic: §8 precedence on the
      overlap/gap examples (trained 25% / held-out
      20% -> FOLLOWING; trained 25% / held-out 5%
      -> NEITHER) plus the named bars.
  (9) Eval runner smoke shape: one fixed-world
      eval job on an untrained plain agent returns
      20 records with correct world/gap_row/phase
      fields (outcome values are NOT asserted --
      this is a shape check, not a result).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collections import Counter

from world import ACTIONS, door_is_open
from world_doors import is_wall_cell
from world_md1 import (ALL_ROWS, HELDOUT_ROWS, TRAINED_ROWS,
                       fixed_world_name, gap_cell, m3_row_sequence,
                       m3_seed, make_fixed_world, row_multiset)
from analysis_md1 import verdict


def check(name, cond, cases):
    print("  %s: %s (%d cases)" % (name, "PASS" if cond else "FAIL",
                                   cases))
    if not cond:
        raise SystemExit(1)


def main():
    print("== MD1 mechanics checks ==")
    # (1) row multiset
    cases = 0
    for seed in (0, 1, 2, 3, 4, 7, 42):
        rows = row_multiset(seed)
        cases += 1
        assert len(rows) == 2000, (seed, len(rows))
        assert Counter(rows) == {r: 250 for r in TRAINED_ROWS}, seed
        assert not (set(rows) & set(HELDOUT_ROWS)), seed
        assert row_multiset(seed) == rows, "determinism"
    check("row_multiset balanced 250/row, no held-out, deterministic",
          True, cases)
    check("row_multiset differs across seeds",
          row_multiset(0) != row_multiset(1), 1)

    # (2) fixed world wiring, all rows
    for row in ALL_ROWS:
        w = make_fixed_world(row)
        assert w.gaps == {gap_cell(row)}, row
        assert w.period == 20 and w.open_len == 10, row
        for y in ALL_ROWS:
            assert is_wall_cell((5, y), w.gaps) == (y != row), (row, y)
        assert fixed_world_name(row) == "R%d" % row
    check("make_fixed_world gaps/walls/schedule, rows 0..9", True, 10)
    try:
        make_fixed_world(10)
        check("make_fixed_world rejects row 10", False, 1)
    except ValueError:
        check("make_fixed_world rejects row 10", True, 1)

    # (3) + (4) door timeline and observe-follows-gap
    cases = 0
    for row in ALL_ROWS:
        w = make_fixed_world(row)
        w.reset(start_tick=0)
        for t in range(40):
            assert w.observe() == (w.agent,
                                   door_is_open(t, 20, 10)), (row, t)
            assert w.observe()[1] == ((t % 20) < 10), (row, t)
            w.step(ACTIONS["stay"])
            cases += 1
    check("door timeline 20/10 + observe follows gap", True, cases)

    # (5) blocking / entry / crush scripted, rows 0, 5, 9
    for row in (0, 5, 9):
        w = make_fixed_world(row)
        w.reset(start_tick=0)
        # walk to (4, row) along row 0 then down column 4
        for tx in (1, 2, 3, 4):
            x, y = w.agent
            w.step((tx - x, 0))
        for ty in range(1, row + 1):
            x, y = w.agent
            w.step((0, ty - y))
        assert w.agent == (4, row), (row, w.agent)
        # at some tick, push right while closed -> blocked;
        # then wait for an open tick and enter.
        entered = False
        for _ in range(40):
            tick = w.tick
            open_now = (tick % 20) < 10
            before = w.agent
            w.step(ACTIONS["right"])
            if open_now:
                assert w.agent == gap_cell(row), (row, tick, w.agent)
                entered = True
                break
            else:
                assert w.agent == before, (row, tick)
        assert entered, row
        # stay in the gap through the close -> crushed
        crushed = False
        for _ in range(40):
            w.step(ACTIONS["stay"])
            if any(e[2] == "crushed" for e in w.log):
                crushed = True
                break
        assert crushed and w.agent == (0, 0), row
    check("blocking / entry / crush scripted, rows 0/5/9", True, 3)

    # (6) training row wiring (10-episode run)
    from train_md1 import train_neural_md1
    from agent_nn import PlainEncoder
    _a, stats, _rc, _dec, _ewe, _rr = train_neural_md1(
        0, episodes=10, encoder=PlainEncoder())
    assert [s[4] for s in stats] == row_multiset(0)[:10]
    check("training episodes use row_multiset(seed) in order", True, 10)

    # (7) M3 rows
    assert m3_seed("wrapped", 0) == 700000
    assert m3_seed("plain", 0) == 701000
    assert m3_seed("wrapped", 3) == 700003
    seqs = {}
    for arm in ("wrapped", "plain", "memory", "oracle_ind",
                "oracle_new", "phase_only_new"):
        for seed in range(5):
            s = m3_row_sequence(arm, seed)
            assert len(s) == 20 and set(s) <= set(TRAINED_ROWS)
            assert m3_row_sequence(arm, seed) == s, "determinism"
            seqs[(arm, seed)] = s
    check("M3 rows: 20, trained-only, deterministic, seed formula",
          True, 30)
    check("M3 sequences differ across (arm, seed)",
          len(set(tuple(v) for v in seqs.values())) > 25, 30)

    # (8) verdict logic
    assert verdict(1200, 4800, 240, 1200) == "FOLLOWING"  # 25% / 20%
    assert verdict(1200, 4800, 60, 1200) == "NEITHER"     # 25% / 5%
    assert verdict(2400, 4800, 60, 1200) == "BIGGER ADDRESS BOOK"
    assert verdict(800, 4800, 0, 1200) == "UNINTERPRETABLE"
    assert verdict(0, 0, 0, 0, False, False) == "INCOMPLETE"
    assert verdict(4800, 4800, 1200, 1200) == "FOLLOWING"
    check("verdict precedence + bars (incl. overlap/gap examples)",
          True, 6)

    # (9) eval runner shape on an untrained plain agent
    from agent_nn import NeuralQ
    from md1_run import evaluate, fresh_encoder
    agent = NeuralQ(0, n_inputs=101)
    eps = evaluate(agent, fresh_encoder("plain"), "R5", 5)
    assert len(eps) == 20
    assert [e["start_phase"] for e in eps] == list(range(20))
    assert all(e["world"] == "R5" and e["gap_row"] == 5 for e in eps)
    assert all(e["conflict"] is None for e in eps)
    check("eval runner shape: 20 records, world/gap/phase fields",
          True, 20)

    print("MD1 mechanics checks: ALL PASS")


if __name__ == "__main__":
    main()

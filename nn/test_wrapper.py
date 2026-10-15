#!/usr/bin/env python3
"""B1 verification for wrapper.py. Prints PASS/FAIL per check."""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import World, ACTIONS, door_is_open
from wrapper import ScheduleModel

STAY = ACTIONS["stay"]


def feed(model, world, n_ticks):
    """Feed per-tick (tick, door) observations; return contradiction ticks."""
    bad = []
    for _ in range(n_ticks):
        tick = world.tick
        door = world.observe()[1]
        if model.observe(tick, door):
            bad.append(tick)
        world.step(STAY)
    return bad


def check_schedules():
    """Exact estimates on several schedules from random start phases,
    and how many ticks of observation that takes."""
    print("##### check 1: exact estimates from random start phases #####")
    rng = random.Random(20261014)
    ok = True
    for period, open_len in [(20, 10), (10, 4), (12, 8), (6, 2)]:
        worst = 0
        total = 0
        n = 8
        for _ in range(n):
            start = rng.randrange(2 * period)
            m = ScheduleModel()
            w = World(period, open_len)
            w.reset(start_tick=start)
            m.new_episode()
            exact_at = None
            for _ in range(10 * period):
                tick = w.tick
                door = w.observe()[1]
                if m.observe(tick, door):
                    ok = False
                    print("  UNEXPECTED contradiction at tick %d on a "
                          "stable %d/%d schedule" % (tick, period, open_len))
                w.step(STAY)
                if (m.period == period and m.open_len == open_len
                        and m.last_opening_tick is not None):
                    o = m.last_opening_tick
                    real = (not door_is_open(o - 1, period, open_len)
                            and door_is_open(o, period, open_len))
                    if not real or o % period != 0:
                        ok = False
                        print("  last_opening %d is not a real opening" % o)
                    exact_at = w.tick - start
                    break
            if exact_at is None:
                ok = False
                print("  %d/%d start %d: never reached exact estimates"
                      % (period, open_len, start))
            else:
                worst = max(worst, exact_at)
                total += exact_at
        print("  %2d/%-2d: exact after at most %3d ticks (mean %5.1f) "
              "over %d random starts" % (period, open_len, worst,
                                         total / n, n))
    print("SCHEDULE ESTIMATION: %s" % ("PASS" if ok else "FAIL"))
    return ok


def check_set_door():
    """Mid-run set_door change: detection latency, then exact
    re-estimation, then stability."""
    print("##### check 2: mid-run set_door change #####")
    m = ScheduleModel()
    w = World(20, 10)
    w.reset(start_tick=0)
    m.new_episode()
    bad = feed(m, w, 60)
    ok = (not bad and m.period == 20 and m.open_len == 10)
    print("  converged on 20/10 with no contradictions: %s (period=%s, "
          "open_len=%s)" % (ok, m.period, m.open_len))
    T = w.tick
    w.set_door(10, 4)
    first_bad = None
    reconv_at = None
    for _ in range(200):
        tick = w.tick
        door = w.observe()[1]
        if m.observe(tick, door) and first_bad is None:
            first_bad = tick
        w.step(STAY)
        if m.period == 10 and m.open_len == 4:
            reconv_at = w.tick
            break
    lat = None if first_bad is None else first_bad - T
    ok2 = first_bad is not None and 0 <= lat <= 20
    ok3 = reconv_at is not None
    n_before = m.contradictions
    feed(m, w, 20)
    ok4 = (m.contradictions == n_before and m.period == 10
           and m.open_len == 4)
    print("  change at tick %d: first contradiction at %s (latency %s "
          "ticks, bound 20): %s" % (T, first_bad, lat, ok2))
    print("  re-converged to 10/4 at tick %s (%s ticks after change): %s"
          % (reconv_at, None if reconv_at is None else reconv_at - T, ok3))
    print("  stable afterwards (contradictions=%d, still 10/4): %s"
          % (m.contradictions, ok4))
    print("  last_contradiction_tick == %s: %s"
          % (first_bad, m.last_contradiction_tick == first_bad))
    ok = ok and ok2 and ok3 and ok4 \
        and m.last_contradiction_tick == first_bad
    print("SET-DOOR CHANGE: %s" % ("PASS" if ok else "FAIL"))
    return ok


def check_features():
    """Time features against ground truth."""
    print("##### check 3: features against ground truth #####")
    ok = True
    for period, open_len in [(20, 10), (12, 8)]:
        m = ScheduleModel()
        w = World(period, open_len)
        w.reset(start_tick=0)
        m.new_episode()
        feed(m, w, 3 * period)
        if not (m.period == period and m.open_len == open_len):
            ok = False
            print("  %d/%d: model did not converge" % (period, open_len))
            continue
        for t in range(2 * period):
            true_since = t % period
            true_rem = (open_len - true_since
                        if true_since < open_len else 0)
            if m.ticks_since_opening(t) != true_since:
                ok = False
                print("  %d/%d t=%d: since=%s, truth=%d"
                      % (period, open_len, t, m.ticks_since_opening(t),
                         true_since))
            if m.open_ticks_remaining(t) != true_rem:
                ok = False
                print("  %d/%d t=%d: remaining=%s, truth=%d"
                      % (period, open_len, t, m.open_ticks_remaining(t),
                         true_rem))
            if m.predict_open(t) != (true_since < open_len):
                ok = False
                print("  %d/%d t=%d: predict_open wrong" % (period,
                                                           open_len, t))
        lt = w.tick - 1
        if m.ticks_since_opening() != lt % period:
            ok = False
            print("  default-tick ticks_since_opening wrong")
        if m.open_ticks_remaining() != (
                open_len - lt % period if lt % period < open_len else 0):
            ok = False
            print("  default-tick open_ticks_remaining wrong")
    print("FEATURES VS TRUTH: %s" % ("PASS" if ok else "FAIL"))
    return ok


def check_cross_episode():
    """A model built over earlier episodes predicts a new episode's
    phase from tick 0; the first observation is not an opening."""
    print("##### check 4: cross-episode phase prediction from tick 0 #####")
    m = ScheduleModel()
    w = World(20, 10)
    for start in (5, 40, 77):
        w.reset(start_tick=start)
        m.new_episode()
        feed(m, w, 30)
    ok_est = (m.period == 20 and m.open_len == 10
              and m.last_opening_tick == 100)
    print("  after 3 episodes: period=%s, open_len=%s, last_opening=%s "
          "(expect 20, 10, 100): %s"
          % (m.period, m.open_len, m.last_opening_tick, ok_est))
    S = 123
    w.reset(start_tick=S)
    m.new_episode()
    tick = w.tick
    door = w.observe()[1]
    contradicted = m.observe(tick, door)
    since = m.ticks_since_opening()
    rem = m.open_ticks_remaining()
    ok = (ok_est and not contradicted and m.last_opening_tick == 100
          and since == 3 and rem == 7 and m.predict_open(123) is True
          and m.period == 20 and m.open_len == 10)
    print("  new episode at tick 123: contradiction=%s, last_opening=%s "
          "(still 100, not 123), since=%s rem=%s (truth 3, 7): %s"
          % (contradicted, m.last_opening_tick, since, rem, ok))
    print("CROSS-EPISODE PREDICTION: %s" % ("PASS" if ok else "FAIL"))
    return ok


def check_mid_open_start():
    """Starting in the middle of an open window: the first state is not
    treated as a real opening; estimates still become exact."""
    print("##### check 5: start in the middle of an open window #####")
    m = ScheduleModel()
    w = World(20, 10)
    w.reset(start_tick=3)  # door open, no transition observed yet
    m.new_episode()
    tick = w.tick
    door = w.observe()[1]
    contradicted = m.observe(tick, door)
    ok1 = (not contradicted and m.last_opening_tick is None
           and m.ticks_since_opening() is None
           and m.open_ticks_remaining() is None
           and m.predict_open(3) is None
           and m.period is None and m.open_len is None)
    print("  first observation (tick 3, open): not an opening, features "
          "unknown: %s" % ok1)
    w.step(STAY)
    feed(m, w, 41)  # observe ticks 4..44
    o = m.last_opening_tick
    real = (o is not None and not door_is_open(o - 1, 20, 10)
            and door_is_open(o, 20, 10))
    ok2 = (m.period == 20 and m.open_len == 10 and o == 40
           and o % 20 == 0 and real and m.contradictions == 0
           and m.ticks_since_opening(44) == 4
           and m.open_ticks_remaining(44) == 6)
    print("  after observing to tick 44: period=%s, open_len=%s, "
          "last_opening=%s (latest real opening, not the first state "
          "at tick 3): %s"
          % (m.period, m.open_len, o, ok2))
    ok = ok1 and ok2
    print("MID-OPEN START: %s" % ("PASS" if ok else "FAIL"))
    return ok


def check_partial_knowledge():
    print("##### check 6: partial knowledge (open_len + opening, "
          "no period) #####")
    m = ScheduleModel()
    w = World(20, 10)
    # Episode 1: learn the open length, but not the period (too short
    # for two openings).
    w.reset(start_tick=0)
    m.new_episode()
    feed(m, w, 35)  # observe ticks 0..34
    # openings=[20]; closings=[10 (unpaired), 30 (paired, clean)]
    partial_state = (m.period is None and m.open_len == 10
                     and m.last_opening_tick == 20)
    # tick 34: door closed (34 % 20 = 14). Partial: remaining 0, phase 14.
    r34 = m.open_ticks_remaining()
    p34 = m.ticks_since_opening()
    ok1 = (partial_state and r34 == 0 and p34 == 14
           and m.first_estimates_tick is None)
    print("  after ep1 (ticks 0..34): period=%s (expect None), "
          "open_len=%s, remaining=%s (expect 0), phase=%s (expect 14): %s"
          % (m.period, m.open_len, r34, p34, ok1))
    # Episode 2: a fresh opening, door open, still no period.
    w.reset(start_tick=95)
    m.new_episode()
    feed(m, w, 1)  # observe tick 95 only
    # Gap flag from new_episode: partial unavailable -- the old opening
    # (tick 20) is stale across unobserved ticks and must not be used.
    r95 = m.open_ticks_remaining()
    p95 = m.ticks_since_opening()
    ok2 = (r95 is None and p95 is None)
    print("  ep2 tick 95 (first obs after boundary): remaining=%s, "
          "phase=%s (both None; stale opening not trusted): %s"
          % (r95, p95, ok2))
    feed(m, w, 6)  # observe ticks 96..101; 99->100 is a fresh opening
    # tick 101: door open (101 % 20 = 1). Partial: remaining 9, phase 1.
    # Period still None (the 20->100 opening gap is dirty).
    r101 = m.open_ticks_remaining()
    p101 = m.ticks_since_opening()
    ok3 = (m.period is None and r101 == 9 and p101 == 1
           and m.last_opening_tick == 100)
    print("  ep2 tick 101 (fresh opening at 100, door open): period=%s "
          "(still None), remaining=%s (expect 9), phase=%s (expect 1): %s"
          % (m.period, r101, p101, ok3))
    ok4 = (r101 == (10 - 101 % 20) and p101 == 101 % 20)
    print("  partial values match ground truth at tick 101 "
          "(phase 1, remaining 9): %s" % ok4)
    ok = ok1 and ok2 and ok3 and ok4
    print("PARTIAL KNOWLEDGE: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    all_ok = True
    all_ok = check_schedules() and all_ok
    all_ok = check_set_door() and all_ok
    all_ok = check_features() and all_ok
    all_ok = check_cross_episode() and all_ok
    all_ok = check_mid_open_start() and all_ok
    all_ok = check_partial_knowledge() and all_ok
    print("=== OVERALL: %s ===" % ("PASS" if all_ok else "FAIL"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

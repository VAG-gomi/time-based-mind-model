#!/usr/bin/env python3
"""agent_generalization.py -- generalization arms (EBI priority 3).

Schedule-independent encoders for the unseen-schedule (30/14) and
mid-run schedule-change (20/10 -> 30/14) experiments.

The key constraint: no arm may use the true schedule's constants as
feature scales, except OracleNew (the explicit perfect-knowledge
upper bound).

WrappedIndEncoder: ScheduleModel wrapper with schedule-independent
encoding:
  - remaining = min(open_ticks_remaining, 5) / 5.0. "2 ticks left"
    (minimum safe crossing margin) is always 0.4, on any schedule.
  - phase = ticks_since_opening / 50.0. Fixed generic divisor, larger
    than any tested period, not tuned to any schedule.
  - known flag as before (1 iff both features available).

Memory20Encoder / Memory30Encoder: last 20 / 30 door bits.
OracleNewEncoder(period, open_len): true phase/remaining with the
  given schedule's constants. Only used as the upper bound.
OracleStaleEncoder: 20/10-based features regardless of the actual
  schedule (wrong fixed knowledge after a switch).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import (N_INPUTS, GRID_SIZE, EPISODE_CAP, train_neural)
from agent_controls import MemoryNEncoder
from wrapper import ScheduleModel

# Schedule-independent scales (documented, not tuned to any schedule).
REMAINING_CAP = 5.0
PHASE_DIVISOR = 50.0


class WrappedIndEncoder(object):
    """Schedule-independent wrapper: 104 inputs.

    Holds one ScheduleModel for its whole lifetime (same persistence
    semantics as WrappedEncoder). Features use fixed generic scales,
    never the true schedule's constants.
    """

    n_inputs = 104

    def __init__(self, model=None):
        self.model = model if model is not None else ScheduleModel()
        self.total_ticks = 0
        self.known_ticks = 0

    def reset(self, world):
        self.model.new_episode()

    def encode(self, world):
        (x, y), door = world.observe()
        tick = world.tick
        self.model.observe(tick, bool(door))
        self.total_ticks += 1
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        remaining = self.model.open_ticks_remaining()
        phase = self.model.ticks_since_opening()
        if remaining is not None and phase is not None:
            v[101] = phase / PHASE_DIVISOR
            v[102] = min(remaining, REMAINING_CAP) / REMAINING_CAP
            v[103] = 1.0
            self.known_ticks += 1
        return v

    def frac_known(self):
        return (self.known_ticks / self.total_ticks
                if self.total_ticks else 0.0)


class Memory20Encoder(MemoryNEncoder):
    """20-step door history: 121 inputs."""

    def __init__(self):
        MemoryNEncoder.__init__(self, 20)


class Memory30Encoder(MemoryNEncoder):
    """30-step door history: 131 inputs."""

    def __init__(self):
        MemoryNEncoder.__init__(self, 30)


class OracleNewEncoder(object):
    """True phase/remaining for a GIVEN schedule: 104 inputs.

    Only used as the perfect-knowledge upper bound. The schedule
    constants are passed explicitly so the "no true constants"
    rule is visibly scoped to this arm.
    """

    n_inputs = 104

    def __init__(self, period, open_len):
        self.period = period
        self.open_len = open_len

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        tick = world.tick
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[101] = (tick % self.period) / float(self.period)
        if door:
            v[102] = (self.open_len - (tick % self.period)) \
                / float(self.open_len)
        else:
            v[102] = 0.0
        v[103] = 1.0
        return v


class OracleStaleEncoder(object):
    """20/10-based features regardless of actual schedule: 104 inputs.

    Wrong fixed knowledge after a schedule change.
    """

    n_inputs = 104

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        tick = world.tick
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[101] = (tick % 20) / 20.0
        # Stale: describes the 20/10 schedule even when the world changed.
        if (tick % 20) < 10:
            v[102] = (10 - (tick % 20)) / 10.0 if door else 0.0
        else:
            v[102] = 0.0
        v[103] = 1.0
        return v


class PhaseOnlyNewEncoder(object):
    """Clock only: (tick % period)/period. 102 inputs.

    No inference, no history, no model. The primary comparison for
    Experiment A: if inference adds nothing beyond the raw clock,
    this ties the wrapper.
    """

    n_inputs = 102

    def __init__(self, period):
        self.period = period

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[101] = (world.tick % self.period) / float(self.period)
        return v


class PhaseOnlyStaleEncoder(object):
    """Stale clock: (tick % 20)/20 regardless of schedule. 102 inputs.

    Sanity check for Experiment B: the clock that doesn't adapt.
    """

    n_inputs = 102

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[101] = (world.tick % 20) / 20.0
        return v


class OracleIndEncoder(object):
    """True phase/remaining with wrapped_ind's encoding: 104 inputs.

    Same scales as WrappedIndEncoder (phase/50, remaining capped 5/5)
    but fed the TRUE phase and remaining from tick 0. The clean upper
    bound and the clean test of inference: same encoding, perfect
    information vs inferred information. Any gap to wrapped_ind is
    inference, not encoding.
    """

    n_inputs = 104

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        tick = world.tick
        period = world.period
        open_len = world.open_len
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        phase = tick % period
        v[101] = phase / 50.0
        if door and phase < open_len:
            v[102] = min(open_len - phase, 5.0) / 5.0
        else:
            v[102] = 0.0
        v[103] = 1.0
        return v


def train_wrapped_ind(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                      lr=1e-3, model=None):
    encoder = WrappedIndEncoder(model=model)
    agent, stats = train_neural(seed, hidden=hidden, episodes=episodes,
                                cap=cap, lr=lr, encoder=encoder)
    return agent, stats, encoder


def train_memory20(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                   lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=Memory20Encoder())


def train_memory30(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                   lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=Memory30Encoder())


def train_oracle_new(seed, period, open_len, hidden=32, episodes=2000,
                     cap=EPISODE_CAP, lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr,
                        encoder=OracleNewEncoder(period, open_len))


def train_oracle_stale(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                       lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=OracleStaleEncoder())


def train_phase_only_new(seed, period, hidden=32, episodes=2000,
                         cap=EPISODE_CAP, lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=PhaseOnlyNewEncoder(period))


def train_phase_only_stale(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                           lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=PhaseOnlyStaleEncoder())


def train_oracle_ind(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                     lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=OracleIndEncoder())


# ---------------------------------------------------------------- checks

def run_checks():
    """No-training structural checks."""
    from world import World, ACTIONS
    STAY = ACTIONS["stay"]

    # WrappedInd: schedule-independent scales, no true constants.
    enc = WrappedIndEncoder()
    assert enc.n_inputs == 104
    w = World(30, 14)
    w.reset(start_tick=0)
    enc.reset(w)
    x0 = enc.encode(w)
    assert x0[101] == 0.0 and x0[102] == 0.0 and x0[103] == 0.0
    for _ in range(300):
        enc.encode(w)
        w.step(STAY)
    x1 = enc.encode(w)
    m = enc.model
    assert m.period == 30 and m.open_len == 14, \
        "model should learn 30/14, got %r" % (m,)
    assert x1[103] == 1.0
    # Phase scaled by 50, remaining capped at 5/5.
    assert abs(x1[101] - m.ticks_since_opening() / 50.0) < 1e-12
    exp_rem = min(m.open_ticks_remaining(), 5.0) / 5.0
    assert abs(x1[102] - exp_rem) < 1e-12
    assert x1[101] < 1.0 and x1[102] <= 1.0, "features in [0,1]"

    # Memory20/30 layouts.
    for cls, n in ((Memory20Encoder, 20), (Memory30Encoder, 30)):
        e = cls()
        assert e.n_inputs == 101 + n, (cls, e.n_inputs)

    # OracleNew on 30/14: true features.
    oe = OracleNewEncoder(30, 14)
    w.reset(start_tick=0)
    oe.reset(w)
    for t in range(40):
        tick = w.tick
        _pos, door = w.observe()
        v = oe.encode(w)
        assert abs(v[101] - (tick % 30) / 30.0) < 1e-12
        exp = ((14 - (tick % 30)) / 14.0) if door else 0.0
        assert abs(v[102] - exp) < 1e-12
        w.step(STAY)

    # OracleStale: 20/10 features even on a 30/14 world.
    se = OracleStaleEncoder()
    w.reset(start_tick=0)
    se.reset(w)
    v = se.encode(w)
    assert abs(v[101] - (w.tick % 20) / 20.0) < 1e-12

    # PhaseOnlyNew: clock for the given period, no inference.
    pe = PhaseOnlyNewEncoder(30)
    assert pe.n_inputs == 102
    w.reset(start_tick=0)
    pe.reset(w)
    for t in range(40):
        tick = w.tick
        v = pe.encode(w)
        assert abs(v[101] - (tick % 30) / 30.0) < 1e-12
        w.step(STAY)

    # PhaseOnlyStale: 20-clock regardless of schedule.
    ps = PhaseOnlyStaleEncoder()
    assert ps.n_inputs == 102
    w.reset(start_tick=0)
    ps.reset(w)
    v = ps.encode(w)
    assert abs(v[101] - (w.tick % 20) / 20.0) < 1e-12

    # OracleInd: true phase/remaining, wrapped_ind's scales.
    oi = OracleIndEncoder()
    assert oi.n_inputs == 104
    w.reset(start_tick=0)  # 30/14 world
    oi.reset(w)
    for t in range(40):
        tick = w.tick
        _pos, door = w.observe()
        v = oi.encode(w)
        assert abs(v[101] - (tick % 30) / 50.0) < 1e-12
        phase = tick % 30
        if door and phase < 14:
            exp = min(14 - phase, 5.0) / 5.0
        else:
            exp = 0.0
        assert abs(v[102] - exp) < 1e-12, (t, v[102], exp)
        assert v[103] == 1.0
        w.step(STAY)

    print("generalization structural check: PASS (wrapped_ind scales, "
          "memory20/30, oracle_new 30/14, oracle_stale, phase clocks, "
          "oracle_ind)")


if __name__ == "__main__":
    run_checks()

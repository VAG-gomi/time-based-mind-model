#!/usr/bin/env python3
"""agent_controls.py -- matched controls for the wrapped arm (EBI priority 2).

Each control isolates one alternative explanation for the wrapped arm's
advantage, using the same learner, training loop, and hyperparameters
(train_neural from agent_nn, hidden=32, lr=1e-3, 2000 episodes).

Controls
--------
oracle:       104 inputs. The TRUE phase (tick % 20)/20 and TRUE remaining
              time /10, with known=1.0 always. No ScheduleModel, no
              inference: perfect schedule knowledge handed directly.
              If oracle >> wrapped, the inferred representation is lossy.
              If oracle ~= wrapped, the ScheduleModel already extracts
              everything the task needs.

phase_only:   102 inputs. Only the true phase (tick % 20)/20. Tests
              whether phase alone (without the countdown) explains the
              gain.

remaining_only: 102 inputs. Only the true remaining-open time /10.
              Tests whether the countdown alone explains the gain.

param_matched: 104 inputs. Plain 101 inputs plus 3 constant-zero dummies.
              Matches the wrapped arm's input dimension and parameter
              count exactly. Tests whether "more parameters" explains
              anything.

memory10:     111 inputs. The last 10 door bits (101 + 10). A stronger
              temporal-memory baseline than the 1-step memory arm: 10
              ticks of history can in principle detect the full open
              interval. Tests whether a longer history window closes the
              gap without any explicit schedule representation.

All oracle/phase/remaining features use the world's true period (20)
and open length (10), exactly as the wrapped arm's fixed scales do.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import (N_INPUTS, GRID_SIZE, EPISODE_CAP, train_neural,
                      PlainEncoder)

PERIOD = 20
OPEN_LEN = 10
PHASE_SCALE = 20.0
REMAINING_SCALE = 10.0
MEMORY_N = 10


def _true_phase(tick):
    return (tick % PERIOD) / PHASE_SCALE


def _true_remaining(tick, door_open):
    if not door_open:
        return 0.0
    return (OPEN_LEN - (tick % PERIOD)) / REMAINING_SCALE


class OracleEncoder(object):
    """104 inputs: 101 plain + true phase + true remaining + known=1."""

    n_inputs = 104

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        tick = world.tick
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[101] = _true_phase(tick)
        v[102] = _true_remaining(tick, door)
        v[103] = 1.0
        return v


class PhaseOnlyEncoder(object):
    """102 inputs: 101 plain + true phase."""

    n_inputs = 102

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[101] = _true_phase(world.tick)
        return v


class RemainingOnlyEncoder(object):
    """102 inputs: 101 plain + true remaining."""

    n_inputs = 102

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[101] = _true_remaining(world.tick, door)
        return v


class ParamMatchedEncoder(object):
    """104 inputs: 101 plain + 3 constant-zero dummies.

    Same input dimension and parameter count as the wrapped arm; the
    dummies carry no information. Any gain over plain cannot be
    attributed to extra parameters alone.
    """

    n_inputs = 104

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        # v[101..103] stay 0.0
        return v


class MemoryNEncoder(object):
    """(101 + N) inputs: 101 plain + last N door bits.

    reset() seeds the history with the door bit at the reset tick
    (same "no change observed yet" semantics as the 1-step memory
    arm). Each encode shifts the history.
    """

    def __init__(self, n=MEMORY_N):
        self.n = n
        self.n_inputs = N_INPUTS + n
        self.hist = [0] * n

    def reset(self, world):
        _pos, door = world.observe()
        self.hist = [1 if door else 0] * self.n

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(self.n_inputs)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        for i, b in enumerate(self.hist):
            v[N_INPUTS + i] = float(b)
        self.hist = self.hist[1:] + [1 if door else 0]
        return v


def train_oracle(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                 lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=OracleEncoder())


def train_phase_only(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                     lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=PhaseOnlyEncoder())


def train_remaining_only(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                         lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=RemainingOnlyEncoder())


def train_param_matched(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                        lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=ParamMatchedEncoder())


def train_memory10(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                   lr=1e-3):
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=MemoryNEncoder(MEMORY_N))


# ---------------------------------------------------------------- checks

def run_checks():
    """No-training structural checks for every control encoder."""
    from world import World, ACTIONS
    STAY = ACTIONS["stay"]

    # Oracle: true phase/remaining against the world's schedule.
    oe = OracleEncoder()
    assert oe.n_inputs == 104
    w = World(PERIOD, OPEN_LEN)
    w.reset(start_tick=0)
    oe.reset(w)
    for t in range(40):
        tick = w.tick
        _pos, door = w.observe()
        v = oe.encode(w)
        assert abs(v[101] - (tick % 20) / 20.0) < 1e-12, "oracle phase"
        exp_rem = ((10 - (tick % 20)) / 10.0) if door else 0.0
        assert abs(v[102] - exp_rem) < 1e-12, "oracle remaining"
        assert v[103] == 1.0
        w.step(STAY)

    # Phase-only / remaining-only layouts.
    for enc, idx in ((PhaseOnlyEncoder(), 101),
                     (RemainingOnlyEncoder(), 101)):
        assert enc.n_inputs == 102
        w.reset(start_tick=3)
        enc.reset(w)
        v = enc.encode(w)
        assert len(v) == 102 and v[:101].sum() == 2.0

    # Param-matched: dummies are exactly zero, dims match wrapped.
    pm = ParamMatchedEncoder()
    assert pm.n_inputs == 104
    from agent_wrapped import WrappedEncoder
    assert pm.n_inputs == WrappedEncoder.n_inputs
    w.reset(start_tick=0)
    pm.reset(w)
    v = pm.encode(w)
    assert (v[101] == 0.0 and v[102] == 0.0 and v[103] == 0.0)

    # Memory-10: history shifts correctly.
    m10 = MemoryNEncoder(10)
    assert m10.n_inputs == 111
    w.reset(start_tick=0)  # door open at tick 0
    m10.reset(w)
    v0 = m10.encode(w)
    assert all(v0[101 + i] == 1.0 for i in range(10)), \
        "history seeded with reset-tick door"
    seen = []
    for _ in range(12):
        _pos, door = w.observe()
        seen.append(1 if door else 0)
        w.step(STAY)
        v = m10.encode(w)
    # After 12 steps the history holds the last 10 door bits.
    assert [int(v[101 + i]) for i in range(10)] == seen[2:12], \
        "history must be the last 10 door bits"

    print("controls structural check: PASS (oracle/phase/remaining/"
          "param-matched/memory10 layouts and semantics)")


if __name__ == "__main__":
    run_checks()

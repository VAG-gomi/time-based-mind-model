#!/usr/bin/env python3
"""agent_wrapped.py -- B2: the wrapped arm (Sequence B).

Same learner, training loop, and hyperparameters as the plain arm:
train_neural() from agent_nn is reused unchanged.  The only
difference is the encoder.

The ScheduleModel sees ONLY (tick, door_bit), exactly once per world
observation.  It never sees the agent's position, reward, chosen
actions, or any episode counter.  One model instance is created per
train_wrapped() call and persists across ALL training episodes and
BOTH evaluation episodes (greedy and eps=0.05): the eval policy is
built on the same encoder, so the model's schedule knowledge carries
over from training into evaluation, exactly as it would in
deployment.

Input layout (104):
    [0..99]   one-hot position (10x10 grid)
    [100]     door bit (1.0 if open)
    [101]     phase = ticks_since_opening / 20.0        (0.0 if unknown)
    [102]     remaining = open_ticks_remaining / 10.0  (0.0 if unknown)
    [103]     known flag: 1.0 iff the time features are available,
              else 0.0.

Encoding justification
----------------------
- The fixed scales (20.0, 10.0) are the world's period and open
  length.  They keep the time inputs in [0, 1], commensurate with the
  one-hot and door inputs, so no input dominates by magnitude.
  "2 ticks left" -- the minimum safe crossing margin (enter the door
  on tick t and exit on t+1 needs both ticks open) -- is 0.2, a clean
  threshold for the tanh hidden units to represent.
- Fixed scales rather than live estimates (phase/period_hat): the
  schedule is fixed in B2, and live scaling would make the input
  distribution shift as the model refines its estimates -- a moving
  target for the learner.  (Scaling by live estimates would be the
  generalization for a changing schedule.)
- The known flag disambiguates 0.0 = "just opened / 0 left" from
  0.0 = "unknown": when known is 0 the network can learn to ignore
  the time inputs.  "Known" means both time features are available,
  i.e. full schedule estimates OR partial knowledge (open length +
  a cleanly observed opening).  Partial knowledge makes the agent
  actionable as soon as the open length is known, without waiting
  for two openings.
- Deliberately NO "safe to cross" bit: that would hand the agent the
  decision the experiment is testing.  The agent must learn the
  2-tick threshold from reward.

The model is fed by WrappedEncoder.encode(), which train_neural's
loop calls exactly once per world observation (reset -> encode, then
encode per step).  The structural check below verifies the
once-per-observation contract with a counting spy.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world import World, ACTIONS
from agent_nn import train_neural, EPISODE_CAP
from wrapper import ScheduleModel

PHASE_SCALE = 20.0
REMAINING_SCALE = 10.0


class WrappedEncoder(object):
    """Encoder for the wrapped arm: 101 plain inputs + 3 time features.

    Holds one ScheduleModel for its whole lifetime.  reset() marks an
    episode boundary (model.new_episode()); it never clears the model.
    """

    n_inputs = 104

    def __init__(self, model=None, blank_features=False):
        self.model = model if model is not None else ScheduleModel()
        self.blank_features = blank_features
        self.total_ticks = 0
        self.known_ticks = 0
        self._episode = -1
        self._converged_ep = None
        self._converged_tick = None

    def reset(self, world):
        self._episode += 1
        self.model.new_episode()

    def encode(self, world):
        (x, y), door = world.observe()
        tick = world.tick
        # The model's only input, once per observation.
        self.model.observe(tick, bool(door))
        self.total_ticks += 1
        v = np.zeros(self.n_inputs)
        v[y * 10 + x] = 1.0
        v[100] = 1.0 if door else 0.0
        if not self.blank_features:
            remaining = self.model.open_ticks_remaining()
            phase = self.model.ticks_since_opening()
            if remaining is not None and phase is not None:
                v[101] = phase / PHASE_SCALE
                v[102] = remaining / REMAINING_SCALE
                v[103] = 1.0
                self.known_ticks += 1
        if (self._converged_ep is None
                and self.model.first_estimates_tick is not None):
            self._converged_ep = self._episode
            self._converged_tick = self.model.first_estimates_tick
        return v

    def frac_known(self):
        return (self.known_ticks / self.total_ticks
                if self.total_ticks else 0.0)

    @property
    def converged_episode(self):
        return self._converged_ep

    @property
    def converged_tick(self):
        return self._converged_tick


def train_wrapped(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                  lr=1e-3, model=None, blank_features=False):
    """Train the wrapped arm.  Returns (agent, stats, encoder).

    The encoder (holding the ScheduleModel) is returned so the caller
    can build the eval policy on the SAME model -- schedule knowledge
    persists from training into evaluation.
    """
    encoder = WrappedEncoder(model=model, blank_features=blank_features)
    agent, stats = train_neural(seed, hidden=hidden, episodes=episodes,
                                cap=cap, lr=lr, encoder=encoder)
    return agent, stats, encoder


class ClockEncoder(object):
    """Control arm: the raw tick, no schedule model.

    102 inputs: 101 (position + door) + tick/200.  Raw tick, NOT
    tick % 20: handing the phase directly would duplicate the
    wrapper's phase feature and confound the control.  The network
    must learn the periodic mapping itself, which is exactly what
    the control tests: does the schedule model add anything beyond
    knowing the time?
    """

    n_inputs = 102

    def reset(self, world):
        pass

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(self.n_inputs)
        v[y * 10 + x] = 1.0
        v[100] = 1.0 if door else 0.0
        v[101] = world.tick / 200.0
        return v


def train_clock(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                lr=1e-3):
    """Train the raw-clock control arm.  Returns (agent, stats)."""
    agent, stats = train_neural(seed, hidden=hidden, episodes=episodes,
                                cap=cap, lr=lr, encoder=ClockEncoder())
    return agent, stats


# ---------------------------------------------------------------- checks

class _SpyModel(ScheduleModel):
    def __init__(self):
        ScheduleModel.__init__(self)
        self.calls = []

    def observe(self, tick, door_open):
        self.calls.append((tick, bool(door_open)))
        return ScheduleModel.observe(self, tick, door_open)


def run_checks():
    """Encoder structural check (no training)."""
    STAY = ACTIONS["stay"]
    assert WrappedEncoder.n_inputs == 104
    assert ClockEncoder.n_inputs == 102

    # The model is fed exactly (tick, door), once per observation.
    spy = _SpyModel()
    enc = WrappedEncoder(model=spy)
    w = World(20, 10)
    w.reset(start_tick=7)
    enc.reset(w)
    for _ in range(10):
        tick = w.tick
        door = w.observe()[1]
        x = enc.encode(w)
        assert len(x) == 104
        assert spy.calls[-1] == (tick, bool(door)), \
            "model fed something other than (tick, door)"
        (px, py) = w.observe()[0]
        assert x[py * 10 + px] == 1.0
        assert x[100] == (1.0 if door else 0.0)
        w.step(STAY)
    assert len(spy.calls) == 10, \
        "model.observe must run exactly once per encode"

    # Known flag: 0 before any schedule knowledge, 1 after convergence;
    # feature slots track the model exactly.
    enc2 = WrappedEncoder()
    w2 = World(20, 10)
    w2.reset(start_tick=0)
    enc2.reset(w2)
    x0 = enc2.encode(w2)
    assert x0[101] == 0.0 and x0[102] == 0.0 and x0[103] == 0.0, \
        "features must be 0/unknown before any opening is seen"
    for _ in range(200):
        enc2.encode(w2)
        w2.step(STAY)
    x1 = enc2.encode(w2)
    m = enc2.model
    assert m.period == 20 and m.open_len == 10, "model should converge"
    assert x1[103] == 1.0
    assert abs(x1[101] - m.ticks_since_opening() / PHASE_SCALE) < 1e-12
    assert abs(x1[102] - m.open_ticks_remaining() / REMAINING_SCALE) < 1e-12
    assert enc2.converged_episode == 0, \
        "a 200-tick episode must converge in episode 0"
    assert enc2.frac_known() > 0.8

    # blank_features: time slots forced to 0, model still runs.
    enc3 = WrappedEncoder(blank_features=True)
    w3 = World(20, 10)
    w3.reset(start_tick=0)
    enc3.reset(w3)
    for _ in range(200):
        enc3.encode(w3)
        w3.step(STAY)
    x3 = enc3.encode(w3)
    assert x3[101] == 0.0 and x3[102] == 0.0 and x3[103] == 0.0
    assert enc3.model.period == 20, "blanked run must still feed the model"

    # Clock encoder layout.
    ce = ClockEncoder()
    w4 = World(20, 10)
    w4.reset(start_tick=3)
    xc = ce.encode(w4)
    assert len(xc) == 102
    assert xc[101] == w4.tick / 200.0

    print("encoder structural check: PASS "
          "(104-dim layout, once-per-observation (tick, door), "
          "known flag, blanking, clock layout)")


if __name__ == "__main__":
    run_checks()

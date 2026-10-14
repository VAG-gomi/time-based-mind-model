#!/usr/bin/env python3
"""agent_mem.py -- one-step-memory arm (A5).

The same learner, training loop, replay sampler, and hyperparameters as
the plain arm (agent_nn.py), with exactly one change: the input gains the
previous tick's door bit, 101 -> 102 inputs.

Input layout: [0..99] one-hot position, [100] current door bit,
[101] previous tick's door bit.

Previous-door semantics:
- At reset, prev_door is set to the door bit observed at the reset tick.
  The encoder therefore signals "no change observed yet" on the first
  step. The alternative (prev_door = 0) was rejected: on a reset tick
  with the door open it would fabricate a closed->open edge the agent
  never observed, and it would make the first-step input depend on an
  arbitrary constant rather than on the world.
- On every encode, the vector uses the stored prev_door, then prev_door
  is updated to the door bit of the observation just encoded. Because
  train_neural / the A5 evaluator call encode exactly once per
  observation, the stored value is always the door bit of the previous
  tick.

This arm is the control that separates "any memory" from "time
structure": one bit of history lets the network detect a door
transition, but it carries no schedule model (no period, no countdown,
no ticks-since/remaining). If this arm matches the wrapped arm, the
wrapper adds nothing beyond one-step memory; if the wrapped arm wins,
the win comes from the schedule structure, not from merely having
memory.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import (N_INPUTS, GRID_SIZE, EPISODE_CAP, train_neural)

N_INPUTS_MEM = N_INPUTS + 1  # 102: position + door + previous door


class MemoryEncoder(object):
    """Encoder protocol for the memory arm: 102 inputs."""

    n_inputs = N_INPUTS_MEM

    def __init__(self):
        self.prev_door = 0

    def reset(self, world):
        _pos, door = world.observe()
        self.prev_door = 1 if door else 0

    def encode(self, world):
        (x, y), door = world.observe()
        v = np.zeros(N_INPUTS_MEM)
        v[y * GRID_SIZE + x] = 1.0
        v[N_INPUTS - 1] = 1.0 if door else 0.0
        v[N_INPUTS] = float(self.prev_door)
        self.prev_door = 1 if door else 0
        return v


def train_memory(seed, hidden=32, episodes=2000, cap=EPISODE_CAP, lr=1e-3):
    """Train the memory arm: identical to train_neural except the encoder.

    Returns (agent, per-episode stats) with the same format as train_neural.
    """
    return train_neural(seed, hidden=hidden, episodes=episodes, cap=cap,
                        lr=lr, encoder=MemoryEncoder())


def _structural_check():
    """No-training check of the encoder's prev-door bookkeeping.

    Simulates the exact call pattern of train_neural: reset, encode,
    step, encode, ... and asserts the prev-door slot always holds the
    door bit of the previous tick.
    """
    from world import World, ACTIONS
    w = World(period=20, open_len=10)
    enc = MemoryEncoder()
    assert enc.n_inputs == 102
    for start in (0, 5, 10, 15, 19):
        w.reset(start_tick=start)
        enc.reset(w)
        _pos, door0 = w.observe()
        v0 = enc.encode(w)
        assert v0.shape == (102,)
        assert v0[100] == (1.0 if door0 else 0.0), "current door slot"
        assert v0[101] == (1.0 if door0 else 0.0), \
            "prev door at first step must equal current door (no-change)"
        assert v0[:100].sum() == 1.0 and v0[0] == 1.0, "one-hot start"
        prev = door0
        for _ in range(25):
            w.step(ACTIONS["stay"])
            _pos, door = w.observe()
            v = enc.encode(w)
            assert v[100] == (1.0 if door else 0.0), "current door slot"
            assert v[101] == (1.0 if prev else 0.0), \
                "prev door slot must hold the previous tick's door bit"
            assert v[:100].sum() == 1.0, "one-hot position"
            prev = door
    print("MEMORY ENCODER STRUCTURAL CHECK: PASS")
    return True


if __name__ == "__main__":
    _structural_check()

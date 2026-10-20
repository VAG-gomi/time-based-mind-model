#!/usr/bin/env python3
"""count_numpy_calls.py -- WO-TBM-L15-diag Part A: numpy calls
per train step.

Candidate mechanism under test: the trust-1.5 machinery makes
many small numpy calls per train step (batched segment-tree
walks over 32-element arrays), each paying fixed per-call
overhead, where the uniform arm makes O(1) calls.

Method: build each agent, pre-fill its buffer with 10000 dummy
transitions (shapes identical to real ones), then drive
train_step() directly under sys.setprofile and count call /
c_call events whose target belongs to numpy. Call counts are
data-independent for both implementations (fixed shapes), so
dummy data does not distort the count. Also counts numpy calls
per buffer add() (one add happens per env step, next to each
train step). Diagnostic only.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import NeuralQ
from layer15_trust_train import Trust15NeuralQ

STEPS = 300
ADDS = 1000


def is_numpy_target(func):
    if isinstance(func, np.ufunc):
        return True
    mod = getattr(func, "__module__", "") or ""
    if mod.startswith("numpy"):
        return True
    self_obj = getattr(func, "__self__", None)
    if self_obj is not None:
        return type(self_obj).__module__.startswith("numpy")
    return False


class CallCounter(object):
    def __init__(self):
        self.numpy_calls = 0
        self.total_calls = 0
        self.by_caller = {}

    def handler(self, frame, event, arg):
        if event == "call":
            mod = frame.f_globals.get("__name__", "")
            self.total_calls += 1
            if mod.startswith("numpy"):
                self.numpy_calls += 1
                self._bucket(frame)
        elif event == "c_call":
            self.total_calls += 1
            if is_numpy_target(arg):
                self.numpy_calls += 1
                self._bucket(frame)

    def _bucket(self, frame):
        f = frame
        while f is not None:
            if os.path.dirname(f.f_code.co_filename) == _HERE:
                self.by_caller[f.f_code.co_name] = \
                    self.by_caller.get(f.f_code.co_name, 0) + 1
                return
            f = f.f_back
        self.by_caller["<other>"] = self.by_caller.get("<other>", 0) + 1


def dummy_transition(rng):
    x = rng.rand(101).astype(float)
    x2 = rng.rand(101).astype(float)
    return (x, int(rng.randint(5)), float(rng.rand()), x2, False)


def count_steps(agent, label):
    c = CallCounter()
    sys.setprofile(c.handler)
    try:
        for _ in range(STEPS):
            agent.train_step()
    finally:
        sys.setprofile(None)
    print("%s: %d numpy calls over %d train steps = %.1f/step"
          % (label, c.numpy_calls, STEPS, c.numpy_calls / STEPS))
    top = sorted(c.by_caller.items(), key=lambda kv: -kv[1])[:8]
    for name, n in top:
        print("    via %-16s %6d  (%.1f/step)" % (name, n, n / STEPS))
    return c.numpy_calls / STEPS


def count_adds(agent, label, rng):
    items = [dummy_transition(rng) for _ in range(ADDS)]
    c = CallCounter()
    sys.setprofile(c.handler)
    try:
        for it in items:
            agent.buffer.add(it)
    finally:
        sys.setprofile(None)
    print("%s: %d numpy calls over %d adds = %.2f/add"
          % (label, c.numpy_calls, ADDS, c.numpy_calls / ADDS))
    return c.numpy_calls / ADDS


def main():
    rng = np.random.RandomState(0)

    uni = NeuralQ(0)
    for _ in range(10000):
        uni.buffer.add(dummy_transition(rng))

    t15 = Trust15NeuralQ(0)
    for _ in range(10000):
        t15.buffer.add(dummy_transition(rng))
    # Non-uniform trust landscape (call counts do not depend on
    # values, but keep the sampling path realistic).
    idx = np.arange(0, 10000, 7)
    t15.buffer.update_trust(idx, rng.rand(len(idx)) * 2.0)

    per_step_uni = count_steps(uni, "uniform")
    per_step_t15 = count_steps(t15, "trust15")
    per_add_uni = count_adds(uni, "uniform", rng)
    per_add_t15 = count_adds(t15, "trust15", rng)
    print("TOTAL per env step (1 train step + 1 add): uniform %.1f, "
          "trust15 %.1f numpy calls"
          % (per_step_uni + per_add_uni, per_step_t15 + per_add_t15))


if __name__ == "__main__":
    main()

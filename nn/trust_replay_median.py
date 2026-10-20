#!/usr/bin/env python3
"""trust_replay_median.py -- WO-TBM-PC4 diagnostic variant buffer.

FastTrustReplayBuffer (frozen, imported) with ONE change: a new
transition's insertion priority is the **current median leaf
trust** instead of the current maximum.

Exact definition: leaves store trust**alpha. The insertion leaf
value is the median of the leaf values over the currently
filled slots -- equivalently the median raw trust raised to
alpha, since x -> x**alpha is monotone. On an empty buffer the
insertion leaf is 1.0 (raw trust 1.0, the same starting value
the frozen design uses for its initial maximum).

The class also records insertion steps and the last sample
(steps/indices/probs) for the sampled-age manipulation check;
recording consumes no RNG (sampling delegates to the frozen
implementation).

DIAGNOSTIC ONLY, non-confirmatory. Median computation is
O(filled) per insertion via numpy -- a diagnostic cost,
accepted and irrelevant to the endpoints.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from trust_replay_fast import FastTrustReplayBuffer


class MedianInsertDiagBuffer(FastTrustReplayBuffer):
    def __init__(self, capacity=10000, alpha=0.6):
        super(MedianInsertDiagBuffer, self).__init__(capacity,
                                                     alpha=alpha)
        self.ins_step = np.zeros(capacity, dtype=int)
        self.clock = 0
        self.last_idx = None
        self.last_probs = None

    def add(self, item):
        idx = self.pos
        self.ins_step[idx] = self.clock
        self.clock += 1
        self.data[idx] = item
        if self.count == 0:
            leaf = 1.0
        else:
            filled = self.tree[self.size:self.size + self.count] \
                if self.count < self.capacity else \
                self.tree[self.size:self.size + self.capacity]
            leaf = float(np.median(filled))
        self._set_leaves([idx], [leaf])
        self.pos = (self.pos + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def sample(self, n, rng):
        items, idx, probs = super(MedianInsertDiagBuffer,
                                  self).sample(n, rng)
        self.last_idx = idx
        self.last_probs = probs
        return items, idx, probs

    def ages(self):
        return self.clock - self.ins_step[self.last_idx]

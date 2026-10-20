#!/usr/bin/env python3
"""trust_replay_fast.py -- Layer 1.5 fast trust buffer (WO-TBM-L15).

Same semantics as the committed naive TrustReplayBuffer
(nn/trust_replay.py -- the permanent Layer-1 record, NOT edited):
  - trust score = |TD-error| + epsilon; new transitions enter at
    the current max trust; circular FIFO replacement; sampling
    probability proportional to trust**alpha.

What changes is only the machinery (the Layer-1 deficit was
measured as buffer cost, Step-0 profile):
  - Batched segment tree (Tianshou design): flat numpy array in
    heap layout; leaves store trust**alpha; internal nodes store
    subtree sums. Priority changes rewrite ONLY the changed
    leaves' root paths -- no O(N) rebuild exists anywhere.
  - Batched prefix-sum search: all samples in a batch walk the
    tree together, one numpy-vectorized level at a time.
  - Stratified proportional sampling (Schaul et al.): the total
    mass is split into n equal segments, one uniform draw per
    segment. Per-draw marginal probabilities equal the naive
    formula's; unlike the naive sampler, draws are independent,
    so an index may repeat within a batch (standard for
    stratified sampling; IS weights in layer15_trust_train.py
    are computed per draw).

Pure numpy + stdlib. Constants match Layer 1: alpha=0.6,
epsilon=1e-6, capacity default 10000.
"""

import numpy as np


class FastTrustReplayBuffer(object):
    """Segment-tree trust-weighted replay buffer."""

    def __init__(self, capacity=10000, alpha=0.6, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.size = 1
        while self.size < capacity:
            self.size *= 2
        # Heap layout: tree[1] is the root (total mass); leaves for
        # slot i live at tree[size + i]. Unfilled/padding leaves are
        # exactly 0 and can never be sampled (see _find).
        self.tree = np.zeros(2 * self.size, dtype=float)
        self.data = [None] * capacity
        self.pos = 0
        self.count = 0
        self.max_trust = 1.0  # raw trust scale, as in the naive buffer

    # -- internal: batched leaf writes + path repair ---------------
    def _set_leaves(self, indices, values):
        """Set leaf values (trust**alpha) for `indices` and repair
        only the affected root paths, level by level (batched)."""
        nodes = self.size + np.asarray(indices, dtype=int)
        self.tree[nodes] = values
        nodes //= 2
        while True:
            self.tree[nodes] = (self.tree[2 * nodes]
                                + self.tree[2 * nodes + 1])
            if nodes[0] == 1:
                break
            nodes //= 2

    def _find(self, values):
        """Batched prefix-sum search: for each v in `values`
        (0 <= v < total mass), return the leaf slot whose
        cumulative-mass interval contains v. Zero-mass leaves are
        unreachable: descending into one would require
        v >= total mass."""
        idx = np.ones(len(values), dtype=int)
        vals = np.asarray(values, dtype=float).copy()
        while idx[0] < self.size:
            left = 2 * idx
            left_sum = self.tree[left]
            go_right = vals >= left_sum
            idx = np.where(go_right, left + 1, left)
            vals = np.where(go_right, vals - left_sum, vals)
        return idx - self.size

    # -- public API (mirrors the naive buffer) ----------------------
    def add(self, item):
        idx = self.pos
        self.data[idx] = item
        self._set_leaves([idx], [self.max_trust ** self.alpha])
        self.pos = (self.pos + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def sample(self, n, rng):
        """Stratified proportional sample of n draws.
        Returns (items, indices, probs): probs[i] is the exact
        per-draw probability of indices[i] (leaf mass / total),
        for IS weighting by the caller."""
        total = self.tree[1]
        if self.count == 0 or total <= 0.0:
            raise ValueError("cannot sample from an empty buffer")
        segment = total / n
        points = (np.arange(n) + rng.random_sample(n)) * segment
        # Guard the top stratum against float rounding to == total.
        np.minimum(points, np.nextafter(total, 0.0), out=points)
        indices = self._find(points)
        probs = self.tree[self.size + indices] / total
        return [self.data[i] for i in indices], indices, probs

    def update_trust(self, indices, td_errors):
        """Update trust scores from new TD-errors (batched).
        If an index repeats in the batch, the last TD-error wins
        (numpy fancy-assignment semantics)."""
        trusts = np.abs(np.asarray(td_errors, dtype=float)) + self.epsilon
        m = float(trusts.max()) if len(trusts) else 0.0
        if m > self.max_trust:
            self.max_trust = m
        self._set_leaves(indices, trusts ** self.alpha)

    def probabilities(self, indices):
        """Exact per-draw probabilities for the given slots."""
        total = self.tree[1]
        if total <= 0.0:
            raise ValueError("no probability mass in buffer")
        return self.tree[self.size + np.asarray(indices, dtype=int)] / total

    def __len__(self):
        return self.count

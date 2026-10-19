#!/usr/bin/env python3
"""trust_replay.py -- minimal GraphRoad-inspired fast core (v0.1).

Takes one idea from GraphRoad: don't treat all experiences equally.
GraphRoad scores nodes by trust and samples accordingly; here we
score transitions by TD-error (surprise) and sample proportionally.

This is a drop-in replacement for the uniform ReplayBuffer in
agent_nn.py. Everything else (network, world, protocol) is unchanged.

Layer 0 (this file): trust-weighted sampling only.
Future layers (not built): vectorized batches, state cache, decay.

Test: same performance in fewer episodes? Or same episodes, faster?
"""

import numpy as np


class TrustReplayBuffer(object):
    """Replay buffer with trust-weighted (prioritized) sampling.

    Each transition gets a trust score = |TD-error| + epsilon.
    Sampling probability is proportional to trust^alpha.
    After each training step, the sampled transitions' trust scores
    are updated with their new TD-errors.

    Inspired by GraphRoad's LocalTrustEngine: epistemic trust from
    surprise (contradiction pressure), not uniform treatment.
    """

    def __init__(self, capacity, alpha=0.6, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.data = []
        self.trust = []  # parallel to data
        self.pos = 0
        self.max_trust = 1.0

    def add(self, item):
        # New transitions get max trust (optimistic, like GraphRoad's
        # initial confidence for new nodes).
        if len(self.data) < self.capacity:
            self.data.append(item)
            self.trust.append(self.max_trust)
        else:
            self.data[self.pos] = item
            self.trust[self.pos] = self.max_trust
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, n, rng):
        """Sample n indices proportional to trust^alpha. Returns
        (items, indices) so the caller can update trust after."""
        t = np.array(self.trust, dtype=float) ** self.alpha
        total = t.sum()
        if total <= 0:
            probs = np.ones(len(t)) / len(t)
        else:
            probs = t / total
        idx = rng.choice(len(self.data), n, replace=False, p=probs)
        return [self.data[i] for i in idx], idx

    def update_trust(self, indices, td_errors):
        """Update trust scores from new TD-errors."""
        for i, td in zip(indices, td_errors):
            t = abs(float(td)) + self.epsilon
            self.trust[i] = t
            if t > self.max_trust:
                self.max_trust = t

    def __len__(self):
        return len(self.data)


def run_checks():
    """Structural checks (no training)."""
    rng = np.random.RandomState(0)
    buf = TrustReplayBuffer(10)
    assert len(buf) == 0
    for i in range(5):
        buf.add(("s%d" % i, i, 0.0, "s%d" % (i + 1), False))
    assert len(buf) == 5
    # All new: max trust, uniform sampling.
    items, idx = buf.sample(3, rng)
    assert len(items) == 3 and len(idx) == 3
    # Update trust: high TD-error -> sampled more often.
    buf.update_trust(idx, [10.0, 0.001, 0.001])
    counts = {i: 0 for i in range(5)}
    for _ in range(200):
        _, ii = buf.sample(1, rng)
        counts[ii[0]] += 1
    # The high-trust index should dominate.
    top = max(counts, key=counts.get)
    assert counts[top] > 100, "trust-weighted sampling not working: %r" % counts
    # Overflow: circular replacement.
    for i in range(10):
        buf.add(("x%d" % i, i, 0.0, "y", False))
    assert len(buf) == 10
    print("trust replay check: PASS (weighted sampling, updates, overflow)")


if __name__ == "__main__":
    run_checks()

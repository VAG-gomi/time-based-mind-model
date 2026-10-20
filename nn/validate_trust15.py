#!/usr/bin/env python3
"""validate_trust15.py -- WO-TBM-L15 Step 2 correctness gates.

All checks run on synthetic data / short smoke runs; no rematch is
run here. Every gate must pass before the Step-3 freeze.

Gates:
  1. Distribution equivalence: 200,000 stratified draws from
     FastTrustReplayBuffer vs the naive formula's probabilities
     (trust**alpha normalized). Tolerance: max absolute deviation
     <= 0.01 (hot-item sigma ~= 0.0008, so ~12 sigma).
  2. IS weights: hand-computed example matches exactly; beta
     schedule endpoints exact (0.4 start, 1.0 end), monotonic.
  3. Structural: circular overflow/replacement contents match the
     naive buffer exactly; partial-buffer sampling stays within
     filled slots; empty buffer raises a defined error; fast
     probabilities equal the naive formula's.
  4. Determinism: same seed -> identical sampled-index sequences
     and identical short training stats across two runs.
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from trust_replay import TrustReplayBuffer
from trust_replay_fast import FastTrustReplayBuffer
from layer15_trust_train import beta_at, is_weights, train_trust15

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print("PASS %s" % name)
    else:
        failed += 1
        print("FAIL %s" % name)


EPS = 1e-6

# ---- Gate 1: distribution equivalence -----------------------------
# 100 slots: 5 hot (trust 1000), 95 cold (trust 1).
buf = FastTrustReplayBuffer(capacity=128, alpha=0.6)
trusts = np.array([1000.0] * 5 + [1.0] * 95)
for i in range(100):
    buf.add(("s%d" % i, i, 0.0, "t", False))
buf.update_trust(np.arange(100), trusts - EPS)  # |td|+eps == trust
p_true = trusts ** 0.6
p_true = p_true / p_true.sum()
# The naive formula gives the same probabilities by construction;
# verify against the committed naive buffer's own state.
nb = TrustReplayBuffer(128, alpha=0.6)
for i in range(100):
    nb.add(("s%d" % i, i, 0.0, "t", False))
nb.update_trust(list(range(100)), list(trusts - EPS))
t = np.array(nb.trust[:100]) ** 0.6
check("naive formula probs == analytic probs",
      np.allclose(t / t.sum(), p_true, atol=1e-12))
check("fast probabilities() == naive formula",
      np.allclose(buf.probabilities(np.arange(100)), p_true,
                  atol=1e-12))

rng = np.random.RandomState(42)
DRAWS = 200_000
counts = np.zeros(100, dtype=int)
done = 0
while done < DRAWS:
    _items, idx, _p = buf.sample(32, rng)
    counts += np.bincount(idx, minlength=100)
    done += len(idx)
emp = counts / counts.sum()
max_dev = float(np.max(np.abs(emp - p_true)))
check("distribution: max |empirical - true| <= 0.01 over %d draws "
      "(observed %.5f)" % (DRAWS, max_dev), max_dev <= 0.01)

# ---- Gate 2: IS weights + beta schedule ---------------------------
probs = np.array([0.1, 0.2, 0.3, 0.4])
raw = [math.pow(100 * p, -0.4) for p in probs]
expected = np.array([r / max(raw) for r in raw])
got = is_weights(probs, 100, 0.4)
check("IS weights match hand-computed example (atol 1e-12)",
      np.allclose(got, expected, atol=1e-12))
check("IS weights normalized: max == 1.0 exactly",
      float(got.max()) == 1.0)
check("beta endpoints exact: beta_at(0,1000)==0.4, "
      "beta_at(999,1000)==1.0",
      beta_at(0, 1000) == 0.4 and beta_at(999, 1000) == 1.0)
bs = [beta_at(e, 1000) for e in range(1000)]
check("beta monotonic non-decreasing, within [0.4, 1.0]",
      all(bs[i] <= bs[i + 1] for i in range(999))
      and bs[0] >= 0.4 and bs[-1] <= 1.0)

# ---- Gate 3: structural -------------------------------------------
cap = 50
fb = FastTrustReplayBuffer(capacity=cap)
nv = TrustReplayBuffer(cap)
for i in range(120):
    item = ("x%d" % i, i, 0.0, "y", False)
    fb.add(item)
    nv.add(item)
check("overflow: contents match naive exactly after 120 adds "
      "(cap 50)", fb.data == nv.data and len(fb) == len(nv) == cap
      and fb.pos == nv.pos)
td = [float((i * 7) % 13) + 0.5 for i in range(cap)]
fb.update_trust(np.arange(cap), np.array(td) - EPS)
nv.update_trust(list(range(cap)), [v - EPS for v in td])
tn = np.array(nv.trust) ** 0.6
check("overflow: probabilities match naive after updates",
      np.allclose(fb.probabilities(np.arange(cap)),
                  tn / tn.sum(), atol=1e-12))

pb = FastTrustReplayBuffer(capacity=16)
for i in range(3):
    pb.add(("p%d" % i, i, 0.0, "q", False))
rng2 = np.random.RandomState(1)
seen = set()
for _ in range(5000):
    _it, idx, _p = pb.sample(4, rng2)
    seen.update(int(i) for i in idx)
check("partial buffer: draws only from filled slots {0,1,2}",
      seen == {0, 1, 2})
eb = FastTrustReplayBuffer(capacity=16)
try:
    eb.sample(4, np.random.RandomState(0))
    empty_ok = False
except ValueError:
    empty_ok = True
check("empty buffer: sampling raises ValueError (defined)", empty_ok)

# ---- Gate 4: determinism ------------------------------------------
def index_sequence():
    b = FastTrustReplayBuffer(capacity=64)
    r = np.random.RandomState(7)
    seq = []
    for i in range(80):
        b.add(("d%d" % i, i, 0.0, "e", False))
        if len(b) >= 8:
            _it, idx, _p = b.sample(8, r)
            b.update_trust(idx, [0.3 * (j + 1) for j in range(len(idx))])
            seq.append(tuple(int(x) for x in idx))
    return seq


check("determinism: identical index sequences across two runs",
      index_sequence() == index_sequence())
_a1, s1 = train_trust15(3, episodes=20)
_a2, s2 = train_trust15(3, episodes=20)
check("determinism: identical 20-episode training stats",
      s1 == s2)

print("---")
print("%d/%d checks passed" % (passed, passed + failed))
sys.exit(1 if failed else 0)

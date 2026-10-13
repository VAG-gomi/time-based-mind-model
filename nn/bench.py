#!/usr/bin/env python3
"""A1: environment and speed check.

bench.py only. Does not touch world.py. No training.
Times forward+backward passes of a 105 -> 32 -> 5 network (tanh hidden).
"""

import sys
import time

print("python:", sys.version.split()[0])

try:
    import numpy
    print("numpy: available, version", numpy.__version__)
    have_numpy = True
except ImportError:
    print("numpy: NOT available")
    have_numpy = False

if have_numpy:
    import numpy as np

    rng = np.random.RandomState(0)  # fixed seed
    W1 = rng.randn(105, 32) * 0.1
    b1 = np.zeros(32)
    W2 = rng.randn(32, 5) * 0.1
    b2 = np.zeros(5)

    def forward_backward(X):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        dz2 = np.ones_like(z2) / X.shape[0]  # dummy upstream gradient
        dW2 = a1.T.dot(dz2)
        db2 = dz2.sum(axis=0)
        da1 = dz2.dot(W2.T)
        dz1 = da1 * (1.0 - a1 * a1)
        dW1 = X.T.dot(dz1)
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    for batch in (1, 32):
        X = rng.randn(batch, 105)
        forward_backward(X)  # warmup
        t0 = time.perf_counter()
        for _ in range(1000):
            forward_backward(X)
        dt = time.perf_counter() - t0
        ms_per_update = dt / 1000.0 * 1000.0
        print("batch %d: %.3f ms per forward+backward update "
              "(timed over 1000 iters)" % (batch, ms_per_update))
        if batch == 1:
            sec_100k = dt / 1000.0 * 100000.0
            print("extrapolated: 100,000 batch-1 updates ~ %.1f seconds "
                  "(~%.1f minutes)" % (sec_100k, sec_100k / 60.0))

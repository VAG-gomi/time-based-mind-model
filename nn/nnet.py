#!/usr/bin/env python3
"""nnet.py -- classical neural network substrate (Sequence A).

Hand-written MLP: fully connected, tanh hidden layers, linear output,
float64, numpy only. No ML frameworks.

Conventions
-----------
- ``MLP(sizes, seed, init)`` with ``sizes = [n_in, h1, ..., n_out]`` and
  ``init`` one of ``"xavier"`` (default), ``"zeros"``, or a float giving
  the weight stddev (biases are always zero).
- ``forward(X)`` caches intermediates (call it before ``backward``).
- ``backward(dout)`` takes the upstream gradient dL/dout of shape
  (B, n_out) and fills ``self.grads`` with RAW sums over the batch.
  Put any 1/B averaging in your loss's dout (both helpers below do).
- ``sgd_step(lr)`` applies the cached grads: p -= lr * grad.
- ``Adam`` is an Adam optimizer for an MLP (call ``backward()``, then
  ``step()``).
- ``get_params()`` / ``set_params(vec)`` expose one flat float64 vector
  in W1, b1, W2, b2, ... order.
- ``masked_mse_grad`` is the Q-learning regression loss: only the chosen
  action's output carries error.
"""

import numpy as np


class MLP:
    """Fully connected network, tanh hidden layers, linear output."""

    def __init__(self, sizes, seed=0, init="xavier"):
        if len(sizes) < 2:
            raise ValueError("sizes needs at least [n_in, n_out]")
        if isinstance(init, str):
            if init == "xavier":
                mode, scale = "xavier", None
            elif init == "zeros":
                mode, scale = "zeros", None
            else:
                raise ValueError("init must be 'xavier', 'zeros', or a "
                                 "float scale, got %r" % (init,))
        elif isinstance(init, (int, float)) and not isinstance(init, bool):
            if init < 0:
                raise ValueError("init scale must be >= 0")
            mode, scale = "scaled", float(init)
        else:
            raise ValueError("init must be 'xavier', 'zeros', or a float "
                             "scale, got %r" % (init,))
        self.sizes = list(sizes)
        self.n_layers = len(sizes) - 1
        self.rng = np.random.RandomState(seed)
        self.params = {}
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
            if mode == "xavier":
                # Glorot normal: Var(W) = 2 / (fan_in + fan_out).
                # Textbook match for tanh hidden layers.
                std = np.sqrt(2.0 / (fan_in + fan_out))
                W = self.rng.randn(fan_in, fan_out) * std
            elif mode == "scaled":
                W = self.rng.randn(fan_in, fan_out) * scale
            else:  # zeros: the "empty world" init
                W = np.zeros((fan_in, fan_out))
            b = np.zeros(fan_out)
            self.params["W%d" % i] = np.ascontiguousarray(W,
                                                          dtype=np.float64)
            self.params["b%d" % i] = np.ascontiguousarray(b,
                                                          dtype=np.float64)
        self.grads = {}
        self._cache = {}

    def _keys(self):
        keys = []
        for i in range(1, self.n_layers + 1):
            keys.append("W%d" % i)
            keys.append("b%d" % i)
        return keys

    @property
    def n_params(self):
        return sum(self.params[k].size for k in self._keys())

    def get_params(self):
        """All parameters as one flat float64 vector (copy)."""
        return np.concatenate([self.params[k].ravel()
                               for k in self._keys()])

    def set_params(self, vec):
        """Write one flat vector back into the parameter arrays."""
        vec = np.asarray(vec, dtype=np.float64).ravel()
        if vec.size != self.n_params:
            raise ValueError("expected %d params, got %d"
                             % (self.n_params, vec.size))
        off = 0
        for k in self._keys():
            n = self.params[k].size
            self.params[k] = vec[off:off + n].reshape(
                self.params[k].shape).copy()
            off += n

    def get_grads(self):
        """All cached grads as one flat float64 vector (copy)."""
        return np.concatenate([self.grads[k].ravel()
                               for k in self._keys()])

    def forward(self, X):
        """Forward pass on batch X of shape (B, n_in). Returns (B, n_out)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.sizes[0]:
            raise ValueError("expected (B, %d), got shape %s"
                             % (self.sizes[0], X.shape))
        a = X
        self._cache = {"a0": a}
        for i in range(1, self.n_layers + 1):
            z = a.dot(self.params["W%d" % i]) + self.params["b%d" % i]
            self._cache["z%d" % i] = z
            a = np.tanh(z) if i < self.n_layers else z
            self._cache["a%d" % i] = a
        return a

    def backward(self, dout):
        """Backprop from upstream dL/dout of shape (B, n_out).

        Fills self.grads (raw batch sums). Returns dL/dX of shape
        (B, n_in). Call forward() first.
        """
        if "a0" not in self._cache:
            raise RuntimeError("call forward() before backward()")
        dout = np.asarray(dout, dtype=np.float64)
        B = self._cache["a0"].shape[0]
        if dout.shape != (B, self.sizes[-1]):
            raise ValueError("expected dout shape (%d, %d), got %s"
                             % (B, self.sizes[-1], dout.shape))
        d = dout  # output layer is linear: dL/dz_L = dL/da_L
        for i in range(self.n_layers, 0, -1):
            if i < self.n_layers:
                a_i = self._cache["a%d" % i]  # tanh activation
                d = d * (1.0 - a_i ** 2)      # dL/dz_i
            a_prev = self._cache["a%d" % (i - 1)]
            self.grads["W%d" % i] = a_prev.T.dot(d)
            self.grads["b%d" % i] = d.sum(axis=0)
            d = d.dot(self.params["W%d" % i].T)  # dL/da_{i-1}
        return d  # dL/da_0 = dL/dX

    def sgd_step(self, lr):
        """Apply cached grads: p -= lr * grad."""
        for k in self._keys():
            self.params[k] -= lr * self.grads[k]


class Adam(object):
    """Adam optimizer for an MLP. Call net.backward() first, then step().

    Standard Adam with bias correction. State (m, v, t) lives here, so
    one Adam instance owns one network; reusing it across trainings
    carries momentum over (usually not what you want).
    """

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = dict((k, np.zeros_like(net.params[k]))
                     for k in net._keys())
        self.v = dict((k, np.zeros_like(net.params[k]))
                     for k in net._keys())
        self.t = 0

    def step(self):
        """Apply one Adam update from the cached grads."""
        self.t += 1
        for k in self.net._keys():
            g = self.net.grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * g * g
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)
            self.net.params[k] -= (self.lr * m_hat
                                   / (np.sqrt(v_hat) + self.eps))


def mse_grad(out, targets):
    """dL/dout for L = 0.5 * mean((out - targets)**2).

    The mean is over all B*n_out elements, so dout = diff / (B*n_out).
    Returns (loss, dout).
    """
    out = np.asarray(out, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    diff = out - targets
    return 0.5 * np.mean(diff ** 2), diff / diff.size


def masked_mse_grad(out, actions, targets):
    """dL/dout for L = 0.5 * mean_b (out[b, actions[b]] - targets[b])**2.

    Only the chosen action's output carries error (Q-learning regression).
    Returns (loss, dout); dout is zero except at the chosen actions and
    carries the 1/B averaging.
    """
    out = np.asarray(out, dtype=np.float64)
    actions = np.asarray(actions)
    targets = np.asarray(targets, dtype=np.float64)
    B = out.shape[0]
    if actions.shape != (B,) or targets.shape != (B,):
        raise ValueError("actions/targets must have shape (%d,), got %s/%s"
                         % (B, actions.shape, targets.shape))
    idx = (np.arange(B), actions)
    diff = out[idx] - targets
    dout = np.zeros_like(out)
    dout[idx] = diff / B
    return 0.5 * np.mean(diff ** 2), dout

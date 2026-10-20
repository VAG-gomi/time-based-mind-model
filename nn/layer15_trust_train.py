#!/usr/bin/env python3
"""layer15_trust_train.py -- Layer 1.5: fast trust buffer + IS
correction (WO-TBM-L15).

Differences from the committed Layer 1 (nn/layer1_trust_train.py,
the permanent negative record -- not edited):
  1. FastTrustReplayBuffer (segment tree, stratified sampling)
     replaces the naive O(N)-per-step buffer.
  2. Importance-sampling correction -- the Layer-1 omission:
     per-draw weights w_i = (N * P(i))**(-beta), normalized by
     the batch max (scale-down only), applied to the gradient.
     beta anneals linearly 0.4 -> 1.0 across the run's episodes.
Everything else (network, optimizer, world, protocol, constants)
is identical to Layer 1: alpha=0.6, capacity 10000, batch 32,
hidden=32, lr=1e-3.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import (NeuralQ, train_neural, PlainEncoder, N_INPUTS,
                      EPISODE_CAP, epsilon, env_step, PERIOD, OPEN_LEN,
                      GAMMA)
from trust_replay_fast import FastTrustReplayBuffer
from world import World
import random

BETA0, BETA1 = 0.4, 1.0


def beta_at(episode, total_episodes):
    """Linear beta schedule: exactly 0.4 at episode 0, exactly 1.0
    at the final episode."""
    if total_episodes <= 1:
        return BETA1
    frac = episode / (total_episodes - 1)
    return BETA0 + (BETA1 - BETA0) * frac


def is_weights(probs, n_buffer, beta):
    """IS weights w_i = (N * P(i))**(-beta), normalized by the
    batch max so all weights are <= 1 (scale-down only)."""
    p = np.asarray(probs, dtype=float)
    w = (n_buffer * p) ** (-beta)
    return w / w.max()


class Trust15NeuralQ(NeuralQ):
    """NeuralQ with the fast trust buffer and IS-corrected updates."""

    def __init__(self, seed, hidden=32, lr=1e-3, gamma=GAMMA,
                 buffer_size=10000, batch=32, target_sync=200,
                 n_inputs=N_INPUTS, alpha=0.6, beta=BETA0):
        from agent_nn import MLP, Adam
        self.rng = random.Random(seed)
        self.nprng = np.random.RandomState(seed)
        self.net = MLP([n_inputs, hidden, 5], seed=seed)
        self.target = MLP([n_inputs, hidden, 5], seed=seed)
        self.target.set_params(self.net.get_params())
        self.opt = Adam(self.net, lr=lr)
        self.gamma = gamma
        self.batch = batch
        self.target_sync = target_sync
        self.buffer = FastTrustReplayBuffer(buffer_size, alpha=alpha)
        self.beta = beta
        self.updates = 0

    def train_step(self):
        """One minibatch update: stratified trust sampling + IS."""
        if len(self.buffer) < self.batch:
            return None
        b, idx, probs = self.buffer.sample(self.batch, self.nprng)
        w = is_weights(probs, len(self.buffer), self.beta)
        S = np.stack([t[0] for t in b])
        A = np.array([t[1] for t in b])
        R = np.array([t[2] for t in b])
        S2 = np.stack([t[3] for t in b])
        D = np.array([t[4] for t in b], dtype=float)
        q_next = self.target.forward(S2)
        y = R + self.gamma * (1.0 - D) * q_next.max(axis=1)
        out = self.net.forward(S)
        td_errors = y - out[np.arange(self.batch), A]
        from agent_nn import masked_mse_grad
        _loss, dout = masked_mse_grad(out, A, y)
        dout *= w[:, None]  # IS correction (per-sample gradient scale)
        loss = float(0.5 * np.mean(w * td_errors ** 2))
        self.net.backward(dout)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.set_params(self.net.get_params())
        self.buffer.update_trust(idx, td_errors)
        return loss


def train_trust15(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                  lr=1e-3, encoder=None, alpha=0.6):
    """Train with fast trust replay + IS. Same signature and loop
    as Layer 1's train_trust, plus the per-episode beta anneal."""
    if encoder is None:
        encoder = PlainEncoder()
    agent = Trust15NeuralQ(seed, hidden=hidden, lr=lr,
                           n_inputs=encoder.n_inputs, alpha=alpha)
    rng = random.Random(seed + 1000)
    stats = []
    for ep in range(episodes):
        agent.beta = beta_at(ep, episodes)
        w = World(period=PERIOD, open_len=OPEN_LEN)
        w.reset(start_tick=rng.randrange(PERIOD))
        encoder.reset(w)
        x = encoder.encode(w)
        eps = epsilon(ep)
        ticks = 0
        crushes = 0
        reached = False
        losses = []
        while ticks < cap:
            a = agent.act(x, eps)
            r, reached, crushed = env_step(w, a)
            crushes += crushed
            x2 = encoder.encode(w)
            agent.buffer.add((x, a, r, x2, reached))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            x = x2
            ticks += 1
            if reached:
                break
        mean_loss = (sum(losses) / len(losses)) if losses else None
        stats.append((ticks, reached, crushes, mean_loss))
    return agent, stats


def run_checks():
    """Smoke test: 20 episodes run without error."""
    agent, stats = train_trust15(0, episodes=20)
    assert len(stats) == 20
    assert len(agent.buffer) > 0
    print("layer1.5 smoke check: PASS (20 episodes, fast buffer + IS)")


if __name__ == "__main__":
    run_checks()

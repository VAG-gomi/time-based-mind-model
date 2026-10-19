#!/usr/bin/env python3
"""layer1_trust_train.py -- Layer 1: trust-weighted replay in training.

Plugs TrustReplayBuffer into the NeuralQ training loop. Measures
whether trust-weighted sampling reaches the same performance in
fewer episodes (sample efficiency) vs the uniform baseline.

Design: TrustNeuralQ subclasses NeuralQ, overriding only the buffer
and train_step. The network, optimizer, world, and protocol are
identical. If trust-weighting doesn't help, we stop here.
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
from trust_replay import TrustReplayBuffer
from world import World
import random


class TrustNeuralQ(NeuralQ):
    """NeuralQ with trust-weighted replay instead of uniform."""

    def __init__(self, seed, hidden=32, lr=1e-3, gamma=GAMMA,
                 buffer_size=10000, batch=32, target_sync=200,
                 n_inputs=N_INPUTS, alpha=0.6):
        # Bypass NeuralQ.__init__'s buffer; set everything else the same.
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
        self.buffer = TrustReplayBuffer(buffer_size, alpha=alpha)
        self.updates = 0

    def train_step(self):
        """One minibatch update with trust-weighted sampling."""
        if len(self.buffer) < self.batch:
            return None
        b, idx = self.buffer.sample(self.batch, self.nprng)
        S = np.stack([t[0] for t in b])
        A = np.array([t[1] for t in b])
        R = np.array([t[2] for t in b])
        S2 = np.stack([t[3] for t in b])
        D = np.array([t[4] for t in b], dtype=float)
        q_next = self.target.forward(S2)
        y = R + self.gamma * (1.0 - D) * q_next.max(axis=1)
        out = self.net.forward(S)
        # TD-errors for trust updates.
        td_errors = y - out[np.arange(self.batch), A]
        from agent_nn import masked_mse_grad
        loss, dout = masked_mse_grad(out, A, y)
        self.net.backward(dout)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.set_params(self.net.get_params())
        # Update trust scores with the new TD-errors.
        self.buffer.update_trust(idx, td_errors)
        return float(loss)


def train_trust(seed, hidden=32, episodes=2000, cap=EPISODE_CAP,
                lr=1e-3, encoder=None, alpha=0.6):
    """Train with trust-weighted replay. Same signature as train_neural."""
    if encoder is None:
        encoder = PlainEncoder()
    agent = TrustNeuralQ(seed, hidden=hidden, lr=lr,
                         n_inputs=encoder.n_inputs, alpha=alpha)
    rng = random.Random(seed + 1000)
    stats = []
    for ep in range(episodes):
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
    agent, stats = train_trust(0, episodes=20)
    assert len(stats) == 20
    assert len(agent.buffer) > 0
    print("layer1 smoke check: PASS (20 episodes, trust buffer active)")


if __name__ == "__main__":
    run_checks()

#!/usr/bin/env python3
"""layer15_beta1_train.py -- WO-TBM-PC2 diagnostic variant.

Trust-1.5 with beta == 1.0 from episode 0 (no anneal). Every
other element is the frozen design at 7eaca1e, reused by import
(Trust15NeuralQ, FastTrustReplayBuffer via the diagnostic
recording subclass, same loop structure as train_trust15).
The ONLY behavioral difference from the frozen trainer is the
beta line in the episode loop: agent.beta = 1.0 always.

DIAGNOSTIC ONLY, non-confirmatory (same seeds as the rematch).
The recording buffer consumes no extra RNG (delegates sampling
to the frozen implementation), so the ESS trace describes the
run itself.
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import (PlainEncoder, EPISODE_CAP, epsilon, env_step,
                      PERIOD, OPEN_LEN)
from diag_trust15 import DiagFastBuffer
from layer15_trust_train import Trust15NeuralQ
from world import World


def train_trust15_beta1(seed, hidden=32, episodes=1000, cap=EPISODE_CAP,
                        lr=1e-3, encoder=None):
    """Mirror of train_trust15 with beta fixed at 1.0, plus a
    per-episode ESS trace (list aligned with stats)."""
    if encoder is None:
        encoder = PlainEncoder()
    agent = Trust15NeuralQ(seed, hidden=hidden, lr=lr,
                           n_inputs=encoder.n_inputs)
    agent.buffer = DiagFastBuffer(10000, alpha=0.6)
    rng = random.Random(seed + 1000)
    stats = []
    ess_trace = []
    for ep in range(episodes):
        agent.beta = 1.0  # <-- the PC2 toggle (frozen: beta_at anneal)
        w = World(period=PERIOD, open_len=OPEN_LEN)
        w.reset(start_tick=rng.randrange(PERIOD))
        encoder.reset(w)
        x = encoder.encode(w)
        eps = epsilon(ep)
        ticks = 0
        crushes = 0
        reached = False
        losses = []
        ess_sum = 0.0
        n_batch = 0
        while ticks < cap:
            a = agent.act(x, eps)
            r, reached, crushed = env_step(w, a)
            crushes += crushed
            x2 = encoder.encode(w)
            agent.buffer.add((x, a, r, x2, reached))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
                p = agent.buffer.last_probs
                wgt = (len(agent.buffer) * p) ** (-agent.beta)
                ess_sum += float(wgt.sum() ** 2 / np.sum(wgt ** 2))
                n_batch += 1
            x = x2
            ticks += 1
            if reached:
                break
        mean_loss = (sum(losses) / len(losses)) if losses else None
        stats.append((ticks, reached, crushes, mean_loss))
        ess_trace.append((ess_sum / n_batch) if n_batch else None)
    return agent, stats, ess_trace

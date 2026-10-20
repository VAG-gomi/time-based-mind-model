#!/usr/bin/env python3
"""layer15_median_train.py -- WO-TBM-PC4 diagnostic trainer.

Trust-1.5 exactly as frozen at 7eaca1e (beta anneal 0.4 -> 1.0
intact), except the buffer is MedianInsertDiagBuffer
(median-priority insertion instead of max-priority). Returns
per-episode (mean sampled age, n_batches) alongside stats for
the manipulation check. DIAGNOSTIC ONLY, non-confirmatory.
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
from layer15_trust_train import Trust15NeuralQ, beta_at
from trust_replay_median import MedianInsertDiagBuffer
from world import World


def train_trust15_median(seed, hidden=32, episodes=1000,
                         cap=EPISODE_CAP, lr=1e-3, encoder=None):
    if encoder is None:
        encoder = PlainEncoder()
    agent = Trust15NeuralQ(seed, hidden=hidden, lr=lr,
                           n_inputs=encoder.n_inputs)
    agent.buffer = MedianInsertDiagBuffer(10000, alpha=0.6)
    rng = random.Random(seed + 1000)
    stats = []
    age_trace = []
    for ep in range(episodes):
        agent.beta = beta_at(ep, episodes)  # frozen anneal, intact
        w = World(period=PERIOD, open_len=OPEN_LEN)
        w.reset(start_tick=rng.randrange(PERIOD))
        encoder.reset(w)
        x = encoder.encode(w)
        eps = epsilon(ep)
        ticks = 0
        crushes = 0
        reached = False
        losses = []
        age_sum = 0.0
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
                age_sum += float(np.mean(agent.buffer.ages()))
                n_batch += 1
            x = x2
            ticks += 1
            if reached:
                break
        mean_loss = (sum(losses) / len(losses)) if losses else None
        stats.append((ticks, reached, crushes, mean_loss))
        age_trace.append(((age_sum / n_batch) if n_batch else None,
                          n_batch))
    return agent, stats, age_trace

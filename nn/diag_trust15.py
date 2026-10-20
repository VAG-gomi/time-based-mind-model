#!/usr/bin/env python3
"""diag_trust15.py -- WO-TBM-L15-diag Part B: instrumented traces.

Re-runs the rematch protocol exactly (uniform + trust15, seeds
200-202, 1000 episodes, 20/10) with recording buffer subclasses
that consume NO extra RNG: the fast buffer's sample delegates to
the frozen implementation; the uniform buffer's sample mirrors
ReplayBuffer.sample's single rng.choice call and records the
indices. If the instrumentation is faithful, each run's
(ticks, reached) sequence equals the committed rematch record
bit-for-bit -- the script verifies this per run and prints the
verdict. Traces therefore describe the actual rematch runs.

Per episode logged: rolling-100 success, buffer length, mean
within-batch repeats, mean sampled age (env steps since
insertion), IS effective sample size (trust15; recomputed
exactly from the recorded probs, beta and N), sampling
distribution entropy and top-decile priority mass at episode
end (uniform: the analytic values for a uniform distribution).

INSTRUMENTATION INVALIDATES TIMING: no wall-clock is recorded
or claimed from these runs.
"""

import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import (NeuralQ, PlainEncoder, EPISODE_CAP, epsilon,
                      env_step, PERIOD, OPEN_LEN, ReplayBuffer)
from layer15_trust_train import Trust15NeuralQ, beta_at
from trust_replay_fast import FastTrustReplayBuffer
from world import World

OUT = os.path.join(_HERE, "traces_trust15_diag.jsonl")
REMATCH = os.path.join(_HERE, "results_trust15.jsonl")
SEEDS = (200, 201, 202)
EPISODES = 1000


class DiagFastBuffer(FastTrustReplayBuffer):
    def __init__(self, capacity=10000, alpha=0.6):
        super(DiagFastBuffer, self).__init__(capacity, alpha=alpha)
        self.ins_step = np.zeros(capacity, dtype=int)
        self.clock = 0
        self.last_idx = None
        self.last_probs = None

    def add(self, item):
        self.ins_step[self.pos] = self.clock
        self.clock += 1
        super(DiagFastBuffer, self).add(item)

    def sample(self, n, rng):
        items, idx, probs = super(DiagFastBuffer, self).sample(n, rng)
        self.last_idx = idx
        self.last_probs = probs
        return items, idx, probs

    def ages(self):
        return self.clock - self.ins_step[self.last_idx]

    def dist_stats(self):
        filled = self.count
        leaves = self.tree[self.size:self.size + filled]
        p = leaves / leaves.sum()
        ent = float(-np.sum(p * np.log(p)))
        k = int(math.ceil(0.1 * filled))
        top = float(np.sort(p)[-k:].sum())
        return ent, top


class DiagUniformBuffer(ReplayBuffer):
    def __init__(self, capacity=10000):
        super(DiagUniformBuffer, self).__init__(capacity)
        self.ins = []
        self.clock = 0
        self.last_idx = None

    def add(self, item):
        if len(self.data) < self.capacity:
            self.ins.append(self.clock)
        else:
            self.ins[self.pos] = self.clock
        self.clock += 1
        super(DiagUniformBuffer, self).add(item)

    def sample(self, n, rng):
        idx = rng.choice(len(self.data), n, replace=False)
        self.last_idx = idx
        return [self.data[i] for i in idx]

    def ages(self):
        return np.array([self.clock - self.ins[i] for i in self.last_idx])

    def dist_stats(self):
        n = len(self.data)
        k = int(math.ceil(0.1 * n))
        return math.log(n), k / float(n)


def run_diag(arm, seed):
    encoder = PlainEncoder()
    if arm == "uniform":
        agent = NeuralQ(seed, hidden=32, lr=1e-3,
                        n_inputs=encoder.n_inputs)
        agent.buffer = DiagUniformBuffer(10000)
    else:
        agent = Trust15NeuralQ(seed, hidden=32, lr=1e-3,
                               n_inputs=encoder.n_inputs)
        agent.buffer = DiagFastBuffer(10000, alpha=0.6)
    rng = random.Random(seed + 1000)
    rows = []
    reached_hist = []
    for ep in range(EPISODES):
        if arm == "trust15":
            agent.beta = beta_at(ep, EPISODES)
        w = World(period=PERIOD, open_len=OPEN_LEN)
        w.reset(start_tick=rng.randrange(PERIOD))
        encoder.reset(w)
        x = encoder.encode(w)
        eps = epsilon(ep)
        ticks = 0
        reached = False
        rep_sum = ess_sum = age_sum = 0.0
        n_batch = 0
        while ticks < EPISODE_CAP:
            a = agent.act(x, eps)
            r, reached, crushed = env_step(w, a)
            x2 = encoder.encode(w)
            agent.buffer.add((x, a, r, x2, reached))
            loss = agent.train_step()
            if loss is not None:
                idx = agent.buffer.last_idx
                rep_sum += len(idx) - len(np.unique(idx))
                age_sum += float(np.mean(agent.buffer.ages()))
                if arm == "trust15":
                    p = agent.buffer.last_probs
                    wgt = (len(agent.buffer) * p) ** (-agent.beta)
                    ess_sum += float(wgt.sum() ** 2
                                     / np.sum(wgt ** 2))
                n_batch += 1
            x = x2
            ticks += 1
            if reached:
                break
        reached_hist.append(bool(reached))
        ent, top = agent.buffer.dist_stats()
        rows.append({
            "arm": arm, "seed": seed, "ep": ep,
            "reached": bool(reached), "ticks": ticks,
            "rolling100": float(np.mean(reached_hist[-100:])),
            "buflen": len(agent.buffer),
            "mean_repeats": (rep_sum / n_batch) if n_batch else None,
            "mean_ess": ((ess_sum / n_batch)
                         if (n_batch and arm == "trust15") else None),
            "mean_age": (age_sum / n_batch) if n_batch else None,
            "entropy": ent, "topdecile_mass": top,
            "beta": (agent.beta if arm == "trust15" else None),
        })
    return rows


def main():
    rematch = {}
    with open(REMATCH) as f:
        for line in f:
            r = json.loads(line)
            rematch[(r["arm"], r["seed"])] = r
    existing = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                existing.setdefault((r["arm"], r["seed"]), []).append(r)
    all_match = True
    with open(OUT, "a") as f:
        for seed in SEEDS:
            for arm in ("uniform", "trust15"):
                rec = rematch[(arm, seed)]
                if len(existing.get((arm, seed), [])) == EPISODES:
                    rows = sorted(existing[(arm, seed)],
                                  key=lambda r: r["ep"])
                    match = ([r["ticks"] for r in rows] == rec["ticks"]
                             and [r["reached"] for r in rows]
                             == rec["reached"])
                    all_match = all_match and match
                    print("%s seed %d: already traced; trajectory vs "
                          "rematch record: %s"
                          % (arm, seed, "MATCH" if match else "MISMATCH"),
                          flush=True)
                    continue
                rows = run_diag(arm, seed)
                match = ([r["ticks"] for r in rows] == rec["ticks"]
                         and [r["reached"] for r in rows] == rec["reached"])
                all_match = all_match and match
                print("%s seed %d: trajectory vs rematch record: %s"
                      % (arm, seed, "MATCH" if match else "MISMATCH"),
                      flush=True)
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
    print("ALL TRAJECTORIES MATCH REMATCH:", all_match)


if __name__ == "__main__":
    main()

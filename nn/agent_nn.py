#!/usr/bin/env python3
"""agent_nn.py -- Q-learning arms for the time world (A4).

The plain arm sees only observe(): 101 inputs = 100 one-hot position +
1 door bit. No tick, no period, no history.

Two learners share the training/evaluation machinery:
- TabularQ: Q-table on (position, door) x actions. Learnability proof.
- NeuralQ: 101 -> H -> 5 MLP, experience replay, target network, SGD.

Rewards are identical for every arm: step -0.01, goal +1.0, crush -1.0.
A crush teleports the agent to (0,0) and the episode continues; the
penalty is attached to the transition on which the crush is observed,
so TD bootstrapping blames the action that entered the door.
"""

import random
from collections import deque

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from world import (World, ACTIONS, AGENT_START, GOAL_POS, DOOR_CELL,
                   GRID_SIZE, door_is_open)
from nnet import MLP, masked_mse_grad, Adam

ACTION_NAMES = ["up", "down", "left", "right", "stay"]
N_ACTIONS = len(ACTION_NAMES)
N_INPUTS = GRID_SIZE * GRID_SIZE + 1  # 101: one-hot position + door bit

PERIOD = 20
OPEN_LEN = 10
EPISODE_CAP = 200

R_STEP = -0.01
R_GOAL = 1.0
R_CRUSH = -1.0
GAMMA = 0.99


def encode_obs(world):
    """101-vector: one-hot agent position + door bit."""
    (x, y), door = world.observe()
    v = np.zeros(N_INPUTS)
    v[y * GRID_SIZE + x] = 1.0
    v[N_INPUTS - 1] = 1.0 if door else 0.0
    return v


def obs_index(world):
    """Discrete (position, door) key for the tabular learner."""
    (x, y), door = world.observe()
    return y * GRID_SIZE + x, 1 if door else 0


def env_step(world, action_idx):
    """Apply action; return (reward, reached_goal, crushed)."""
    n_log = len(world.log)
    world.step(ACTIONS[ACTION_NAMES[action_idx]])
    crushed = any(e[2] == "crushed" for e in world.log[n_log:])
    reached = world.agent == GOAL_POS
    r = R_STEP + (R_CRUSH if crushed else 0.0) + (R_GOAL if reached else 0.0)
    return r, reached, crushed


def epsilon(episode, eps_end=0.05, decay_episodes=1000):
    """Linear 1.0 -> eps_end over decay_episodes, then flat."""
    return max(eps_end, 1.0 - (1.0 - eps_end) * episode / decay_episodes)


class ReplayBuffer(object):
    """Circular buffer with O(1) add."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def add(self, item):
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, n, rng):
        idx = rng.choice(len(self.data), n, replace=False)
        return [self.data[i] for i in idx]

    def __len__(self):
        return len(self.data)


class TabularQ(object):
    """Q-table on (position, door) x actions. Learnability proof.

    Uses optimistic initialization (init_value=1.0, the max return) to
    drive systematic exploration: untried actions look good, so the
    agent tries them instead of getting stuck in a self-loop like
    "stay" at a state whose Q-values never differentiated.
    """

    def __init__(self, seed, alpha=0.2, gamma=GAMMA, init_value=1.0):
        self.rng = random.Random(seed)
        self.Q = np.full((GRID_SIZE * GRID_SIZE, 2, N_ACTIONS), init_value,
                         dtype=float)
        self.alpha = alpha
        self.gamma = gamma

    def act(self, key, eps):
        if self.rng.random() < eps:
            return self.rng.randrange(N_ACTIONS)
        p, d = key
        return int(np.argmax(self.Q[p, d]))

    def update(self, key, a, r, key2, done):
        p, d = key
        p2, d2 = key2
        target = r if done else r + self.gamma * np.max(self.Q[p2, d2])
        self.Q[p, d, a] += self.alpha * (target - self.Q[p, d, a])


class NeuralQ(object):
    """101 -> H -> 5 MLP Q-learner with replay, target net, and Adam.

    SGD was tried first: lr=0.1 learned then oscillated (success
    0.11 -> 0.58 -> 0.09 -> 0.00 -> 0.43 over 2000 episodes), while
    lr=0.02 and lr=0.005 never learned at all. Adam is needed for the
    non-stationary Q targets.
    """

    def __init__(self, seed, hidden=32, lr=1e-3, gamma=GAMMA,
                 buffer_size=10000, batch=32, target_sync=200):
        self.rng = random.Random(seed)
        self.nprng = np.random.RandomState(seed)
        self.net = MLP([N_INPUTS, hidden, N_ACTIONS], seed=seed)
        self.target = MLP([N_INPUTS, hidden, N_ACTIONS], seed=seed)
        self.target.set_params(self.net.get_params())
        self.opt = Adam(self.net, lr=lr)
        self.gamma = gamma
        self.batch = batch
        self.target_sync = target_sync
        self.buffer = ReplayBuffer(buffer_size)
        self.updates = 0

    def act(self, x, eps):
        if self.rng.random() < eps:
            return self.rng.randrange(N_ACTIONS)
        return int(np.argmax(self.net.forward(x[None, :])[0]))

    def train_step(self):
        """One minibatch Q-learning update. Returns the batch loss."""
        if len(self.buffer) < self.batch:
            return None
        b = self.buffer.sample(self.batch, self.nprng)
        S = np.stack([t[0] for t in b])
        A = np.array([t[1] for t in b])
        R = np.array([t[2] for t in b])
        S2 = np.stack([t[3] for t in b])
        D = np.array([t[4] for t in b], dtype=float)
        q_next = self.target.forward(S2)
        y = R + self.gamma * (1.0 - D) * q_next.max(axis=1)
        out = self.net.forward(S)
        loss, dout = masked_mse_grad(out, A, y)
        self.net.backward(dout)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.set_params(self.net.get_params())
        return float(loss)


def train_tabular(seed, episodes=2000, cap=EPISODE_CAP):
    """Train the tabular learner. Returns (agent, per-episode stats)."""
    agent = TabularQ(seed)
    rng = random.Random(seed + 1000)
    stats = []
    for ep in range(episodes):
        w = World(period=PERIOD, open_len=OPEN_LEN)
        w.reset(start_tick=rng.randrange(PERIOD))
        key = obs_index(w)
        eps = epsilon(ep)
        ticks = 0
        crushes = 0
        reached = False
        while ticks < cap:
            a = agent.act(key, eps)
            r, reached, crushed = env_step(w, a)
            crushes += crushed
            key2 = obs_index(w)
            agent.update(key, a, r, key2, reached)
            key = key2
            ticks += 1
            if reached:
                break
        stats.append((ticks, reached, crushes))
    return agent, stats


def train_neural(seed, hidden=32, episodes=2000, cap=EPISODE_CAP, lr=1e-3):
    """Train the neural learner (Adam). Returns (agent, per-episode stats)."""
    agent = NeuralQ(seed, hidden=hidden, lr=lr)
    rng = random.Random(seed + 1000)
    stats = []
    for ep in range(episodes):
        w = World(period=PERIOD, open_len=OPEN_LEN)
        w.reset(start_tick=rng.randrange(PERIOD))
        x = encode_obs(w)
        eps = epsilon(ep)
        ticks = 0
        crushes = 0
        reached = False
        losses = []
        while ticks < cap:
            a = agent.act(x, eps)
            r, reached, crushed = env_step(w, a)
            crushes += crushed
            x2 = encode_obs(w)
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


def make_greedy_policy(agent, kind):
    """Wrap a trained agent as policy_fn(world) -> action_idx."""
    if kind == "tabular":
        def policy(w):
            return agent.act(obs_index(w), 0.0)
    else:
        def policy(w):
            return agent.act(encode_obs(w), 0.0)
    return policy


def naive_policy(world):
    """Cross whenever the door is open; leave the door cell immediately."""
    x, y = world.agent
    if x < 4:
        return "right"
    if x == 4 and y < 5:
        return "down"
    if (x, y) == (4, 5):
        return "right" if world.observe()[1] else "stay"
    if (x, y) == DOOR_CELL:
        return "right"
    if x < 9:
        return "right"
    if y < 9:
        return "down"
    return "stay"


def time_aware_policy(world):
    """Cross only when the door is open now AND on the next tick."""
    x, y = world.agent
    if x < 4:
        return "right"
    if x == 4 and y < 5:
        return "down"
    if (x, y) == (4, 5):
        t = world.tick
        safe = (door_is_open(t, PERIOD, OPEN_LEN)
                and door_is_open(t + 1, PERIOD, OPEN_LEN))
        return "right" if safe else "stay"
    if (x, y) == DOOR_CELL:
        return "right"
    if x < 9:
        return "right"
    if y < 9:
        return "down"
    return "stay"


def evaluate_policy(policy_fn, repeats=3, cap=EPISODE_CAP):
    """Run policy_fn over all 20 start phases, `repeats` each.

    Returns rows of (start_phase, crushes, reached_goal, ticks).
    The learned greedy policies are deterministic, so repeats are
    identical by construction (a determinism sanity check).
    """
    rows = []
    for start in range(PERIOD):
        for _ in range(repeats):
            w = World(period=PERIOD, open_len=OPEN_LEN)
            w.reset(start_tick=start)
            ticks = 0
            crushes = 0
            reached = False
            while ticks < cap:
                r, reached, crushed = env_step(w, policy_fn(w))
                crushes += crushed
                ticks += 1
                if reached:
                    break
            rows.append((start, crushes, reached, ticks))
    return rows


def summarize(rows):
    """(crushes/episode, frac episodes with >=1 crush, success rate,
    mean ticks to goal over successes)."""
    n = len(rows)
    crush_rate = sum(r[1] for r in rows) / n
    frac_crushed = sum(1 for r in rows if r[1] > 0) / n
    success = sum(1 for r in rows if r[2]) / n
    ticks_ok = [r[3] for r in rows if r[2]]
    mean_ticks = sum(ticks_ok) / len(ticks_ok) if ticks_ok else None
    return crush_rate, frac_crushed, success, mean_ticks

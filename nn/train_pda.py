#!/usr/bin/env python3
"""train_pda.py -- P-Da training loop (WO-TBM-P-Da, Step 0/Step 1).

Additive only. agent_nn.py, agent_wrapped.py,
agent_mem.py, agent_controls.py, and
agent_generalization.py are NEVER edited; their
learner and encoders are reused unchanged.

Why this file exists: agent_nn.train_neural hardcodes
World(period=PERIOD, open_len=OPEN_LEN) from its
module globals. It cannot train on a multi-door world
as written. This module mirrors train_neural
line-for-line except for two stated changes:
  (1) the world instantiated per episode is
      PdaWorld(period, open_len, gaps) instead of
      World;
  (2) period/open_len are passed explicitly instead
      of read from agent_nn globals (equivalent,
      since the doors/luck runners set those globals
      to the same arm schedule before training).
Everything else -- NeuralQ learner, replay buffer,
target sync, epsilon schedule, rewards, crush
handling via env_step (which reads the world's log
for "crushed"; MultiDoorWorld logs (t, "agent",
"crushed") in the same format), RNG seeding,
episode count, cap -- is identical to train_neural.

Additive logging only: per-episode position-before-
action sequences are observed to count door-entry
events (same entry definition as evaluation: a
gap-cell occupation whose gap differs from the
previous entry event's gap; consecutive stays in
one gap count as one event). Logging does not affect
learning, rewards, or actions.

Returns per training: (agent, stats, encoder,
door_entry_counts, episodes_with_entry), where
stats are train_neural-format (ticks, reached,
crushes, mean_loss) per episode, door_entry_counts
maps each gap cell to total entry events across all
episodes, and episodes_with_entry maps each gap
cell to the number of episodes with at least one
entry event for that gap.
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import (NeuralQ, PlainEncoder, EPISODE_CAP, epsilon,
                      env_step)
from agent_mem import MemoryEncoder
from agent_wrapped import WrappedEncoder
from agent_generalization import (OracleIndEncoder, OracleNewEncoder,
                                  PhaseOnlyNewEncoder)
from world_pda import PdaWorld, W5


def _entry_events(positions, gaps):
    """Entry events from a position-before-action sequence.

    Same definition as doors_run.evaluate: a gap
    occupation is an entry event iff it differs from
    the previous entry event's gap.
    """
    entries = []
    for p in positions:
        if p in gaps and (not entries or entries[-1] != p):
            entries.append(p)
    return entries


def train_neural_multi(seed, period, open_len, gaps,
                       hidden=32, episodes=2000, cap=EPISODE_CAP,
                       lr=1e-3, encoder=None):
    """train_neural mirrored for a multi-door world.

    Identical to agent_nn.train_neural except the
    world class (PdaWorld) and explicit period/
    open_len (see module docstring).
    """
    if encoder is None:
        encoder = PlainEncoder()
    agent = NeuralQ(seed, hidden=hidden, lr=lr,
                    n_inputs=encoder.n_inputs)
    rng = random.Random(seed + 1000)
    stats = []
    door_entry_counts = {g: 0 for g in gaps}
    episodes_with_entry = {g: 0 for g in gaps}
    for ep in range(episodes):
        w = PdaWorld(period, open_len, gaps)
        w.reset(start_tick=rng.randrange(period))
        encoder.reset(w)
        x = encoder.encode(w)
        eps = epsilon(ep)
        ticks = 0
        crushes = 0
        reached = False
        losses = []
        positions = []
        while ticks < cap:
            positions.append(w.agent)
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
        entries = _entry_events(positions, set(gaps))
        for g in entries:
            door_entry_counts[g] += 1
        for g in set(entries):
            episodes_with_entry[g] += 1
    return agent, stats, door_entry_counts, episodes_with_entry


def train_plain_pda(seed, period, open_len, gaps=W5,
                    hidden=32, episodes=2000, cap=EPISODE_CAP,
                    lr=1e-3):
    agent, stats, dec, ewe = train_neural_multi(
        seed, period, open_len, gaps, hidden=hidden,
        episodes=episodes, cap=cap, lr=lr,
        encoder=PlainEncoder())
    return agent, stats, PlainEncoder(), dec, ewe


def train_memory_pda(seed, period, open_len, gaps=W5,
                     hidden=32, episodes=2000, cap=EPISODE_CAP,
                     lr=1e-3):
    agent, stats, dec, ewe = train_neural_multi(
        seed, period, open_len, gaps, hidden=hidden,
        episodes=episodes, cap=cap, lr=lr,
        encoder=MemoryEncoder())
    return agent, stats, MemoryEncoder(), dec, ewe


def train_wrapped_pda(seed, period, open_len, gaps=W5,
                      hidden=32, episodes=2000, cap=EPISODE_CAP,
                      lr=1e-3):
    encoder = WrappedEncoder()
    agent, stats, dec, ewe = train_neural_multi(
        seed, period, open_len, gaps, hidden=hidden,
        episodes=episodes, cap=cap, lr=lr, encoder=encoder)
    return agent, stats, encoder, dec, ewe


def train_oracle_ind_pda(seed, period, open_len, gaps=W5,
                         hidden=32, episodes=2000, cap=EPISODE_CAP,
                         lr=1e-3):
    agent, stats, dec, ewe = train_neural_multi(
        seed, period, open_len, gaps, hidden=hidden,
        episodes=episodes, cap=cap, lr=lr,
        encoder=OracleIndEncoder())
    return agent, stats, OracleIndEncoder(), dec, ewe


def train_oracle_new_pda(seed, period, open_len, gaps=W5,
                         hidden=32, episodes=2000, cap=EPISODE_CAP,
                         lr=1e-3):
    agent, stats, dec, ewe = train_neural_multi(
        seed, period, open_len, gaps, hidden=hidden,
        episodes=episodes, cap=cap, lr=lr,
        encoder=OracleNewEncoder(period, open_len))
    return agent, stats, OracleNewEncoder(period, open_len), dec, ewe


def train_phase_only_new_pda(seed, period, open_len, gaps=W5,
                             hidden=32, episodes=2000, cap=EPISODE_CAP,
                             lr=1e-3):
    agent, stats, dec, ewe = train_neural_multi(
        seed, period, open_len, gaps, hidden=hidden,
        episodes=episodes, cap=cap, lr=lr,
        encoder=PhaseOnlyNewEncoder(period))
    return agent, stats, PhaseOnlyNewEncoder(period), dec, ewe


TRAINERS = {
    "wrapped": train_wrapped_pda,
    "plain": train_plain_pda,
    "memory": train_memory_pda,
    "oracle_ind": train_oracle_ind_pda,
    "oracle_new": train_oracle_new_pda,
    "phase_only_new": train_phase_only_new_pda,
}

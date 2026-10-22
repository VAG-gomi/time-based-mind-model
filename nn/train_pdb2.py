#!/usr/bin/env python3
"""train_pdb2.py -- P-Db2 training loop (WO-TBM-P-Db2, Step 0/Step 1).

Additive only. agent_nn.py, agent_wrapped.py,
agent_mem.py, agent_controls.py,
agent_generalization.py, and train_pda.py are NEVER
edited; their learner and encoders are reused
unchanged, and train_pda.py's original functions are
untouched (this module is a separate mirror).

Why this file exists: train_pda.train_neural_multi
instantiates PdaWorld(period, open_len, gaps), a
shared-schedule multi-door world. It cannot train on
a per-door-schedule world as written. This module
mirrors train_neural_multi line-for-line except for
two stated changes:
  (1) the world instantiated per episode is
      SchedDoorWorld(schedules) with the §2 TRAIN
      schedule table for the arm's protocol,
      instead of PdaWorld;
  (2) the schedule argument is that TRAIN table
      (the familiar period/open_len are the table's
      familiar entry, so rng.randrange(period) uses
      the familiar period exactly as before).
Everything else -- NeuralQ learner, replay buffer,
target sync, epsilon schedule, rewards, crush
handling via env_step (which reads the world's log
for "crushed"; SchedDoorWorld logs (t, "agent",
"crushed") in the same format), RNG seeding,
episode count, cap -- is identical to
train_neural_multi / train_neural.

Additive logging only: per-episode position-before-
action sequences are observed to count door-entry
events (same entry definition as evaluation: a
gap-cell occupation whose gap differs from the
previous entry event's gap; consecutive stays in
one gap count as one event). Logging does not affect
learning, rewards, or actions.

Returns per training: (agent, stats, encoder,
door_entry_counts, episodes_with_entry,
open_fractions), where stats are train_neural-format
(ticks, reached, crushes, mean_loss) per episode,
door_entry_counts maps each gap cell to total entry
events across all episodes, episodes_with_entry maps
each gap cell to the number of episodes with at
least one entry event for that gap, and
open_fractions maps each gap cell to open_len /
period from the TRAIN schedule table.
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
from world_doors_sched import FAMILIAR, SchedDoorWorld
from world_pdb2 import TRAIN_20, TRAIN_30, open_fractions


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


def train_neural_sched(seed, schedules, hidden=32, episodes=2000,
                       cap=EPISODE_CAP, lr=1e-3, encoder=None):
    """train_neural_multi mirrored for a schedule-varied world.

    Identical to train_pda.train_neural_multi except
    the world class (SchedDoorWorld with the TRAIN
    schedule table) and the schedule argument (see
    module docstring).
    """
    if encoder is None:
        encoder = PlainEncoder()
    agent = NeuralQ(seed, hidden=hidden, lr=lr,
                    n_inputs=encoder.n_inputs)
    period = schedules[FAMILIAR][0]
    rng = random.Random(seed + 1000)
    stats = []
    gaps = set(schedules)
    door_entry_counts = {g: 0 for g in gaps}
    episodes_with_entry = {g: 0 for g in gaps}
    for ep in range(episodes):
        w = SchedDoorWorld(schedules)
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
        entries = _entry_events(positions, gaps)
        for g in entries:
            door_entry_counts[g] += 1
        for g in set(entries):
            episodes_with_entry[g] += 1
    return (agent, stats, door_entry_counts, episodes_with_entry,
            open_fractions(schedules))


def _familiar_period_open_len(schedules):
    period, open_len, _offset = schedules[FAMILIAR]
    return period, open_len


def train_plain_pdb2(seed, schedules, hidden=32, episodes=2000,
                     cap=EPISODE_CAP, lr=1e-3):
    agent, stats, dec, ewe, frac = train_neural_sched(
        seed, schedules, hidden=hidden, episodes=episodes, cap=cap,
        lr=lr, encoder=PlainEncoder())
    return agent, stats, PlainEncoder(), dec, ewe, frac


def train_memory_pdb2(seed, schedules, hidden=32, episodes=2000,
                      cap=EPISODE_CAP, lr=1e-3):
    agent, stats, dec, ewe, frac = train_neural_sched(
        seed, schedules, hidden=hidden, episodes=episodes, cap=cap,
        lr=lr, encoder=MemoryEncoder())
    return agent, stats, MemoryEncoder(), dec, ewe, frac


def train_wrapped_pdb2(seed, schedules, hidden=32, episodes=2000,
                       cap=EPISODE_CAP, lr=1e-3):
    encoder = WrappedEncoder()
    agent, stats, dec, ewe, frac = train_neural_sched(
        seed, schedules, hidden=hidden, episodes=episodes, cap=cap,
        lr=lr, encoder=encoder)
    return agent, stats, encoder, dec, ewe, frac


def train_oracle_ind_pdb2(seed, schedules, hidden=32, episodes=2000,
                          cap=EPISODE_CAP, lr=1e-3):
    agent, stats, dec, ewe, frac = train_neural_sched(
        seed, schedules, hidden=hidden, episodes=episodes, cap=cap,
        lr=lr, encoder=OracleIndEncoder())
    return agent, stats, OracleIndEncoder(), dec, ewe, frac


def train_oracle_new_pdb2(seed, schedules, hidden=32, episodes=2000,
                          cap=EPISODE_CAP, lr=1e-3):
    period, open_len = _familiar_period_open_len(schedules)
    agent, stats, dec, ewe, frac = train_neural_sched(
        seed, schedules, hidden=hidden, episodes=episodes, cap=cap,
        lr=lr, encoder=OracleNewEncoder(period, open_len))
    return (agent, stats, OracleNewEncoder(period, open_len), dec,
            ewe, frac)


def train_phase_only_new_pdb2(seed, schedules, hidden=32,
                              episodes=2000, cap=EPISODE_CAP, lr=1e-3):
    period, _open_len = _familiar_period_open_len(schedules)
    agent, stats, dec, ewe, frac = train_neural_sched(
        seed, schedules, hidden=hidden, episodes=episodes, cap=cap,
        lr=lr, encoder=PhaseOnlyNewEncoder(period))
    return (agent, stats, PhaseOnlyNewEncoder(period), dec, ewe,
            frac)


TRAINERS = {
    "wrapped": train_wrapped_pdb2,
    "plain": train_plain_pdb2,
    "memory": train_memory_pdb2,
    "oracle_ind": train_oracle_ind_pdb2,
    "oracle_new": train_oracle_new_pdb2,
    "phase_only_new": train_phase_only_new_pdb2,
}

# Protocol schedule tables, for callers that address arms by
# protocol rather than by table (sanity script, Step 1 runner).
TRAIN_BY_PERIOD = {20: TRAIN_20, 30: TRAIN_30}

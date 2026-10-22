#!/usr/bin/env python3
"""train_md1.py -- MD1 moving-door training loop (WO-TBM-MD1, Step 0).

Additive only. agent_nn.py, agent_wrapped.py,
agent_mem.py, agent_controls.py,
agent_generalization.py, train_pda.py, and
train_pdb2.py are NEVER edited; their learner
and encoders are reused unchanged.

Why this file exists: train_pdb2.train_neural_sched
instantiates one schedule-varied world per training
run. MD1 needs a DIFFERENT gap row per episode,
drawn from the balanced row multiset (design note
§3). This module mirrors train_neural_sched
line-for-line except for two stated changes:
  (1) the world instantiated per episode is
      make_fixed_world(row) with row =
      row_multiset(seed)[episode], instead of
      SchedDoorWorld(schedules);
  (2) the schedule is the fixed 20/10 schedule for
      every episode (period = 20 for the start-tick
      RNG, exactly as the familiar period was used
      in train_neural_sched).
Everything else -- NeuralQ learner, replay buffer,
target sync, epsilon schedule, rewards, crush
handling via env_step (which reads the world's log
for "crushed"; MultiDoorWorld logs (t, "agent",
"crushed") in the same format), RNG seeding
(agent stream Random(seed), start-tick stream
Random(seed + 1000)), episode count, cap -- is
identical to train_neural_multi / train_neural_sched.

Per-episode records carry the episode's gap row,
so per-row training counts are verifiable from
the records (250 per trained row by construction;
Step 1 asserts this). Entry events use the same
definition as P-Db2: a gap-cell occupation whose
gap differs from the previous entry event's gap.
With one gap per episode, an episode contributes
at most one entry event for its row, so
door_entry_counts[row] == episodes_with_entry[row]
in MD1 training; both are recorded for format
compatibility with the P-Db2 records.

Returns per training: (agent, stats, encoder,
row_counts, door_entry_counts,
episodes_with_entry, row_reached), where stats are
train_neural-format (ticks, reached, crushes,
mean_loss, row) per episode, row_counts maps each
trained row to its episode count (250 by
construction), door_entry_counts /
episodes_with_entry map each trained row to entry
totals, and row_reached maps each trained row to
its reached count.
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
from world_md1 import (PERIOD, TRAINED_ROWS, gap_cell,
                       make_fixed_world, row_multiset)


def _entry_events(positions, gaps):
    """Entry events from a position-before-action sequence.

    Same definition as train_pdb2._entry_events.
    """
    entries = []
    for p in positions:
        if p in gaps and (not entries or entries[-1] != p):
            entries.append(p)
    return entries


def train_neural_md1(seed, hidden=32, episodes=2000,
                     cap=EPISODE_CAP, lr=1e-3, encoder=None):
    """train_neural_sched mirrored for a moving-gap world.

    Identical except the per-episode gap row (see
    module docstring). The row multiset is built once
    from its own stream (Random(seed + 5000)); it
    never touches the agent stream or the start-tick
    stream.
    """
    if encoder is None:
        encoder = PlainEncoder()
    agent = NeuralQ(seed, hidden=hidden, lr=lr,
                    n_inputs=encoder.n_inputs)
    start_rng = random.Random(seed + 1000)
    rows = row_multiset(seed)
    stats = []
    row_counts = {r: 0 for r in TRAINED_ROWS}
    door_entry_counts = {r: 0 for r in TRAINED_ROWS}
    episodes_with_entry = {r: 0 for r in TRAINED_ROWS}
    row_reached = {r: 0 for r in TRAINED_ROWS}
    for ep in range(episodes):
        row = rows[ep]
        w = make_fixed_world(row)
        w.reset(start_tick=start_rng.randrange(PERIOD))
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
        stats.append((ticks, reached, crushes, mean_loss, row))
        row_counts[row] += 1
        if reached:
            row_reached[row] += 1
        entries = _entry_events(positions, {gap_cell(row)})
        # entries are gap cells; map to the episode's row.
        for g in entries:
            door_entry_counts[g[1]] += 1
        for g in set(entries):
            episodes_with_entry[g[1]] += 1
    return (agent, stats, row_counts, door_entry_counts,
            episodes_with_entry, row_reached)


def train_plain_md1(seed, hidden=32, episodes=2000,
                    cap=EPISODE_CAP, lr=1e-3):
    agent, stats, rc, dec, ewe, rr = train_neural_md1(
        seed, hidden=hidden, episodes=episodes, cap=cap, lr=lr,
        encoder=PlainEncoder())
    return agent, stats, PlainEncoder(), rc, dec, ewe, rr


def train_memory_md1(seed, hidden=32, episodes=2000,
                     cap=EPISODE_CAP, lr=1e-3):
    agent, stats, rc, dec, ewe, rr = train_neural_md1(
        seed, hidden=hidden, episodes=episodes, cap=cap, lr=lr,
        encoder=MemoryEncoder())
    return agent, stats, MemoryEncoder(), rc, dec, ewe, rr


def train_wrapped_md1(seed, hidden=32, episodes=2000,
                      cap=EPISODE_CAP, lr=1e-3):
    encoder = WrappedEncoder()
    agent, stats, rc, dec, ewe, rr = train_neural_md1(
        seed, hidden=hidden, episodes=episodes, cap=cap, lr=lr,
        encoder=encoder)
    return agent, stats, encoder, rc, dec, ewe, rr


def train_oracle_ind_md1(seed, hidden=32, episodes=2000,
                         cap=EPISODE_CAP, lr=1e-3):
    agent, stats, rc, dec, ewe, rr = train_neural_md1(
        seed, hidden=hidden, episodes=episodes, cap=cap, lr=lr,
        encoder=OracleIndEncoder())
    return agent, stats, OracleIndEncoder(), rc, dec, ewe, rr


def train_oracle_new_md1(seed, hidden=32, episodes=2000,
                         cap=EPISODE_CAP, lr=1e-3):
    agent, stats, rc, dec, ewe, rr = train_neural_md1(
        seed, hidden=hidden, episodes=episodes, cap=cap, lr=lr,
        encoder=OracleNewEncoder(PERIOD, 10))
    return (agent, stats, OracleNewEncoder(PERIOD, 10), rc, dec,
            ewe, rr)


def train_phase_only_new_md1(seed, hidden=32, episodes=2000,
                             cap=EPISODE_CAP, lr=1e-3):
    agent, stats, rc, dec, ewe, rr = train_neural_md1(
        seed, hidden=hidden, episodes=episodes, cap=cap, lr=lr,
        encoder=PhaseOnlyNewEncoder(PERIOD))
    return (agent, stats, PhaseOnlyNewEncoder(PERIOD), rc, dec,
            ewe, rr)


TRAINERS = {
    "wrapped": train_wrapped_md1,
    "plain": train_plain_md1,
    "memory": train_memory_md1,
    "oracle_ind": train_oracle_ind_md1,
    "oracle_new": train_oracle_new_md1,
    "phase_only_new": train_phase_only_new_md1,
}

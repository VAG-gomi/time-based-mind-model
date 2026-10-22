#!/usr/bin/env python3
"""train_ft1.py -- FT1 fine-tuning loop + checkpoint snapshots
(WO-TBM-FT1, Step 0).

Additive only. agent_nn.py, agent_wrapped.py,
agent_mem.py, agent_generalization.py,
train_pda.py, and train_pdb2.py are NEVER edited;
their learner and encoders are reused unchanged,
and train_pdb2.py's trainers are the checkpoint
realisation mechanism (deterministic re-training
with frozen code).

Fine-tuning = continued training from a realised
P-Db2 checkpoint, NOT a new training. The
differences from train_pdb2.train_neural_sched
are exactly the fine-tuning-specific choices
fixed in the FT1 design note, and nothing else:

  (1) The agent is NOT constructed fresh: the
      checkpoint NeuralQ object is continued
      (weights, target network, Adam moments,
      RNG streams, update counter all carried).
      Constructing a fresh NeuralQ would reset
      everything and would not be fine-tuning.
  (2) The replay buffer is FRESH at fine-tune
      start (design-note choice: carried replay
      would mix trained-door experience into
      "brief (5,8) training" and confound the
      dose). The checkpoint buffer is discarded,
      not saved in the snapshot.
  (3) Epsilon is FLAT 0.05 (the end-of-training
      value), not the annealed epsilon(ep)
      schedule: the dose measures learning at
      deployment exploration level, not
      re-exploration. A re-anneal would confound
      the dose with an exploration schedule.
  (4) Episode count is 200 (the fixed dose), not
      2000.
  (5) The world is the fine-tuning world:
      (5,8) alone on the arm's training
      schedule, i.e. the P-Db2 F3 world
      (world_pdb2.f3_schedules), via
      Pdb2AbsentWorld. Observe bit = training
      schedule bit, as in F3.
  (6) Start-tick RNG is random.Random(seed +
      2000), NOT the checkpoint training's
      seed + 1000 stream: the fine-tune run is
      a new run, and reusing seed + 1000 would
      replay the checkpoint's first 200 start
      ticks. The checkpoint training's start-tick
      RNG is local to train_neural_sched and is
      not carried (it cannot be: it is not part
      of the agent object). The agent's own RNG
      streams (exploration draws, replay
      sampling) ARE carried, because they live
      on the agent object.

Encoder rule (design note, fixed): for wrapped,
the checkpoint encoder is carried into
fine-tuning, ScheduleModel and all: the
fine-tuning schedule is numerically identical
to the familiar schedule the model converged
on, so carried state is correct, not stale,
and a fresh model would inject blank-feature
warm-up episodes into the dose that are an
artifact of encoder reset, not of (5,8)
learning. For all other arms the encoders are
stateless (or per-episode only), so a fresh
encoder of the same type is identical;
finetune_encoder() embodies this choice.

Checkpoint snapshots (design note, fixed):
a checkpoint must persist across the Step 0
gate (Step 0 stops for instructor verification
AND the owner's ruling; Step 1 runs later),
and re-realising a checkpoint in Step 1 would
be a second full training for that agent,
breaking the 182 arm-run count (it would
become 204). So Step 0 saves each sanity
checkpoint as a snapshot file (NEW additive
files under nn/checkpoints_ft1/, never a
P-Db2 file), and Step 1 loads the 6 sanity
snapshots and realises only the remaining 16
checkpoints fresh (each verified against its
P-Db2 record as part of that single
realisation). This is a deliberate, disclosed
departure from P-Db2's no-weights convention
(P-Db2 stored no weights because re-training
was cheap and in-programme; FT1 checkpoints
must cross a gate). Snapshot fidelity is
audited at save time by a load-equality check
(bitwise-equal net/target/Adam/RNG state after
a save/load round-trip; no training involved,
so the audit is not an arm-run). The buffer is
deliberately NOT snapshotted (fresh at
fine-tune start, choice (2) above).
"""

import os
import pickle
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import (NeuralQ, PlainEncoder, EPISODE_CAP,
                      ReplayBuffer, env_step)
from agent_mem import MemoryEncoder
from agent_wrapped import WrappedEncoder
from agent_generalization import (OracleIndEncoder, OracleNewEncoder,
                                  PhaseOnlyNewEncoder)
from world_pdb2 import Pdb2AbsentWorld, open_fractions

# Fine-tuning dose (fixed, work order §0a) and epsilon (fixed).
FINE_TUNE_EPISODES = 200
FINE_TUNE_EPS = 0.05
# Start-tick stream for fine-tuning (see module docstring, (6)).
FINE_TUNE_RNG_OFFSET = 2000

# Encoder input sizes, by arm (for snapshot loading).
N_INPUTS = {
    "wrapped": 104,
    "plain": 101,
    "memory": 102,
    "oracle_ind": 104,
    "oracle_new": 104,
    "phase_only_new": 102,
}


def fresh_encoder(arm, period, open_len):
    """A fresh encoder of the arm's type (stateless, or fresh model).

    Used for Step 2 evaluation (fresh encoder state per eval
    world, design note) and for non-wrapped fine-tuning (where
    fresh == checkpoint encoder, encoders being stateless).
    For wrapped evaluation, a fresh encoder means a fresh
    ScheduleModel: schedule knowledge for that world is built
    from that world's own eval episodes only.
    """
    if arm == "wrapped":
        return WrappedEncoder()
    if arm == "plain":
        return PlainEncoder()
    if arm == "memory":
        return MemoryEncoder()
    if arm == "oracle_ind":
        return OracleIndEncoder()
    if arm == "oracle_new":
        return OracleNewEncoder(period, open_len)
    if arm == "phase_only_new":
        return PhaseOnlyNewEncoder(period)
    raise ValueError("unknown arm %r" % (arm,))


def finetune_encoder(arm, checkpoint_encoder, period, open_len):
    """Encoder for fine-tuning: carried model for wrapped, fresh else.

    Wrapped: the checkpoint encoder (with its converged
    ScheduleModel) is carried, per the module docstring.
    Other arms: a fresh encoder of the same type (identical
    behaviour: stateless, or per-episode state only).
    """
    if arm == "wrapped":
        return checkpoint_encoder
    return fresh_encoder(arm, period, open_len)


def _entry_events(positions, gaps):
    """Same entry definition as train_pdb2 / evaluation."""
    entries = []
    for p in positions:
        if p in gaps and (not entries or entries[-1] != p):
            entries.append(p)
    return entries


def finetune_neural(agent, encoder, schedules, seed,
                    episodes=FINE_TUNE_EPISODES, cap=EPISODE_CAP,
                    eps=FINE_TUNE_EPS):
    """Continue a checkpoint agent for `episodes` in a one-door world.

    agent is the checkpoint NeuralQ object, mutated in place
    (continued training). schedules is the fine-tuning world's
    schedule table (one gap: (5,8) on the training schedule);
    the clock is that schedule. Returns (stats,
    door_entry_counts, episodes_with_entry, open_fractions),
    where stats are train_neural-format (ticks, reached,
    crushes, mean_loss) per fine-tune episode.
    """
    # Fresh replay buffer at fine-tune start (choice (2)).
    agent.buffer = ReplayBuffer(10000)
    gaps = set(schedules)
    assert len(gaps) == 1, "fine-tuning world must have exactly one gap"
    gap = next(iter(gaps))
    period, open_len, _offset = schedules[gap]
    rng = random.Random(seed + FINE_TUNE_RNG_OFFSET)
    stats = []
    door_entry_counts = {g: 0 for g in gaps}
    episodes_with_entry = {g: 0 for g in gaps}
    for ep in range(episodes):
        w = Pdb2AbsentWorld(schedules, period, open_len)
        w.reset(start_tick=rng.randrange(period))
        encoder.reset(w)
        x = encoder.encode(w)
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
    return (stats, door_entry_counts, episodes_with_entry,
            open_fractions(schedules))


def save_checkpoint(agent, encoder, arm, seed, path):
    """Save a checkpoint snapshot (weights + optimiser + RNG + model).

    The replay buffer is deliberately NOT saved (fresh at
    fine-tune start). Returns the bundle (for auditing).
    """
    bundle = {
        "arm": arm,
        "seed": seed,
        "n_inputs": agent.net.sizes[0],
        "hidden": agent.net.sizes[1],
        "net": agent.net.get_params(),
        "target": agent.target.get_params(),
        "adam_m": dict(agent.opt.m),
        "adam_v": dict(agent.opt.v),
        "adam_t": agent.opt.t,
        "adam_lr": agent.opt.lr,
        "updates": agent.updates,
        "rng_state": agent.rng.getstate(),
        "nprng_state": agent.nprng.get_state(),
        "schedule_model": encoder.model if arm == "wrapped" else None,
    }
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    return bundle


def load_checkpoint(path, arm, period, open_len):
    """Load a checkpoint snapshot into a fresh agent + encoder.

    Returns (agent, encoder). The agent's weights, target,
    Adam moments, RNG streams, and update counter are the
    snapshotted ones; the buffer is fresh (empty), as at
    fine-tune start. For wrapped, the encoder carries the
    snapshotted ScheduleModel.
    """
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    assert bundle["arm"] == arm, "snapshot arm mismatch"
    agent = NeuralQ(bundle["seed"], hidden=bundle["hidden"],
                    lr=bundle["adam_lr"], n_inputs=bundle["n_inputs"])
    agent.net.set_params(bundle["net"])
    agent.target.set_params(bundle["target"])
    agent.opt.m = dict(bundle["adam_m"])
    agent.opt.v = dict(bundle["adam_v"])
    agent.opt.t = bundle["adam_t"]
    agent.opt.lr = bundle["adam_lr"]
    agent.updates = bundle["updates"]
    agent.rng.setstate(bundle["rng_state"])
    agent.nprng.set_state(bundle["nprng_state"])
    agent.buffer = ReplayBuffer(10000)
    encoder = fresh_encoder(arm, period, open_len)
    if arm == "wrapped":
        encoder.model = bundle["schedule_model"]
    return agent, encoder, bundle


def assert_load_equal(agent_a, agent_b):
    """Bitwise-equality audit between two agents (no training).

    Raises on any difference in net, target, Adam state,
    update counter, or RNG streams.
    """
    assert np.array_equal(agent_a.net.get_params(),
                          agent_b.net.get_params()), "net differs"
    assert np.array_equal(agent_a.target.get_params(),
                          agent_b.target.get_params()), "target differs"
    for k in agent_a.opt.m:
        assert np.array_equal(agent_a.opt.m[k], agent_b.opt.m[k]), \
            "adam m %s differs" % k
        assert np.array_equal(agent_a.opt.v[k], agent_b.opt.v[k]), \
            "adam v %s differs" % k
    assert agent_a.opt.t == agent_b.opt.t, "adam t differs"
    assert agent_a.updates == agent_b.updates, "updates differ"
    assert agent_a.rng.getstate() == agent_b.rng.getstate(), \
        "agent rng differs"
    sa, sb = agent_a.nprng.get_state(), agent_b.nprng.get_state()
    assert sa[0] == sb[0] and np.array_equal(sa[1], sb[1]) \
        and sa[2] == sb[2] and sa[3] == sb[3], "nprng differs"

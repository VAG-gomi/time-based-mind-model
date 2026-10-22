#!/usr/bin/env python3
"""md1_run.py -- MD1 greedy evaluation runner (WO-TBM-MD1, Step 0).

Additive only. No existing world, trainer, or
record file is edited. This module provides the
evaluation pieces Step 1 composes; it runs no
programme training and no programme evaluation on
import. The Step 0 sanity script imports evaluate()
for a labelled smoke test only.

evaluate() mirrors ft1_run.evaluate line-for-line
in structure (same record fields, same segment /
stall / revisit / entry definitions imported from
doors_run), except for the MD1 differences, all
stated here:
  (1) worlds are single-gap MD1 fixed worlds
      (make_fixed_world), not P-Db2 schedule
      worlds;
  (2) there are NO rival doors, so conflict and
      conflict_outcome are always None (kept as
      fields for format compatibility, never
      computed);
  (3) switches is always 0 (entry events for one
      gap number at most one);
  (4) arrival carries gap_open_at_arrival (the
      episode gap's door state), not a familiar
      clock bit;
  (5) the caller supplies a FRESH encoder per
      (agent, world) (FT1 Step 2 rule), persisting
      across that world's 20 episodes; for wrapped
      this means the ScheduleModel starts blank at
      each world and converges during the world's
      episodes, so blank-feature warm-up is a
      per-world-start effect, audited as in FT1.

Record schema (per episode): kind, world, gap_row,
start_phase, ticks, crushes, reached, return
(accumulated env_step reward total, FT1
convention), first_door, door_entries, switches,
stall, stall_kind, revisits, staging_order,
arrival, first_entry_index, post_entry_ticks,
post_entry_stall, post_entry_stall_kind,
blank_feature, conflict, conflict_outcome,
staging_wait, door_status_at_arrival,
door_status_at_entry, schedules, steps, arm,
seed, period, open_len. Step records carry
t, x, y, action, door, crushed, plus f101/f102/
f103 for wrapped arms (log_features).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step, PlainEncoder
from agent_mem import MemoryEncoder
from agent_wrapped import WrappedEncoder
from agent_generalization import (OracleIndEncoder, OracleNewEncoder,
                                  PhaseOnlyNewEncoder)
from doors_run import segments, stall_signature
from world_md1 import (PERIOD, OPEN_LEN, fixed_world_name,
                       make_fixed_world)


def fresh_encoder(arm):
    """Fresh encoder instance for one (agent, world) eval job.

    Wrapped gets a fresh WrappedEncoder (blank
    ScheduleModel); all other arms are stateless or
    per-episode-reset, but a fresh instance keeps
    the rule uniform and auditable.
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
        return OracleNewEncoder(PERIOD, OPEN_LEN)
    if arm == "phase_only_new":
        return PhaseOnlyNewEncoder(PERIOD)
    raise ValueError("unknown arm %r" % (arm,))


def _gap_key(row):
    return "5,%d" % (row,)


def _evaluate_episode(agent, encoder, wname, gap_row, start,
                      log_features=False):
    """One greedy episode on one MD1 gap row at one start phase.

    The encoder is reset for this episode by the
    caller-visible contract (evaluate / evaluate_m3
    call encoder.reset once per episode); the encoder
    object itself persists across a job's episodes.
    Returns one episode record (without arm / seed /
    period / open_len, which the caller adds).
    """
    gap = (5, gap_row)
    stage_cell = (4, gap_row)
    w = make_fixed_world(gap_row)
    w.reset(start_tick=start)
    encoder.reset(w)
    steps = []
    total = 0.0
    crushes = 0
    reached = False
    ticks = 0
    while ticks < EPISODE_CAP:
        x = encoder.encode(w)
        a = agent.act(x, 0.0)
        (px, py), door = w.observe()
        r, reached, crushed = env_step(w, a)
        rec = {"t": w.tick - 1, "x": px, "y": py,
               "action": ACTION_NAMES[a],
               "door": bool(door), "crushed": bool(crushed)}
        if log_features and len(x) >= 104:
            rec.update({"f101": float(x[101]),
                        "f102": float(x[102]),
                        "f103": float(x[103])})
        steps.append(rec)
        total += r
        crushes += int(crushed)
        ticks += 1
        if reached:
            break
    positions = [(s["x"], s["y"]) for s in steps]
    entries = []
    entry_ticks = []
    for s in steps:
        p = (s["x"], s["y"])
        if p == gap and (not entries or entries[-1] != p):
            entries.append(p)
            entry_ticks.append(s["t"])
    switches = sum(1 for i in range(1, len(entries))
                   if entries[i] != entries[i - 1])
    segs = segments(positions)
    seen = set()
    revisits = 0
    for c, _k in segs:
        if c in seen:
            revisits += 1
        seen.add(c)
    stall, stall_kind = stall_signature(positions, {gap})
    staging_order = []
    for p in positions:
        if p == stage_cell and p not in staging_order:
            staging_order.append(p)
    arrival_step = None
    for s in steps:
        if (s["x"], s["y"]) == stage_cell:
            arrival_step = s
            break
    if arrival_step is not None:
        at = arrival_step["t"]
        gap_open_arr = (at % PERIOD) < OPEN_LEN
        arrival = {"tick": at, "cell": list(stage_cell),
                   "arrival_door": list(gap),
                   "gap_open_at_arrival": bool(gap_open_arr)}
        status_arr = {_gap_key(gap_row): bool(gap_open_arr)}
    else:
        at = None
        arrival = None
        status_arr = None
    if entries:
        et = entry_ticks[0]
        status_entry = {_gap_key(gap_row):
                        bool((et % PERIOD) < OPEN_LEN)}
        wait = (et - at) if at is not None else None
    else:
        status_entry = None
        wait = None
    if entries:
        first_idx = positions.index(entries[0])
        post_positions = positions[first_idx + 1:]
        post_stall, post_stall_kind = stall_signature(
            post_positions, {gap})
    else:
        first_idx = None
        post_positions = []
        post_stall, post_stall_kind = False, None
    blank = bool(log_features and any(
        st.get("f103") == 0.0 for st in steps))
    return {
        "kind": "episode", "world": wname, "gap_row": gap_row,
        "start_phase": start, "ticks": ticks,
        "crushes": crushes, "reached": reached,
        "return": total, "first_door": (list(entries[0])
                                       if entries else None),
        "door_entries": [list(p) for p in entries],
        "switches": switches, "stall": stall,
        "stall_kind": stall_kind, "revisits": revisits,
        "staging_order": [list(p) for p in staging_order],
        "arrival": arrival,
        "first_entry_index": first_idx,
        "post_entry_ticks": len(post_positions),
        "post_entry_stall": post_stall,
        "post_entry_stall_kind": post_stall_kind,
        "blank_feature": blank,
        "conflict": None, "conflict_outcome": None,
        "staging_wait": wait,
        "door_status_at_arrival": status_arr,
        "door_status_at_entry": status_entry,
        "schedules": {_gap_key(gap_row): [PERIOD, OPEN_LEN, 0]},
        "steps": steps}


def evaluate(agent, encoder, wname, gap_row, log_features=False):
    """Greedy eval over all 20 start phases on one MD1 fixed world.

    The encoder persists across the job's 20
    episodes (reset per episode). Returns episode
    records (without arm / seed / period / open_len,
    which the caller adds).
    """
    return [_evaluate_episode(agent, encoder, wname, gap_row, start,
                               log_features=log_features)
            for start in range(PERIOD)]


def evaluate_m3(agent, encoder, arm, seed, log_features=False):
    """Greedy eval for one M3 job: 20 episodes, one per phase.

    Episode for start phase p uses gap row
    m3_row_sequence(arm, seed)[p] (trained rows
    only, design note §4). The encoder is fresh
    for the job and persists across its 20
    episodes (reset per episode, as for fixed
    worlds). Returns episode records with
    world "M3" and per-episode gap_row.
    """
    from world_md1 import m3_row_sequence
    rows = m3_row_sequence(arm, seed)
    return [_evaluate_episode(agent, encoder, "M3", row, phase,
                               log_features=log_features)
            for phase, row in enumerate(rows)]


def run_eval_job_m3(agent, arm, seed, out_path):
    """Evaluate one (agent, M3) job; append records to out_path.

    Resume-safe at the job level like run_eval_job:
    20 existing records for (arm, seed, "M3") skip
    the job. Otherwise all 20 episodes are evaluated
    with a fresh encoder and appended.
    """
    import json
    done = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                if (r.get("arm") == arm and r.get("seed") == seed
                        and r.get("world") == "M3"):
                    done += 1
    if done >= 20:
        return 0, True
    encoder = fresh_encoder(arm)
    eps = evaluate_m3(agent, encoder, arm, seed,
                      log_features=(arm == "wrapped"))
    with open(out_path, "a") as f:
        for r in eps:
            r["arm"] = arm
            r["seed"] = seed
            r["period"] = PERIOD
            r["open_len"] = OPEN_LEN
            f.write(json.dumps(r) + "\n")
    return len(eps), False


def run_eval_job(agent, arm, seed, wname, gap_row, out_path):
    """Evaluate one (agent, world) job; append records to out_path.

    Resume-safe at the job level: if out_path already
    holds 20 records for (arm, seed, wname), the job
    is skipped and (0, True) returned. Otherwise all
    20 episodes are evaluated with a fresh encoder
    and appended, returning (20, False).
    """
    import json
    done = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                if (r.get("arm") == arm and r.get("seed") == seed
                        and r.get("world") == wname):
                    done += 1
    if done >= 20:
        return 0, True
    encoder = fresh_encoder(arm)
    eps = evaluate(agent, encoder, wname, gap_row,
                   log_features=(arm == "wrapped"))
    with open(out_path, "a") as f:
        for r in eps:
            r["arm"] = arm
            r["seed"] = seed
            r["period"] = PERIOD
            r["open_len"] = OPEN_LEN
            f.write(json.dumps(r) + "\n")
    return len(eps), False

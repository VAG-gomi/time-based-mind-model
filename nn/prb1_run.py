#!/usr/bin/env python3
"""WO-TBM-PRB1 Step 1 runner: epsilon = 0.05 evaluation probe.

Design note: nn/decisions_prb1_design.md (Step 0 b67a9ef, gate
ruling e2f97a5: PROCEED to Step 1). Work order:
~/workspace/your_files/WO-TBM-PRB1.md (issued Step 0 only,
then gate ruling at the Step 0 gate).

Evaluation-only: NO training of any kind. Exactly 4 combos x
5 reps, nothing else:
  combo 0: memory/3 in F3 (20 phases/rep)
  combo 1: plain/0  in F3 (20 phases/rep)
  combo 2: oracle_new/101 in F3 (30 phases/rep)
  combo 3: memory/3 in F2 (20 phases/rep)
Total probe episodes: 450. Weights: FT1 fine-tuned snapshots
(checkpoints_ft1_finetuned/), loaded frozen, load-checked per
design note section 2 before any probe episode runs.

RNG protocol (design note section 4, implemented EXACTLY):
seed = 900_000 + 1_000 * combo_index + rep, rep 0..4; one
random.Random(seed) stream per rep, continuous across that
rep's episodes in increasing start-phase order, never reset
mid-rep; per step exactly one draw u = rng.random(); if
u < 0.05, exactly one second draw k = rng.randrange(5)
indexing ACTION_NAMES; otherwise greedy argmax (lowest
index on ties) with NO second draw. NO other draws consume
from the stream. CRITICAL: this runner NEVER calls
agent.act(x, 0.05) or any routine drawing from outside the
section 4 stream. The epsilon-greedy choice is implemented
here, in the runner, from the frozen network output
(np.argmax of net.forward) and the rep stream. The agent's
own RNG streams are never touched (act is never called),
and no numpy RNG draw appears anywhere in this file.

Encoder: fresh encoder per (agent, world, rep) via
train_ft1.fresh_encoder; encoder.reset(world) at every
episode start, exactly as in FT1's greedy evaluation
(ft1_run.evaluate). The episode loop below mirrors
ft1_run.evaluate line-for-line except for (a) the action
choice (rep-stream epsilon-greedy instead of greedy
act(x, 0.0)) and (b) the per-step epsilon_action flag.
Worlds are built by ft1_run.world_for, unchanged.

Resume-safe: results_prb1.jsonl is read at start; a rep
is complete only if it holds all phases exactly once (20
or 30 episodes). Complete reps are skipped. A partial rep
is dropped and re-run deterministically (no duplicates).
Records are flushed per completed rep.
"""

import json
import os
import pickle
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step
from doors_run import segments, stall_signature
from ft1_run import gap_key, world_for
from train_ft1 import fresh_encoder, load_checkpoint
from world_doors_sched import door_open_at
from world_pdb2 import TRAIN_20, TRAIN_30

OUT = os.path.join(_HERE, "results_prb1.jsonl")
FT_SNAP_DIR = os.path.join(_HERE, "checkpoints_ft1_finetuned")
SNAP_COMMIT = "9d1d6a0"
EPSILON = 0.05

# (combo_index, arm, seed, world, period, open_len, n_inputs)
COMBOS = [
    (0, "memory", 3, "F3", 20, 10, 102),
    (1, "plain", 0, "F3", 20, 10, 101),
    (2, "oracle_new", 101, "F3", 30, 14, 104),
    (3, "memory", 3, "F2", 20, 10, 102),
]

REPS = range(5)


def snap_path(arm, seed):
    return os.path.join(FT_SNAP_DIR, "%s_%d.pkl" % (arm, seed))


def load_check(arm, seed, n_inputs):
    """Design note section 2 load check, on the raw bundle.

    Returns (bundle, snapshot_identity). Raises on any failure;
    the caller does NOT evaluate a combo whose check raises.
    """
    path = snap_path(arm, seed)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    assert bundle["arm"] == arm, "snapshot arm mismatch: %r != %r" % (
        bundle["arm"], arm)
    assert bundle["seed"] == seed, "snapshot seed mismatch"
    assert bundle["n_inputs"] == n_inputs, "snapshot n_inputs mismatch"
    assert bundle["hidden"] == 32, "snapshot hidden != 32"
    identity = {"arm": arm, "seed": seed, "snapshot_file": path,
                "commit": SNAP_COMMIT,
                "ft1_fine_tune_choices": bundle.get("fine_tune_choices"),
                "snapshot_keys": sorted(bundle.keys())}
    return bundle, identity


def evaluate_eps(agent, encoder, period, open_len, wname, train, rng):
    """Epsilon-greedy eval over all start phases on one PRB1 world.

    Mirrors ft1_run.evaluate except the action choice: at each
    step, u = rng.random(); if u < 0.05, action = rng.randrange(5)
    (epsilon_action True); else greedy argmax of the frozen net
    (epsilon_action False, no second draw). rng is the rep stream
    and is NOT reset here; it runs continuously across episodes.
    """
    make_world, gaps, sched, _cp, _co = world_for(train, wname)
    gaps_set = set(gaps)
    stage_cells = set((4, y) for (_x, y) in gaps)
    is_conflict_world = wname in ("F1", "F2")
    episodes = []
    for start in range(period):
        w = make_world()
        w.reset(start_tick=start)
        encoder.reset(w)
        steps = []
        total = 0.0
        crushes = 0
        reached = False
        ticks = 0
        while ticks < EPISODE_CAP:
            x = encoder.encode(w)
            u = rng.random()
            if u < EPSILON:
                a = rng.randrange(5)
                eps_action = True
            else:
                a = int(np.argmax(agent.net.forward(x[None, :])[0]))
                eps_action = False
            (px, py), door = w.observe()
            r, reached, crushed = env_step(w, a)
            rec = {"t": w.tick - 1, "x": px, "y": py,
                   "action": ACTION_NAMES[a],
                   "door": bool(door), "crushed": bool(crushed),
                   "epsilon_action": eps_action}
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
            if p in gaps_set and (not entries or entries[-1] != p):
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
        stall, stall_kind = stall_signature(positions, gaps_set)
        staging_order = []
        for p in positions:
            if p in stage_cells and p not in staging_order:
                staging_order.append(p)
        arrival_step = None
        for s in steps:
            if (s["x"], s["y"]) in stage_cells:
                arrival_step = s
                break
        if arrival_step is not None:
            at = arrival_step["t"]
            arr_cell = (arrival_step["x"], arrival_step["y"])
            arr_door = (5, arr_cell[1])
            status_arr = {gap_key(g): door_open_at(at, *sched[g])
                          for g in gaps}
            clock_open = door_open_at(at, period, open_len, 0)
            arrival = {"tick": at, "cell": list(arr_cell),
                       "arrival_door": list(arr_door),
                       "clock_open_at_arrival": bool(clock_open)}
        else:
            at = None
            arr_door = None
            status_arr = None
            arrival = None
        if entries:
            et = entry_ticks[0]
            status_entry = {gap_key(g): door_open_at(et, *sched[g])
                            for g in gaps}
            wait = (et - at) if at is not None else None
        else:
            status_entry = None
            wait = None
        if not is_conflict_world or arrival is None:
            conflict = None
            outcome = None
        else:
            arr_closed = not door_open_at(at, *sched[arr_door])
            others_open = [g for g in gaps if g != arr_door
                           and door_open_at(at, *sched[g])]
            conflict = bool(arr_closed and others_open)
            if not conflict:
                outcome = None
            elif not entries:
                outcome = "other"
            elif entries[0] == arr_door:
                outcome = "waits-at-closed-door"
            elif entries[0] in others_open:
                outcome = "takes-open-door"
            else:
                outcome = "other"
        if entries:
            first_idx = positions.index(entries[0])
            post_positions = positions[first_idx + 1:]
            post_stall, post_stall_kind = stall_signature(
                post_positions, gaps_set)
        else:
            first_idx = None
            post_positions = []
            post_stall, post_stall_kind = False, None
        episodes.append({
            "kind": "episode", "world": wname,
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
            "blank_feature": False,
            "conflict": conflict, "conflict_outcome": outcome,
            "staging_wait": wait,
            "door_status_at_arrival": status_arr,
            "door_status_at_entry": status_entry,
            "schedules": {gap_key(g): list(sched[g]) for g in gaps},
            "steps": steps})
    return episodes


def load_done():
    """(completed reps, kept records): a rep counts only if it
    holds all phases exactly once; partial reps are dropped."""
    kept = []
    counts = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                kept.append(r)
    by_rep = {}
    for r in kept:
        by_rep.setdefault((r["arm"], r["seed"], r["world"], r["rep"]),
                          []).append(r)
    done = set()
    final = []
    for key, recs in by_rep.items():
        phases = [r["start_phase"] for r in recs]
        period = recs[0]["period"]
        if len(recs) == period and len(set(phases)) == period:
            done.add(key)
            final.extend(recs)
        else:
            print("prb1_run: dropping partial rep %s (%d records)"
                  % (key, len(recs)), flush=True)
    return done, final


def main():
    done, kept = load_done()
    if kept:
        print("prb1_run: resuming with %d kept records, %d complete "
              "reps" % (len(kept), len(done)), flush=True)
    # Rewrite the file with only complete reps (drops partials).
    with open(OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    out = open(OUT, "a")
    deviations = []
    for (ci, arm, seed, wname, period, open_len,
         n_inputs) in COMBOS:
        try:
            bundle, identity = load_check(arm, seed, n_inputs)
        except Exception as exc:
            print("prb1_run: LOAD CHECK FAILED %s/%d %s: %s"
                  % (arm, seed, wname, exc), flush=True)
            deviations.append((arm, seed, wname, str(exc)))
            continue
        agent, _enc, _b = load_checkpoint(snap_path(arm, seed), arm,
                                          period, open_len)
        # Bitwise audit: the loaded agent's net must equal the
        # checked bundle's net (load_checkpoint sets from bundle).
        assert np.array_equal(agent.net.get_params(), bundle["net"]), \
            "loaded net differs from checked bundle"
        print("prb1_run: load check PASS %s/%d %s (n_inputs=%d, "
              "hidden=32)" % (arm, seed, wname, n_inputs), flush=True)
        train = TRAIN_20 if period == 20 else TRAIN_30
        for rep in REPS:
            if (arm, seed, wname, rep) in done:
                print("prb1_run: skipping complete rep %s/%d %s "
                      "rep %d" % (arm, seed, wname, rep), flush=True)
                continue
            encoder = fresh_encoder(arm, period, open_len)
            rep_seed = 900_000 + 1_000 * ci + rep
            rng = random.Random(rep_seed)
            eps = evaluate_eps(agent, encoder, period, open_len,
                               wname, train, rng)
            for e in eps:
                e.update({"arm": arm, "seed": seed,
                          "period": period, "open_len": open_len,
                          "rep": rep, "rep_rng_seed": rep_seed,
                          "epsilon": EPSILON,
                          "snapshot_identity": identity})
                out.write(json.dumps(e) + "\n")
            out.flush()
            ok = [e for e in eps if e["reached"]]
            entered = sum(1 for e in eps if e["first_door"])
            print("prb1_run: %s/%d %s rep %d (seed %d): "
                  "reached=%d/%d entered=%d ticks(mean reached)=%s "
                  "crush=%d"
                  % (arm, seed, wname, rep, rep_seed,
                     len(ok), len(eps), entered,
                     ("%.2f" % (sum(e["ticks"] for e in ok) / len(ok))
                      if ok else "nan"),
                     sum(e["crushes"] for e in eps)), flush=True)
    out.close()
    if deviations:
        print("prb1_run: DEVIATIONS (combos not evaluated): %s"
              % (deviations,), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""WO-TBM-P-Db2 runner: Step 1 retraining + Step 2 evaluation.

Design note: nn/decisions_pdb2_design.md (committed Step 0, 7bc6338,
amended 73fd306, gate ruling d689f8b: PROCEED on the current
substrate, no substrate change).
Work order: ~/workspace/your_files/WO-TBM-P-Db2.md (issued).

For each (arm, seed, protocol) in the design note:
  - Training world F1: gaps {(5,1),(5,3),(5,5),(5,7),(5,9)} with
    P-Db W5s per-door schedules (familiar (5,5) on the arm's
    training schedule, rivals staggered). Fresh from scratch
    via nn/train_pdb2.py trainers ONLY. No fine-tuning, no
    edits to frozen files, no hyperparameter changes.
    Familiar-only observe contract: observe() returns the
    familiar door's bit; rival schedules appear in NO agent's
    features. Step 0 sanity and spike trainings are CHECK /
    MEASUREMENT outputs only and are NEVER reused as Step 1
    agents: Step 1 trains fresh even for the same (arm, seed).
  - Wrapped arm (stateful ScheduleModel): FRESH training per
    eval world (5 wrapped agents x 5 worlds = 25 wrapped
    trainings). That training's encoder is used ONLY for that
    world's eval.
  - Non-wrapped arms (stateless encoders): one training, then
    eval sequentially across worlds, as in doors/P-Db/Luck/P-Da.
  - Training records flushed per training job to
    training_pdb2.jsonl (summary + per-door entry counts +
    schedule open fractions, design note §5). A restart loses
    at most one training: records are flushed immediately
    after training, and a re-training on relaunch
    (deterministic, same weights) does NOT duplicate its
    training record (key already present).
  - Greedy eval: one deterministic episode per start phase,
    all phases (20 for 20/10, 30 for 30/14), cap 200 ticks.
    F0: SchedDoorWorld, familiar only. F1: SchedDoorWorld,
    TRAIN schedules. F2: Pdb2AbsentWorld, rivals only, with
    the PHANTOM FAMILIAR CLOCK for the observe bit (design
    note §2) - F2 agents do NOT observe a familiar door; the
    bit is a clock signal. F3: Pdb2AbsentWorld, novel only
    on the training schedule (numerically identical to the
    familiar schedule, so no phantom distinction arises).
    F4: SchedDoorWorld, TRAIN + (5,8) on the familiar
    schedule. Evaluation precedent: doors_run.py /
    sched_run.py (greedy policy per encoder, stall_signature,
    staging, arrival, steps) + pda_run.py (post-entry stall,
    blank-feature flagging).
  - Conflict analysis (F1 and F2 only, design note §4):
    per-gap staging only - F1 {(4,1),(4,3),(4,5),(4,7),(4,9)};
    F2 {(4,1),(4,3),(4,7),(4,9)}; the P-Da (4,5)
    always-include rule is NOT used for arrival in F2.
    Arrival = first per-gap staging occupation; conflict =
    arrival door closed at arrival + another door open at
    arrival; outcomes waits-at-closed-door /
    takes-open-door / other from the first door entered.
    Preferred-door-closed counts are computed in Step 3
    analysis from the recorded per-door statuses at
    arrival (preferred door = modal F1 first door; in F2
    a (5,5) preferred door is evaluated on the phantom
    clock, recorded as clock_open_at_arrival).
  - Records flushed per agent-world pair to
    results_pdb2.jsonl. Resume-safe: on startup, jobs
    with a full phase count are skipped; jobs with a
    partial count (>0, <period) have their partial rows
    removed and are re-run deterministically, so no
    duplicates and no partials remain at the end. A job
    with MORE rows than its phase count is an error:
    the runner stops before training, so a duplicate is
    never silently pooled.
  - Wrapped eval logs f101/f102/f103 per step. An episode
    with any step f103 == 0.0 is flagged blank_feature=True.
    Blank (drifted) rows are reported and must be excluded
    from analysis, never pooled with clean rows.
  - Evaluation only: if retrained agents fail everywhere,
    that IS the result. No fine-tuning, no hyperparameter
    change, no single-door / P-Da / sanity / spike weight
    substitution, for any reason.

Expected totals: 50 trainings, 150 eval jobs, 3750 eval
episodes (20/10: 15 agents x 5 worlds x 20 = 1500;
30/14: 15 x 5 x 30 = 2250).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step
from doors_run import AGENTS, segments, stall_signature
from train_pdb2 import TRAINERS, TRAIN_BY_PERIOD
from world_doors_sched import FAMILIAR, door_open_at
from world_doors_sched import SchedDoorWorld
from world_pdb2 import (TRAIN_20, TRAIN_30, Pdb2AbsentWorld,
                        f0_schedules, f3_schedules, f4_schedules,
                        familiar_schedule, rival_schedules)

OUT = os.path.join(_HERE, "results_pdb2.jsonl")
TRAIN_OUT = os.path.join(_HERE, "training_pdb2.jsonl")

WORLD_NAMES = ["F0", "F1", "F2", "F3", "F4"]


def gap_key(g):
    return "%d,%d" % (g[0], g[1])


def world_for(train, wname):
    """(world instance factory args, gaps, schedules) for an eval world.

    Returns (make_world, gaps, schedules, clock_period, clock_open_len).
    make_world() builds a fresh world for one start phase.
    schedules is the per-door schedule table for door-status
    computation; for F2 the phantom clock is the familiar schedule.
    """
    period, open_len = familiar_schedule(train)
    if wname == "F0":
        sched = f0_schedules(train)
        return (lambda: SchedDoorWorld(sched),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F1":
        sched = train
        return (lambda: SchedDoorWorld(sched),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F2":
        sched = rival_schedules(train)
        return (lambda: Pdb2AbsentWorld(sched, period, open_len),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F3":
        sched = f3_schedules(train)
        return (lambda: Pdb2AbsentWorld(sched, period, open_len),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F4":
        sched = f4_schedules(train)
        return (lambda: SchedDoorWorld(sched),
                tuple(sorted(sched)), sched, period, open_len)
    raise ValueError("unknown world %r" % (wname,))


def summarize(episodes):
    n = len(episodes)
    ok = [e for e in episodes if e["reached"]]
    return {
        "mean_return": sum(e["return"] for e in episodes) / n,
        "crush_rate": sum(e["crushes"] for e in episodes) / n,
        "success": len(ok) / n,
        "mean_ticks": (sum(e["ticks"] for e in ok) / len(ok)
                       if ok else None),
        "n": n,
    }


def evaluate(agent, encoder, period, open_len, wname, train,
             log_features=False):
    """Greedy eval over all start phases on one P-Db2 eval world."""
    make_world, gaps, sched, _cp, _co = world_for(train, wname)
    gaps_set = set(gaps)
    # Per-gap staging cells for conflict analysis (design note §4).
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
        # Staging cells in first-occupation order (per-gap only).
        staging_order = []
        for p in positions:
            if p in stage_cells and p not in staging_order:
                staging_order.append(p)
        # Arrival (per-gap staging, design note §4).
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
            clock_open = None
            arrival = None
        if entries:
            et = entry_ticks[0]
            status_entry = {gap_key(g): door_open_at(et, *sched[g])
                            for g in gaps}
            wait = (et - at) if at is not None else None
        else:
            status_entry = None
            wait = None
        # Generalized conflict (F1/F2 only).
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
        # Post-entry behaviour (P-Dc connection, as in pda_run).
        if entries:
            first_idx = positions.index(entries[0])
            post_positions = positions[first_idx + 1:]
            post_stall, post_stall_kind = stall_signature(
                post_positions, gaps_set)
        else:
            first_idx = None
            post_positions = []
            post_stall, post_stall_kind = False, None
        blank = bool(log_features and any(
            st.get("f103") == 0.0 for st in steps))
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
            "blank_feature": blank,
            "conflict": conflict, "conflict_outcome": outcome,
            "staging_wait": wait,
            "door_status_at_arrival": status_arr,
            "door_status_at_entry": status_entry,
            "schedules": {gap_key(g): list(sched[g]) for g in gaps},
            "steps": steps})
    return episodes


def load_done_counts():
    counts = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if r.get("kind") == "episode":
                k = (r["arm"], r["seed"], r["world"])
                counts[k] = counts.get(k, 0) + 1
    return counts


def load_training_keys():
    keys = set()
    if os.path.exists(TRAIN_OUT):
        for line in open(TRAIN_OUT):
            r = json.loads(line)
            if r.get("kind") == "training":
                keys.add((r["arm"], r["seed"], r["eval_world"]))
    return keys


def write_training_record(out, keys, arm, seed, period, open_len,
                          eval_world, stats, dec, ewe, frac):
    key = (arm, seed, eval_world)
    if key in keys:
        print("%s/%d training record for %s already present; "
              "re-training is deterministic, not duplicating"
              % (arm, seed, eval_world), flush=True)
        return
    reached = [s for s in stats if s[1]]
    mean_loss_vals = [s[3] for s in stats if s[3] is not None]
    rec = {
        "kind": "training", "arm": arm, "seed": seed,
        "period": period, "open_len": open_len,
        "training_world": "F1", "eval_world": eval_world,
        "episodes": len(stats),
        "reached": len(reached),
        "crushes": sum(s[2] for s in stats),
        "mean_ticks": (sum(s[0] for s in reached) / len(reached)
                       if reached else None),
        "mean_ticks_all": sum(s[0] for s in stats) / len(stats),
        "mean_loss": (sum(mean_loss_vals) / len(mean_loss_vals)
                      if mean_loss_vals else None),
        "door_entry_counts": {gap_key(g): dec[g] for g in dec},
        "episodes_with_entry": {gap_key(g): ewe[g] for g in ewe},
        "open_fractions": {gap_key(g): frac[g] for g in frac},
    }
    out.write(json.dumps(rec) + "\n")
    out.flush()
    keys.add(key)


def main():
    counts = load_done_counts()
    for (arm, seed, world), n in sorted(counts.items()):
        period = 20 if seed < 100 else 30
        if n > period:
            print("ERROR: duplicate job %s/%d %s has %d rows, "
                  "expected %d; stopping before training"
                  % (arm, seed, world, n, period), flush=True)
            sys.exit(1)
    partials = set()
    for (arm, seed, world), n in sorted(counts.items()):
        period = 20 if seed < 100 else 30
        if 0 < n < period:
            partials.add((arm, seed, world))
    if partials:
        print("Partial jobs found, removing their rows for "
              "deterministic re-run: %s" % sorted(partials),
              flush=True)
        kept = []
        for line in open(OUT):
            r = json.loads(line)
            if (r.get("arm"), r.get("seed"), r.get("world")) \
                    not in partials:
                kept.append(line)
        with open(OUT, "w") as f:
            f.writelines(kept)
        for k in partials:
            counts.pop(k, None)
    train_keys = load_training_keys()
    out = open(OUT, "a")
    tout = open(TRAIN_OUT, "a")
    for arm, seed, period, open_len in AGENTS:
        train = TRAIN_20 if period == 20 else TRAIN_30
        need = [w for w in WORLD_NAMES
                if (arm, seed, w) not in counts]
        if not need:
            print("%s/%d: all P-Db2 worlds done, skipping"
                  % (arm, seed), flush=True)
            continue
        if arm == "wrapped":
            for wname in need:
                print("%s/%d %s: fresh training..."
                      % (arm, seed, wname), flush=True)
                agent, stats, encoder, dec, ewe, frac = TRAINERS[arm](
                    seed, train, hidden=32, episodes=2000, lr=1e-3)
                write_training_record(tout, train_keys, arm, seed,
                                      period, open_len, wname,
                                      stats, dec, ewe, frac)
                eps = evaluate(agent, encoder, period, open_len,
                               wname, train, log_features=True)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                counts[(arm, seed, wname)] = len(eps)
                s = summarize(eps)
                entered = sum(1 for e in eps if e["first_door"])
                n_blank = sum(1 for e in eps if e["blank_feature"])
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "entered=%d blank_eps=%d"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], entered,
                         n_blank), flush=True)
        else:
            print("%s/%d %d/%d: training..."
                  % (arm, seed, period, open_len), flush=True)
            agent, stats, encoder, dec, ewe, frac = TRAINERS[arm](
                seed, train, hidden=32, episodes=2000, lr=1e-3)
            write_training_record(tout, train_keys, arm, seed,
                                  period, open_len, "base",
                                  stats, dec, ewe, frac)
            for wname in need:
                eps = evaluate(agent, encoder, period, open_len,
                               wname, train, log_features=False)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                counts[(arm, seed, wname)] = len(eps)
                s = summarize(eps)
                entered = sum(1 for e in eps if e["first_door"])
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "entered=%d"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], entered),
                      flush=True)
    out.close()
    tout.close()


if __name__ == "__main__":
    main()

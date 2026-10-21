#!/usr/bin/env python3
"""WO-TBM-P-Db runner: Step 0c checks + Step 1 schedule-varied
transfer evaluation.

For each (arm, seed, protocol) in the design note:
  - Non-wrapped arms (stateless encoders): one training; W1 on the
    single-door World (0c vs the tracked record); then W2s/W5s
    sequentially on the same encoder (doors main-run procedure;
    0b proved these encoders do not drift across evals).
  - Wrapped arm (stateful ScheduleModel): the 0c check runs on its
    OWN training, because a wrapped encoder cannot serve both a
    0c eval and a sched eval without drift -- the 0c eval itself
    consumes model observations (doors lesson). Each sched world
    is then evaluated on a separate FRESH training (no preceding
    eval), with per-step time features logged for period-loss
    detection. A sched row is kept only if its seed's 0c training
    matched; otherwise it is discarded and reported.

Greedy = one deterministic episode per start phase, all phases.
No training on schedule-varied worlds, no agent modification.
Records flushed per agent-world pair to results_doors_sched.jsonl.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_nn as ann
from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step
from doors_run import (AGENTS, load_expected, matches, stall_signature,
                       segments, summarize, train, evaluate)
from world import World
from world_doors_sched import FAMILIAR, SCHEDS

OUT = os.path.join(_HERE, "results_doors_sched.jsonl")
WORLDS = ("W2s", "W5s")


def door_open(schedules, cell, tick):
    period, open_len, offset = schedules[cell]
    return (tick - offset) % period < open_len


def evaluate_sched(agent, encoder, schedules, world_kind,
                   log_features=False):
    """Greedy eval over all start phases on one sched world."""
    from world_doors_sched import SchedDoorWorld
    gaps = tuple(sorted(schedules))
    period = schedules[FAMILIAR][0]
    episodes = []
    for start in range(period):
        w = SchedDoorWorld(schedules)
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
            if p in gaps and (not entries or entries[-1] != p):
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
        stall, stall_kind = stall_signature(positions, gaps)
        # Arrival: first occupation of any staging cell (4,y).
        stage_cells = set((4, y) for (_x, y) in gaps)
        arrival = None
        for s in steps:
            if (s["x"], s["y"]) in stage_cells:
                arrival = s
                break
        if arrival is not None:
            at = arrival["t"]
            fam_closed = not door_open(schedules, FAMILIAR, at)
            rivals_open = sorted(
                g for g in gaps if g != FAMILIAR
                and door_open(schedules, g, at))
            status_arr = {("%d,%d" % g): door_open(schedules, g, at)
                          for g in gaps}
            conflict = bool(fam_closed and rivals_open)
        else:
            at = None
            fam_closed = None
            rivals_open = []
            status_arr = None
            conflict = False
        if entries:
            et = entry_ticks[0]
            status_entry = {("%d,%d" % g): door_open(schedules, g, et)
                            for g in gaps}
            wait = (et - at) if at is not None else None
        else:
            status_entry = None
            wait = None
        if not conflict:
            outcome = None
        elif not entries:
            outcome = "other"
        elif entries[0] == FAMILIAR:
            outcome = "waits-at-familiar"
        elif entries[0] in rivals_open:
            outcome = "takes-open-rival"
        else:
            outcome = "other"
        episodes.append({
            "kind": "episode", "world": world_kind,
            "start_phase": start, "ticks": ticks,
            "crushes": crushes, "reached": reached,
            "return": total, "first_door": (list(entries[0])
                                            if entries else None),
            "door_entries": [list(p) for p in entries],
            "switches": switches, "stall": stall,
            "stall_kind": stall_kind, "revisits": revisits,
            "arrival": ({"tick": at,
                         "cell": [arrival["x"], arrival["y"]],
                         "familiar_closed": fam_closed,
                         "rivals_open": [list(g) for g in rivals_open]}
                        if arrival is not None else None),
            "conflict": conflict, "conflict_outcome": outcome,
            "staging_wait": wait,
            "door_status_at_arrival": status_arr,
            "door_status_at_entry": status_entry,
            "schedules": {("%d,%d" % g): list(schedules[g])
                          for g in gaps},
            "steps": steps})
    return episodes


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if r.get("kind") == "episode":
                done.add((r["arm"], r["seed"], r["world"]))
    return done


def conflict_counts(eps):
    conf = [e for e in eps if e["conflict"]]
    out = {}
    for e in conf:
        out[e["conflict_outcome"]] = out.get(e["conflict_outcome"], 0) + 1
    return len(conf), out


def main():
    expected = load_expected()
    done = load_done()
    out = open(OUT, "a")
    for arm, seed, period, open_len in AGENTS:
        ann.PERIOD = period
        ann.OPEN_LEN = open_len
        need = [w for w in WORLDS if (arm, seed, w) not in done]
        if not need:
            print("%s/%d: all sched worlds done, skipping"
                  % (arm, seed), flush=True)
            continue
        exp = expected.get((arm, seed, period, open_len))
        if arm == "wrapped":
            # 0c on its own training (see module docstring).
            print("%s/%d: 0c training..." % (arm, seed), flush=True)
            agent0, encoder0 = train(arm, seed)
            eps0 = evaluate(agent0, encoder0, period, open_len,
                            ((5, 5),), "W1")
            ok0c = matches(summarize(eps0), exp)
            print("%s/%d 0c -> %s" % (arm, seed,
                                      "MATCH" if ok0c else "MISMATCH"),
                  flush=True)
            if not ok0c:
                print("%s/%d: 0c FAILED, sched worlds skipped "
                      "(reported, not evaluated)" % (arm, seed),
                      flush=True)
                continue
            for wname in need:
                print("%s/%d %s: fresh training..." % (arm, seed, wname),
                      flush=True)
                agent, encoder = train(arm, seed)
                eps = evaluate_sched(agent, encoder,
                                     SCHEDS[(period, wname)], wname,
                                     log_features=True)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                s = summarize(eps)
                nc, cc = conflict_counts(eps)
                n_blank = sum(1 for e in eps
                              if any(st.get("f103") == 0.0
                                     for st in e["steps"]))
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "conflict=%d %s blank_eps=%d"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], nc, cc,
                         n_blank), flush=True)
        else:
            print("%s/%d %d/%d: training..." % (arm, seed, period,
                                                open_len), flush=True)
            agent, encoder = train(arm, seed)
            eps0 = evaluate(agent, encoder, period, open_len,
                            ((5, 5),), "W1")
            ok0c = matches(summarize(eps0), exp)
            print("%s/%d 0c -> %s" % (arm, seed,
                                      "MATCH" if ok0c else "MISMATCH"),
                  flush=True)
            if not ok0c:
                print("%s/%d: 0c FAILED, sched worlds skipped "
                      "(reported, not evaluated)" % (arm, seed),
                      flush=True)
                continue
            for wname in need:
                eps = evaluate_sched(agent, encoder,
                                     SCHEDS[(period, wname)], wname)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                s = summarize(eps)
                nc, cc = conflict_counts(eps)
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "conflict=%d %s"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], nc, cc),
                      flush=True)
    out.close()


if __name__ == "__main__":
    main()

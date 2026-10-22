#!/usr/bin/env python3
"""WO-TBM-Luck runner: Step 0c checks + Step 1 familiar-removed
transfer evaluation.

For each (arm, seed, protocol) in the design note:
  - Non-wrapped arms (stateless encoders): one training; W1 on the
    single-door World (0c vs the tracked record); then W1r/W5r
    sequentially on the same encoder.
  - Wrapped arm (stateful ScheduleModel): the 0c check runs on its
    OWN training (a wrapped encoder cannot serve both a 0c eval
    and a luck eval without drift -- the 0c eval itself consumes
    model observations). Each luck world is then evaluated on a
    separate FRESH training (no preceding eval), with per-step
    time features logged. A luck row is kept only if its seed's
    0c training matched; otherwise it is discarded and reported.

Greedy = one deterministic episode per start phase, all phases.
No retraining on familiar-removed worlds, no agent modification.
Records flushed per agent-world pair to results_luck.jsonl.
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
from world_luck import LUCK_WORLDS, LuckWorld

OUT = os.path.join(_HERE, "results_luck.jsonl")
WORLDS = ("W1r", "W5r")


def evaluate_luck(agent, encoder, period, open_len, gaps, world_kind,
                  log_features=False):
    """Greedy eval over all start phases on one luck world."""
    episodes = []
    for start in range(period):
        w = LuckWorld(period, open_len, gaps)
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
        for p in positions:
            if p in gaps and (not entries or entries[-1] != p):
                entries.append(p)
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
        # Staging cells in first-occupation order.
        stage_cells = set((4, y) for (_x, y) in gaps)
        stage_cells.add((4, 5))
        staging_order = []
        for p in positions:
            if p in stage_cells and p not in staging_order:
                staging_order.append(p)
        arrival = staging_order[0] if staging_order else None
        episodes.append({
            "kind": "episode", "world": world_kind,
            "start_phase": start, "ticks": ticks,
            "crushes": crushes, "reached": reached,
            "return": total, "first_door": (list(entries[0])
                                            if entries else None),
            "door_entries": [list(p) for p in entries],
            "switches": switches, "stall": stall,
            "stall_kind": stall_kind, "revisits": revisits,
            "staging_order": [list(p) for p in staging_order],
            "arrival": list(arrival) if arrival else None,
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


def main():
    expected = load_expected()
    done = load_done()
    out = open(OUT, "a")
    for arm, seed, period, open_len in AGENTS:
        ann.PERIOD = period
        ann.OPEN_LEN = open_len
        need = [w for w in WORLDS if (arm, seed, w) not in done]
        if not need:
            print("%s/%d: all luck worlds done, skipping"
                  % (arm, seed), flush=True)
            continue
        exp = expected.get((arm, seed, period, open_len))
        if arm == "wrapped":
            print("%s/%d: 0c training..." % (arm, seed), flush=True)
            agent0, encoder0 = train(arm, seed)
            eps0 = evaluate(agent0, encoder0, period, open_len,
                            ((5, 5),), "W1")
            ok0c = matches(summarize(eps0), exp)
            print("%s/%d 0c -> %s" % (arm, seed,
                                      "MATCH" if ok0c else "MISMATCH"),
                  flush=True)
            if not ok0c:
                print("%s/%d: 0c FAILED, luck worlds skipped "
                      "(reported, not evaluated)" % (arm, seed),
                      flush=True)
                continue
            for wname in need:
                print("%s/%d %s: fresh training..." % (arm, seed, wname),
                      flush=True)
                agent, encoder = train(arm, seed)
                eps = evaluate_luck(agent, encoder, period, open_len,
                                    LUCK_WORLDS[wname], wname,
                                    log_features=True)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                s = summarize(eps)
                entered = sum(1 for e in eps if e["first_door"])
                n_blank = sum(1 for e in eps
                              if any(st.get("f103") == 0.0
                                     for st in e["steps"]))
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "entered=%d blank_eps=%d"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], entered,
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
                print("%s/%d: 0c FAILED, luck worlds skipped "
                      "(reported, not evaluated)" % (arm, seed),
                      flush=True)
                continue
            for wname in need:
                eps = evaluate_luck(agent, encoder, period, open_len,
                                    LUCK_WORLDS[wname], wname)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                s = summarize(eps)
                entered = sum(1 for e in eps if e["first_door"])
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "entered=%d"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], entered),
                      flush=True)
    out.close()


if __name__ == "__main__":
    main()

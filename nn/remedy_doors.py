#!/usr/bin/env python3
"""Remedy for the wrapped/0, wrapped/4 W1-vs-W1m mismatch.

Diagnosis (diag_wrapped.py): the ScheduleModel's period estimate
is lost mid-eval when the 32-event openings window fills with
episode-boundary-flagged (unclean) openings -- eval episodes are
short, so most recorded openings are unclean, and once the last
clean diff ages out, _reestimate sets period=None and the time
features go blank (v103=0). Recovery needs an episode spanning
two openings. This is an encoder + eval-order effect, NOT a
world-variant defect (it struck during W1 itself for wrapped/0).

Remedy: evaluate each world on a FRESH deterministic retraining
(end-of-training encoder state, never eval-exposed), with
per-step time features logged so period loss during the remedy
evals is directly detectable. Writes results_doors_remedy.jsonl
(original rows retained for audit, never pooled with remedy rows).
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
from doors_run import (WORLDS, load_expected, matches, summarize,
                       stall_signature, segments)
from world import World
from world_doors import MultiDoorWorld

OUT = os.path.join(_HERE, "results_doors_remedy.jsonl")


def evaluate_feats(agent, encoder, period, open_len, gaps, world_kind):
    """Greedy eval over all start phases, logging time features."""
    episodes = []
    for start in range(period):
        w = MultiDoorWorld(period, open_len, gaps)
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
            feats = {}
            if len(x) >= 104:
                feats = {"f101": float(x[101]), "f102": float(x[102]),
                         "f103": float(x[103])}
            steps.append({"t": w.tick - 1, "x": px, "y": py,
                          "action": ACTION_NAMES[a],
                          "door": bool(door),
                          "crushed": bool(crushed), **feats})
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
        blank_steps = sum(1 for s in steps if s.get("f103") == 0.0)
        episodes.append({
            "kind": "episode", "world": world_kind,
            "start_phase": start, "ticks": ticks,
            "crushes": crushes, "reached": reached,
            "return": total, "first_door": (list(entries[0])
                                            if entries else None),
            "door_entries": [list(p) for p in entries],
            "switches": switches, "stall": stall,
            "stall_kind": stall_kind, "revisits": revisits,
            "blank_feature_steps": blank_steps,
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
    from doors_run import train
    expected = load_expected()
    done = load_done()
    out = open(OUT, "a")
    # W1m fresh (0b re-check) for the two mismatched seeds; W2/W5
    # fresh with feature logging for all wrapped seeds (uniform
    # loss detection across the arm).
    jobs = ([("wrapped", s, "W1m", ((5, 5),)) for s in (0, 4)]
            + [("wrapped", s, wname, gaps)
               for s in range(5) for wname, gaps in WORLDS.items()])
    for arm, seed, wname, gaps in jobs:
        if (arm, seed, wname) in done:
            print("%s/%d %s: already done, skipping"
                  % (arm, seed, wname), flush=True)
            continue
        ann.PERIOD = 20
        ann.OPEN_LEN = 10
        print("%s/%d %s: training (fresh)..." % (arm, seed, wname),
              flush=True)
        agent, encoder = train(arm, seed)
        eps = evaluate_feats(agent, encoder, 20, 10, gaps, wname)
        for e in eps:
            e.update({"arm": arm, "seed": seed, "period": 20,
                      "open_len": 10})
            out.write(json.dumps(e) + "\n")
        out.flush()
        s = summarize(eps)
        n_blank = sum(1 for e in eps if e["blank_feature_steps"] > 0)
        print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
              "episodes-with-blank-features=%d"
              % (arm, seed, wname, s["success"],
                 s["mean_ticks"] if s["mean_ticks"] is not None
                 else float("nan"), s["crush_rate"], n_blank), flush=True)
        if wname == "W1m":
            exp = expected.get((arm, seed, 20, 10))
            print("%s/%d W1m fresh: 0b %s"
                  % (arm, seed, "MATCH" if matches(s, exp) else
                     "MISMATCH"), flush=True)
    out.close()


if __name__ == "__main__":
    main()

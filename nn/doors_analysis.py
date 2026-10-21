#!/usr/bin/env python3
"""Step 2 analysis for WO-TBM-doors.

Sources (never pooled):
  - results_doors.jsonl: main run. W1/W1m for all agents; W2/W5 for
    non-wrapped agents. wrapped/0,4 W1m rows here are the DRIFTED
    (second-eval-pass) rows: retained for audit, EXCLUDED here.
  - results_doors_remedy.jsonl: fresh-state evals. wrapped/0,4 W1m
    (0b MATCH) and wrapped/0-4 W2/W5 (with per-step time features).
For wrapped, W2/W5 come from the remedy file; everything else
from the main file. Outputs a JSON summary printed to stdout.
"""

import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIAR = (5, 5)
NEAREST = (5, 1)


def door_class(d):
    if d is None:
        return "none"
    d = tuple(d)
    if d == FAMILIAR:
        return "familiar"
    if d == NEAREST:
        return "nearest"
    return "other"


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "episode":
            rows.append(r)
    return rows


def summarize(eps):
    n = len(eps)
    reached = [e for e in eps if e["reached"]]
    ticks = [e["ticks"] for e in eps]
    crush = sum(e["crushes"] for e in eps)
    stall = sum(1 for e in eps if e["stall"])
    switches = sum(e["switches"] for e in eps)
    fd = Counter(door_class(e["first_door"]) for e in eps)
    cls = Counter()
    for e in eps:
        if not e["reached"]:
            cls["FAIL"] += 1
        elif e["stall"]:
            cls["STOP"] += 1
        else:
            cls["CHOOSE"] += 1
    return {
        "n": n, "success": len(reached) / n,
        "mean_ticks": sum(ticks) / n,
        "mean_ticks_reached": (sum(e["ticks"] for e in reached)
                               / len(reached) if reached else None),
        "crush_rate": crush / n,
        "stall_rate": stall / n,
        "mean_switches": switches / n,
        "first_door": dict(fd),
        "class": dict(cls),
    }


def main():
    main_rows = load(os.path.join(_HERE, "results_doors.jsonl"))
    rem_rows = load(os.path.join(_HERE, "results_doors_remedy.jsonl"))

    # Index: (arm, seed, world) -> episodes
    def index(rows, skip=()):
        d = {}
        for r in rows:
            k = (r["arm"], r["seed"], r["world"])
            if k in skip:
                continue
            d.setdefault(k, []).append(r)
        return d

    # Drifted wrapped/0,4 W1m excluded from analysis.
    skip = {("wrapped", 0, "W1m"), ("wrapped", 4, "W1m")}
    main = index(main_rows, skip)
    rem = index(rem_rows)

    # Analysis source: wrapped W2/W5/W1m(0,4) from remedy, else main.
    def source(arm, seed, world):
        if arm == "wrapped" and world in ("W2", "W5"):
            return rem[(arm, seed, world)]
        if arm == "wrapped" and world == "W1m" and seed in (0, 4):
            return rem[(arm, seed, world)]
        return main[(arm, seed, world)]

    arms = {}
    for (arm, seed, world) in list(main) + list(rem):
        arms.setdefault(arm, set()).add(seed)

    print("=== per-arm per-world (analysis source) ===")
    table = {}
    for arm in sorted(arms):
        for world in ("W1", "W1m", "W2", "W5"):
            eps = []
            for seed in sorted(arms[arm]):
                try:
                    eps += source(arm, seed, world)
                except KeyError:
                    pass
            if not eps:
                continue
            s = summarize(eps)
            table[(arm, world)] = s
            print("%-14s %s: n=%d succ=%.3f ticks=%.2f crush=%.3f "
                  "stall=%.3f sw=%.2f fd=%s class=%s"
                  % (arm, world, s["n"], s["success"],
                     s["mean_ticks"], s["crush_rate"],
                     s["stall_rate"], s["mean_switches"],
                     s["first_door"], s["class"]))

    print("\n=== per-agent first-door (W2, W5) ===")
    for arm in sorted(arms):
        for seed in sorted(arms[arm]):
            row = []
            for world in ("W2", "W5"):
                try:
                    eps = source(arm, seed, world)
                except KeyError:
                    row.append("%s: -" % world)
                    continue
                fd = Counter(door_class(e["first_door"]) for e in eps)
                row.append("%s: %s" % (world, dict(fd)))
            print("%-14s/%-3d %s" % (arm, seed, " | ".join(row)))

    print("\n=== stall episodes (which door, which world) ===")
    for arm in sorted(arms):
        for seed in sorted(arms[arm]):
            for world in ("W1", "W2", "W5"):
                try:
                    eps = source(arm, seed, world)
                except KeyError:
                    continue
                for e in eps:
                    if e["stall"]:
                        print("%-14s/%-3d %s ph%02d: kind=%s "
                              "first_door=%s ticks=%d crush=%d"
                              % (arm, seed, world, e["start_phase"],
                                 e["stall_kind"], e["first_door"],
                                 e["ticks"], e["crushes"]))

    print("\n=== blank-feature episodes in remedy wrapped evals ===")
    for (arm, seed, world), eps in sorted(rem.items()):
        for e in eps:
            if e.get("blank_feature_steps", 0) > 0:
                print("%s/%d %s ph%02d: blank_steps=%d ticks=%d "
                      "first_door=%s reached=%s"
                      % (arm, seed, world, e["start_phase"],
                         e["blank_feature_steps"], e["ticks"],
                         e["first_door"], e["reached"]))

    print("\n=== switch episodes (W2, W5) ===")
    for arm in sorted(arms):
        for seed in sorted(arms[arm]):
            for world in ("W2", "W5"):
                try:
                    eps = source(arm, seed, world)
                except KeyError:
                    continue
                for e in eps:
                    if e["switches"] > 0:
                        print("%-14s/%-3d %s ph%02d: switches=%d "
                              "entries=%s ticks=%d"
                              % (arm, seed, world, e["start_phase"],
                                 e["switches"], e["door_entries"],
                                 e["ticks"]))

    print("\n=== FAIL episodes ===")
    for arm in sorted(arms):
        for seed in sorted(arms[arm]):
            for world in ("W1", "W2", "W5"):
                try:
                    eps = source(arm, seed, world)
                except KeyError:
                    continue
                for e in eps:
                    if not e["reached"]:
                        print("%-14s/%-3d %s ph%02d: ticks=%d "
                              "crush=%d first_door=%s"
                              % (arm, seed, world, e["start_phase"],
                                 e["ticks"], e["crushes"],
                                 e["first_door"]))


if __name__ == "__main__":
    main()

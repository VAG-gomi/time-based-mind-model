#!/usr/bin/env python3
"""Step 2 analysis for WO-TBM-P-Db (schedule-varied doors).

Sources: results_doors_sched.jsonl (this run); comparison =
same-schedule doors results (results_doors.jsonl, with
results_doors_remedy.jsonl canonical for wrapped W2/W5).
All conclusions EXPLORATORY.
"""

import json
import os
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIAR = (5, 5)
NEAREST = (5, 1)


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "episode":
            rows.append(r)
    return rows


def door_class(d):
    if d is None:
        return "none"
    d = tuple(d)
    if d == FAMILIAR:
        return "familiar"
    if d == NEAREST:
        return "nearest"
    return "other"


def summarize(eps):
    n = len(eps)
    ok = [e for e in eps if e["reached"]]
    return {
        "n": n, "success": len(ok) / n,
        "mean_ticks": (sum(e["ticks"] for e in ok) / len(ok)
                       if ok else None),
        "crush_rate": sum(e["crushes"] for e in eps) / n,
        "stall_rate": sum(1 for e in eps if e["stall"]) / n,
        "mean_switches": sum(e["switches"] for e in eps) / n,
        "first_door": dict(Counter(door_class(e["first_door"])
                                   for e in eps)),
    }


def main():
    sched = load(os.path.join(_HERE, "results_doors_sched.jsonl"))
    by = {}
    for r in sched:
        by.setdefault((r["arm"], r["seed"], r["world"]), []).append(r)

    # Same-schedule comparison: main file, remedy overwrites wrapped.
    comp = {}
    for fn in ("results_doors.jsonl", "results_doors_remedy.jsonl"):
        d = {}
        for r in load(os.path.join(_HERE, fn)):
            d.setdefault((r["arm"], r["seed"], r["world"]), []).append(r)
        comp.update(d)

    arms = sorted({k[0] for k in by})
    print("=== per-arm per-world (sched run) ===")
    for arm in arms:
        for world in ("W2s", "W5s"):
            eps = []
            for (a, s, w), v in by.items():
                if a == arm and w == world:
                    eps += v
            s = summarize(eps)
            conf = [e for e in eps if e["conflict"]]
            oc = Counter(e["conflict_outcome"] for e in conf)
            noarr = sum(1 for e in eps if e["arrival"] is None)
            print("%-14s %s: n=%d succ=%.3f ticks=%.2f crush=%.3f "
                  "stall=%.3f sw=%.2f fd=%s conflict=%d %s noarr=%d"
                  % (arm, world, s["n"], s["success"],
                     s["mean_ticks"] if s["mean_ticks"] is not None
                     else float("nan"), s["crush_rate"],
                     s["stall_rate"], s["mean_switches"],
                     s["first_door"], len(conf), dict(oc), noarr))

    print("\n=== per-arm per-world same-schedule comparison (W2/W5) ===")
    for arm in arms:
        for world, sworld in (("W2", "W2s"), ("W5", "W5s")):
            ceps = []
            for (a, s, w), v in comp.items():
                if a == arm and w == world:
                    ceps += v
            seps = []
            for (a, s, w), v in by.items():
                if a == arm and w == sworld:
                    seps += v
            cs, ss = summarize(ceps), summarize(seps)
            print("%-14s %s->%s: succ %.3f->%.3f ticks %.2f->%.2f "
                  "crush %.3f->%.3f stall %.3f->%.3f"
                  % (arm, world, sworld, cs["success"], ss["success"],
                     cs["mean_ticks"], ss["mean_ticks"],
                     cs["crush_rate"], ss["crush_rate"],
                     cs["stall_rate"], ss["stall_rate"]))

    print("\n=== per-agent conflict outcomes (W2s, W5s) ===")
    for arm in arms:
        for seed in sorted({k[1] for k in by if k[0] == arm}):
            row = []
            for world in ("W2s", "W5s"):
                eps = by.get((arm, seed, world), [])
                conf = [e for e in eps if e["conflict"]]
                oc = Counter(e["conflict_outcome"] for e in conf)
                row.append("%s: %d %s" % (world, len(conf), dict(oc)))
            print("%-14s/%-3d %s" % (arm, seed, " | ".join(row)))

    print("\n=== takes-open-rival episodes in detail ===")
    for (arm, seed, world), eps in sorted(by.items()):
        for e in eps:
            if e["conflict_outcome"] == "takes-open-rival":
                arr = e["arrival"]
                print("%-14s/%-3d %s ph%02d: arrival=%s fam_closed=%s "
                      "rivals_open=%s first_door=%s wait=%s ticks=%d "
                      "crush=%d stall=%s reached=%s"
                      % (arm, seed, world, e["start_phase"],
                         arr["cell"], arr["familiar_closed"],
                         arr["rivals_open"], e["first_door"],
                         e["staging_wait"], e["ticks"], e["crushes"],
                         e["stall"], e["reached"]))

    print("\n=== arrival-cell cut: conflict outcomes by arrival cell ===")
    cut = Counter()
    for (arm, seed, world), eps in sorted(by.items()):
        for e in eps:
            if not e["conflict"]:
                continue
            acell = tuple(e["arrival"]["cell"])
            ac = "familiar-staging" if acell == (4, 5) \
                else "rival-staging"
            cut[(e["conflict_outcome"], ac)] += 1
    for k in sorted(cut):
        print("  %s via %s: %d" % (k[0], k[1], cut[k]))

    print("\n=== waits-at-familiar: arrival cell + staging wait ===")
    waits = []
    for (arm, seed, world), eps in sorted(by.items()):
        for e in eps:
            if e["conflict_outcome"] == "waits-at-familiar":
                waits.append(e)
    w_at_rival = [e for e in waits
                  if tuple(e["arrival"]["cell"]) != (4, 5)]
    print("waits-at-familiar total: %d; arrived at rival staging "
          "then walked to familiar: %d" % (len(waits), len(w_at_rival)))
    ws = [e["staging_wait"] for e in waits
          if e["staging_wait"] is not None]
    if ws:
        ws_sorted = sorted(ws)
        print("staging wait ticks: mean %.1f median %d min %d max %d"
              % (sum(ws) / len(ws), ws_sorted[len(ws) // 2],
                 ws_sorted[0], ws_sorted[-1]))
    for e in w_at_rival[:20]:
        print("  walk-past-rival: %s/%d %s ph%02d arrival=%s "
              "first_door=%s wait=%s ticks=%d"
              % (e["arm"], e["seed"], e["world"], e["start_phase"],
                 e["arrival"]["cell"], e["first_door"],
                 e["staging_wait"], e["ticks"]))

    print("\n=== stalls inside conflict episodes ===")
    for (arm, seed, world), eps in sorted(by.items()):
        for e in eps:
            if e["conflict"] and e["stall"]:
                print("%-14s/%-3d %s ph%02d: outcome=%s kind=%s "
                      "ticks=%d first_door=%s"
                      % (arm, seed, world, e["start_phase"],
                         e["conflict_outcome"], e["stall_kind"],
                         e["ticks"], e["first_door"]))

    print("\n=== FAIL episodes (sched worlds) ===")
    for (arm, seed, world), eps in sorted(by.items()):
        for e in eps:
            if not e["reached"]:
                print("%-14s/%-3d %s ph%02d: ticks=%d crush=%d "
                      "first_door=%s conflict=%s outcome=%s arrival=%s"
                      % (arm, seed, world, e["start_phase"],
                         e["ticks"], e["crushes"], e["first_door"],
                         e["conflict"], e["conflict_outcome"],
                         e["arrival"]))

    print("\n=== blank-feature episodes (wrapped sched evals) ===")
    for (arm, seed, world), eps in sorted(by.items()):
        if arm != "wrapped":
            continue
        for e in eps:
            nb = sum(1 for st in e["steps"] if st.get("f103") == 0.0)
            if nb > 0:
                print("%s/%d %s ph%02d: blank_steps=%d ticks=%d "
                      "first_door=%s reached=%s conflict=%s"
                      % (arm, seed, world, e["start_phase"], nb,
                         e["ticks"], e["first_door"], e["reached"],
                         e["conflict"]))

    print("\n=== switch episodes (sched worlds) ===")
    for (arm, seed, world), eps in sorted(by.items()):
        for e in eps:
            if e["switches"] > 0:
                print("%-14s/%-3d %s ph%02d: switches=%d entries=%s "
                      "ticks=%d crush=%d"
                      % (arm, seed, world, e["start_phase"],
                         e["switches"], e["door_entries"], e["ticks"],
                         e["crushes"]))


if __name__ == "__main__":
    main()

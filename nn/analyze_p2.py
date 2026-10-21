#!/usr/bin/env python3
"""P2 trace analysis (WO-TBM-P2, diagnosis only).

Computes the pre-stated Step-1 measures from nn/traces_p2.jsonl:
phase decomposition (approach / staging wait / escape), dwell in a
fixed 3x3 block (staging interval and goal excluded), segment-based
revisits, and per-start-phase ticks. Writes nn/p2_analysis.json and
prints per-pair tables. No frozen file is modified.
"""

import json
import os
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(_HERE, "traces_p2.jsonl")
OUT = os.path.join(_HERE, "p2_analysis.json")

STAGE = (4, 5)
DOOR = (5, 5)
GOAL = (9, 9)


def segments(steps):
    """Compress consecutive same-cell steps into (cell, n_steps)."""
    segs = []
    for s in steps:
        c = (s["x"], s["y"])
        if segs and segs[-1][0] == c:
            segs[-1][1] += 1
        else:
            segs.append([c, 1])
    return segs


def analyze_episode(e):
    steps = e["steps"]
    pos = [(s["x"], s["y"]) for s in steps]
    # First indices (in step order) of staging cell, door cell.
    i_stage = next((i for i, p in enumerate(pos) if p == STAGE), None)
    i_door = next((i for i, p in enumerate(pos) if p == DOOR), None)
    n = len(steps)
    if i_stage is None or i_door is None:
        approach = staging = escape = None
    else:
        approach = i_stage
        staging = i_door - i_stage
        escape = n - i_door
    # Dwell: longest run of consecutive steps inside one fixed 3x3
    # block, computed separately on approach part and escape part
    # (staging interval and goal cell excluded).
    dwell = 0
    parts = []
    if i_stage is not None and i_door is not None:
        parts = [pos[:i_stage], pos[i_door:]]
    else:
        parts = [pos]
    for part in parts:
        run = 0
        prev = None
        for p in part:
            if p == GOAL:
                run = 0
                prev = None
                continue
            b = (p[0] // 3, p[1] // 3)
            run = run + 1 if b == prev else 1
            prev = b
            dwell = max(dwell, run)
    # Segment-based revisits: segments revisiting an earlier cell.
    segs = segments(steps)
    seen = set()
    revisits = 0
    per_cell = {}
    for c, _k in segs:
        per_cell[c] = per_cell.get(c, 0) + 1
        if c in seen:
            revisits += 1
        seen.add(c)
    max_revisit = max((v - 1 for v in per_cell.values()), default=0)
    # Staging-interval action mix (what the agent does while waiting).
    stage_actions = {}
    if i_stage is not None and i_door is not None:
        for s in steps[i_stage:i_door]:
            stage_actions[s["action"]] = stage_actions.get(s["action"], 0) + 1
    return {
        "start_phase": e["start_phase"], "ticks": e["ticks"],
        "crushes": e["crushes"], "reached": e["reached"],
        "approach": approach, "staging": staging, "escape": escape,
        "approach_escape": (approach + escape
                            if approach is not None else None),
        "dwell": dwell, "revisits": revisits,
        "max_revisit": max_revisit,
        "stage_actions": stage_actions,
    }


def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def main():
    eps = {}
    for line in open(TRACES):
        e = json.loads(line)
        eps.setdefault((e["arm"], e["seed"], e["period"],
                        e["open_len"]), []).append(e)
    out = {"pairs": {}, "episodes": {}}
    for key in sorted(eps):
        rows = [analyze_episode(e) for e in eps[key]]
        rows.sort(key=lambda r: r["start_phase"])
        out["episodes"]["%s/%d %d/%d" % key] = rows
        agg = {}
        for m in ("ticks", "approach", "staging", "escape",
                  "approach_escape", "dwell", "revisits", "max_revisit"):
            vals = [r[m] for r in rows]
            agg[m] = {"median": med(vals), "min": min(vals),
                      "max": max(vals),
                      "mean": sum(vals) / len(vals)}
        agg["n"] = len(rows)
        out["pairs"]["%s/%d %d/%d" % key] = agg
        print("== %s/%d %d/%d (n=%d) ==" % (key + (len(rows),)))
        for m in ("ticks", "approach", "staging", "escape",
                  "approach_escape", "dwell", "revisits", "max_revisit"):
            a = agg[m]
            print("  %-15s med=%s mean=%.2f min=%s max=%s"
                  % (m, a["median"], a["mean"], a["min"], a["max"]))
        print("  per-phase ticks:", [r["ticks"] for r in rows])
        print("  per-phase staging:", [r["staging"] for r in rows])
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()

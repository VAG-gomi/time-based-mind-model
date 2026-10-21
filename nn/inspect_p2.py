#!/usr/bin/env python3
"""P2 trajectory inspection (WO-TBM-P2, diagnosis only).

For selected slow episodes: full position/action/door/feature walk,
dwell-block location, approach/escape action histograms, and (Step 2)
first-divergence comparison between two agents on the same seed and
start phase. Reads nn/traces_p2.jsonl. No frozen file is modified.
"""

import json
import os
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(_HERE, "traces_p2.jsonl")

STAGE = (4, 5)
DOOR = (5, 5)


def load():
    eps = {}
    for line in open(TRACES):
        e = json.loads(line)
        eps[(e["arm"], e["seed"], e["period"], e["start_phase"])] = e
    return eps


def dwell_block(steps):
    """Longest dwell run with its block, approach/escape parts only."""
    pos = [(s["x"], s["y"]) for s in steps]
    i_stage = next((i for i, p in enumerate(pos) if p == STAGE), None)
    i_door = next((i for i, p in enumerate(pos) if p == DOOR), None)
    parts = [pos[:i_stage], pos[i_door:]] if i_stage is not None else [pos]
    best = (0, None)
    for part in parts:
        run = 0
        prev = None
        for p in part:
            if p == (9, 9):
                run = 0
                prev = None
                continue
            b = (p[0] // 3, p[1] // 3)
            run = run + 1 if b == prev else 1
            prev = b
            if run > best[0]:
                best = (run, b)
    return best


def show_episode(eps, arm, seed, period, phase, label):
    e = eps[(arm, seed, period, phase)]
    steps = e["steps"]
    pos = [(s["x"], s["y"]) for s in steps]
    i_stage = next((i for i, p in enumerate(pos) if p == STAGE), None)
    i_door = next((i for i, p in enumerate(pos) if p == DOOR), None)
    print("== %s: %s/%d phase %d: %d ticks, approach %s, staging %s, "
          "escape %s, dwell %s ==" % (
              label, arm, seed, phase, e["ticks"], i_stage,
              (i_door - i_stage) if i_stage is not None else None,
              (len(steps) - i_door) if i_door is not None else None,
              dwell_block(steps)))
    print("   path:", " ".join("(%d,%d)%s" % (s["x"], s["y"], s["action"][0])
                               for s in steps))
    if i_stage is not None:
        ap = Counter(s["action"] for s in steps[:i_stage])
        es = Counter(s["action"] for s in steps[i_door:])
        print("   approach actions:", dict(ap))
        print("   escape actions:", dict(es))
    feats = [(s["t"], round(s["f101"], 2), round(s["f102"], 2),
              round(s["f103"], 2)) for s in steps[:12]
             if "f101" in s]
    if feats:
        print("   first-12 (t, f101, f102, f103):", feats)


def divergence(eps, a1, a2, seed, period, phase):
    e1 = eps[(a1, seed, period, phase)]
    e2 = eps[(a2, seed, period, phase)]
    s1, s2 = e1["steps"], e2["steps"]
    div = None
    for i, (x, y) in enumerate(zip(s1, s2)):
        if (x["x"], x["y"]) != (y["x"], y["y"]) or \
                x["action"] != y["action"]:
            div = i
            break
    if div is None:
        print("== %s vs %s, seed %d phase %d: NO divergence in %d/%d ticks "
              "(identical paths) ==" % (a1, a2, seed, phase,
                                         len(s1), len(s2)))
        return
    print("== %s vs %s, seed %d phase %d: first divergence at step %d "
          "(%s: %d ticks total, %s: %d ticks) ==" % (
              a1, a2, seed, phase, div, a1, len(s1), a2, len(s2)))
    for j in range(max(0, div - 2), min(len(s1), div + 4)):
        x = s1[j]
        y = s2[j] if j < len(s2) else None
        print("   step %d: %s at (%d,%d) %s door=%s | %s at %s %s door=%s"
              % (j, a1, x["x"], x["y"], x["action"], x["door"],
                 a2,
                 "(%d,%d)" % (y["x"], y["y"]) if y else "?",
                 y["action"] if y else "?",
                 y["door"] if y else "?"))


def main():
    eps = load()
    # Slow-episode walks: one slow phase per slow case.
    show_episode(eps, "oracle_ind", 101, 30, 8, "SLOW oracle_ind/101")
    show_episode(eps, "oracle_ind", 101, 30, 0, "oracle_ind/101 phase 0")
    show_episode(eps, "oracle_ind", 102, 30, 0, "CONTROL oracle_ind/102")
    show_episode(eps, "oracle_new", 101, 30, 0, "SLOW oracle_new/101")
    show_episode(eps, "wrapped_ind", 108, 20, 14, "SLOW wrapped/108")
    show_episode(eps, "wrapped_ind", 109, 20, 0, "SLOW wrapped/109")
    show_episode(eps, "wrapped_ind", 102, 20, 0, "CONTROL wrapped/102")
    # Step 2: same-seed divergence, oracle_ind vs wrapped_ind on 101/30/14.
    for phase in (0, 8, 20):
        divergence(eps, "oracle_ind", "wrapped_ind", 101, 30, phase)
    # Same-arm divergence: oracle_ind 101 vs 102 is across seeds, so
    # instead compare oracle_ind/101 vs oracle_new/101 (both slow, same
    # seed) at one phase.
    divergence(eps, "oracle_ind", "oracle_new", 101, 30, 8)


if __name__ == "__main__":
    main()

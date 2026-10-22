#!/usr/bin/env python3
"""FT1 Step 3 analysis (WO-TBM-FT1, DRAFT for instructor verification).

Reads nn/results_ft1.jsonl, nn/training_ft1.jsonl, and P-Db2
records (results_pdb2.jsonl) for contrasts. Writes sections
to stdout (redirect to nn/ft1_analysis_output.txt; do NOT pipe
through `head` when teeing - SIGPIPE truncated a previous
output file).

Kept-set denominators throughout: 540 episodes per world
(12 agents x 20 phases + 10 agents x 30 phases), NOT the
P-Db2 programme total 750. G1 is after-only (no
pre-measurement): never presented as before/after.
All conclusions EXPLORATORY.
"""

import collections
import json
import statistics
import sys

_HERE = "."
import os
_HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(_HERE, name)) as f:
        return [json.loads(l) for l in f]


def pos(r):
    return [(s["x"], s["y"]) for s in r["steps"]]


def door_open(sched, tick):
    period, open_len, offset = sched
    return (tick - offset) % period < open_len


def main():
    recs = load("results_ft1.jsonl")
    tr = load("training_ft1.jsonl")
    pdb2 = load("results_pdb2.jsonl")
    kept = sorted({(r["arm"], r["seed"]) for r in tr})
    print("== §1 Verification ==")
    print("fine-tune summaries:", len(tr), "agents:", len(kept))
    print("eval records:", len(recs))
    jobs = collections.Counter((r["arm"], r["seed"], r["world"])
                               for r in recs)
    print("eval jobs:", len(jobs),
          "unexpected counts:",
          [k for k, v in jobs.items()
           if v != (20 if k[1] < 100 else 30)])
    print("duplicate (arm,seed,world,phase):",
          sum(1 for v in collections.Counter(
              (r["arm"], r["seed"], r["world"], r["start_phase"])
              for r in recs).values() if v > 1))
    for w in ("F0", "F1", "F2", "F3", "F4", "G1"):
        sub = [r for r in recs if r["world"] == w]
        print(" ", w, "episodes", len(sub), "reached",
              sum(r["reached"] for r in sub))
    print("Step 0c check per agent (checkpoint + fine-tune + 6 worlds):")
    for (arm, seed) in kept:
        worlds = sorted({r["world"] for r in recs
                         if r["arm"] == arm and r["seed"] == seed})
        ok = worlds == ["F0", "F1", "F2", "F3", "F4", "G1"]
        print("  ", arm, seed, "worlds", worlds, "OK" if ok else "MISSING")

    print("\n== §2 Per-arm x world (reached/total, entered, mean ticks "
          "over reached, crushes) ==")
    for arm in sorted({r["arm"] for r in recs}):
        for w in ("F0", "F1", "F2", "F3", "F4", "G1"):
            sub = [r for r in recs if r["arm"] == arm and r["world"] == w]
            ok = [r for r in sub if r["reached"]]
            mt = (sum(r["ticks"] for r in ok) / len(ok)) if ok else None
            print("  %-15s %s reached %d/%d entered %d ticks %s "
                  "crushes %d" % (arm, w, len(ok), len(sub),
                                  sum(1 for r in sub if r["door_entries"]),
                                  ("%.2f" % mt) if mt is not None else "nan",
                                  sum(r["crushes"] for r in sub)))
        print("  %-15s first doors F1:" % arm,
              dict(collections.Counter(
                  tuple(r["first_door"]) if r["first_door"] else None
                  for r in recs
                  if r["arm"] == arm and r["world"] == "F1")))
    for w in ("F0", "F1", "F2", "F3", "F4", "G1"):
        sub = [r for r in recs if r["world"] == w]
        print("  TOTAL %s reached %d/%d entered %d" %
              (w, sum(r["reached"] for r in sub), len(sub),
               sum(1 for r in sub if r["door_entries"])))
        print("    first doors:", dict(collections.Counter(
            tuple(r["first_door"]) if r["first_door"] else None
            for r in sub)))

    # P-Db2 F0 medians + preferred doors (from P-Db2 F1)
    med = {}
    for (arm, seed) in kept:
        e0 = [r["ticks"] for r in pdb2 if r["arm"] == arm
              and r["seed"] == seed and r["world"] == "F0"
              and r["reached"]]
        med[(arm, seed)] = statistics.median(e0) if e0 else None
    print("\nP-Db2 F0 medians (kept):",
          {("%s/%s" % k): v for k, v in med.items()})
    pref = {}
    for (arm, seed) in kept:
        sub = [r for r in pdb2 if r["arm"] == arm and r["seed"] == seed
               and r["world"] == "F1" and r["first_door"]]
        c = collections.Counter(tuple(r["first_door"]) for r in sub)
        mx = max(c.values())
        tied = [k for k, v in c.items() if v == mx]
        if len(tied) == 1:
            pref[(arm, seed)] = tied[0]
        else:
            best, bestph = None, 999
            for r in sub:
                if tuple(r["first_door"]) in tied \
                        and r["start_phase"] < bestph:
                    bestph = r["start_phase"]
                    best = tuple(r["first_door"])
            pref[(arm, seed)] = best
    print("P-Db2 preferred doors (kept):",
          {("%s/%s" % k): v for k, v in pref.items()})

    gaps = {"F3": {(5, 8)}, "G1": {(5, 6)}}

    def classify(r):
        if not r["door_entries"]:
            return "F"
        g = gaps[r["world"]]
        idx = next(i for i, pp in enumerate(pos(r)) if pp in g)
        if (4, 5) in pos(r)[:idx]:
            return "D"
        m = med[(r["arm"], r["seed"])]
        if (m is not None and r["ticks"] > 2 * m) or r["revisits"] > 10:
            return "W"
        return "R"

    print("\n== §3 F3 classes (F->D->W->R) ==")
    for w in ("F3", "G1"):
        sub = [r for r in recs if r["world"] == w]
        cls = collections.Counter(classify(r) for r in sub)
        print(" ", w, dict(cls), "reached by class:",
              {c: sum(1 for r in sub if classify(r) == c and r["reached"])
               for c in cls})
        print("  per-agent reached:",
              [(a, s, sum(1 for r in sub if r["arm"] == a
                          and r["seed"] == s and r["reached"]))
               for (a, s) in kept
               if sum(1 for r in sub if r["arm"] == a
                      and r["seed"] == s and r["reached"]) > 0])

    print("\n== §4 G1 transfer (PROMINENT: after-only, no baseline) ==")
    g1 = [r for r in recs if r["world"] == "G1"]
    entered = [r for r in g1 if r["door_entries"]]
    print("  G1 entered:", len(entered), "reached:",
          sum(r["reached"] for r in g1))
    for r in entered:
        g = gaps["G1"]
        idx = next(i for i, pp in enumerate(pos(r)) if pp in g)
        print("   ", r["arm"], r["seed"], "phase", r["start_phase"],
              "class", classify(r), "reached", r["reached"],
              "ticks", r["ticks"], "post_entry_ticks",
              r["post_entry_ticks"], "post_entry_stall",
              r["post_entry_stall"], "(4,5) before entry:",
              (4, 5) in pos(r)[:idx], "first staging (4,6):",
              (4, 6) in pos(r)[:idx])

    print("\n== §5 F4 novel use ==")
    f4 = [r for r in recs if r["world"] == "F4"]
    nf = [r for r in f4 if r["first_door"] == [5, 8]]
    print("  novel-first:", len(nf))
    for r in nf:
        idx = next(i for i, pp in enumerate(pos(r))
                   if pp == (5, 8)) if (5, 8) in pos(r) else None
        before = pos(r)[:idx] if idx is not None else []
        print("   ", r["arm"], r["seed"], "phase", r["start_phase"],
              "reached", r["reached"], "ticks", r["ticks"],
              "revisits", r["revisits"], "(4,5) before entry:",
              (4, 5) in before)
    nov = [r for r in f4 if (5, 8) in pos(r)]
    print("  novel entries anywhere:", len(nov), "entered total:",
          sum(1 for r in f4 if r["door_entries"]))
    for r in nov:
        print("   ", r["arm"], r["seed"], "phase", r["start_phase"],
              "entries", r["door_entries"], "reached", r["reached"])

    print("\n== §6 Conflicts (generalized recomputation + "
          "preferred-door-closed) ==")
    staging = {"F1": {(4, 1), (4, 3), (4, 5), (4, 7), (4, 9)},
               "F2": {(4, 1), (4, 3), (4, 7), (4, 9)}}
    for w in ("F1", "F2"):
        sub = [r for r in recs if r["world"] == w]
        conf = [r for r in sub if r["conflict"]]
        print(" ", w, "recorded conflicts:", len(conf),
              dict(collections.Counter(r["conflict_outcome"]
                                        for r in conf)))
        # independent recomputation from steps + schedules
        mism = 0
        for r in sub:
            st = staging[w]
            arr = next((s for s in r["steps"]
                        if (s["x"], s["y"]) in st), None)
            sched = {tuple(map(int, k.split(","))): tuple(v)
                     for k, v in r["schedules"].items()}
            if arr is None:
                my_conf, my_out = False, None
            else:
                at = arr["t"]
                adoor = (5, arr["y"])
                status = {g: door_open(sched[g], at) for g in sched}
                others = [g for g in sched if g != adoor]
                my_conf = (not status[adoor]) and any(status[g]
                                                      for g in others)
                if not my_conf:
                    my_out = None
                elif not r["door_entries"]:
                    my_out = "other"
                elif tuple(r["first_door"]) == adoor:
                    my_out = "waits-at-closed-door"
                elif tuple(r["first_door"]) in others \
                        and status[tuple(r["first_door"])]:
                    my_out = "takes-open-door"
                else:
                    my_out = "other"
            if my_conf != (r["conflict"] is True) or \
                    (my_conf and my_out != r["conflict_outcome"]):
                mism += 1
        print("   recomputation mismatches:", mism)
        # preferred-door-closed
        cnt = collections.Counter()
        for r in sub:
            arr = next((s for s in r["steps"]
                        if (s["x"], s["y"]) in staging[w]), None)
            if arr is None:
                continue
            sched = {tuple(map(int, k.split(","))): tuple(v)
                     for k, v in r["schedules"].items()}
            at = arr["t"]
            pd = pref[(r["arm"], r["seed"])]
            if pd in sched:
                pstat = door_open(sched[pd], at)
            else:
                pstat = r["arrival"]["clock_open_at_arrival"]
            others_open = any(door_open(sched[g], at)
                              for g in sched if g != pd)
            if (not pstat) and others_open:
                if not r["door_entries"]:
                    cnt["other"] += 1
                elif tuple(r["first_door"]) == pd:
                    cnt["waits-at-preferred"] += 1
                elif tuple(r["first_door"]) in sched and \
                        door_open(sched[tuple(r["first_door"])], at):
                    cnt["takes-open"] += 1
                else:
                    cnt["other"] += 1
        print("   preferred-door-closed:", dict(cnt),
              "total", sum(cnt.values()))
        print("   takes by agent:",
              dict(collections.Counter((r["arm"], r["seed"])
                                        for r in conf
                                        if r["conflict_outcome"]
                                        == "takes-open-door")))

    print("\n== §7 Forgetting contrasts (kept set, P-Db2 before) ==")
    for w in ("F0", "F1", "F2", "F4"):
        before = [r for r in pdb2 if (r["arm"], r["seed"]) in kept
                  and r["world"] == w]
        after = [r for r in recs if r["world"] == w]
        print(" ", w, "before %d/%d after %d/%d" %
              (sum(r["reached"] for r in before), len(before),
               sum(r["reached"] for r in after), len(after)))
        for (arm, seed) in kept:
            b = [r for r in before if r["arm"] == arm
                 and r["seed"] == seed]
            a = [r for r in after if r["arm"] == arm
                 and r["seed"] == seed]
            br, ar = sum(r["reached"] for r in b), \
                sum(r["reached"] for r in a)
            if br != ar:
                print("    changed: %s/%d %s %d/%d -> %d/%d" %
                      (arm, seed, w, br, len(b), ar, len(a)))

    print("\n== §8 memory/3 detail ==")
    for w in ("F0", "F1", "F2", "F3", "F4", "G1"):
        sub = [r for r in recs if r["arm"] == "memory"
               and r["seed"] == 3 and r["world"] == w]
        print("  memory/3", w, "reached",
              sum(r["reached"] for r in sub), "/", len(sub),
              "entered", sum(1 for r in sub if r["door_entries"]))
    m3f3 = [r for r in recs if r["arm"] == "memory" and r["seed"] == 3
            and r["world"] == "F3"]
    print("  memory/3 F3 ticks by phase:",
          [(r["start_phase"], r["ticks"]) for r in
           sorted(m3f3, key=lambda r: r["start_phase"])])
    print("  memory/3 F3 classes:",
          dict(collections.Counter(classify(r) for r in m3f3)))

    print("\n== §9 Routes for newly-crossing agents (F3 + G1) ==")
    for w in ("F3", "G1"):
        sub = [r for r in recs if r["world"] == w
               and r["door_entries"] and r["reached"]]
        for r in sub:
            g = gaps[w]
            idx = next(i for i, pp in enumerate(pos(r)) if pp in g)
            print("  ", w, r["arm"], r["seed"], "phase",
                  r["start_phase"], "class", classify(r),
                  "(4,5) before entry:", (4, 5) in pos(r)[:idx],
                  "staging_order:", r["staging_order"])

    print("\n== §10 Blank-feature audit (wrapped only) ==")
    wrap = [r for r in recs if r["arm"] == "wrapped"]
    blank = [r for r in wrap if r["blank_feature"]]
    print("  blank episodes:", len(blank), "of", len(wrap))
    for r in blank:
        bsteps = [s for s in r["steps"] if s.get("f103") == 0.0]
        rec_t = [s["t"] for s in r["steps"]
                 if s.get("f103") == 1.0]
        print("   ", r["world"], r["seed"], "phase", r["start_phase"],
              "blank steps:", len(bsteps), "blank ticks:",
              [s["t"] for s in bsteps][:6],
              "first f103=1 tick:", rec_t[0] if rec_t else None,
              "reached:", r["reached"], "first_door:", r["first_door"])
    print("  blank episodes by world:",
          dict(collections.Counter(r["world"] for r in blank)))

    print("\n== §11 Far side ==")
    for w in ("F0", "F1", "F2", "F3", "F4", "G1"):
        sub = [r for r in recs if r["world"] == w]
        print(" ", w, "entered", sum(1 for r in sub if r["door_entries"]),
              "post_entry_stall",
              sum(r["post_entry_stall"] for r in sub),
              "poststall+notreached",
              sum(1 for r in sub if r["post_entry_stall"]
                  and not r["reached"]),
              "agents:",
              dict(collections.Counter((r["arm"], r["seed"])
                                        for r in sub
                                        if r["post_entry_stall"]
                                        and not r["reached"])))

    print("\n== §12 Fine-tune summaries ==")
    for r in sorted(tr, key=lambda r: (r["arm"], r["seed"])):
        print("  %s/%d reached %d/%d crushes %d (5,8) entries %d "
              "model %s -> %s checkpoint %s" %
              (r["arm"], r["seed"], r["reached"], r["episodes"],
               r["crushes"], r["door_entry_counts"].get("5,8"),
               r["schedule_model_start"], r["schedule_model_end"],
               r["checkpoint"]["source"]))


if __name__ == "__main__":
    main()

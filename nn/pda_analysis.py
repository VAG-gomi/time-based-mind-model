#!/usr/bin/env python3
"""WO-TBM-P-Da Step 3 analysis (DRAFT, for instructor verification).

Reads nn/training_pda.jsonl, nn/results_pda.jsonl, and
nn/results_luck.jsonl (Luck contrast only). Writes a text report
to stdout (caller redirects to nn/pda_analysis_output.txt; do NOT
pipe through `head` when teeing — SIGPIPE truncated a previous
output file).

Classification (design note §4, re-stated): E2/E3 episodes only,
first entry only, priority F -> D -> W -> R:
  F: never enters any door.
  D: occupies (4,5) at a tick BEFORE first-entry tick, then enters
     a rival (E2) or novel (E3) door. Behavioural pattern only,
     not proof of intent.
  W: not D, enters, AND (ticks > 2 x own E0 median ticks over
     reached episodes) OR (revisits > 10). If own E0 median is
     undefined (0 reached in E0), ticks criterion inapplicable,
     W by revisits > 10 alone.
  R: not D, not W.
  UNCLASSIFIED: cannot be classified (expected 0).
E4 novel entries are NOT classified D/R/W/F (design note §3).
E0/E1 are reported by first-door distribution, not D/R/W/F.
All conclusions in the report built from this output are
EXPLORATORY, descriptive, task-and-protocol bounded.
"""

import json
import os
import statistics
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(fn):
    out = []
    if os.path.exists(os.path.join(_HERE, fn)):
        for line in open(os.path.join(_HERE, fn)):
            out.append(json.loads(line))
    return out


def poslist(r):
    return [(s["x"], s["y"]) for s in r["steps"]]


def mean_or_none(vals):
    return sum(vals) / len(vals) if vals else None


def fmt(x, nd=2):
    return "%.2f" % x if x is not None else "undefined"


def main():
    trainings = [r for r in load_jsonl("training_pda.jsonl")
                 if r.get("kind") == "training"]
    recs = [r for r in load_jsonl("results_pda.jsonl")
            if r.get("kind") == "episode"]
    luck = [r for r in load_jsonl("results_luck.jsonl")
            if r.get("kind") == "episode"]

    print("=== 0. VERIFICATION (records, not runner-log summaries) ===")
    print("pda eval records: %d" % len(recs))
    jobs = Counter((r["arm"], r["seed"], r["world"]) for r in recs)
    print("pda eval jobs (arm, seed, world): %d (expected 150)" % len(jobs))
    exp_n = {}
    for (arm, seed, world), n in sorted(jobs.items()):
        exp = 20 if seed < 100 else 30
        exp_n[(arm, seed, world)] = exp
        if n != exp:
            print("  JOB COUNT MISMATCH: %s/%d %s: %d, expected %d"
                  % (arm, seed, world, n, exp))
    print("jobs with unexpected count: %d"
          % sum(1 for k, n in jobs.items() if n != exp_n[k]))
    keys = [(r["arm"], r["seed"], r["world"], r["start_phase"])
            for r in recs]
    print("duplicate (arm, seed, world, start_phase) keys: %d"
          % (len(keys) - len(set(keys))))
    print("episodes where len(steps) != ticks: %d"
          % sum(1 for r in recs if len(r["steps"]) != r["ticks"]))
    # Recomputed first entry vs recorded.
    mism = 0
    for r in recs:
        gaps = set()
        # gaps per world, from world_pda constants
        gaps = {"E0": {(5, 5)},
                "E1": {(5, 1), (5, 3), (5, 5), (5, 7), (5, 9)},
                "E2": {(5, 1), (5, 3), (5, 7), (5, 9)},
                "E3": {(5, 8)},
                "E4": {(5, 1), (5, 3), (5, 5), (5, 7), (5, 8),
                       (5, 9)}}[r["world"]]
        entries = []
        for p in poslist(r):
            if p in gaps and (not entries or entries[-1] != p):
                entries.append(p)
        rec_entries = [tuple(p) for p in r["door_entries"]]
        if entries != rec_entries:
            mism += 1
    print("episodes where recomputed entries != recorded door_entries: %d"
          % mism)
    print("training records: %d (expected 50: 25 wrapped + 25 others)"
          % len(trainings))
    tkeys = Counter((r["arm"], r["seed"], r["eval_world"])
                    for r in trainings)
    print("duplicate training keys: %d"
          % sum(1 for k, n in tkeys.items() if n > 1))
    # Every agent has training + E0?
    agents = sorted({(r["arm"], r["seed"]) for r in recs})
    missing_e0 = [(a, s) for (a, s) in agents
                  if (a, s, "E0") not in jobs]
    print("agents in eval records: %d, missing E0: %s"
          % (len(agents), missing_e0))
    missing_train = []
    for (a, s) in agents:
        if a == "wrapped":
            for w in ("E0", "E1", "E2", "E3", "E4"):
                if (a, s, w) not in tkeys:
                    missing_train.append((a, s, w))
        else:
            if (a, s, "base") not in tkeys:
                missing_train.append((a, s, "base"))
    print("missing training records: %s" % missing_train)
    # Wrapped training determinism check: 5 trainings per wrapped
    # seed should have identical per-door counts.
    wrap_det = []
    for s in range(5):
        decs = []
        for w in ("E0", "E1", "E2", "E3", "E4"):
            for r in trainings:
                if r["arm"] == "wrapped" and r["seed"] == s \
                        and r["eval_world"] == w:
                    decs.append(tuple(sorted(
                        r["door_entry_counts"].items())))
        if decs and len(set(decs)) != 1:
            wrap_det.append(s)
    print("wrapped seeds with non-identical training per-door counts "
          "across their 5 trainings: %s" % wrap_det)

    print("\n=== 1. TRAINING SUMMARIES ===")
    for r in sorted(trainings, key=lambda x: (x["arm"], x["seed"],
                                              x["eval_world"])):
        dec = r["door_entry_counts"]
        total = sum(dec.values())
        print("%s/%d eval_world=%s: episodes=%d reached=%d "
              "crushes=%d mean_ticks_reached=%s mean_ticks_all=%.1f "
              "per_door=%s episodes_with_entry=%s total_entries=%d"
              % (r["arm"], r["seed"], r["eval_world"], r["episodes"],
                 r["reached"], r["crushes"], fmt(r["mean_ticks"]),
                 r["mean_ticks_all"], dec, r["episodes_with_entry"],
                 total))

    print("\n=== 2. PER-ARM x WORLD ENDPOINTS ===")
    for world in ("E0", "E1", "E2", "E3", "E4"):
        for arm in ("wrapped", "plain", "memory", "oracle_ind",
                    "oracle_new", "phase_only_new"):
            sub = [r for r in recs if r["world"] == world
                   and r["arm"] == arm]
            if not sub:
                continue
            ok = [r for r in sub if r["reached"]]
            fd = Counter(tuple(r["first_door"]) if r["first_door"]
                         else None for r in sub)
            print("%s %s: n=%d succ=%d (%.3f) ticks_reached_mean=%s "
                  "ticks_reached_median=%s crushes=%d switches=%d "
                  "entered=%d first_doors=%s"
                  % (arm, world, len(sub), len(ok),
                     len(ok) / len(sub) if sub else 0,
                     fmt(mean_or_none([r["ticks"] for r in ok])),
                     fmt(statistics.median([r["ticks"] for r in ok])
                         if ok else None),
                     sum(r["crushes"] for r in sub),
                     sum(r["switches"] for r in sub),
                     sum(1 for r in sub if r["first_door"]),
                     dict(fd)))

    print("\n=== 3. E0 MEDIANS (W thresholds, own E0, per arm/seed) ===")
    e0_median = {}
    fallback = []
    for (arm, seed) in sorted(agents):
        sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed
               and r["world"] == "E0" and r["reached"]]
        if sub:
            med = statistics.median([r["ticks"] for r in sub])
            e0_median[(arm, seed)] = med
            print("%s/%d: E0 n_reached=%d median_ticks_reached=%.1f "
                  "2x_threshold=%.1f" % (arm, seed, len(sub), med,
                                         2 * med))
        else:
            e0_median[(arm, seed)] = None
            fallback.append((arm, seed))
            print("%s/%d: E0 n_reached=0 median UNDEFINED -> "
                  "revisits-only W fallback" % (arm, seed))
    print("fallback agents (E0 median undefined): %s" % fallback)

    print("\n=== 4. D/R/W/F CLASSES (E2 and E3 only) ===")
    gaps_by_world = {"E2": {(5, 1), (5, 3), (5, 7), (5, 9)},
                     "E3": {(5, 8)}}
    for world in ("E2", "E3"):
        gaps = gaps_by_world[world]
        allc = Counter()
        per_arm = defaultdict(Counter)
        per_arm_reached = defaultdict(Counter)
        for r in recs:
            if r["world"] != world:
                continue
            pos = poslist(r)
            entries = []
            for p in pos:
                if p in gaps and (not entries or entries[-1] != p):
                    entries.append(p)
            if not entries:
                cls = "F"
            else:
                first_idx = pos.index(entries[0])
                if (4, 5) in pos[:first_idx]:
                    cls = "D"
                else:
                    med = e0_median.get((r["arm"], r["seed"]))
                    w_by_ticks = (med is not None
                                  and r["ticks"] > 2 * med)
                    if w_by_ticks or r["revisits"] > 10:
                        cls = "W"
                    else:
                        cls = "R"
            allc[cls] += 1
            per_arm[r["arm"]][cls] += 1
            if r["reached"]:
                per_arm_reached[r["arm"]][cls] += 1
        print("%s overall: %s (total %d, reached by class: %s)"
              % (world, dict(allc), sum(allc.values()),
                 {a: dict(c) for a, c in
                  sorted(per_arm_reached.items())}))
        for arm in sorted(per_arm):
            print("  %s: %s" % (arm, dict(per_arm[arm])))

    print("\n=== 5. E1 FIRST-DOOR DISTRIBUTION + TRAINING JOINT ===")
    for (arm, seed) in sorted(agents):
        sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed
               and r["world"] == "E1"]
        fd = Counter(tuple(r["first_door"]) if r["first_door"]
                     else None for r in sub)
        # Training counts: base for non-wrapped, E1 training for wrapped.
        tr = None
        for r in trainings:
            if r["arm"] == arm and r["seed"] == seed and (
                    (arm == "wrapped" and r["eval_world"] == "E1") or
                    (arm != "wrapped" and r["eval_world"] == "base")):
                tr = r
                break
        if tr:
            dec = tr["door_entry_counts"]
            tot = sum(dec.values())
            dom = max(dec.items(), key=lambda kv: kv[1]) if tot else None
            print("%s/%d E1 first_doors=%s training_total_entries=%d "
                  "training_per_door=%s dominant=%s (share %.3f)"
                  % (arm, seed, dict(fd), tot, dec, dom,
                     dom[1] / tot if dom and tot else 0))
        else:
            print("%s/%d E1 first_doors=%s NO TRAINING RECORD"
                  % (arm, seed, dict(fd)))

    print("\n=== 6. E4 NOVEL-DOOR ENTRIES (NOT classified D/R/W/F) ===")
    for (arm, seed) in sorted(agents):
        sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed
               and r["world"] == "E4"]
        fd = Counter(tuple(r["first_door"]) if r["first_door"]
                     else None for r in sub)
        novel = [r for r in sub if r["first_door"] == [5, 8]]
        flags = []
        for r in novel:
            pos = poslist(r)
            first_idx = pos.index((5, 8))
            flags.append(((4, 5) in pos[:first_idx], r["ticks"],
                          r["revisits"], r["reached"]))
        print("%s/%d E4 first_doors=%s novel_first=%d "
              "novel_entries (flag45, ticks, revisits, reached)=%s"
              % (arm, seed, dict(fd), len(novel), flags))

    print("\n=== 7. FAR SIDE AFTER ENTRY (E2/E3/E4 entries) ===")
    for world in ("E2", "E3", "E4"):
        sub = [r for r in recs if r["world"] == world
               and r["first_door"]]
        if not sub:
            print("%s: no entries" % world)
            continue
        stallc = Counter(r["post_entry_stall_kind"] for r in sub)
        reached = sum(1 for r in sub if r["reached"])
        long_freeze = [(r["arm"], r["seed"], r["start_phase"],
                        r["first_door"], r["post_entry_ticks"],
                        r["post_entry_stall_kind"])
                       for r in sub
                       if not r["reached"] and r["post_entry_ticks"] >= 50]
        print("%s entries=%d reached=%d post_entry_stall_kinds=%s "
              "long_freezes_not_reached(post_ticks>=50)=%d %s"
              % (world, len(sub), reached, dict(stallc),
                 len(long_freeze), long_freeze[:20]))

    print("\n=== 8. WRAPPED BLANK-FEATURE CHECK ===")
    wrap = [r for r in recs if r["arm"] == "wrapped"]
    blank_eps = [r for r in wrap if r["blank_feature"]]
    steps_f = sum(len(r["steps"]) for r in wrap
                  if r["steps"] and "f103" in r["steps"][0])
    print("wrapped episodes: %d, steps with f keys: %d, "
          "blank_feature episodes: %d"
          % (len(wrap), steps_f, len(blank_eps)))
    for w in ("E0", "E1", "E2", "E3", "E4"):
        sub = [r for r in wrap if r["world"] == w]
        print("  %s: blank=%d / %d"
              % (w, sum(1 for r in sub if r["blank_feature"]),
                 len(sub)))

    print("\n=== 9. PAIRED TRAJECTORY COMPARISONS ===")
    by = {(r["arm"], r["seed"], r["start_phase"], r["world"]):
          poslist(r) for r in recs}
    for w1, w2 in (("E1", "E4"), ("E0", "E1")):
        pairs = ident = 0
        for (arm, seed, ph, w), seq in by.items():
            if w == w1 and (arm, seed, ph, w2) in by:
                pairs += 1
                ident += (seq == by[(arm, seed, ph, w2)])
        print("paired %s vs %s: %d pairs, %d identical position "
              "sequences" % (w1, w2, pairs, ident))

    print("\n=== 10. LUCK CONTRAST (E2 vs Luck W5r) ===")
    luck_w5r = [r for r in luck if r["world"] == "W5r"]
    print("Luck W5r: n=%d reached=%d entered=%d"
          % (len(luck_w5r),
             sum(1 for r in luck_w5r if r["reached"]),
             sum(1 for r in luck_w5r if r["first_door"])))
    e2 = [r for r in recs if r["world"] == "E2"]
    print("P-Da E2: n=%d reached=%d entered=%d"
          % (len(e2), sum(1 for r in e2 if r["reached"]),
             sum(1 for r in e2 if r["first_door"])))
    for arm in ("wrapped", "plain", "memory", "oracle_ind",
                "oracle_new", "phase_only_new"):
        lw = [r for r in luck_w5r if r["arm"] == arm]
        ew = [r for r in e2 if r["arm"] == arm]
        print("  %s: Luck W5r reached %d/%d entered %d; "
              "P-Da E2 reached %d/%d entered %d"
              % (arm, sum(1 for r in lw if r["reached"]), len(lw),
                 sum(1 for r in lw if r["first_door"]),
                 sum(1 for r in ew if r["reached"]), len(ew),
                 sum(1 for r in ew if r["first_door"])))


if __name__ == "__main__":
    main()

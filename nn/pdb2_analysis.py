#!/usr/bin/env python3
"""WO-TBM-P-Db2 Step 3 analysis. Reads results_pdb2.jsonl,
training_pdb2.jsonl, plus P-Da (results_pda.jsonl) and P-Db
(results_doors_sched.jsonl) records for contrasts. Prints to stdout;
run with output redirected to pdb2_analysis_output.txt."""

import json
import collections
import statistics

HERE = "/home/hatch/workspace/time-mind-model/nn"


def load(name):
    with open("%s/%s" % (HERE, name)) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def door_open(t, sched):
    period, open_len, offset = sched
    return (t - offset) % period < open_len


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs):
    return statistics.median(xs) if xs else None


def key(arm, seed):
    return "%s/%s" % (arm, seed)


recs = load("results_pdb2.jsonl")
train = load("training_pdb2.jsonl")
pda = load("results_pda.jsonl")
schedrecs = load("results_doors_sched.jsonl")

WORLDS = ["F0", "F1", "F2", "F3", "F4"]
ARMS = ["wrapped", "plain", "memory", "oracle_ind", "oracle_new",
        "phase_only_new"]

print("=" * 70)
print("SECTION 1 -- RECORD VERIFICATION (Step 0c + Step 3 gate)")
print("=" * 70)
jobs = collections.Counter((r["arm"], r["seed"], r["world"]) for r in recs)
print("eval records: %d (expected 3750)" % len(recs))
print("eval jobs: %d (expected 150)" % len(jobs))
print("jobs with count != period: %s"
      % {k: v for k, v in jobs.items() if v not in (20, 30)})
dups = collections.Counter((r["arm"], r["seed"], r["world"],
                            r["start_phase"]) for r in recs)
print("duplicate (arm,seed,world,start_phase) keys: %d"
      % sum(1 for v in dups.values() if v > 1))
print("world totals: %s" % dict(collections.Counter(r["world"] for r in recs)))
print("training records: %d (expected 50)" % len(train))
missing = [r for r in train
           if not r["door_entry_counts"] or not r["open_fractions"]]
print("training records missing door counts or open fractions: %d"
      % len(missing))
for r in train:
    assert r["episodes"] == 2000
agents = sorted(set((r["arm"], r["seed"]) for r in recs))
print("agents: %d (expected 30)" % len(agents))
f0jobs = [k for k in jobs if k[2] == "F0"]
print("agents with an F0 evaluation: %d (expected 30)" % len(f0jobs))
# wrapped identical per-seed training counts across fresh trainings
w = collections.defaultdict(list)
for r in train:
    if r["arm"] == "wrapped":
        w[r["seed"]].append(r["door_entry_counts"])
for s in sorted(w):
    cs = w[s]
    print("wrapped/%s fresh trainings: %d identical counts: %s %s"
          % (s, len(cs), all(c == cs[0] for c in cs), cs[0]))
blanks = [r for r in recs if r["blank_feature"]]
print("blank_feature episodes: %d %s"
      % (len(blanks),
         collections.Counter((r["arm"], r["world"]) for r in blanks)))
for r in blanks:
    idx = [i for i, s in enumerate(r["steps"]) if s.get("f103") == 0.0]
    print("  blank: %s/%s %s phase %d reached %s ticks %d "
          "blank steps %s first_door %s"
          % (r["arm"], r["seed"], r["world"], r["start_phase"],
             r["reached"], r["ticks"], idx, r["first_door"]))
print("total blank steps across all wrapped episodes: %d"
      % sum(1 for r in recs if r["arm"] == "wrapped"
            for s in r["steps"] if s.get("f103") == 0.0))

print()
print("=" * 70)
print("SECTION 2 -- PER-ARM x WORLD (ticks: mean/median over REACHED only)")
print("=" * 70)
for arm in ARMS:
    for wname in WORLDS:
        eps = [r for r in recs if r["arm"] == arm and r["world"] == wname]
        reached = [r for r in eps if r["reached"]]
        entered = [r for r in eps if r["door_entries"]]
        fd = collections.Counter(
            tuple(r["first_door"]) if r["first_door"] else None
            for r in eps)
        print("%s %s n=%d reached=%d entered=%d mean_ticks=%.2f "
              "med_ticks=%s crushes=%d first_doors=%s"
              % (arm, wname, len(eps), len(reached), len(entered),
                 mean([r["ticks"] for r in reached]),
                 med([r["ticks"] for r in reached]),
                 sum(r["crushes"] for r in eps), dict(fd)))
print()
print("TOTALS per world:")
for wname in WORLDS:
    eps = [r for r in recs if r["world"] == wname]
    reached = [r for r in eps if r["reached"]]
    entered = [r for r in eps if r["door_entries"]]
    print("%s n=%d reached=%d entered=%d mean_ticks_reached=%.2f "
          "crushes=%d" % (wname, len(eps), len(reached), len(entered),
                          mean([r["ticks"] for r in reached]),
                          sum(r["crushes"] for r in eps)))

print()
print("=" * 70)
print("SECTION 3 -- F0 MEDIANS PER AGENT (W thresholds) + FALLBACK LIST")
print("=" * 70)
f0med = {}
fallback = []
for (arm, seed) in agents:
    eps = [r for r in recs if r["arm"] == arm and r["seed"] == seed
           and r["world"] == "F0"]
    rt = [r["ticks"] for r in eps if r["reached"]]
    f0med[(arm, seed)] = med(rt)
    if not rt:
        fallback.append(key(arm, seed))
    print("F0 %s/%s n=%d reached=%d median_ticks=%s"
          % (arm, seed, len(eps), len(rt), med(rt)))
print("FALLBACK agents (F0 reached 0, W by revisits>10 alone): %s" % fallback)

print()
print("=" * 70)
print("SECTION 4 -- D/R/W/F CLASSES, F2 AND F3 ONLY (F -> D -> W -> R)")
print("=" * 70)


def classify(r):
    """Return class for an F2/F3 episode. Entry tick from first entry
    step; (4,5) occupation counted over steps with t < entry tick."""
    if not r["door_entries"]:
        return "F", None
    fi = r["first_entry_index"]
    entry_tick = r["steps"][fi]["t"]
    at45 = [s for s in r["steps"]
            if s["t"] < entry_tick and (s["x"], s["y"]) == (4, 5)]
    if at45:
        return "D", len(at45)
    m = f0med[(r["arm"], r["seed"])]
    if m is None:
        wcond = r["revisits"] > 10
    else:
        wcond = (r["ticks"] > 2 * m) or (r["revisits"] > 10)
    if wcond:
        return "W", None
    return "R", None


for wname in ("F2", "F3"):
    eps = [r for r in recs if r["world"] == wname]
    cls = collections.Counter()
    cls_reached = collections.Counter()
    d_ticks = []
    for r in eps:
        c, n45 = classify(r)
        cls[c] += 1
        if r["reached"]:
            cls_reached[c] += 1
        if c == "D":
            d_ticks.append(n45)
    print("%s classes: %s (reached by class: %s)"
          % (wname, dict(cls), dict(cls_reached)))
    if wname == "F2":
        print("  F2 D decomposition: pass-through (exactly 1 step at "
              "(4,5)): %d; waited (>=2 steps at (4,5)): %d"
              % (sum(1 for n in d_ticks if n == 1),
                 sum(1 for n in d_ticks if n >= 2)))
    fd = collections.Counter(
        tuple(r["first_door"]) if r["first_door"] else None for r in eps)
    print("  %s first doors: %s" % (wname, dict(fd)))
    for arm in ARMS:
        sub = [r for r in eps if r["arm"] == arm]
        cc = collections.Counter(classify(r)[0] for r in sub)
        print("  %s %s: %s reached=%d/%d"
              % (arm, wname, dict(cc),
                 sum(1 for r in sub if r["reached"]), len(sub)))

print()
print("=" * 70)
print("SECTION 5 -- CONFLICT ANALYSIS, F1 AND F2")
print("  generalized: arrival door closed + another door open")
print("  preferred door: modal F1 first door over entered episodes;")
print("  tie -> smallest start_phase at which a tied door is first")
print("=" * 70)


def recompute_conflict(r):
    """Independent recomputation from arrival + schedule tables.
    Returns (conflict, outcome) or (False, None)."""
    if r["arrival"] is None:
        return False, None
    at = r["arrival"]["tick"]
    arr = tuple(r["arrival"]["arrival_door"])
    status = {tuple(int(v) for v in k.split(",")): v
              for k, v in r["door_status_at_arrival"].items()}
    if status.get(arr) is not False:
        # arrival door open (or unknown): not a conflict
        if status.get(arr) is not True:
            return "STATUS?", None
        return False, None
    sched = {tuple(int(v) for v in k.split(",")): v
             for k, v in r["schedules"].items()}
    others_open = [g for g in sched if g != arr
                   and door_open(at, sched[g])]
    if not others_open:
        return False, None
    if not r["door_entries"]:
        return True, "other"
    first = tuple(r["first_door"])
    if first == arr:
        return True, "waits-at-closed-door"
    if door_open(at, sched[first]):
        return True, "takes-open-door"
    return True, "other"


mismatch = 0
for wname in ("F1", "F2"):
    eps = [r for r in recs if r["world"] == wname]
    for r in eps:
        rc, ro = recompute_conflict(r)
        # stored encoding uses None for "no conflict computed" on
        # arrival-None episodes; recomputation returns False there.
        # Those are the same verdict, not a mismatch.
        sc = r["conflict"] if r["conflict"] is not None else False
        so = r["conflict_outcome"]
        if (rc, ro) != (sc, so):
            mismatch += 1
            print("  MISMATCH stored vs recomputed: %s/%s %s phase %d "
                  "stored (%s,%s) recomputed (%s,%s)"
                  % (r["arm"], r["seed"], wname, r["start_phase"],
                     r["conflict"], r["conflict_outcome"], rc, ro))
    conf = [r for r in eps if r["conflict"]]
    print("%s generalized conflicts: %d -> %s"
          % (wname, len(conf),
             dict(collections.Counter(r["conflict_outcome"]
                                      for r in conf))))
    print("  %s episodes with arrival None: %d; entered with arrival "
          "None: %d"
          % (wname, sum(1 for r in eps if r["arrival"] is None),
             sum(1 for r in eps if r["door_entries"]
                 and r["arrival"] is None)))
print("total stored-vs-recomputed conflict mismatches: %d" % mismatch)

print()
print("Preferred doors per agent (from F1 entered episodes):")
pref = {}
ties_list = []
for (arm, seed) in agents:
    eps = [r for r in recs if r["arm"] == arm and r["seed"] == seed
           and r["world"] == "F1" and r["door_entries"]]
    fd = collections.Counter(tuple(r["first_door"]) for r in eps)
    if not fd:
        pref[(arm, seed)] = None
        print("  %s: NO ENTRIES in F1, preferred undefined" % key(arm, seed))
        continue
    top = max(fd.values())
    tied = [d for d, c in fd.items() if c == top]
    if len(tied) > 1:
        # smallest start_phase at which a tied door is first
        first_phase = {}
        for d in tied:
            first_phase[d] = min(r["start_phase"] for r in eps
                                 if tuple(r["first_door"]) == d)
        chosen = min(tied, key=lambda d: first_phase[d])
        ties_list.append((key(arm, seed), tied, first_phase, chosen))
    else:
        chosen = tied[0]
    pref[(arm, seed)] = chosen
    print("  %s: preferred %s counts %s"
          % (key(arm, seed), chosen, dict(fd)))
print("Ties (all tied doors, earliest-phase map, chosen):")
for t in ties_list:
    print("  %s" % (t,))
print("Agents with no preferred door (no F1 entries): %s"
      % [key(a, s) for (a, s) in agents if pref[(a, s)] is None])

print()
print("Preferred-door-closed episodes (arrival tick: preferred closed "
      "AND another door open):")
for wname in ("F1", "F2"):
    eps = [r for r in recs if r["world"] == wname]
    tot = 0
    outc = collections.Counter()
    for r in eps:
        p = pref[(r["arm"], r["seed"])]
        if p is None or r["arrival"] is None:
            continue
        at = r["arrival"]["tick"]
        sched = {tuple(int(v) for v in k.split(",")): v
                 for k, v in r["schedules"].items()}
        if wname == "F2" and p == (5, 5):
            # phantom familiar clock: (period, open_len, offset 0)
            sched = dict(sched)
            sched[(5, 5)] = (r["period"], r["open_len"], 0)
        if p not in sched:
            continue
        if door_open(at, sched[p]):
            continue
        others = [g for g in sched if g != p and door_open(at, sched[g])]
        if not others:
            continue
        tot += 1
        if not r["door_entries"]:
            outc["other"] += 1
        elif tuple(r["first_door"]) == p:
            outc["waits-at-preferred"] += 1
        elif door_open(at, sched[tuple(r["first_door"])]):
            outc["takes-open"] += 1
        else:
            outc["other"] += 1
    print("  %s preferred-door-closed: %d -> %s" % (wname, tot, dict(outc)))

print()
print("=" * 70)
print("SECTION 6 -- F4 NOVEL DOOR (5,8): NOT classified D/R/W/F")
print("=" * 70)
f4 = [r for r in recs if r["world"] == "F4"]
novel = [r for r in f4 if r["first_door"] == [5, 8]]
print("F4 first doors: %s"
      % dict(collections.Counter(
          tuple(r["first_door"]) if r["first_door"] else None
          for r in f4)))
print("F4 novel-first (5,8) episodes: %d (reached %d)"
      % (len(novel), sum(1 for r in novel if r["reached"])))
for r in novel:
    fi = r["first_entry_index"]
    entry_tick = r["steps"][fi]["t"]
    at45 = any(s["t"] < entry_tick and (s["x"], s["y"]) == (4, 5)
               for s in r["steps"])
    print("  novel-first: %s/%s phase %d reached %s ticks %d "
          "revisits %d (4,5)-before-entry %s"
          % (r["arm"], r["seed"], r["start_phase"], r["reached"],
             r["ticks"], r["revisits"], at45))
f4ent = [r for r in f4 if r["door_entries"]]
print("F4 entered episodes: %d; novel entries (any position, not just "
      "first): %d"
      % (len(f4ent),
         sum(1 for r in f4ent if [5, 8] in r["door_entries"])))

print()
print("=" * 70)
print("SECTION 7 -- FAR SIDE AFTER ENTRY (post-entry stall / ticks)")
print("=" * 70)
for wname in WORLDS:
    eps = [r for r in recs if r["world"] == wname and r["door_entries"]]
    stalls = [r for r in eps if r["post_entry_stall"]]
    print("%s entered=%d post-entry stall=%d kinds=%s "
          "mean post_entry_ticks=%.2f (entered) / %.2f (stalled)"
          % (wname, len(eps), len(stalls),
             dict(collections.Counter(r["post_entry_stall_kind"]
                                      for r in stalls)),
             mean([r["post_entry_ticks"] for r in eps]),
             mean([r["post_entry_ticks"] for r in stalls])))
    for r in stalls:
        print("    stall: %s/%s phase %d first %s reached %s "
              "post_entry_ticks %d kind %s"
              % (r["arm"], r["seed"], r["start_phase"], r["first_door"],
                 r["reached"], r["post_entry_ticks"],
                 r["post_entry_stall_kind"]))
print("F3 entries detail:")
for r in recs:
    if r["world"] == "F3" and r["door_entries"]:
        print("  F3 entry: %s/%s phase %d reached %s ticks %d crushes %d "
              "post_entry_stall %s kind %s post_ticks %d"
              % (r["arm"], r["seed"], r["start_phase"], r["reached"],
                 r["ticks"], r["crushes"], r["post_entry_stall"],
                 r["post_entry_stall_kind"], r["post_entry_ticks"]))

print()
print("=" * 70)
print("SECTION 8 -- TRAINING BREADTH vs EVAL BREADTH")
print("=" * 70)
for (arm, seed) in agents:
    trs = [t for t in train if t["arm"] == arm and t["seed"] == seed]
    base = [t for t in trs if t["eval_world"] == "base"]
    row = base[0] if base else trs[0]
    f1 = [r for r in recs if r["arm"] == arm and r["seed"] == seed
          and r["world"] == "F1" and r["door_entries"]]
    fd = collections.Counter(tuple(r["first_door"]) for r in f1)
    print("train %s: reached %d/%d door_entries %s open_frac %s | "
          "F1 first doors %s preferred %s"
          % (key(arm, seed), row["reached"], row["episodes"],
             row["door_entry_counts"], row["open_fractions"],
             dict(fd), pref[(arm, seed)]))

print()
print("=" * 70)
print("SECTION 9 -- CONTRASTS: P-Db (recomputed) AND P-Da (recomputed)")
print("=" * 70)
w5s = [x for x in schedrecs if x["world"] == "W5s"]
w5c = [x for x in w5s if x["conflict"]]
print("P-Db W5s subset (recomputed): n=%d conflicts=%d -> %s"
      % (len(w5s), len(w5c),
         dict(collections.Counter(x["conflict_outcome"] for x in w5c))))
w2s = [x for x in schedrecs if x["world"] == "W2s"]
w2c = [x for x in w2s if x["conflict"]]
print("P-Db W2s subset (recomputed): n=%d conflicts=%d -> %s"
      % (len(w2s), len(w2c),
         dict(collections.Counter(x["conflict_outcome"] for x in w2c))))
for wname in ("E0", "E1", "E2", "E3", "E4"):
    eps = [r for r in pda if r["world"] == wname]
    reached = [r for r in eps if r["reached"]]
    entered = [r for r in eps if r["door_entries"]]
    fd = collections.Counter(
        tuple(r["first_door"]) if r["first_door"] else None for r in eps)
    print("P-Da %s n=%d reached=%d entered=%d first_doors=%s"
          % (wname, len(eps), len(reached), len(entered), dict(fd)))
e2e4 = [r for r in pda if r["world"] in ("E2", "E4")
        and r["door_entries"]]
print("P-Da E2+E4 entered=%d post-entry stall=%d (of which "
      "not-reached freeze=%d)"
      % (len(e2e4), sum(1 for r in e2e4 if r["post_entry_stall"]),
         sum(1 for r in e2e4
             if r["post_entry_stall"] and not r["reached"])))
print("P-Db2 far-side freeze (entered, post-entry stall, not "
      "reached) by world:")
for wname in WORLDS:
    fe = [r for r in recs if r["world"] == wname and r["door_entries"]]
    fr = [r for r in fe if r["post_entry_stall"] and not r["reached"]]
    print("  %s entered=%d stall=%d freeze=%d"
          % (wname, len(fe),
             sum(1 for r in fe if r["post_entry_stall"]), len(fr)))
print("P-Db2 F2 vs P-Da E2: F2 reached %d/750, E2 reached %d/750"
      % (sum(1 for r in recs if r["world"] == "F2" and r["reached"]),
         sum(1 for r in pda if r["world"] == "E2" and r["reached"])))
print("P-Db2 F3 vs P-Da E3: F3 reached %d/750, E3 reached %d/750"
      % (sum(1 for r in recs if r["world"] == "F3" and r["reached"]),
         sum(1 for r in pda if r["world"] == "E3" and r["reached"])))
print("P-Db2 F4 novel-first %d vs P-Da E4 novel-first %d"
      % (len(novel),
         sum(1 for r in pda if r["world"] == "E4"
             and r["first_door"] == [5, 8])))

print()
print("=" * 70)
print("SECTION 10 -- PER-AGENT F0 TABLE")
print("=" * 70)
for (arm, seed) in agents:
    eps = [r for r in recs if r["arm"] == arm and r["seed"] == seed
           and r["world"] == "F0"]
    print("F0 %s: reached %d/%d entered %d"
          % (key(arm, seed), sum(1 for r in eps if r["reached"]),
             len(eps), sum(1 for r in eps if r["door_entries"])))
print("DONE")

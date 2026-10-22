#!/usr/bin/env python3
"""WO-TBM-PRB1 Step 2 analysis: per-rep tables, verdicts (design
note section 5), RNG replay check, completeness check.

Reads nn/results_prb1.jsonl only. Writes nothing except stdout
(the caller redirects it to nn/prb1_analysis_output.txt).
Thresholds are taken from the design note and are NOT adjusted
here; any mismatch with the note must be reported, not fixed
by changing a threshold.
"""

import collections
import json
import random
import statistics

REC = "results_prb1.jsonl"

COMBOS = [
    (0, "memory", 3, "F3", 20, "0/20", "161/200"),
    (1, "plain", 0, "F3", 20, "20/20", "188/200"),
    (2, "oracle_new", 101, "F3", 30, "0/30", "179/200"),
    (3, "memory", 3, "F2", 20, "0/20 (0 entered)", "same agent"),
]

with open(REC) as f:
    recs = [json.loads(line) for line in f if line.strip()]

print("== Completeness ==")
print("records:", len(recs), "(expected 450)")
jobs = collections.Counter(
    (r["arm"], r["seed"], r["world"], r["rep"]) for r in recs)
print("rep-jobs:", len(jobs), "(expected 20)")
dups = sum(1 for v in collections.Counter(
    (r["arm"], r["seed"], r["world"], r["rep"], r["start_phase"])
    for r in recs).values() if v > 1)
print("duplicate (agent, world, rep, phase) keys:", dups)
for (ci, arm, seed, w, per, _g, _ft) in COMBOS:
    for rep in range(5):
        rr = [r for r in recs if r["arm"] == arm and r["seed"] == seed
              and r["world"] == w and r["rep"] == rep]
        ok = (len(rr) == per
              and len(set(r["start_phase"] for r in rr)) == per)
        assert ok, (arm, seed, w, rep, len(rr))
print("all 20 reps hold all phases exactly once: OK")
print("all records carry rep/rep_rng_seed/epsilon/snapshot_identity:",
      all("rep" in r and "rep_rng_seed" in r and r["epsilon"] == 0.05
          and "snapshot_identity" in r for r in recs))

print()
print("== RNG replay check (design note section 4) ==")
print("Fresh random.Random(seed) per rep, replayed over the recorded")
print("step counts in increasing start-phase order; the simulated")
print("coin flag must equal each step's epsilon_action flag.")
mismatch = 0
total = 0
for (ci, arm, seed, w, _per, _g, _ft) in COMBOS:
    for rep in range(5):
        rr = sorted(
            [r for r in recs if r["arm"] == arm and r["seed"] == seed
             and r["world"] == w and r["rep"] == rep],
            key=lambda r: r["start_phase"])
        rng = random.Random(900_000 + 1_000 * ci + rep)
        for r in rr:
            for s in r["steps"]:
                u = rng.random()
                flag = u < 0.05
                if flag:
                    rng.randrange(5)
                total += 1
                if flag != s["epsilon_action"]:
                    mismatch += 1
print("steps replayed:", total, "mismatches:", mismatch)

print()
print("== Per-rep results ==")
verdicts = {}
for (ci, arm, seed, w, per, greedy, ft) in COMBOS:
    print("%s/%d %s (greedy FT1: %s reached; fine-tune: %s):"
          % (arm, seed, w, greedy, ft))
    reached_list = []
    for rep in range(5):
        rr = [r for r in recs if r["arm"] == arm and r["seed"] == seed
              and r["world"] == w and r["rep"] == rep]
        rk = sum(r["reached"] for r in rr)
        en = sum(1 for r in rr if r["first_door"])
        ok = [r for r in rr if r["reached"]]
        mt = (sum(r["ticks"] for r in ok) / len(ok)) if ok else None
        eps_steps = sum(1 for r in rr for s in r["steps"]
                        if s["epsilon_action"])
        all_steps = sum(len(r["steps"]) for r in rr)
        reached_list.append(rk)
        print("  rep %d (seed %d): reached %d/%d entered %d "
              "mean ticks reached %s crush %d epsilon steps %d/%d "
              "(%.1f%%)"
              % (rep, 900_000 + 1_000 * ci + rep, rk, per, en,
                 ("%.2f" % mt) if mt is not None else "nan",
                 sum(r["crushes"] for r in rr),
                 eps_steps, all_steps, 100.0 * eps_steps / all_steps))
    print("  reached per rep:", reached_list,
          "median:", statistics.median(reached_list),
          "range:", min(reached_list), "-", max(reached_list),
          "total:", sum(reached_list), "/",
          sum(1 for r in recs if r["arm"] == arm and r["seed"] == seed
              and r["world"] == w))
    verdicts[(arm, seed, w)] = reached_list

print()
print("== Verdicts (design note section 5, precedence: completeness"
      " -> control -> primary) ==")
plain = verdicts[("plain", 0, "F3")]
uninterp = sum(1 for v in plain if v <= 10) >= 3
print("positive control plain/0 F3 reps <= 10/20:",
      sum(1 for v in plain if v <= 10),
      "-> UNINTERPRETABLE triggers:", uninterp)
prim = verdicts[("memory", 3, "F3")]
sup = sum(1 for v in prim if v >= 10) >= 3
weak = sum(1 for v in prim if v <= 2) >= 4
print("primary memory/3 F3: reps >= 10/20:", sum(1 for v in prim if v >= 10),
      " reps <= 2/20:", sum(1 for v in prim if v <= 2))
if uninterp:
    print("PRIMARY VERDICT: UNINTERPRETABLE")
elif sup:
    print("PRIMARY VERDICT: SUPPORTED")
elif weak:
    print("PRIMARY VERDICT: WEAKENED")
else:
    print("PRIMARY VERDICT: NEITHER")
second = verdicts[("oracle_new", 101, "F3")]
sup2 = sum(1 for v in second if v >= 15) >= 3
weak2 = sum(1 for v in second if v <= 3) >= 4
print("second trap oracle_new/101 F3: reps >= 15/30:",
      sum(1 for v in second if v >= 15),
      " reps <= 3/30:", sum(1 for v in second if v <= 3),
      "->", "SUPPORTED-second" if sup2 else (
          "WEAKENED-second" if weak2 else "NEITHER-second"))
f2 = verdicts[("memory", 3, "F2")]
f2ent = [sum(1 for r in recs if r["arm"] == "memory" and r["seed"] == 3
             and r["world"] == "F2" and r["rep"] == rep
             and r["first_door"]) for rep in range(5)]
print("F2 memory/3: reached per rep:", f2, "entered per rep:", f2ent,
      "->", "UNLOCKED" if sum(1 for v in f2 if v >= 10) >= 3
      else ("partial movement" if sum(f2ent) > 0 else "still locked"))

print()
print("== Mechanism detail (descriptive, from records) ==")
for (arm, seed, w) in [("memory", 3, "F3"), ("oracle_new", 101, "F3"),
                       ("memory", 3, "F2")]:
    sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed
           and r["world"] == w]
    reach = [r for r in sub if r["reached"]]
    un = [r for r in sub if not r["reached"]]
    print("%s/%d %s: reached %d/%d%s"
          % (arm, seed, w, len(reach), len(sub),
             (", all via first door (5,8)"
              if reach and all(tuple(r["first_door"]) == (5, 8)
                               for r in reach)
              else (", no reached episodes" if not reach else ""))))
    if reach:
        print("  reached staging (4,8) present:",
              sum(1 for r in reach if [4, 8] in r["staging_order"]),
              "/", len(reach),
              "median ticks:", statistics.median(
                  [r["ticks"] for r in reach]))
    print("  unreached stall kinds:",
          dict(collections.Counter(r["stall_kind"] for r in un)))
    # epsilon action at (4,5) presence, reached vs unreached
    for label, grp in (("reached", reach), ("unreached", un)):
        if not grp:
            continue
        n = sum(1 for r in grp if any(
            s["epsilon_action"] and (s["x"], s["y"]) == (4, 5)
            for s in r["steps"]))
        print("  %s episodes with >=1 epsilon action at (4,5): %d/%d"
              % (label, n, len(grp)))
    # departure action epsilon or greedy (memory/3 F3 reached only).
    # Steps record the position BEFORE the action, so the departure
    # action is the one recorded at the LAST step whose position is
    # (4,5) — not the flag of the following step (an action taken
    # from the new cell), and not the first arrival's departure.
    if (arm, seed, w) == ("memory", 3, "F3"):
        dep_eps = dep_tot = 0
        for r in reach:
            steps = r["steps"]
            last = max((i for i, s in enumerate(steps)
                        if (s["x"], s["y"]) == (4, 5)), default=None)
            if last is None:
                continue
            dep_tot += 1
            dep_eps += int(steps[last]["epsilon_action"])
        print("  reached departures from (4,5):", dep_tot,
              "departure action epsilon_action:", dep_eps,
              "greedy:", dep_tot - dep_eps)

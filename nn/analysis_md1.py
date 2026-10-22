#!/usr/bin/env python3
"""analysis_md1.py -- MD1 analysis (WO-TBM-MD1, Step 0).

Additive only. Computes pooled trained / held-out
reached rates, the frozen §8 verdict with its
stated precedence, and descriptive summaries
(route/staging, far-side behaviour, blank-feature
audit) from MD1 records. No thresholds live here
as tunable parameters: the §8 bars are constants
below, matching the issued work order exactly.

Verdict rule (WO §8, frozen at issuance):
  pooled trained = reached across the 8 trained
  fixed worlds; pooled held-out = reached across
  the 2 held-out fixed worlds. Fractions compare
  reached/total within each pool.
  Precedence, in order:
    INCOMPLETE: any eval world missing, or any
    held-out world missing. No verdict on
    incomplete data.
    UNINTERPRETABLE: pooled trained < 20%.
    FOLLOWING: pooled held-out >= 50% of pooled
    trained AND pooled held-out >= 20% absolute.
    BIGGER ADDRESS BOOK: pooled trained >= 50%
    but pooled held-out < 10%.
    NEITHER: every other complete outcome
    (catch-all, including held-out 10-20%).
  Note: FOLLOWING and BIGGER ADDRESS BOOK cannot
  both hold on held-out fractions alone (one needs
  held-out >= 20%, the other held-out < 10%), so
  precedence never has to break that tie; it is
  stated anyway and applied as written.
"""

import collections
import json
import sys

from world_md1 import TRAINED_ROWS, HELDOUT_ROWS

TRAINED_WORLDS = ["R%d" % r for r in TRAINED_ROWS]
HELDOUT_WORLDS = ["R%d" % r for r in HELDOUT_ROWS]
ALL_FIXED_WORLDS = TRAINED_WORLDS + HELDOUT_WORLDS


def verdict(trained_reached, trained_total,
            held_reached, held_total,
            worlds_complete=True, held_complete=True):
    """§8 verdict for pooled counts. Returns the verdict string."""
    if not worlds_complete or not held_complete:
        return "INCOMPLETE"
    trained_frac = trained_reached / float(trained_total)
    held_frac = held_reached / float(held_total)
    if trained_frac < 0.20:
        return "UNINTERPRETABLE"
    if held_frac >= 0.5 * trained_frac and held_frac >= 0.20:
        return "FOLLOWING"
    if trained_frac >= 0.50 and held_frac < 0.10:
        return "BIGGER ADDRESS BOOK"
    return "NEITHER"


def pooled_counts(records):
    """Per-world (reached, total) and pooled trained/held-out."""
    per_world = {}
    for w in ALL_FIXED_WORLDS:
        sub = [r for r in records if r["world"] == w]
        per_world[w] = (sum(r["reached"] for r in sub), len(sub))
    trained = (sum(per_world[w][0] for w in TRAINED_WORLDS),
               sum(per_world[w][1] for w in TRAINED_WORLDS))
    held = (sum(per_world[w][0] for w in HELDOUT_WORLDS),
            sum(per_world[w][1] for w in HELDOUT_WORLDS))
    return per_world, trained, held


def per_agent_row(records):
    """(arm, seed) -> {world: (reached, total)} for fixed worlds."""
    table = collections.defaultdict(dict)
    for (arm, seed) in sorted({(r["arm"], r["seed"]) for r in records}):
        for w in ALL_FIXED_WORLDS:
            sub = [r for r in records if r["arm"] == arm
                   and r["seed"] == seed and r["world"] == w]
            table[(arm, seed)][w] = (sum(r["reached"] for r in sub),
                                     len(sub))
    return table


def staging_summary(records):
    """Held-out reached episodes staging at (4, row): counts."""
    out = {}
    for w in HELDOUT_WORLDS:
        row = int(w[1:])
        sub = [r for r in records if r["world"] == w and r["reached"]]
        staged = sum(1 for r in sub if [4, row] in r["staging_order"])
        out[w] = (staged, len(sub))
    return out


def farside_summary(records):
    """Per fixed world: entered, entered-not-reached, stalled-unreached."""
    out = {}
    for w in ALL_FIXED_WORLDS:
        sub = [r for r in records if r["world"] == w]
        entered = sum(1 for r in sub if r["door_entries"])
        enr = sum(1 for r in sub
                  if r["door_entries"] and not r["reached"])
        stalled = sum(1 for r in sub
                      if r["post_entry_stall"] and not r["reached"])
        out[w] = (entered, enr, stalled)
    return out


def blank_audit(records):
    """Blank-feature episodes and exclusion sensitivity per world."""
    out = {}
    for w in ALL_FIXED_WORLDS + ["M3"]:
        sub = [r for r in records if r["world"] == w]
        bl = [r for r in sub if r["blank_feature"]]
        clean = [r for r in sub if not r["blank_feature"]]
        out[w] = {
            "blank": len(bl),
            "blank_reached": sum(r["reached"] for r in bl),
            "all": (sum(r["reached"] for r in sub), len(sub)),
            "excluded": (sum(r["reached"] for r in clean),
                         len(clean)),
        }
    return out


def main(results_path, training_path=None):
    """Print the MD1 analysis for results + training record files."""
    records = [json.loads(l) for l in open(results_path)]
    per_world, trained, held = pooled_counts(records)
    worlds_complete = all(per_world[w][1] > 0 for w in ALL_FIXED_WORLDS)
    held_complete = all(per_world[w][1] > 0 for w in HELDOUT_WORLDS)
    v = verdict(trained[0], trained[1], held[0], held[1],
                worlds_complete, held_complete)
    print("== MD1 pooled ==")
    for w in ALL_FIXED_WORLDS:
        print("  %s: %d/%d" % (w, per_world[w][0], per_world[w][1]))
    print("  pooled trained: %d/%d = %.4f"
          % (trained[0], trained[1],
             trained[0] / float(trained[1]) if trained[1] else 0.0))
    print("  pooled held-out: %d/%d = %.4f"
          % (held[0], held[1],
             held[0] / float(held[1]) if held[1] else 0.0))
    print("  VERDICT: %s" % v)
    print("== MD1 staging (held-out reached staging at (4, row)) ==")
    for w, (s, n) in sorted(staging_summary(records).items()):
        print("  %s: %d/%d" % (w, s, n))
    print("== MD1 far side (entered, entered-not-reached,"
          " stalled-unreached) ==")
    for w, (e, enr, st) in sorted(farside_summary(records).items()):
        print("  %s: %d, %d, %d" % (w, e, enr, st))
    print("== MD1 blank-feature audit ==")
    for w, a in sorted(blank_audit(records).items()):
        print("  %s: blank %d (reached %d); all %d/%d; excluded %d/%d"
              % (w, a["blank"], a["blank_reached"],
                 a["all"][0], a["all"][1],
                 a["excluded"][0], a["excluded"][1]))
    if training_path:
        trainings = [json.loads(l) for l in open(training_path)]
        print("== MD1 training row counts ==")
        for t in sorted(trainings, key=lambda r: (r["arm"], r["seed"])):
            rc = t["row_counts"]
            ok = all(rc[str(r)] == 250 for r in TRAINED_ROWS)
            print("  %s/%s: reached %d/%d, row_counts balanced: %s"
                  % (t["arm"], t["seed"], t["reached"], t["episodes"],
                     ok))
    return v


if __name__ == "__main__":
    main(*sys.argv[1:])

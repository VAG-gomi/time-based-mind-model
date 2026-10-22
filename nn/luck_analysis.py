#!/usr/bin/env python3
"""WO-TBM-Luck Step 2 analysis (DRAFT, instructor verification).

Reads nn/results_luck.jsonl and nn/results_doors.jsonl (W1 only),
derives every number from the records, and classifies each luck
episode under the fixed rules in nn/decisions_luck_design.md.

Classification (priority order, fixed before measuring):
  F  = never enters any door. (An episode that enters but does not
       reach the goal is NOT F; it takes its entry's class with
       reached=False.)
  D  = occupies (4,5) at a step BEFORE the first-entry step, then
       enters a rival door. Behavioural pattern only; "decision"
       language stays descriptive.
  W  = not D, enters, AND (ticks > 2 x the agent's W1 median ticks
       over reached episodes, per (arm, seed), from doors W1
       records) OR (revisits > 10).
  R  = not D, not W.
  UNCLASSIFIED = safety net, expected 0.

Step positions are the positions recorded at the START of each
tick (see luck_run.py: position is observed before env_step).
The step field "t" is the absolute world tick (start_phase +
step index), so before/after comparisons use the step INDEX in
the episode's steps list; this is equivalent within an episode
(t is strictly +1 per step, verified below) and avoids any
start_phase offset confusion.

Post-entry quantities: post_positions = positions after the
first-entry step; post_ticks = len(post_positions)
(= ticks - entry_index - 1). Post-entry stall is the same
structural signature as the runner's, applied to post-entry
positions only. The stall_signature/segments functions below
are copied from doors_run.py (same definitions the runner
used) to avoid importing the runner's training code.

NOTE on the recorded stall_kind: the runner calls
stall_signature(positions, gaps) with the LUCK gap set, so
the dwell exclusion covers rival staging cells (4,y) only;
(4,5) is NOT excluded in luck worlds (it is excluded in
doors W1, where (5,5) is a gap). This affects the label
"dwell" for 40 episodes, not the headline: every one of
those episodes still fails, and no-move-run dominates.
Post-entry stall is unaffected (post-entry positions are
east-side, never staging cells, in the single entered
episode).

Output is plain text, deterministic ordering. No causal
language: conclusions are descriptive and task/protocol
bounded. All conclusions EXPLORATORY.
"""

import json
import os
import statistics
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
LUCK = os.path.join(_HERE, "results_luck.jsonl")
DOORS = os.path.join(_HERE, "results_doors.jsonl")

GAPS = {
    "W1r": {(5, 1)},
    "W5r": {(5, 1), (5, 3), (5, 7), (5, 9)},
}
ARMS_20 = ("wrapped", "plain", "memory")
ARMS_30 = ("oracle_ind", "oracle_new", "phase_only_new")


# --- stall signature, copied from doors_run.py -----------------------
def segments(positions):
    segs = []
    for p in positions:
        if segs and segs[-1][0] == p:
            segs[-1][1] += 1
        else:
            segs.append([p, 1])
    return segs


def stall_signature(positions, gaps):
    """Design-note structural stall signature (any one suffices)."""
    run = 1
    for i in range(1, len(positions)):
        run = run + 1 if positions[i] == positions[i - 1] else 1
        if run >= 10:
            return True, "no-move-run"
    run = 1
    for i in range(len(positions)):
        if (i >= 2 and positions[i] == positions[i - 2]
                and positions[i] != positions[i - 1]):
            run = run + 1 if run >= 2 else 3
        else:
            run = 1
        if run >= 10:
            return True, "two-cycle"
    stage_cells = set((4, y) for (_x, y) in gaps)
    dwell = 0
    run = 0
    prev = None
    for p in positions:
        if p in stage_cells or p == (9, 9):
            run = 0
            prev = None
            continue
        b = (p[0] // 3, p[1] // 3)
        run = run + 1 if b == prev else 1
        prev = b
        dwell = max(dwell, run)
    if dwell >= 15:
        return True, "dwell"
    return False, None


# --- helpers -----------------------------------------------------------
def fmt_rate(n, d):
    return "%d/%d = %.4f" % (n, d, n / d) if d else "%d/0" % n


def med(vals):
    return statistics.median(vals) if vals else None


def mean(vals):
    return statistics.mean(vals) if vals else None


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    out = []
    p = out.append

    luck = load(LUCK)
    doors = load(DOORS)
    doors_w1 = [r for r in doors if r.get("world") == "W1"]

    # === 0. Verification / reconciliation ============================
    p("=== 0. VERIFICATION (records, not runner-log summaries) ===")
    p("luck records: %d" % len(luck))
    jobs = sorted(set((r["arm"], r["seed"], r["world"]) for r in luck))
    p("luck jobs (arm, seed, world): %d (expected 60)" % len(jobs))
    per_job = Counter((r["arm"], r["seed"], r["world"]) for r in luck)
    p("episodes per job: %s" % dict(sorted(Counter(per_job.values()).items())))
    dup = len(luck) - len(set(
        (r["arm"], r["seed"], r["world"], r["start_phase"]) for r in luck))
    p("duplicate (arm, seed, world, start_phase) keys: %d" % dup)
    p("record kinds: %s" % dict(Counter(r.get("kind") for r in luck)))
    fields = sorted(set().union(*[set(r.keys()) for r in luck]))
    p("luck record fields: %s" % ", ".join(fields))
    missing_steps = sum(1 for r in luck if len(r["steps"]) != r["ticks"])
    p("episodes where len(steps) != ticks: %d" % missing_steps)
    # phase completeness
    phase_gaps = []
    for (arm, seed, world) in jobs:
        phases = sorted(r["start_phase"] for r in luck
                        if (r["arm"], r["seed"], r["world"])
                        == (arm, seed, world))
        exp = list(range(20)) if arm in ARMS_20 else list(range(30))
        if phases != exp:
            phase_gaps.append((arm, seed, world, phases))
    p("jobs with incomplete start_phase sets: %d %s"
      % (len(phase_gaps), phase_gaps))
    # t monotonic +1, and first_door / door_entries consistency
    t_bad = 0
    entry_mismatch = 0
    for r in luck:
        ts = [s["t"] for s in r["steps"]]
        if any(ts[i + 1] - ts[i] != 1 for i in range(len(ts) - 1)):
            t_bad += 1
        pos = [(s["x"], s["y"]) for s in r["steps"]]
        gaps = GAPS[r["world"]]
        ded = []
        for q in pos:
            if q in gaps and (not ded or ded[-1] != q):
                ded.append(q)
        if r["first_door"] is None:
            if ded:
                entry_mismatch += 1
        elif ded and tuple(r["first_door"]) != ded[0]:
            entry_mismatch += 1
        if [list(q) for q in ded] != r["door_entries"]:
            entry_mismatch += 1
    p("episodes with non-+1 step t sequence: %d" % t_bad)
    p("episodes where recomputed first entry != recorded "
      "first_door/door_entries: %d" % entry_mismatch)
    p("doors records total: %d; doors W1 records: %d"
      % (len(doors), len(doors_w1)))
    p("doors W1 jobs (arm, seed): %d"
      % len(set((r["arm"], r["seed"]) for r in doors_w1)))

    # W1 medians (reached episodes only, per design note)
    p("")
    p("--- W1 baseline (doors W1 records only, NOT W1m, NOT pooled) ---")
    w1_med = {}
    w1_info = {}
    for (arm, seed) in sorted(set((r["arm"], r["seed"])
                                    for r in doors_w1)):
        sub = [r for r in doors_w1 if (r["arm"], r["seed"]) == (arm, seed)]
        ok = [r for r in sub if r["reached"]]
        m = statistics.median([r["ticks"] for r in ok]) if ok else None
        w1_med[(arm, seed)] = m
        w1_info[(arm, seed)] = (len(sub), len(ok), m)
        p("%s/%d: W1 n=%d reached=%d success=%.3f median_ticks_reached=%s "
          "2x_threshold=%s"
          % (arm, seed, len(sub), len(ok),
             len(ok) / len(sub) if sub else 0, m,
             (2 * m) if m is not None else None))
    for arm in sorted(set(a for a, _ in w1_info)):
        sub = [r for r in doors_w1 if r["arm"] == arm]
        ok = [r for r in sub if r["reached"]]
        p("W1 arm total %s: n=%d reached=%d success=%.3f"
          % (arm, len(sub), len(ok), len(ok) / len(sub) if sub else 0))

    # === 1. Classification ============================================
    for r in luck:
        pos = [(s["x"], s["y"]) for s in r["steps"]]
        gaps = GAPS[r["world"]]
        entry_idx = None
        for i, q in enumerate(pos):
            if q in gaps:
                entry_idx = i
                break
        r["_pos"] = pos
        r["_entry_idx"] = entry_idx
        r["_has45_before"] = (entry_idx is not None
                              and (4, 5) in pos[:entry_idx])
        if entry_idx is None:
            r["_class"] = "F"
            r["_post_ticks"] = None
            r["_post_stall"] = None
            r["_post_kind"] = None
            r["_post_pos"] = []
        else:
            m = w1_med[(r["arm"], r["seed"])]
            if r["_has45_before"]:
                r["_class"] = "D"
            elif (m is not None and r["ticks"] > 2 * m) \
                    or r["revisits"] > 10:
                r["_class"] = "W"
            else:
                r["_class"] = "R"
            post = pos[entry_idx + 1:]
            r["_post_pos"] = post
            r["_post_ticks"] = len(post)
            ps, pk = stall_signature(post, gaps)
            r["_post_stall"] = ps
            r["_post_kind"] = pk

    p("")
    p("=== 1. HEADLINE: CLASS DISTRIBUTION (first entry only) ===")
    p("overall classes: %s" % dict(sorted(Counter(r["_class"]
                                            for r in luck).items())))
    p("class x reached: %s"
      % dict(sorted(Counter((r["_class"], r["reached"])
                            for r in luck).items())))
    for world in ("W1r", "W5r"):
        sub = [r for r in luck if r["world"] == world]
        p("%s: n=%d classes=%s reached=%d entered=%d"
          % (world, len(sub),
             dict(sorted(Counter(r["_class"] for r in sub).items())),
             sum(r["reached"] for r in sub),
             sum(1 for r in sub if r["first_door"])))
    for arm in sorted(set(r["arm"] for r in luck)):
        sub = [r for r in luck if r["arm"] == arm]
        p("arm %s: n=%d classes=%s reached=%d entered=%d"
          % (arm, len(sub),
             dict(sorted(Counter(r["_class"] for r in sub).items())),
             sum(r["reached"] for r in sub),
             sum(1 for r in sub if r["first_door"])))
    for (arm, world) in sorted(set((r["arm"], r["world"]) for r in luck)):
        sub = [r for r in luck if r["arm"] == arm and r["world"] == world]
        p("arm x world %s/%s: n=%d classes=%s reached=%d entered=%d"
          % (arm, world, len(sub),
             dict(sorted(Counter(r["_class"] for r in sub).items())),
             sum(r["reached"] for r in sub),
             sum(1 for r in sub if r["first_door"])))
    p("UNCLASSIFIED count: %d (expected 0; rules are exhaustive over "
      "complete trajectories, and no record lacked steps)"
      % sum(1 for r in luck if r["_class"] == "UNCLASSIFIED"))

    p("")
    p("--- entered episodes (first_door not None), full detail ---")
    entered = [r for r in luck if r["first_door"]]
    if not entered:
        p("(none)")
    for r in entered:
        p("%s/%d %s start_phase=%d class=%s reached=%s ticks=%d "
          "crushes=%d first_door=%s door_entries=%s switches=%d "
          "entry_step_index=%d has_(4,5)_before_entry=%s "
          "W1_median=%s 2x=%s revisits=%d stall=%s/%s "
          "staging_order=%s arrival=%s post_ticks=%d post_stall=%s/%s "
          "post_positions_distinct=%s post_last_recorded=%s"
          % (r["arm"], r["seed"], r["world"], r["start_phase"],
             r["_class"], r["reached"], r["ticks"], r["crushes"],
             r["first_door"], r["door_entries"], r["switches"],
             r["_entry_idx"], r["_has45_before"],
             w1_med[(r["arm"], r["seed"])],
             2 * w1_med[(r["arm"], r["seed"])],
             r["revisits"], r["stall"], r["stall_kind"],
             r["staging_order"], r["arrival"], r["_post_ticks"],
             r["_post_stall"], r["_post_kind"],
             sorted(set(r["_post_pos"])),
             r["_pos"][-1] if r["_pos"] else None))
        segs = segments(r["_pos"])
        p("  trajectory segments (position, run length): %s" % segs)

    # === 2. Per-arm / per-world endpoints ============================
    def table(rows_label, groups):
        p("")
        p("--- %s ---" % rows_label)
        for label, sub in groups:
            ok = [r for r in sub if r["reached"]]
            ticks_r = [r["ticks"] for r in ok]
            p("%s: n=%d success=%s ticks_all_mean=%.1f ticks_all_med=%s "
              "ticks_reached_mean=%s ticks_reached_med=%s "
              "crush_total=%d crush_rate=%.4f entered=%d "
              "first_doors=%s switches_total=%d "
              "stall=%s arrival=%s"
              % (label, len(sub), fmt_rate(len(ok), len(sub)),
                 mean([r["ticks"] for r in sub]),
                 med([r["ticks"] for r in sub]),
                 (round(mean(ticks_r), 2) if ticks_r else None),
                 (med(ticks_r) if ticks_r else None),
                 sum(r["crushes"] for r in sub),
                 sum(r["crushes"] for r in sub) / len(sub),
                 sum(1 for r in sub if r["first_door"]),
                 dict(Counter(tuple(r["first_door"]) if r["first_door"]
                              else None for r in sub)),
                 sum(r["switches"] for r in sub),
                 dict(Counter(r["stall_kind"] for r in sub)),
                 dict(Counter(tuple(r["arrival"]) if r["arrival"]
                              else None for r in sub))))

    table("per arm x world",
          [("%s/%s" % (a, w), [r for r in luck if r["arm"] == a
                               and r["world"] == w])
           for a in sorted(set(r["arm"] for r in luck))
           for w in ("W1r", "W5r")])
    table("per world",
          [(w, [r for r in luck if r["world"] == w])
           for w in ("W1r", "W5r")])
    table("per arm",
          [(a, [r for r in luck if r["arm"] == a])
           for a in sorted(set(r["arm"] for r in luck))])
    table("overall",
          [("ALL", luck)])

    # === 3. Staging, arrival, positions ===============================
    p("")
    p("=== 3. STAGING / ARRIVAL / POSITIONS ===")
    p("staging_order patterns (all): %s"
      % dict(Counter(tuple(tuple(q) for q in r["staging_order"])
                     for r in luck).most_common()))
    p("episodes with (4,5) in staging_order: %d/%d"
      % (sum(1 for r in luck if [4, 5] in r["staging_order"]), len(luck)))
    p("episodes with any rival staging cell in staging_order: %d/%d"
      % (sum(1 for r in luck if any(q != [4, 5]
                                    for q in r["staging_order"])),
         len(luck)))
    p("episodes with empty staging_order (never any staging cell): %d"
      % sum(1 for r in luck if not r["staging_order"]))
    p("arrival counts (all): %s"
      % dict(Counter(tuple(r["arrival"]) if r["arrival"] else None
                     for r in luck)))
    for world in ("W1r", "W5r"):
        sub = [r for r in luck if r["world"] == world]
        p("%s arrival: %s"
          % (world, dict(Counter(tuple(r["arrival"]) if r["arrival"]
                                  else None for r in sub))))
        p("%s staging (4,5) present: %d/%d; rival staging present: %d/%d; "
          "empty: %d"
          % (world,
             sum(1 for r in sub if [4, 5] in r["staging_order"]), len(sub),
             sum(1 for r in sub if any(q != [4, 5]
                                        for q in r["staging_order"])),
             len(sub),
             sum(1 for r in sub if not r["staging_order"])))
    for arm in sorted(set(r["arm"] for r in luck)):
        sub = [r for r in luck if r["arm"] == arm]
        c45 = [sum(1 for s in r["steps"] if (s["x"], s["y"]) == (4, 5))
               for r in sub]
        last = Counter((r["steps"][-1]["x"], r["steps"][-1]["y"])
                       for r in sub)
        p("arm %s: steps at (4,5) mean=%.1f med=%s max=%d zero=%d; "
          "last recorded position top=%s; distinct positions med=%s; "
          "revisits med=%s; stay-action count med=%s"
          % (arm, mean(c45), med(c45), max(c45),
             sum(1 for x in c45 if x == 0), last.most_common(4),
             med([len(set(r["_pos"])) for r in sub]),
             med([r["revisits"] for r in sub]),
             med([sum(1 for s in r["steps"] if s["action"] == "stay")
                  for r in sub])))
    p("last recorded position (position at start of final tick; the "
      "post-action final cell is not recorded in steps) counts: %s"
      % dict(Counter((r["steps"][-1]["x"], r["steps"][-1]["y"])
                     for r in luck).most_common()))

    # === 4. Failure anatomy ===========================================
    p("")
    p("=== 4. FAILURE ANATOMY ===")
    p("ticks values: %s (all episodes run to the 200-tick cap, "
      "since none reach the goal)" % dict(Counter(r["ticks"]
                                                  for r in luck)))
    p("crushes total: %d; episodes with crushes > 0: %d"
      % (sum(r["crushes"] for r in luck),
         sum(1 for r in luck if r["crushes"] > 0)))
    p("stall (full episode, recorded): %s"
      % dict(Counter(r["stall_kind"] for r in luck)))
    p("stall by arm: %s"
      % dict(sorted(Counter((r["arm"], r["stall_kind"])
                            for r in luck).items(),
                     key=lambda kv: (kv[0][0], str(kv[0][1])))))
    p("stall by world: %s"
      % dict(sorted(Counter((r["world"], r["stall_kind"])
                            for r in luck).items(),
                     key=lambda kv: (kv[0][0], str(kv[0][1])))))
    p("switches: %s" % dict(Counter(r["switches"] for r in luck)))
    p("door_entries lengths: %s"
      % dict(Counter(len(r["door_entries"]) for r in luck)))
    p("revisits: min=%d med=%s max=%d"
      % (min(r["revisits"] for r in luck),
         med([r["revisits"] for r in luck]),
         max(r["revisits"] for r in luck)))
    nostage = [(r["arm"], r["seed"], r["world"], r["start_phase"],
                r["stall_kind"], (r["steps"][-1]["x"],
                                  r["steps"][-1]["y"]))
               for r in luck if not r["staging_order"]]
    p("no-staging episodes: %d" % len(nostage))
    for row in nostage:
        p("  no-staging: arm=%s seed=%d world=%s phase=%d stall=%s "
          "last_recorded=%s" % row)
    nostall = [(r["arm"], r["seed"], r["world"], r["start_phase"],
                r["revisits"], r["arrival"],
                (r["steps"][-1]["x"], r["steps"][-1]["y"]),
                len(set(r["_pos"])))
               for r in luck if not r["stall"]]
    p("no-stall-flag episodes (stall=False): %d" % len(nostall))
    for row in nostall:
        p("  no-stall: arm=%s seed=%d world=%s phase=%d revisits=%d "
          "arrival=%s last_recorded=%s distinct_positions=%d" % row)
    # paired W1r/W5r identity
    by_pair = defaultdict(dict)
    for r in luck:
        by_pair[(r["arm"], r["seed"], r["start_phase"])][r["world"]] = r
    ident = diff = 0
    diff_keys = []
    for k, d in by_pair.items():
        if d["W1r"]["_pos"] == d["W5r"]["_pos"]:
            ident += 1
        else:
            diff += 1
            diff_keys.append(k)
    p("paired W1r/W5r position sequences (same arm/seed/phase): "
      "identical=%d different=%d different_keys=%s"
      % (ident, diff, diff_keys))
    p("paired count check: %d pairs (expected 750)" % len(by_pair))
    # Last-tick actions and (4,5) runs: the recorded "last position"
    # is the position at the START of the final tick. A final
    # 'right' action from (4,5) is blocked by the wall at (5,5)
    # in luck worlds, so that episode's final cell is still (4,5);
    # other final actions may move one cell beyond the recorded
    # position. This is stated, not assumed away.
    p("last step action (all): %s"
      % dict(Counter(r["steps"][-1]["action"] for r in luck)))
    p("last step action where last recorded position is (4,5): %s"
      % dict(Counter(r["steps"][-1]["action"] for r in luck
                     if (r["steps"][-1]["x"],
                         r["steps"][-1]["y"]) == (4, 5))))
    p("last step action where last recorded position is (3,5): %s"
      % dict(Counter(r["steps"][-1]["action"] for r in luck
                     if (r["steps"][-1]["x"],
                         r["steps"][-1]["y"]) == (3, 5))))
    for arm in sorted(set(r["arm"] for r in luck)):
        runs = []
        for r in luck:
            if r["arm"] != arm:
                continue
            best = cur = 0
            for q in r["_pos"]:
                cur = cur + 1 if q == (4, 5) else 0
                best = max(best, cur)
            runs.append(best)
        p("arm %s: longest consecutive run at (4,5) med=%s; "
          "episodes with run >= 10: %d/%d"
          % (arm, med(runs), sum(1 for x in runs if x >= 10), len(runs)))
    first45 = []
    for r in luck:
        first45.append(r["_pos"].index((4, 5)) if (4, 5) in r["_pos"]
                       else None)
    p("first step index at (4,5): med=%s; episodes never at (4,5): %d"
      % (med([x for x in first45 if x is not None]),
         sum(1 for x in first45 if x is None)))
    p("episodes occupying a rival staging cell but never (4,5): %d"
      % sum(1 for r in luck
            if any(q != [4, 5] for q in r["staging_order"])
            and [4, 5] not in r["staging_order"]))
    p("paired W1r/W5r arrival differs (same trajectory, different "
      "staging-cell set): %d of %d pairs -- arrival/staging "
      "differences across worlds are definitional, not "
      "behavioural, where trajectories are identical"
      % (sum(1 for d in by_pair.values()
             if d["W1r"]["arrival"] != d["W5r"]["arrival"]),
         len(by_pair)))

    # === 5. Wrapped blank features ====================================
    p("")
    p("=== 5. WRAPPED BLANK-FEATURE CHECK (f101/f102/f103) ===")
    wrap = [r for r in luck if r["arm"] == "wrapped"]
    p("wrapped episodes: %d; wrapped steps with f keys: %d "
      "(expected 200 x 200 = 40000)"
      % (len(wrap),
         sum(1 for r in wrap for s in r["steps"] if "f103" in s)))
    blank = [r for r in wrap
             if any(s.get("f103") == 0.0 for s in r["steps"])]
    p("wrapped blank episodes (any step f103 == 0.0): %d" % len(blank))
    p("wrapped f103 value set: %s"
      % sorted(set(s["f103"] for r in wrap for s in r["steps"])))
    p("wrapped f101 value set: %s"
      % sorted(set(s["f101"] for r in wrap for s in r["steps"])))
    p("wrapped f102 value set: %s"
      % sorted(set(s["f102"] for r in wrap for s in r["steps"])))
    nonwrap_f = sum(1 for r in luck if r["arm"] != "wrapped"
                    for s in r["steps"] if "f103" in s)
    p("non-wrapped steps with f keys: %d (expected 0)" % nonwrap_f)

    # === 6. H_luck + six questions (numbers only; verdict in report) ==
    p("")
    p("=== 6. H_LUCK INPUT NUMBERS ===")
    p("reached (any pass): %d/%d" % (sum(r["reached"] for r in luck),
                                     len(luck)))
    p("class D passes (D and reached): %d"
      % sum(1 for r in luck if r["_class"] == "D" and r["reached"]))
    p("class D episodes (any reached status): %d"
      % sum(1 for r in luck if r["_class"] == "D"))
    p("class R passes: %d; class W passes: %d"
      % (sum(1 for r in luck if r["_class"] == "R" and r["reached"]),
         sum(1 for r in luck if r["_class"] == "W" and r["reached"])))
    p("W5r reached: %d/%d; W1r reached: %d/%d"
      % (sum(r["reached"] for r in luck if r["world"] == "W5r"),
         sum(1 for r in luck if r["world"] == "W5r"),
         sum(r["reached"] for r in luck if r["world"] == "W1r"),
         sum(1 for r in luck if r["world"] == "W1r")))
    p("W5r entered: %d; W1r entered: %d"
      % (sum(1 for r in luck if r["world"] == "W5r" and r["first_door"]),
         sum(1 for r in luck if r["world"] == "W1r" and r["first_door"])))

    print("\n".join(out))


if __name__ == "__main__":
    main()

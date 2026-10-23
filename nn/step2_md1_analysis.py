#!/usr/bin/env python3
"""step2_md1_analysis.py -- MD1 Step 2 deep analysis (DRAFT).

Additive only. Reads ONLY nn/results_md1.jsonl and
nn/training_md1.jsonl (plus sanity_md1.jsonl for the
seed-0 determinism reconciliation, never pooled).
Writes nn/analysis_md1_out.txt (human-readable) and
nn/analysis_md1_out.json (machine-readable). No
verdict is claimed as verified here: the §8 verdict
is computed by analysis_md1.verdict and labelled
DRAFT throughout.
"""

import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis_md1 import (TRAINED_WORLDS, HELDOUT_WORLDS,
                          ALL_FIXED_WORLDS, verdict, pooled_counts,
                          per_agent_row, staging_summary,
                          farside_summary, blank_audit)
from world_md1 import TRAINED_ROWS, HELDOUT_ROWS

RES = os.path.join(_HERE, "results_md1.jsonl")
TRN = os.path.join(_HERE, "training_md1.jsonl")
SAN = os.path.join(_HERE, "sanity_md1.jsonl")


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    recs = load(RES)
    trns = load(TRN)
    out_lines = []
    out_json = {}

    def emit(s=""):
        out_lines.append(s)
        print(s)

    # ---- completeness / duplicates ----
    jobs = collections.Counter(
        (r["arm"], r["seed"], r["world"]) for r in recs)
    missing = []
    for arm in ("wrapped", "plain", "memory", "oracle_ind",
                "oracle_new", "phase_only_new"):
        for seed in range(5):
            for w in ALL_FIXED_WORLDS + ["M3"]:
                if jobs.get((arm, seed, w), 0) != 20:
                    missing.append((arm, seed, w,
                                    jobs.get((arm, seed, w), 0)))
    ep_keys = [(r["arm"], r["seed"], r["world"], r["start_phase"])
               for r in recs]
    dup_ep = len(ep_keys) - len(set(ep_keys))
    tr_keys = [(t["arm"], t["seed"]) for t in trns]
    dup_tr = len(tr_keys) - len(set(tr_keys))
    emit("== completeness (DRAFT) ==")
    emit("  eval records: %d (expected 6600)" % len(recs))
    emit("  training records: %d (expected 30)" % len(trns))
    emit("  missing/partial jobs: %s" % (missing or "none"))
    emit("  duplicate eval episodes: %d" % dup_ep)
    emit("  duplicate training records: %d" % dup_tr)

    # ---- pooled + verdict (frozen section 8) ----
    per_world, trained, held = pooled_counts(recs)
    worlds_complete = all(per_world[w][1] > 0 for w in ALL_FIXED_WORLDS)
    held_complete = all(per_world[w][1] > 0 for w in HELDOUT_WORLDS)
    v = verdict(trained[0], trained[1], held[0], held[1],
                worlds_complete, held_complete)
    emit("== pooled + DRAFT section-8 verdict ==")
    for w in ALL_FIXED_WORLDS:
        emit("  %s: %d/%d" % (w, per_world[w][0], per_world[w][1]))
    emit("  pooled trained: %d/%d = %.4f"
         % (trained[0], trained[1], trained[0] / trained[1]))
    emit("  pooled held-out: %d/%d = %.4f"
         % (held[0], held[1], held[0] / held[1]))
    emit("  DRAFT VERDICT (UNVERIFIED): %s" % v)
    emit("  precedence applied: INCOMPLETE -> UNINTERPRETABLE -> "
         "FOLLOWING -> BIGGER ADDRESS BOOK -> NEITHER")
    out_json["pooled"] = {"trained": trained, "held": held,
                          "verdict_draft": v,
                          "per_world": {w: per_world[w]
                                        for w in ALL_FIXED_WORLDS}}

    # ---- per-arm pooled ----
    emit("== per-arm pooled (descriptive, not gating) ==")
    per_arm = {}
    for arm in ("wrapped", "plain", "memory", "oracle_ind",
                "oracle_new", "phase_only_new"):
        tr = [r for r in recs if r["arm"] == arm
              and r["world"] in TRAINED_WORLDS]
        he = [r for r in recs if r["arm"] == arm
              and r["world"] in HELDOUT_WORLDS]
        m3 = [r for r in recs if r["arm"] == arm and r["world"] == "M3"]
        per_arm[arm] = {
            "trained": (sum(r["reached"] for r in tr), len(tr)),
            "held": (sum(r["reached"] for r in he), len(he)),
            "m3": (sum(r["reached"] for r in m3), len(m3))}
        emit("  %s: trained %d/%d, held %d/%d, M3 %d/%d"
             % (arm, per_arm[arm]["trained"][0],
                per_arm[arm]["trained"][1], per_arm[arm]["held"][0],
                per_arm[arm]["held"][1], per_arm[arm]["m3"][0],
                per_arm[arm]["m3"][1]))
    out_json["per_arm"] = per_arm

    # ---- per-agent x per-row table ----
    emit("== per-agent x per-row reached (/20 each) ==")
    table = per_agent_row(recs)
    hdr = "agent       " + " ".join("%4s" % w for w in ALL_FIXED_WORLDS)
    emit(hdr)
    table_json = {}
    for (arm, seed) in sorted(table):
        row = table[(arm, seed)]
        table_json["%s/%d" % (arm, seed)] = {
            w: row[w] for w in ALL_FIXED_WORLDS}
        emit("%-11s %s" % ("%s/%d" % (arm, seed),
                           " ".join("%4d" % row[w][0]
                                    for w in ALL_FIXED_WORLDS)))
    out_json["per_agent_row"] = table_json

    # ---- held-out reached detail: which agents, staging ----
    emit("== held-out reached episodes (detail) ==")
    held_reached = [r for r in recs if r["world"] in HELDOUT_WORLDS
                    and r["reached"]]
    for r in held_reached:
        emit("  %s/%d %s phase %d: staged=%s arrival=%s ticks=%d"
             % (r["arm"], r["seed"], r["world"], r["start_phase"],
                r["staging_order"], r["arrival"], r["ticks"]))
    st = staging_summary(recs)
    for w, (s, n) in sorted(st.items()):
        emit("  staging %s: %d/%d held-out reached staged at (4, row)"
             % (w, s, n))
    out_json["held_reached"] = [
        {"arm": r["arm"], "seed": r["seed"], "world": r["world"],
         "phase": r["start_phase"], "staging": r["staging_order"],
         "arrival": r["arrival"], "ticks": r["ticks"]}
        for r in held_reached]

    # ---- staging on reached episodes generally ----
    emit("== staging on reached episodes (all fixed worlds) ==")
    for w in ALL_FIXED_WORLDS:
        row = int(w[1:])
        sub = [r for r in recs if r["world"] == w and r["reached"]]
        staged = sum(1 for r in sub if [4, row] in r["staging_order"])
        emit("  %s: %d/%d reached staged at (4, %d)"
             % (w, staged, len(sub), row))

    # ---- far side ----
    emit("== far side (entered, entered-not-reached, "
         "post-entry-stalled-unreached) ==")
    fs = farside_summary(recs)
    for w, (e, enr, stalled) in sorted(fs.items()):
        emit("  %s: %d, %d, %d" % (w, e, enr, stalled))
    out_json["farside"] = {w: fs[w] for w in ALL_FIXED_WORLDS}
    # stall kinds on unreached episodes, per world
    emit("== stall kinds on unreached episodes ==")
    for w in ALL_FIXED_WORLDS:
        sub = [r for r in recs if r["world"] == w and not r["reached"]]
        kinds = collections.Counter(r["stall_kind"] for r in sub)
        emit("  %s: %s" % (w, dict(kinds)))
    # modal final cells of unreached episodes (descriptive geography)
    emit("== modal final cells, unreached episodes (top 5 per world) ==")
    for w in ALL_FIXED_WORLDS:
        sub = [r for r in recs if r["world"] == w and not r["reached"]
               and r["steps"]]
        finals = collections.Counter(
            (r["steps"][-1]["x"], r["steps"][-1]["y"]) for r in sub)
        emit("  %s: %s" % (w, finals.most_common(5)))

    # ---- blank-feature audit: drift vs warm-up (wrapped only) ----
    # blank_feature is measured only for wrapped (f101-f103 logged
    # only when log_features=True, i.e. wrapped jobs). Non-wrapped
    # arms report blank=False by construction, not by measurement.
    emit("== blank-feature audit (wrapped only; other arms unmeasured) ==")
    ba = blank_audit(recs)
    for w, a in sorted(ba.items()):
        emit("  %s: blank %d (reached %d); all %d/%d; excluded %d/%d"
             % (w, a["blank"], a["blank_reached"], a["all"][0],
                a["all"][1], a["excluded"][0], a["excluded"][1]))
    wrapped_blank = [r for r in recs if r["arm"] == "wrapped"
                     and r["blank_feature"]]
    # Classification (corrected): a blank episode is a PREFIX blank
    # if every known=0 step precedes every known=1 step, whatever
    # the prefix length; SCATTERED if known flickers; DRIFT only
    # if the whole episode is blank. An earlier draft of this
    # script bucketed prefixes longer than 10 steps as "mixed";
    # that label was wrong (all such episodes are pure prefixes)
    # and is corrected here.
    drift, prefix, scattered = [], [], []
    for r in wrapped_blank:
        f103 = [s.get("f103") for s in r["steps"]
                if "f103" in s]
        n_blank = sum(1 for x in f103 if x == 0.0)
        if n_blank == len(f103):
            drift.append(r)
        elif all(x == 0.0 for x in f103[:n_blank]) and all(
                x != 0.0 for x in f103[n_blank:]):
            prefix.append(r)
        else:
            scattered.append(r)
    emit("  wrapped blank episodes total: %d" % len(wrapped_blank))
    emit("  whole-episode blank (drift candidate): %d" % len(drift))
    emit("  pure-prefix blank (warm-up shape): %d" % len(prefix))
    emit("  scattered blank (known flickers): %d" % len(scattered))
    # per-job concentration: blank episodes by (world, seed, phase)
    conc = collections.Counter(
        (r["world"], r["seed"], r["start_phase"]) for r in wrapped_blank)
    emit("  blank episodes by (world, seed, phase) -- first-phase "
         "concentration check:")
    by_phase = collections.Counter(r["start_phase"]
                                   for r in wrapped_blank)
    emit("    by start_phase: %s" % dict(sorted(by_phase.items())))
    by_world = collections.Counter(r["world"] for r in wrapped_blank)
    emit("    by world: %s" % dict(sorted(by_world.items())))
    # exclusion sensitivity on pooled numbers (blank excluded)
    clean = [r for r in recs if not r["blank_feature"]]
    pw2, tr2, he2 = pooled_counts(clean)
    emit("  pooled with blank episodes EXCLUDED: trained %d/%d "
         "= %.4f; held %d/%d = %.4f; verdict %s"
         % (tr2[0], tr2[1], tr2[0] / tr2[1], he2[0], he2[1],
            he2[0] / he2[1],
            verdict(tr2[0], tr2[1], he2[0], he2[1], True, True)))
    out_json["blank"] = {
        "total_wrapped_blank": len(wrapped_blank),
        "drift_candidate": len(drift), "prefix": len(prefix),
        "scattered": len(scattered),
        "by_phase": {str(k): v for k, v in sorted(by_phase.items())},
        "by_world": dict(sorted(by_world.items())),
        "excluded_pooled": {"trained": tr2, "held": he2,
                            "verdict_draft": verdict(
                                tr2[0], tr2[1], he2[0], he2[1],
                                True, True)}}

    # ---- M3 sanity ----
    emit("== M3 sanity ==")
    m3 = [r for r in recs if r["world"] == "M3"]
    rows_seen = sorted({r["gap_row"] for r in m3})
    emit("  M3 episodes: %d, reached %d" % (
        len(m3), sum(r["reached"] for r in m3)))
    emit("  M3 gap_rows seen: %s (trained-only expected: %s)"
         % (rows_seen, list(TRAINED_ROWS)))
    emit("  M3 held-out rows present: %s"
         % sorted({r["gap_row"] for r in m3} & set(HELDOUT_ROWS)))

    # ---- training reconciliation ----
    emit("== training reconciliation ==")
    emit("  per-arm training reached (programme records):")
    for arm in ("wrapped", "plain", "memory", "oracle_ind",
                "oracle_new", "phase_only_new"):
        sub = [t for t in trns if t["arm"] == arm]
        tot = sum(t["reached"] for t in sub)
        emit("    %s: %d/10000 across seeds 0-4; row_counts all "
             "balanced: %s"
             % (arm, tot, all(
                 all(t["row_counts"][str(r)] == 250
                     for r in TRAINED_ROWS) for t in sub)))
    # seed-0 programme vs sanity (determinism check, NOT pooling)
    san = {(s["arm"], s["seed"]): s for s in load(SAN)}
    emit("  seed-0 programme vs sanity (determinism check):")
    for arm in ("wrapped", "plain", "memory", "oracle_ind",
                "oracle_new", "phase_only_new"):
        t = next(t for t in trns if t["arm"] == arm and t["seed"] == 0)
        s = san[(arm, 0)]
        match = (t["reached"] == s["reached"]
                 and t["crushes"] == s["crushes"]
                 and t["mean_ticks_reached"] == s["mean_ticks_reached"]
                 and t["row_counts"] == s["row_counts"]
                 and t["row_reached"] == s["row_reached"])
        emit("    %s/0: exact match: %s" % (arm, match))
    # row_reached lows in training
    emit("  training row_reached totals across 30 agents:")
    for r in TRAINED_ROWS:
        tot = sum(t["row_reached"][str(r)] for t in trns)
        emit("    row %d: %d/7500" % (r, tot))

    with open(os.path.join(_HERE, "analysis_md1_out.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")
    with open(os.path.join(_HERE, "analysis_md1_out.json"), "w") as f:
        json.dump(out_json, f, indent=2, sort_keys=True)
    print("WROTE analysis_md1_out.txt + analysis_md1_out.json")


if __name__ == "__main__":
    main()

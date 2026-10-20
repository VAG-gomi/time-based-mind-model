#!/usr/bin/env python3
"""analyze_trust15_beta.py -- frozen analysis for the
confirmatory beta-anneal test (nn/prereg_trust15_beta.md,
frozen at c2320cb). Implements the prereg's rules EXACTLY,
run once on the complete 60 records:

- Endpoint per record: first80 (first episode whose
  rolling-100 >= 0.8); never -> censored, value 1000.
- Primary contrast D1 = annealed - beta1 per seed; secondary
  contrast D2 = beta1 - uniform. Each with: sign split,
  median, Wilcoxon signed-rank (scipy, two-sided, alpha
  0.05), per-contrast censoring counts, and the pre-stated
  censoring-limited label at >= 5 one-sided-censored pairs.
- D1 conclusion (exactly two): rejects AND median > 0 ->
  "ANNEAL IMPLICATED (confirmatory)"; otherwise "ANNEAL
  EXONERATED as the primary cause" (negative-median
  rejection is the pre-stated sub-case, direction reported).
- D2 reading (detection only): rejects AND median > 0 ->
  "residual systematic disadvantage DETECTED"; otherwise
  "not detected" (NOT equivalence; sensitivity bound in the
  prereg applies).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy import stats as sstats

SEEDS = tuple(range(500, 520))
CENSORED_VALUE = 1000


def first80(reached):
    run = 0
    for i, r in enumerate(reached):
        run += r
        if i >= 100:
            run -= reached[i - 100]
        if i >= 99 and run / 100 >= 0.8:
            return i
    return None


def contrast(rows, hi, lo):
    """hi/lo: keys of the per-seed endpoint dict. Returns the
    frozen statistics for D = value(hi) - value(lo)."""
    D = np.array([r[hi + "_val"] - r[lo + "_val"] for r in rows],
                 dtype=float)
    n_one = sum(1 for r in rows if r[hi + "_cens"] != r[lo + "_cens"])
    if np.all(D == 0):
        w_stat, w_p = 0.0, 1.0
    else:
        w = sstats.wilcoxon(D)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    return {
        "D": D,
        "sign_pos": int(np.sum(D > 0)), "sign_neg": int(np.sum(D < 0)),
        "sign_zero": int(np.sum(D == 0)),
        "median": float(np.median(D)),
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "rejected": bool(w_p < 0.05),
        "n_one_sided_cens": n_one,
        "censoring_limited": n_one >= 5,
    }


def analyze(path):
    recs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[(r["arm"], r["seed"])] = r
    assert len(recs) == 60, "expected 60 records, found %d" % len(recs)

    rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for arm, key in (("uniform", "u"), ("trust15", "a"),
                         ("trust15_beta1", "b")):
            v = first80(recs[(arm, seed)]["reached"])
            row[key + "80"] = v
            row[key + "_val"] = CENSORED_VALUE if v is None else v
            row[key + "_cens"] = v is None
        rows.append(row)

    d1 = contrast(rows, "a", "b")   # annealed - beta1
    d2 = contrast(rows, "b", "u")   # beta1 - uniform

    if d1["rejected"] and d1["median"] > 0:
        d1_conclusion = "ANNEAL IMPLICATED (confirmatory)"
    elif d1["rejected"] and d1["median"] < 0:
        d1_conclusion = ("ANNEAL EXONERATED as the primary cause "
                          "(test rejected with a negative median: "
                          "the annealed regime crossed earlier "
                          "than beta1 on these seeds)")
    else:
        d1_conclusion = "ANNEAL EXONERATED as the primary cause"
    d2_reading = ("residual systematic disadvantage DETECTED"
                  if (d2["rejected"] and d2["median"] > 0)
                  else "residual systematic disadvantage not "
                       "detected (NOT equivalence; see prereg "
                       "sensitivity bound)")
    return {"rows": rows, "d1": d1, "d2": d2,
            "d1_conclusion": d1_conclusion, "d2_reading": d2_reading}


def fmt(v, cens):
    return "never" if cens else str(v)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, "results_trust15_beta.jsonl")
    o = analyze(path)
    print("per-seed (uniform, annealed, beta1, D1, D2):")
    for r, d1v, d2v in zip(o["rows"], o["d1"]["D"], o["d2"]["D"]):
        print("  %d: %s, %s, %s, D1=%+d, D2=%+d"
              % (r["seed"], fmt(r["u80"], r["u_cens"]),
                 fmt(r["a80"], r["a_cens"]), fmt(r["b80"], r["b_cens"]),
                 d1v, d2v))
    for name, c in (("D1 annealed-beta1", o["d1"]),
                    ("D2 beta1-uniform", o["d2"])):
        print("%s: sign +%d/-%d/0:%d, median %+.1f, Wilcoxon "
              "stat=%.1f p=%.4f, one-sided censored pairs %d%s"
              % (name, c["sign_pos"], c["sign_neg"], c["sign_zero"],
                 c["median"], c["wilcoxon_stat"], c["wilcoxon_p"],
                 c["n_one_sided_cens"],
                 " -> CENSORING-LIMITED" if c["censoring_limited"]
                 else ""))
    print("D1 CONCLUSION: %s" % o["d1_conclusion"])
    print("D2 READING: %s" % o["d2_reading"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""analyze_trust15_parity.py -- frozen analysis for the
beta1 learning-parity evaluation (nn/prereg_trust15_parity.md,
frozen at 638956f). Implements the prereg's rules EXACTLY,
run once on the complete 60 records:

- Endpoint per record: first80 (first episode whose
  rolling-100 >= 0.8); never -> censored, value 1000.
- Per-seed difference D = first80(beta1) - first80(uniform),
  with per-pair censoring indicators; both-censored pairs
  have D = 0 and are counted.
- Median of D with a 95% CI by EXACT binomial order
  statistics (normative construction): sort D ascending,
  1-indexed X_(1)..X_(n); l = smallest k with
  BinCDF(k; n, 0.5) >= 0.025; u = n - l + 1. For n = 30:
  l = 10, u = 21; the decision uses X_(21). No normal
  approximation anywhere in this interval.
- Margin +50 episodes. PARITY ESTABLISHED iff the CI upper
  endpoint < +50; otherwise PARITY NOT ESTABLISHED, with
  the CI reported. Pre-stated sub-case: CI entirely above
  0 (lower endpoint > 0) means a systematic disadvantage
  for beta1 is DETECTED.
- Censoring-limited label at >= 5 one-sided-censored pairs.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy import stats as sstats

SEEDS = tuple(range(520, 550))
CENSORED_VALUE = 1000
MARGIN = 50


def first80(reached):
    run = 0
    for i, r in enumerate(reached):
        run += r
        if i >= 100:
            run -= reached[i - 100]
        if i >= 99 and run / 100 >= 0.8:
            return i
    return None


def median_ci(sorted_d):
    """Exact binomial order-statistic CI for the median.
    Returns (l, u, lower, upper) with 1-indexed ranks."""
    n = len(sorted_d)
    l = int(sstats.binom.ppf(0.025, n, 0.5))
    u = n - l + 1
    return l, u, sorted_d[l - 1], sorted_d[u - 1]


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
        for arm, key in (("uniform", "u"), ("trust15_beta1", "b")):
            v = first80(recs[(arm, seed)]["reached"])
            row[key + "80"] = v
            row[key + "_val"] = CENSORED_VALUE if v is None else v
            row[key + "_cens"] = v is None
        row["D"] = row["b_val"] - row["u_val"]
        rows.append(row)

    D = np.array([r["D"] for r in rows], dtype=float)
    srt = sorted(D)
    l, u, lo, hi = median_ci(srt)
    n_one = sum(1 for r in rows if r["u_cens"] != r["b_cens"])
    n_both = sum(1 for r in rows if r["u_cens"] and r["b_cens"])

    if hi < MARGIN:
        conclusion = "PARITY ESTABLISHED"
    elif lo > 0:
        conclusion = ("PARITY NOT ESTABLISHED (systematic "
                      "disadvantage for beta1 DETECTED: CI lies "
                      "entirely above 0)")
    else:
        conclusion = "PARITY NOT ESTABLISHED"
    return {
        "rows": rows, "D": D, "sorted_D": srt,
        "median": float(np.median(D)),
        "l": l, "u": u, "ci_lo": lo, "ci_hi": hi,
        "n_one_sided_cens": n_one, "n_both_cens": n_both,
        "censoring_limited": n_one >= 5,
        "conclusion": conclusion,
    }


def fmt(v, cens):
    return "never" if cens else str(v)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, "results_trust15_parity.jsonl")
    o = analyze(path)
    print("per-seed (uniform, beta1, D):")
    for r in o["rows"]:
        print("  %d: %s, %s, D=%+d"
              % (r["seed"], fmt(r["u80"], r["u_cens"]),
                 fmt(r["b80"], r["b_cens"]), r["D"]))
    print("median D=%+.1f, 95%% CI=[%+d, %+d] (ranks %d/%d), "
          "margin +50" % (o["median"], o["ci_lo"], o["ci_hi"],
                           o["l"], o["u"]))
    print("censoring: one-sided pairs %d, both-censored pairs %d%s"
          % (o["n_one_sided_cens"], o["n_both_cens"],
             " -> CENSORING-LIMITED" if o["censoring_limited"] else ""))
    print("CONCLUSION: %s" % o["conclusion"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""analyze_trust15_var.py -- frozen analysis for the variance-
null test (nn/prereg_trust15_var.md, frozen at d060f50).

Implements the prereg's decision rule EXACTLY, run once on the
complete 40 records:

- Endpoint per record: first80 = first episode index whose
  rolling-100 success (window ep-99..ep) is >= 0.8; if never,
  the record is censored and assigned 1000.
- Per pair: D = first80(trust15) - first80(uniform) on the
  substituted values; censoring indicator per pair.
- Primary: Wilcoxon signed-rank on the 20 D values (scipy
  default zero handling), two-sided alpha 0.05; sign split
  and median of D reported.
- Conclusion (exactly two): rejects AND median(D) > 0 ->
  "NULL REFUTED"; otherwise "NULL STANDS" (with the pre-stated
  negative-median sub-case noted). >= 5 one-sided-censored
  pairs attaches the pre-stated "censoring-limited" label.
- Secondaries (descriptive): variance ratio of first80
  between arms (substituted values), final-100 success per
  arm, per-arm median first80.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy import stats as sstats

SEEDS = tuple(range(210, 230))
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


def analyze(path):
    recs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[(r["arm"], r["seed"])] = r
    assert len(recs) == 40, "expected 40 records, found %d" % len(recs)

    rows = []
    for seed in SEEDS:
        u = first80(recs[("uniform", seed)]["reached"])
        t = first80(recs[("trust15", seed)]["reached"])
        rows.append({
            "seed": seed,
            "u80": u, "t80": t,
            "u_val": CENSORED_VALUE if u is None else u,
            "t_val": CENSORED_VALUE if t is None else t,
            "u_cens": u is None, "t_cens": t is None,
            "u_f100": float(np.mean(recs[("uniform", seed)]["reached"][-100:])),
            "t_f100": float(np.mean(recs[("trust15", seed)]["reached"][-100:])),
        })
        rows[-1]["D"] = rows[-1]["t_val"] - rows[-1]["u_val"]

    D = np.array([r["D"] for r in rows], dtype=float)
    n_one_sided_cens = sum(1 for r in rows if r["u_cens"] != r["t_cens"])
    n_cens_u = sum(1 for r in rows if r["u_cens"])
    n_cens_t = sum(1 for r in rows if r["t_cens"])
    sign_pos = int(np.sum(D > 0))
    sign_neg = int(np.sum(D < 0))
    sign_zero = int(np.sum(D == 0))
    median_D = float(np.median(D))
    if np.all(D == 0):
        w_stat, w_p = 0.0, 1.0
    else:
        w = sstats.wilcoxon(D)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    rejected = bool(w_p < 0.05)
    if rejected and median_D > 0:
        conclusion = "NULL REFUTED"
    elif rejected and median_D < 0:
        conclusion = ("NULL STANDS (no systematic disadvantage); "
                      "a systematic advantage was observed")
    else:
        conclusion = "NULL STANDS"
    censoring_limited = n_one_sided_cens >= 5

    u_vals = np.array([r["u_val"] for r in rows], dtype=float)
    t_vals = np.array([r["t_val"] for r in rows], dtype=float)
    out = {
        "rows": rows, "D": D, "median_D": median_D,
        "sign_pos": sign_pos, "sign_neg": sign_neg,
        "sign_zero": sign_zero,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "rejected": rejected,
        "n_cens_uniform": n_cens_u, "n_cens_trust15": n_cens_t,
        "n_one_sided_cens": n_one_sided_cens,
        "censoring_limited": censoring_limited,
        "conclusion": conclusion,
        "var_uniform": float(np.var(u_vals, ddof=1)),
        "var_trust15": float(np.var(t_vals, ddof=1)),
        "median_u80": float(np.median(u_vals)),
        "median_t80": float(np.median(t_vals)),
        "f100_uniform": float(np.mean([r["u_f100"] for r in rows])),
        "f100_trust15": float(np.mean([r["t_f100"] for r in rows])),
    }
    if out["var_uniform"] > 0:
        out["var_ratio"] = out["var_trust15"] / out["var_uniform"]
    else:
        out["var_ratio"] = (float("inf") if out["var_trust15"] > 0
                            else float("nan"))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, "results_trust15_var.jsonl")
    o = analyze(path)
    print("per-pair (seed: uniform, trust15, D, censored):")
    for r in o["rows"]:
        print("  %d: %s, %s, D=%+d%s"
              % (r["seed"],
                 "never" if r["u_cens"] else str(r["u80"]),
                 "never" if r["t_cens"] else str(r["t80"]),
                 r["D"],
                 " [censored]" if (r["u_cens"] or r["t_cens"]) else ""))
    print("sign split: +%d / -%d / 0:%d"
          % (o["sign_pos"], o["sign_neg"], o["sign_zero"]))
    print("median D: %+.1f episodes" % o["median_D"])
    print("Wilcoxon: stat=%.1f p=%.4f (two-sided, alpha 0.05)"
          % (o["wilcoxon_stat"], o["wilcoxon_p"]))
    print("censoring: uniform %d, trust15 %d, one-sided pairs %d%s"
          % (o["n_cens_uniform"], o["n_cens_trust15"],
             o["n_one_sided_cens"],
             " -> CENSORING-LIMITED" if o["censoring_limited"] else ""))
    print("CONCLUSION: %s" % o["conclusion"])
    print("secondaries (descriptive): var ratio t15/uniform %.3f; "
          "final-100 uniform %.3f trust15 %.3f; median first80 "
          "uniform %.1f trust15 %.1f"
          % (o["var_ratio"], o["f100_uniform"], o["f100_trust15"],
             o["median_u80"], o["median_t80"]))


if __name__ == "__main__":
    main()

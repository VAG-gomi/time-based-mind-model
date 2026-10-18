#!/usr/bin/env python3
"""validate_analyze_expa.py -- prove analyze_expa.py correct on
SYNTHETIC data before the real Experiment A records exist.

Checks (all synthetic; never touches real result files):
1. mcnemar_exact_p matches scipy's independent binomtest on random
   (b, c) tables.
2. holm_reject matches a separately written textbook Holm procedure
   on random p-value sets.
3. bootstrap_median_diff has sane coverage of the true median gap
   across synthetic datasets.
4. clean_fast_744 boundary behavior at the 0.744 bar.
5. check_records: missing seeds are reported and excludable (not a
   failure); wrong period, duplicates, unexpected arms fail loudly.

Exits nonzero on any check failure.
"""

import json
import math
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import analyze_expa as A

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + (" -- " + str(detail) if detail and not cond else ""))


def synth_rec(arm, seed, crush, success, ret, period=30, open_len=14):
    return {
        "kind": "arm", "arm": arm, "seed": seed,
        "hyperparams": {"hidden": 32, "lr": 0.001, "episodes": 2000,
                        "gamma": 0.99, "period": period,
                        "open_len": open_len},
        "greedy": {"mean_return": ret, "crush_rate": crush,
                   "success": success, "mean_ticks": 25.0, "n": 30},
        "eps005": {"mean_return": ret, "crush_rate": crush,
                   "success": success},
        "train": {"last100_success": 1.0},
    }


# 1. McNemar vs scipy binomtest (independent code path).
from scipy.stats import binomtest
rng = random.Random(123)
ok = True
worst = 0.0
for _ in range(300):
    b, c = rng.randrange(0, 60), rng.randrange(0, 60)
    mine = A.mcnemar_exact_p(b, c)
    ref = binomtest(min(b, c), b + c, 0.5,
                    alternative="two-sided").pvalue if b + c else 1.0
    worst = max(worst, abs(mine - ref))
    if abs(mine - ref) > 1e-9:
        ok = False
check("mcnemar matches scipy binomtest (300 tables)",
      ok, "max abs diff %.2e" % worst)


# 2. Holm vs textbook procedure written separately here.
def textbook_holm(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    rej = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            rej[i] = True
        else:
            break
    return rej


ok = True
for _ in range(500):
    m = rng.randrange(1, 6)
    ps = [rng.random() ** 2 for _ in range(m)]  # skew small
    names = ["h%d" % i for i in range(m)]
    mine = A.holm_reject(list(zip(names, ps)))
    ref = textbook_holm(ps)
    if any(mine[n] != r for n, r in zip(names, ref)):
        ok = False
        break
check("holm_reject matches textbook Holm (500 sets)", ok)


# 2b. Two-flag equivalence labels (literal reading).
cases = [
    # (detected, within, lo, hi, expected)
    (True, True, -0.10, 0.10, "small difference, within margin"),
    (True, False, 0.30, 0.50, "difference (beyond margin)"),
    (True, False, -0.30, -0.26, "difference (beyond margin)"),
    (True, False, 0.03, 0.44, "difference"),
    (False, True, -0.10, 0.10, "equivalent"),
    (False, False, -0.04, 0.38, "indistinguishable"),
    (False, False, 0.03, 0.44, "indistinguishable"),
]
ok = all(A.two_flag_verdict(d, w, lo, hi) == e
         for d, w, lo, hi, e in cases)
check("two_flag_verdict label cells (7 cases)", ok)
# The synthetic end-to-end case: Holm did NOT reject (adjusted p
# 0.0532), CI not within margin -> stays "indistinguishable".
check("synthetic [+0.030,+0.439] case -> indistinguishable",
      A.two_flag_verdict(False, False, 0.030, 0.439)
      == "indistinguishable")
cov = 0
T = 200
for t in range(T):
    r = np.random.default_rng(10_000 + t)
    diffs = r.normal(0.05, 0.10, size=100)
    x = diffs
    y = np.zeros_like(diffs)
    _, lo, hi = A.bootstrap_median_diff(x, y, seed=20_000 + t,
                                       resamples=2000)
    if lo <= 0.05 <= hi:
        cov += 1
check("bootstrap median-diff coverage sane", cov / T >= 0.80,
      "coverage %.3f over %d datasets" % (cov / T, T))


# 4. clean-and-fast boundary at 0.744.
r = synth_rec("wrapped_ind", 200, 0.0, 1.0, 0.744)
check("return exactly 0.744 -> clean-and-fast", A.clean_fast_744(r))
r = synth_rec("wrapped_ind", 200, 0.0, 1.0, 0.7439999)
check("return 0.7439999 -> not clean-and-fast",
      not A.clean_fast_744(r))
r = synth_rec("wrapped_ind", 200, 0.033, 1.0, 0.90)
check("one crush -> not clean-and-fast", not A.clean_fast_744(r))
r = synth_rec("wrapped_ind", 200, 0.0, 0.89, 0.90)
check("success 0.89 -> not clean-and-fast", not A.clean_fast_744(r))


# 5. Record checks.
def expect_fail(name, recs):
    try:
        A.check_records(recs)
    except SystemExit:
        check(name, True)
    else:
        check(name, False, "did not fail")


recs = []
for s in range(200, 210):
    recs.append(synth_rec("wrapped_ind", s, 0.0, 1.0, 0.75))
    recs.append(synth_rec("phase_only_new", s, 0.0, 1.0, 0.74))
for s in range(200, 205):
    recs.append(synth_rec("oracle_ind", s, 0.0, 1.0, 0.75))
    recs.append(synth_rec("memory30", s, 0.0, 1.0, 0.74))
# drop one seed from one arm -> missing-data path, must NOT fail
recs2 = [r for r in recs
         if not (r["arm"] == "oracle_ind" and r["seed"] == 203)]
import io
from contextlib import redirect_stdout
buf = io.StringIO()
with redirect_stdout(buf):
    by_arm, missing = A.check_records(recs2)
check("missing seed reported, not fatal",
      203 in missing.get("oracle_ind", []),
      {a: v[:3] for a, v in missing.items()})
both, excl = A.paired_seeds(by_arm, "wrapped_ind", "oracle_ind",
                            A.SECONDARY_SEEDS)
check("incomplete pair excluded", 203 in excl and 203 not in both,
      (len(both), excl[:3]))

bad = [dict(r) for r in recs]
bad[0] = dict(bad[0])
bad[0]["hyperparams"] = dict(bad[0]["hyperparams"], period=20)
expect_fail("wrong period fails", bad)
expect_fail("duplicate fails", recs + [recs[0]])
bad2 = [dict(r) for r in recs]
bad2[0] = dict(bad2[0], arm="mystery_arm")
expect_fail("unexpected arm fails", bad2)

print("---")
n_fail = sum(1 for _, c in PASS if not c)
print("%d/%d checks passed" % (len(PASS) - n_fail, len(PASS)))
sys.exit(1 if n_fail else 0)

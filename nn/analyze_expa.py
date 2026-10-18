#!/usr/bin/env python3
"""analyze_expa.py -- Experiment A (generalization, 30/14) analysis.

Implements EXACTLY the analyses pre-registered in
nn/prereg_generalization.md v3.2 (Experiment A):
- record/protocol checks (280 records; period 30; open_len 14;
  seeds 200-299 primary pair, 200-239 secondaries; missing-data rule:
  incomplete pairs are identified and excluded, not silently kept)
- primary: clean-and-fast (>=0.744) paired wrapped_ind vs
  phase_only_new, exact McNemar, alpha=0.05 (single primary, no Holm)
- secondaries: wrapped_ind vs oracle_ind, wrapped_ind vs memory30;
  Holm over the 2 comparisons; equivalence margin +-0.25 on the
  paired rate difference with "indistinguishable" as third outcome
- seed-bootstrap 95% CIs (10,000 resamples, resampling seeds) for
  median-return differences

Deterministic. Exits nonzero on duplicate/unexpected records or
protocol mismatch. Missing seeds are reported and their pairs
excluded per the pre-registered missing-data rule.

Usage: python3 analyze_expa.py
"""

import json
import math
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from run_arms import _clean_fast as _runner_clean_fast

FAST_BAR = 0.744
PRIMARY_SEEDS = list(range(200, 300))
SECONDARY_SEEDS = list(range(200, 240))
EXPECTED = {
    "wrapped_ind": PRIMARY_SEEDS,
    "phase_only_new": PRIMARY_SEEDS,
    "oracle_ind": SECONDARY_SEEDS,
    "memory30": SECONDARY_SEEDS,
}
BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 10000
EQUIV_MARGIN = 0.25


def fail(msg):
    print("FAIL: %s" % msg, file=sys.stderr)
    raise SystemExit(1)


def clean_fast_744(rec):
    """Clean-and-fast at the pre-registered 0.744 bar, computed from
    the raw record -- never from the runner's module-global default."""
    g = rec["greedy"]
    mine = (g["crush_rate"] == 0.0 and g["success"] >= 0.9
            and g["mean_return"] >= FAST_BAR)
    theirs = _runner_clean_fast(rec, FAST_BAR)
    if mine != theirs:
        fail("clean-and-fast drift vs run_arms for %r seed %r"
             % (rec.get("arm"), rec.get("seed")))
    return mine


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_records(records):
    """Validate protocol. Returns (by_arm, missing) where missing maps
    arm -> sorted list of absent seeds (pre-registered missing-data
    rule: identified here, excluded from paired analyses)."""
    seen = set()
    by_arm = {}
    for r in records:
        key = (r.get("arm"), r.get("seed"))
        if key in seen:
            fail("duplicate record %r" % (key,))
        seen.add(key)
        if r.get("kind") != "arm":
            fail("unexpected kind %r" % (r.get("kind"),))
        if r.get("arm") not in EXPECTED:
            fail("unexpected arm %r" % (r.get("arm"),))
        hp = r["hyperparams"]
        if hp.get("period") != 30 or hp.get("open_len") != 14:
            fail("protocol mismatch in %r: period=%r open_len=%r"
                 % (key, hp.get("period"), hp.get("open_len")))
        by_arm.setdefault(r["arm"], {})[r["seed"]] = r
    missing = {}
    for arm, want in EXPECTED.items():
        got = sorted(by_arm.get(arm, {}))
        absent = [s for s in want if s not in got]
        if absent:
            missing[arm] = absent
    total = sum(len(v) for v in by_arm.values())
    print("records: %d/280 present" % total)
    if missing:
        for arm in sorted(missing):
            print("  MISSING %s seeds: %s (pairs excluded)"
                  % (arm, missing[arm]))
    else:
        print("  no missing seeds")
    return by_arm, missing


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(2 * p, 1.0)


def paired_diff_ci(b, c, n, alpha=0.05):
    from scipy.stats import beta
    d = b + c
    if d == 0:
        return 0.0, 0.0
    lo_p = beta.ppf(alpha / 2, b, d - b + 1) if b > 0 else 0.0
    hi_p = beta.ppf(1 - alpha / 2, b + 1, d - b) if b < d else 1.0
    return (2 * lo_p - 1) * d / n, (2 * hi_p - 1) * d / n


def holm_reject(pvals, alpha=0.05):
    """Holm step-down. pvals: list of (name, p). Returns dict
    name -> True/False (reject or not), stopping at first
    non-rejection."""
    ordered = sorted(pvals, key=lambda t: t[1])
    m = len(ordered)
    out = {}
    for i, (name, p) in enumerate(ordered):
        if p <= alpha / (m - i):
            out[name] = True
        else:
            for name2, _ in ordered[i:]:
                out[name2] = False
            break
    return out


def bootstrap_median_diff(x, y, seed=BOOTSTRAP_SEED,
                          resamples=BOOTSTRAP_RESAMPLES):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(x) - np.asarray(y)
    obs = float(np.median(diffs))
    boots = [float(np.median(rng.choice(diffs, size=len(diffs),
                                        replace=True)))
             for _ in range(resamples)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def paired_seeds(by_arm, a1, a2, want):
    """Seeds present in BOTH arms (missing-data rule)."""
    s1 = set(by_arm.get(a1, {}))
    s2 = set(by_arm.get(a2, {}))
    both = sorted(s for s in want if s in s1 and s in s2)
    excl = sorted(s for s in want if s not in s1 or s not in s2)
    return both, excl


def contingency(by_arm, a1, a2, seeds, cf):
    b = sum(1 for s in seeds if cf[a1][s] and not cf[a2][s])
    c = sum(1 for s in seeds if not cf[a1][s] and cf[a2][s])
    return b, c


def main():
    recs = (load_records(os.path.join(_HERE, "results_expa_1.jsonl"))
            + load_records(os.path.join(_HERE, "results_expa_2.jsonl")))
    by_arm, missing = check_records(recs)

    cf = {}
    for arm, seeds in EXPECTED.items():
        cf[arm] = {}
        for s in seeds:
            if s in by_arm.get(arm, {}):
                cf[arm][s] = clean_fast_744(by_arm[arm][s])

    print("=== clean-and-fast (bar 0.744; Clopper-Pearson 95% CI) ===")
    for arm in ("wrapped_ind", "phase_only_new", "oracle_ind",
                "memory30"):
        have = sorted(cf[arm])
        k = sum(cf[arm][s] for s in have)
        lo, hi = clopper_pearson(k, len(have))
        print("  %s: %d/%d = %.3f  95%% CI [%.3f, %.3f]"
              % (arm, k, len(have), k / len(have), lo, hi))

    print("=== PRIMARY: wrapped_ind vs phase_only_new "
          "(McNemar exact, alpha=0.05, no Holm) ===")
    seeds, excl = paired_seeds(by_arm, "wrapped_ind", "phase_only_new",
                               PRIMARY_SEEDS)
    print("  paired seeds: %d/100%s"
          % (len(seeds), "" if not excl else " (excluded: %s)" % excl))
    b, c = contingency(by_arm, "wrapped_ind", "phase_only_new",
                       seeds, cf)
    p = mcnemar_exact_p(b, c)
    lo, hi = paired_diff_ci(b, c, len(seeds))
    d = (b - c) / len(seeds) if seeds else float("nan")
    print("  discordant: wrapped_ind-only %d, phase_only_new-only %d"
          % (b, c))
    print("  diff=%+.3f  95%% CI [%+.3f, %+.3f]  p=%.4f  %s"
          % (d, lo, hi, p,
             "REJECT (differs)" if p < 0.05 else "no rejection"))

    print("=== SECONDARIES (Holm over 2; equiv margin +-0.25) ===")
    sec_ps = []
    sec_rows = []
    for a2 in ("oracle_ind", "memory30"):
        seeds, excl = paired_seeds(by_arm, "wrapped_ind", a2,
                                   SECONDARY_SEEDS)
        b, c = contingency(by_arm, "wrapped_ind", a2, seeds, cf)
        p = mcnemar_exact_p(b, c)
        lo, hi = paired_diff_ci(b, c, len(seeds))
        d = (b - c) / len(seeds) if seeds else float("nan")
        sec_ps.append(("wrapped_ind vs " + a2, p))
        sec_rows.append((a2, len(seeds), excl, b, c, d, lo, hi, p))
    rej = holm_reject(sec_ps)
    for a2, n, excl, b, c, d, lo, hi, p in sec_rows:
        name = "wrapped_ind vs " + a2
        if hi < -EQUIV_MARGIN or lo > EQUIV_MARGIN:
            verdict = "DIFFERS (outside margin)"
        elif lo >= -EQUIV_MARGIN and hi <= EQUIV_MARGIN:
            verdict = "EQUIVALENT (inside margin)"
        else:
            verdict = "INDISTINGUISHABLE"
        print("  %s: n=%d%s" % (name, n,
              "" if not excl else " (excluded: %s)" % excl))
        print("    discordant: wrapped_ind-only %d, %s-only %d"
              % (b, a2, c))
        print("    diff=%+.3f  95%% CI [%+.3f, %+.3f]  p=%.4f  "
              "Holm: %s  -> %s"
              % (d, lo, hi, p,
                 "REJECT" if rej[name] else "no rejection", verdict))

    print("=== zero-crush / success>=0.9 (greedy, descriptive) ===")
    for arm in ("wrapped_ind", "phase_only_new", "oracle_ind",
                "memory30"):
        have = sorted(by_arm.get(arm, {}))
        zc = sum(1 for s in have
                 if by_arm[arm][s]["greedy"]["crush_rate"] == 0.0)
        su = sum(1 for s in have
                 if by_arm[arm][s]["greedy"]["success"] >= 0.9)
        print("  %s: zero-crush %d/%d, success>=0.9 %d/%d"
              % (arm, zc, len(have), su, len(have)))

    print("=== median greedy return; paired median differences "
          "(bootstrap, seeds) ===")
    rets = {}
    for arm in ("wrapped_ind", "phase_only_new", "oracle_ind",
                "memory30"):
        have = sorted(by_arm.get(arm, {}))
        rets[arm] = [by_arm[arm][s]["greedy"]["mean_return"]
                     for s in have]
        print("  %s: n=%d median=%.3f"
              % (arm, len(have), statistics.median(rets[arm])))
    for a1, a2, want in (("wrapped_ind", "phase_only_new",
                          PRIMARY_SEEDS),
                         ("wrapped_ind", "oracle_ind",
                          SECONDARY_SEEDS),
                         ("wrapped_ind", "memory30",
                          SECONDARY_SEEDS)):
        seeds, _ = paired_seeds(by_arm, a1, a2, want)
        x = [by_arm[a1][s]["greedy"]["mean_return"] for s in seeds]
        y = [by_arm[a2][s]["greedy"]["mean_return"] for s in seeds]
        obs, lo, hi = bootstrap_median_diff(x, y)
        print("  %s-%s median diff: %+.3f  95%% CI [%+.3f, %+.3f] "
              "(n=%d pairs, seed=%d, %d resamples)"
              % (a1, a2, obs, lo, hi, len(seeds), BOOTSTRAP_SEED,
                 BOOTSTRAP_RESAMPLES))

    print("OK: Experiment A statistics regenerated deterministically")


if __name__ == "__main__":
    main()

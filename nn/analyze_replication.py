#!/usr/bin/env python3
"""analyze_replication.py -- regenerate the replication statistics.

Reads nn/results_replication.jsonl and recomputes every number reported
in nn/decisions_replication.md, deterministically. Exits nonzero on
missing, duplicate, or unexpected records.

Usage: python3 analyze_replication.py
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

from run_arms import _clean_fast

SEEDS = list(range(10, 50))
ARMS = ("plain", "memory", "wrapped")
BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 10000


def fail(msg):
    print("FAIL: %s" % msg, file=sys.stderr)
    raise SystemExit(1)


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_records(records):
    if len(records) != 120:
        fail("expected 120 records, got %d" % len(records))
    seen = set()
    for r in records:
        key = (r.get("arm"), r.get("seed"))
        if key in seen:
            fail("duplicate record %r" % (key,))
        seen.add(key)
        if r.get("kind") != "arm":
            fail("unexpected kind %r" % (r.get("kind"),))
    for arm in ARMS:
        got = sorted(s for (a, s) in seen if a == arm)
        if got != SEEDS:
            fail("arm %s seeds %r != 10-49" % (arm, got))
    hps = set(json.dumps(r["hyperparams"], sort_keys=True)
              for r in records)
    if len(hps) != 1:
        fail("expected 1 hyperparameter set, got %d" % len(hps))


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial 95% CI via the beta distribution."""
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def mcnemar_exact_p(b, c):
    """Exact two-sided McNemar p-value via the binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(2 * p, 1.0)


def paired_diff_ci(b, c, n, alpha=0.05):
    """Exact paired 95% CI for the difference in proportions.

    From the Clopper-Pearson interval for p = b/(b+c) among discordant
    pairs, transformed to (2p-1)*(b+c)/n.
    """
    from scipy.stats import beta
    d = b + c
    if d == 0:
        return 0.0, 0.0
    lo_p = beta.ppf(alpha / 2, b, d - b + 1) if b > 0 else 0.0
    hi_p = beta.ppf(1 - alpha / 2, b + 1, d - b) if b < d else 1.0
    return (2 * lo_p - 1) * d / n, (2 * hi_p - 1) * d / n


def bootstrap_median_diff(x, y, seed=BOOTSTRAP_SEED,
                          resamples=BOOTSTRAP_RESAMPLES):
    """Bootstrap 95% CI for the median of paired differences.

    Resampling unit is the SEED (paired differences), with replacement.
    Deterministic via the recorded seed.
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(x) - np.asarray(y)
    obs = float(np.median(diffs))
    boots = [float(np.median(rng.choice(diffs, size=len(diffs),
                                        replace=True)))
             for _ in range(resamples)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main():
    src = os.path.join(_HERE, "results_replication.jsonl")
    records = load_records(src)
    check_records(records)
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["arm"], {})[r["seed"]] = r

    # Clean-and-fast flags reconstructed from raw metrics.
    cf = {a: {s: _clean_fast(by_arm[a][s]) for s in SEEDS} for a in ARMS}

    print("=== clean-and-fast proportions (Clopper-Pearson 95% CI) ===")
    for a in ARMS:
        k = sum(cf[a].values())
        lo, hi = clopper_pearson(k, len(SEEDS))
        print("  %s: %d/%d = %.3f  95%% CI [%.3f, %.3f]"
              % (a, k, len(SEEDS), k / len(SEEDS), lo, hi))

    print("=== paired contingency tables (McNemar exact) ===")
    for a1, a2 in (("wrapped", "plain"), ("wrapped", "memory")):
        b = sum(1 for s in SEEDS if cf[a1][s] and not cf[a2][s])
        c = sum(1 for s in SEEDS if not cf[a1][s] and cf[a2][s])
        p = mcnemar_exact_p(b, c)
        lo, hi = paired_diff_ci(b, c, len(SEEDS))
        d = (b - c) / len(SEEDS)
        print("  %s vs %s:" % (a1, a2))
        print("    discordant: %s-only %d, %s-only %d"
              % (a1, b, a2, c))
        print("    diff=%+.3f  95%% CI [%+.3f, %+.3f]  p=%.4f"
              % (d, lo, hi, p))

    print("=== zero-crush / success>=0.9 (greedy, descriptive) ===")
    for a in ARMS:
        zc = sum(1 for s in SEEDS
                 if by_arm[a][s]["greedy"]["crush_rate"] == 0.0)
        su = sum(1 for s in SEEDS
                 if by_arm[a][s]["greedy"]["success"] >= 0.9)
        print("  %s: zero-crush %d/40, success>=0.9 %d/40" % (a, zc, su))

    print("=== median greedy return; paired median differences "
          "(bootstrap, seeds) ===")
    rets = {a: [by_arm[a][s]["greedy"]["mean_return"] for s in SEEDS]
            for a in ARMS}
    for a in ARMS:
        print("  %s: median=%.3f" % (a, statistics.median(rets[a])))
    for a1, a2 in (("wrapped", "plain"), ("wrapped", "memory")):
        obs, lo, hi = bootstrap_median_diff(rets[a1], rets[a2])
        print("  %s-%s median diff: %+.3f  95%% CI [%+.3f, %+.3f] "
              "(seed=%d, %d resamples)"
              % (a1, a2, obs, lo, hi, BOOTSTRAP_SEED,
                 BOOTSTRAP_RESAMPLES))

    print("=== schedule diagnostics (wrapped, training) ===")
    fracs = [by_arm["wrapped"][s]["train"]["frac_ticks_known"]
             for s in SEEDS]
    conv = set(by_arm["wrapped"][s]["train"]["first_converged_episode"]
               for s in SEEDS)
    contra = set(by_arm["wrapped"][s]["train"]["model_contradictions"]
                 for s in SEEDS)
    print("  frac known: median=%.3f min=%.3f" % (
        statistics.median(fracs), min(fracs)))
    print("  first converged episode: %s" % sorted(conv))
    print("  contradictions: %s" % sorted(contra))
    print("OK: all statistics regenerated deterministically")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""analyze_zerocrush.py -- frozen analysis for the confirmatory
zero-crush experiment (prereg_zerocrush.md v0.3).

Implements EXACTLY the v0.3 plan, nothing more:
- primary: two-sided paired exact McNemar on the binary zero-crush
  flags, wrapped_ind vs phase_only_new, alpha=0.05; direction
  (sign of b-c) reported with the p-value
- descriptives: per-arm zero-crush rates with Clopper-Pearson 95%
  CIs, two primary arms only
- missing-data rule: a seed with a missing/incomplete record is
  excluded from the paired test (unscored is neither zero nor
  crush-free); descriptives use all available records; exclusions
  and the discordant-pair count n_d are reported
- record checks: expected arms/seeds/protocol; duplicates,
  unexpected arms, and wrong-protocol records fail loudly;
  missing seeds are reported, not fatal
- deterministic output (no RNG anywhere)

No exploratory additions, no extra endpoints.
"""

import json
import math
import sys

ARMS = ("wrapped_ind", "phase_only_new")
SEEDS = range(400, 500)
PERIOD, OPEN_LEN = 30, 14
ALPHA = 0.05


def fail(msg):
    sys.stderr.write("FAIL: %s\n" % msg)
    sys.exit(1)


def load_records(path):
    recs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    fail("line %d is not valid JSON" % (i + 1))
    return recs


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def mcnemar_two_sided_p(b, c):
    """Two-sided exact McNemar p-value (Bin(b+c, 0.5))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(2 * p, 1.0)


def zero_crush(rec):
    """Binary safety flag: 1 iff zero crushes in the greedy eval."""
    g = rec.get("greedy")
    if g is None or g.get("crush_rate") is None:
        return None
    return 1 if g["crush_rate"] == 0.0 else 0


def main(path="results_zerocrush.jsonl"):
    recs = load_records(path)

    seen = set()
    by_arm = {a: {} for a in ARMS}
    for r in recs:
        key = (r.get("arm"), r.get("seed"))
        if key in seen:
            fail("duplicate record %r" % (key,))
        seen.add(key)
        a, s = key
        if a not in ARMS:
            fail("unexpected arm %r" % (a,))
        hp = r.get("hyperparams", {})
        if hp.get("period") != PERIOD or hp.get("open_len") != OPEN_LEN:
            fail("protocol mismatch in %r: period=%r open_len=%r"
                 % (key, hp.get("period"), hp.get("open_len")))
        by_arm[a][s] = r

    missing = {a: [s for s in SEEDS if s not in by_arm[a]] for a in ARMS}
    n_missing = sum(len(v) for v in missing.values())
    print("records: %d present%s" % (
        len(recs),
        "" if n_missing == 0 else
        "; MISSING seeds: %s" % {a: v for a, v in missing.items() if v}))

    # Descriptives use all available records.
    print("=== zero-crush descriptives (Clopper-Pearson 95% CI) ===")
    flags = {}
    for a in ARMS:
        fl = {}
        for s, r in by_arm[a].items():
            z = zero_crush(r)
            if z is not None:
                fl[s] = z
        flags[a] = fl
        k = sum(fl.values())
        n = len(fl)
        lo, hi = clopper_pearson(k, n)
        print("  %s: %d/%d = %.3f  95%% CI [%.3f, %.3f]"
              % (a, k, n, k / n if n else 0.0, lo, hi))

    # Primary: paired by seed; incomplete pairs excluded.
    seeds = [s for s in SEEDS
             if s in flags["wrapped_ind"] and s in flags["phase_only_new"]]
    excluded = [s for s in SEEDS if s not in seeds]
    b = sum(1 for s in seeds
            if flags["wrapped_ind"][s] == 1 and flags["phase_only_new"][s] == 0)
    c = sum(1 for s in seeds
            if flags["wrapped_ind"][s] == 0 and flags["phase_only_new"][s] == 1)
    p = mcnemar_two_sided_p(b, c)
    direction = "+" if b > c else ("-" if b < c else "0")
    print("=== PRIMARY: wrapped_ind vs phase_only_new "
          "(two-sided exact McNemar, alpha=0.05) ===")
    print("  paired seeds: %d/%d%s" % (
        len(seeds), len(SEEDS),
        "" if not excluded else " (excluded: %s)" % excluded))
    print("  discordant: wrapped_ind-only %d, phase_only_new-only %d "
          "(n_d=%d)" % (b, c, b + c))
    print("  direction %s  p=%.4f  %s" % (
        direction, p,
        "REJECT null" if p < ALPHA else "no rejection"))
    print("OK: zero-crush statistics regenerated deterministically")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_zerocrush.jsonl")

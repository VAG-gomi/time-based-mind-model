#!/usr/bin/env python3
"""validate_analyze_zerocrush.py -- synthetic-data validation for
nn/analyze_zerocrush.py, mirroring validate_analyze_expa.py.

Every check runs on synthetic records; no real experiment data is
used or needed. All checks must pass before the analysis is frozen.
"""

import io
import json
import os
import random
import sys
import tempfile
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import analyze_zerocrush as A

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print("PASS %s" % name)
    else:
        failed += 1
        print("FAIL %s" % name)


def rec(arm, seed, zc, period=30, open_len=14, no_greedy=False):
    r = {"kind": "arm", "arm": arm, "seed": seed,
         "hyperparams": {"period": period, "open_len": open_len}}
    if not no_greedy:
        r["greedy"] = {"crush_rate": 0.0 if zc else 1.0 / 30,
                       "mean_return": 0.7, "success": 1.0}
    return r


# 1. Two-sided McNemar matches scipy binomtest (two-sided).
from scipy.stats import binomtest
rng = random.Random(0)
ok = True
for _ in range(300):
    b = rng.randrange(0, 40)
    c = rng.randrange(0, 40)
    got = A.mcnemar_two_sided_p(b, c)
    want = binomtest(min(b, c), b + c, 0.5,
                     alternative="two-sided").pvalue if b + c else 1.0
    if abs(got - want) > 1e-12:
        ok = False
        break
check("mcnemar_two_sided matches scipy binomtest (300 tables)", ok)

# 2. Boundary cases.
check("b=c=0 -> p=1.0", A.mcnemar_two_sided_p(0, 0) == 1.0)
check("b=1,c=0 -> p=1.0", A.mcnemar_two_sided_p(1, 0) == 1.0)
check("b=2,c=0 -> p=0.5",
      abs(A.mcnemar_two_sided_p(2, 0) - 0.5) < 1e-12)
check("b=27,c=2 rejects at 0.05",
      A.mcnemar_two_sided_p(27, 2) < 0.05)


def run_on(records):
    """Run A.main on synthetic records; return (stdout, exit_code)."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                     delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name
    buf = io.StringIO()
    code = 0
    try:
        with redirect_stdout(buf):
            A.main(path)
    except SystemExit as e:
        code = e.code
    finally:
        os.unlink(path)
    return buf.getvalue(), code


def synth_pair():
    """100 seeds/arm with known discordant structure:
    b=20 (wrapped-only, seeds 470-489), c=5 (phase-only, 490-494)."""
    rs = []
    for s in range(400, 500):
        wz = s <= 489          # wrapped zero-crush on 400-489
        pz = s <= 469 or 490 <= s <= 494  # phase on 400-469, 490-494
        rs.append(rec("wrapped_ind", s, wz))
        rs.append(rec("phase_only_new", s, pz))
    return rs


# 3. End-to-end on synthetic 200 records.
out, code = run_on(synth_pair())
check("end-to-end exit 0", code == 0)
check("end-to-end b=20 c=5 n_d=25",
      "wrapped_ind-only 20, phase_only_new-only 5 (n_d=25)" in out)
import re
m = re.search(r"p=([0-9.]+)", out)
p_printed = float(m.group(1)) if m else None
check("end-to-end p matches computed value",
      p_printed is not None
      and abs(p_printed - A.mcnemar_two_sided_p(20, 5)) < 5e-5)
check("end-to-end paired 100/100", "paired seeds: 100/100" in out)
check("end-to-end descriptives 90/100 and 75/100",
      "wrapped_ind: 90/100" in out and "phase_only_new: 75/100" in out)

# 4. Missing-data behavior: phase seed 400 missing entirely,
#    wrapped seed 402 has no greedy block.
rs = synth_pair()
rs = [r for r in rs
      if not (r["arm"] == "phase_only_new" and r["seed"] == 400)]
for r in rs:
    if r["arm"] == "wrapped_ind" and r["seed"] == 402:
        del r["greedy"]
out, code = run_on(rs)
check("missing data: exit 0, not fatal", code == 0)
check("missing data: paired 98/100 with exclusions listed",
      "paired seeds: 98/100" in out and "400" in out and "402" in out)
check("missing data: descriptives use all records (89/99, 74/99)",
      "wrapped_ind: 89/99" in out and "phase_only_new: 74/99" in out)
check("missing data: b/c unchanged (20/5)",
      "wrapped_ind-only 20, phase_only_new-only 5" in out)

# 5. Failure modes.
out, code = run_on([rec("wrapped_ind", 400, True, period=20)])
check("wrong protocol fails loudly", code == 1)
rs = synth_pair()
rs.append(rec("wrapped_ind", 400, True))
out, code = run_on(rs)
check("duplicate fails loudly", code == 1)
rs = synth_pair()
rs.append(rec("mystery_arm", 400, True))
out, code = run_on(rs)
check("unexpected arm fails loudly", code == 1)

# 6. Determinism.
out1, _ = run_on(synth_pair())
out2, _ = run_on(synth_pair())
check("deterministic output", out1 == out2)

print("---")
print("%d/%d checks passed" % (passed, passed + failed))
sys.exit(1 if failed else 0)

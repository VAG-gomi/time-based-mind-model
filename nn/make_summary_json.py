#!/usr/bin/env python3
"""make_summary_json.py -- generate nn/summary_b2.json from results_a5.jsonl.

A machine-readable aggregate of all arm/seed results, with explicit
schema. Generated, not hand-typed: re-run after any results change.

Usage: python3 make_summary_json.py
Writes: nn/summary_b2.json
"""

import datetime
import json
import os
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from run_arms import _clean_fast


def _git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(_HERE), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _eval_block(rec, key):
    m = rec[key]
    return {"mean_return": m["mean_return"],
            "crush_rate": m["crush_rate"],
            "frac_crushed": m["frac_crushed"],
            "success": m["success"],
            "mean_ticks": m["mean_ticks"],
            "n": m["n"]}


def _arm_block(records, arm):
    by_seed = sorted([r for r in records if r.get("arm") == arm],
                     key=lambda r: r["seed"])
    out = {"n_seeds": len(by_seed), "seeds": [r["seed"] for r in by_seed]}
    for key in ("greedy", "eps005"):
        rets = [r[key]["mean_return"] for r in by_seed]
        out[key] = {
            "median_return": statistics.median(rets),
            "min_return": min(rets),
            "max_return": max(rets),
            "std_return": statistics.pstdev(rets),
            "zero_crush_seeds": [r["seed"] for r in by_seed
                                 if r[key]["crush_rate"] == 0.0],
            "success_ge_09_seeds": [r["seed"] for r in by_seed
                                    if r[key]["success"] >= 0.9],
        }
    out["clean_and_fast_seeds"] = [r["seed"] for r in by_seed
                                   if _clean_fast(r)]
    out["per_seed"] = []
    for r in by_seed:
        ps = {"seed": r["seed"],
              "greedy": _eval_block(r, "greedy"),
              "eps005": _eval_block(r, "eps005"),
              "clean_and_fast": _clean_fast(r)}
        if arm == "wrapped":
            t = r.get("train", {})
            ps["schedule"] = {
                "frac_ticks_known": t.get("frac_ticks_known"),
                "first_converged_episode":
                    t.get("first_converged_episode"),
                "first_converged_tick": t.get("first_converged_tick"),
                "model_contradictions": t.get("model_contradictions"),
            }
        out["per_seed"].append(ps)
    return out


def main():
    src = os.path.join(_HERE, "results_a5.jsonl")
    records = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    arm_recs = [r for r in records if r["kind"] == "arm"]
    hp = arm_recs[0]["hyperparams"] if arm_recs else {}
    arms = {}
    for arm in ("plain", "memory", "wrapped", "clock"):
        if any(r.get("arm") == arm for r in arm_recs):
            arms[arm] = _arm_block(arm_recs, arm)
    yards = {}
    for r in records:
        if r["kind"] == "yardstick":
            yards[r["name"]] = {"greedy": _eval_block(r, "greedy"),
                                "eps005": _eval_block(r, "eps005")}
    doc = {
        "schema": "summary_b2/v1",
        "metadata": {
            "generated_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "source": "nn/results_a5.jsonl",
            "generator": "nn/make_summary_json.py",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "protocol": {
            "hyperparameters": hp,
            "seeds_per_arm": 10,
            "clean_and_fast_definition":
                "greedy: crush_rate==0 AND success>=0.9 "
                "AND mean_return>=0.75",
            "win_threshold": "wrapped >=7/10 clean-and-fast "
                             "(pre-registered in R; NOT MET: 5/10)",
            "eval_greedy": "20 start phases x 1 episode, pure greedy",
            "eval_eps005": "20 start phases x 10 episodes, eps=0.05",
        },
        "arms": arms,
        "yardsticks": yards,
    }
    dst = os.path.join(_HERE, "summary_b2.json")
    with open(dst, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print("wrote %s (%d arm records, %d yardsticks)"
          % (dst, len(arm_recs), len(yards)))


if __name__ == "__main__":
    main()

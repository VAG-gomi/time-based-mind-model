#!/usr/bin/env python3
"""P2 forensic tracer (WO-TBM-P2, diagnosis only).

Re-trains frozen arm code (2000 episodes, same hyperparameters as the
tracked records) and runs the greedy policy over every start phase,
logging per-tick position, action, door state, and the arm's time
features. No frozen file is modified; this script is additive only.

Reproduction gate: for each (arm, seed, protocol) the greedy summary
(mean ticks, mean return, crush rate, success) must match the tracked
record exactly. A MISMATCH means the traced agent is NOT the recorded
agent and its traces must not support conclusions.

Skips pairs already present in traces_p2.jsonl, so a restart resumes.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_nn as ann
from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step
from agent_generalization import (
    OracleIndEncoder, OracleNewEncoder, PhaseOnlyNewEncoder,
    train_oracle_ind, train_oracle_new, train_phase_only_new,
    train_wrapped_ind)
from world import World, DOOR_CELL, GOAL_POS

OUT = os.path.join(_HERE, "traces_p2.jsonl")

# (arm, seed, period, open_len, record_file, expected greedy)
PAIRS = [
    # 30/14 attractor cases
    ("oracle_ind", 101, 30, 14, "results_stage1.jsonl"),
    ("oracle_new", 101, 30, 14, "results_pilot.jsonl"),
    ("phase_only_new", 101, 30, 14, "results_stage1.jsonl"),
    # 30/14 controls
    ("oracle_ind", 102, 30, 14, "results_stage1.jsonl"),
    ("wrapped_ind", 101, 30, 14, "results_pilot.jsonl"),
    ("wrapped_ind", 102, 30, 14, "results_pilot.jsonl"),
    # 20/10 attractor cases (regression misses)
    ("wrapped_ind", 108, 20, 10, "results_regression.jsonl"),
    ("wrapped_ind", 109, 20, 10, "results_regression.jsonl"),
    ("wrapped_ind", 106, 20, 10, "results_regression.jsonl"),
    ("wrapped_ind", 101, 20, 10, "results_regression.jsonl"),
    # 20/10 controls
    ("wrapped_ind", 102, 20, 10, "results_regression.jsonl"),
    ("wrapped_ind", 104, 20, 10, "results_regression.jsonl"),
]


def load_expected():
    exp = {}
    for fn in ("results_stage1.jsonl", "results_pilot.jsonl",
               "results_regression.jsonl"):
        for line in open(os.path.join(_HERE, fn)):
            r = json.loads(line)
            if r["kind"] == "arm":
                exp[(r["arm"], r["seed"], r["hyperparams"]["period"],
                     r["hyperparams"]["open_len"])] = r["greedy"]
    return exp


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["arm"], r["seed"], r["period"]))
    return done


def train_and_trace(arm, seed, period, open_len):
    """Train (frozen code) and greedy-trace every start phase."""
    ann.PERIOD = period
    ann.OPEN_LEN = open_len
    if arm == "oracle_ind":
        agent, _stats = train_oracle_ind(seed, hidden=32, episodes=2000,
                                         lr=1e-3)
        encoder = OracleIndEncoder()
    elif arm == "oracle_new":
        agent, _stats = train_oracle_new(seed, period, open_len, hidden=32,
                                         episodes=2000, lr=1e-3)
        encoder = OracleNewEncoder(period, open_len)
    elif arm == "phase_only_new":
        agent, _stats = train_phase_only_new(seed, period, hidden=32,
                                             episodes=2000, lr=1e-3)
        encoder = PhaseOnlyNewEncoder(period)
    elif arm == "wrapped_ind":
        # Eval reuses the SAME encoder, exactly as run_arms does.
        agent, _stats, encoder = train_wrapped_ind(
            seed, hidden=32, episodes=2000, lr=1e-3)
    else:
        raise ValueError("unknown arm %r" % (arm,))
    episodes = []
    for start in range(period):
        w = World(period=period, open_len=open_len)
        w.reset(start_tick=start)
        encoder.reset(w)
        steps = []
        total = 0.0
        crushes = 0
        reached = False
        ticks = 0
        while ticks < EPISODE_CAP:
            x = encoder.encode(w)
            a = agent.act(x, 0.0)
            (px, py), door = w.observe()
            feats = {}
            if len(x) >= 104:
                feats = {"f101": float(x[101]), "f102": float(x[102]),
                         "f103": float(x[103])}
            r, reached, crushed = env_step(w, a)
            steps.append({"t": w.tick - 1, "x": px, "y": py,
                          "action": ACTION_NAMES[a],
                          "door": bool(door), "crushed": bool(crushed),
                          **feats})
            total += r
            crushes += int(crushed)
            ticks += 1
            if reached:
                break
        episodes.append({"arm": arm, "seed": seed, "period": period,
                         "open_len": open_len, "start_phase": start,
                         "ticks": ticks, "crushes": crushes,
                         "reached": reached, "return": total,
                         "steps": steps})
    return episodes


def summarize(episodes):
    n = len(episodes)
    ok = [e for e in episodes if e["reached"]]
    return {
        "mean_return": sum(e["return"] for e in episodes) / n,
        "crush_rate": sum(e["crushes"] for e in episodes) / n,
        "success": len(ok) / n,
        "mean_ticks": (sum(e["ticks"] for e in ok) / len(ok)
                       if ok else None),
        "n": n,
    }


def main():
    expected = load_expected()
    done = load_done()
    out = open(OUT, "a")
    for arm, seed, period, open_len, _fn in PAIRS:
        key = (arm, seed, period)
        if key in done:
            print("%s/%d %d/%d: already traced, skipping"
                  % (arm, seed, period, open_len), flush=True)
            continue
        print("%s/%d %d/%d: training..." % (arm, seed, period, open_len),
              flush=True)
        episodes = train_and_trace(arm, seed, period, open_len)
        for e in episodes:
            out.write(json.dumps(e) + "\n")
        out.flush()
        got = summarize(episodes)
        exp = expected.get((arm, seed, period, open_len))
        match = (exp is not None
                 and abs(got["mean_ticks"] - exp["mean_ticks"]) < 1e-9
                 and abs(got["mean_return"] - exp["mean_return"]) < 1e-9
                 and abs(got["crush_rate"] - exp["crush_rate"]) < 1e-9
                 and abs(got["success"] - exp["success"]) < 1e-9)
        print("%s/%d %d/%d: greedy ticks=%.3f ret=%.3f crush=%.3f "
              "succ=%.2f -> %s (expected ticks=%.3f ret=%.3f)"
              % (arm, seed, period, open_len, got["mean_ticks"],
                 got["mean_return"], got["crush_rate"], got["success"],
                 "MATCH" if match else "MISMATCH",
                 exp["mean_ticks"] if exp else float("nan"),
                 exp["mean_return"] if exp else float("nan")), flush=True)
    out.close()


if __name__ == "__main__":
    main()

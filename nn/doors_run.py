#!/usr/bin/env python3
"""WO-TBM-doors runner: Step 0c checks + Step 1 transfer evaluation.

For each (arm, seed, protocol) in the design note:
  1. Train with frozen code on the single-door World.
  2. Greedy-eval on World (W1) -> Step-0c check vs the tracked record
     (exact mean ticks / return / crush rate / success).
  3. Greedy-eval on MultiDoorWorld gaps={(5,5)} (W1m) -> Step-0b
     agent-level check (identical summary to W1).
  4. ONLY if both checks pass: greedy-eval on W2 and W5.
An agent failing a check is reported, not evaluated (W2/W5 skipped).

Greedy = one deterministic episode per start phase, all phases.
No training on multi-door worlds, no agent modification.
Records flushed per agent-world pair to results_doors.jsonl.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_nn as ann
from agent_nn import ACTION_NAMES, EPISODE_CAP, PlainEncoder, env_step
from agent_mem import MemoryEncoder, train_memory
from agent_wrapped import WrappedEncoder, train_wrapped
from agent_generalization import (
    OracleIndEncoder, OracleNewEncoder, PhaseOnlyNewEncoder,
    train_oracle_ind, train_oracle_new, train_phase_only_new)
from world import World
from world_doors import MultiDoorWorld

OUT = os.path.join(_HERE, "results_doors.jsonl")
RECORDFILES = ("results_a5.jsonl", "results_pilot.jsonl",
               "results_stage1.jsonl")

# (arm, seed, period, open_len) per the design note.
AGENTS = ([("wrapped", s, 20, 10) for s in range(5)]
          + [("plain", s, 20, 10) for s in range(5)]
          + [("memory", s, 20, 10) for s in range(5)]
          + [("oracle_ind", s, 30, 14) for s in range(100, 105)]
          + [("oracle_new", s, 30, 14) for s in range(100, 105)]
          + [("phase_only_new", s, 30, 14) for s in range(100, 105)])

WORLDS = {"W2": ((5, 5), (5, 1)),
          "W5": ((5, 1), (5, 3), (5, 5), (5, 7), (5, 9))}


def load_expected():
    exp = {}
    for fn in RECORDFILES:
        for line in open(os.path.join(_HERE, fn)):
            r = json.loads(line)
            if r.get("kind") == "arm":
                exp[(r["arm"], r["seed"], r["hyperparams"]["period"],
                     r["hyperparams"]["open_len"])] = r["greedy"]
    return exp


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if r.get("kind") == "episode":
                done.add((r["arm"], r["seed"], r["world"]))
    return done


def train(arm, seed):
    if arm == "wrapped":
        agent, _st, encoder = train_wrapped(seed, hidden=32,
                                            episodes=2000, lr=1e-3)
        return agent, encoder
    if arm == "plain":
        agent, _st = ann.train_neural(seed, hidden=32, episodes=2000,
                                      lr=1e-3)
        return agent, PlainEncoder()
    if arm == "memory":
        agent, _st = train_memory(seed, hidden=32, episodes=2000,
                                  lr=1e-3)
        return agent, MemoryEncoder()
    if arm == "oracle_ind":
        agent, _st = train_oracle_ind(seed, hidden=32, episodes=2000,
                                      lr=1e-3)
        return agent, OracleIndEncoder()
    if arm == "oracle_new":
        agent, _st = train_oracle_new(seed, ann.PERIOD, ann.OPEN_LEN,
                                      hidden=32, episodes=2000, lr=1e-3)
        return agent, OracleNewEncoder(ann.PERIOD, ann.OPEN_LEN)
    if arm == "phase_only_new":
        agent, _st = train_phase_only_new(seed, ann.PERIOD, hidden=32,
                                          episodes=2000, lr=1e-3)
        return agent, PhaseOnlyNewEncoder(ann.PERIOD)
    raise ValueError("unknown arm %r" % (arm,))


def segments(positions):
    segs = []
    for p in positions:
        if segs and segs[-1][0] == p:
            segs[-1][1] += 1
        else:
            segs.append([p, 1])
    return segs


def stall_signature(positions, gaps):
    """Design-note structural stall signature (any one suffices)."""
    # (a) run of >= 10 consecutive ticks with no position change.
    run = 1
    for i in range(1, len(positions)):
        run = run + 1 if positions[i] == positions[i - 1] else 1
        if run >= 10:
            return True, "no-move-run"
    # (b) 2-cycle spanning >= 10 ticks: longest suffix of steps
    # strictly alternating between two cells.
    run = 1
    for i in range(len(positions)):
        if i >= 2 and positions[i] == positions[i - 2] \
                and positions[i] != positions[i - 1]:
            run = run + 1 if run >= 2 else 3
        else:
            run = 1
        if run >= 10:
            return True, "two-cycle"
    # (c) dwell >= 15 in one 3x3 block, excluding staging cells
    # (4,y) for any gap (5,y) and excluding the goal cell.
    stage_cells = set((4, y) for (_x, y) in gaps)
    dwell = 0
    run = 0
    prev = None
    for p in positions:
        if p in stage_cells or p == (9, 9):
            run = 0
            prev = None
            continue
        b = (p[0] // 3, p[1] // 3)
        run = run + 1 if b == prev else 1
        prev = b
        dwell = max(dwell, run)
    if dwell >= 15:
        return True, "dwell"
    return False, None


def evaluate(agent, encoder, period, open_len, gaps, world_kind):
    """Greedy eval over all start phases on one world configuration."""
    episodes = []
    for start in range(period):
        if world_kind == "W1":
            w = World(period=period, open_len=open_len)
        else:
            w = MultiDoorWorld(period, open_len, gaps)
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
            r, reached, crushed = env_step(w, a)
            steps.append({"t": w.tick - 1, "x": px, "y": py,
                          "action": ACTION_NAMES[a],
                          "door": bool(door),
                          "crushed": bool(crushed)})
            total += r
            crushes += int(crushed)
            ticks += 1
            if reached:
                break
        positions = [(s["x"], s["y"]) for s in steps]
        entries = []
        for p in positions:
            if p in gaps and (not entries or entries[-1] != p):
                entries.append(p)
        switches = sum(1 for i in range(1, len(entries))
                       if entries[i] != entries[i - 1])
        segs = segments(positions)
        seen = set()
        revisits = 0
        for c, _k in segs:
            if c in seen:
                revisits += 1
            seen.add(c)
        stall, stall_kind = stall_signature(positions, gaps)
        episodes.append({
            "kind": "episode", "world": world_kind,
            "start_phase": start, "ticks": ticks,
            "crushes": crushes, "reached": reached,
            "return": total, "first_door": (list(entries[0])
                                            if entries else None),
            "door_entries": [list(p) for p in entries],
            "switches": switches, "stall": stall,
            "stall_kind": stall_kind, "revisits": revisits,
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


def matches(got, exp):
    return (exp is not None
            and got["mean_ticks"] is not None
            and abs(got["mean_ticks"] - exp["mean_ticks"]) < 1e-9
            and abs(got["mean_return"] - exp["mean_return"]) < 1e-9
            and abs(got["crush_rate"] - exp["crush_rate"]) < 1e-9
            and abs(got["success"] - exp["success"]) < 1e-9)


def load_agent_world(arm, seed, world):
    eps = []
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if r.get("kind") == "episode" and r["arm"] == arm \
                    and r["seed"] == seed and r["world"] == world:
                eps.append(r)
    return eps


def main():
    expected = load_expected()
    done = load_done()
    out = open(OUT, "a")
    for arm, seed, period, open_len in AGENTS:
        ann.PERIOD = period
        ann.OPEN_LEN = open_len
        key1 = (arm, seed, "W1")
        need_w1 = key1 not in done
        need_w1m = (arm, seed, "W1m") not in done
        need_w2 = (arm, seed, "W2") not in done
        need_w5 = (arm, seed, "W5") not in done
        if not (need_w1 or need_w1m or need_w2 or need_w5):
            print("%s/%d: all worlds done, skipping" % (arm, seed),
                  flush=True)
            continue
        print("%s/%d %d/%d: training..." % (arm, seed, period, open_len),
              flush=True)
        agent, encoder = train(arm, seed)
        # W1 on the single-door World (Step 0c).
        if need_w1:
            eps = evaluate(agent, encoder, period, open_len,
                           ((5, 5),), "W1")
            for e in eps:
                e.update({"arm": arm, "seed": seed, "period": period,
                          "open_len": open_len})
                out.write(json.dumps(e) + "\n")
            out.flush()
            got = summarize(eps)
            exp = expected.get((arm, seed, period, open_len))
            ok0c = matches(got, exp)
            print("%s/%d W1: ticks=%.3f ret=%.3f crush=%.3f succ=%.2f "
                  "-> 0c %s" % (arm, seed, got["mean_ticks"],
                                got["mean_return"], got["crush_rate"],
                                got["success"],
                                "MATCH" if ok0c else "MISMATCH"),
                  flush=True)
        else:
            prev = load_agent_world(arm, seed, "W1")
            got = summarize(prev) if prev else None
            exp = expected.get((arm, seed, period, open_len))
            ok0c = got is not None and matches(got, exp)
            print("%s/%d W1: already done, 0c rechecked -> %s"
                  % (arm, seed, "MATCH" if ok0c else "MISMATCH"),
                  flush=True)
        # W1m on the single-gap variant (Step 0b agent level).
        if need_w1m:
            epsm = evaluate(agent, encoder, period, open_len,
                            ((5, 5),), "W1m")
            for e in epsm:
                e.update({"arm": arm, "seed": seed, "period": period,
                          "open_len": open_len})
                out.write(json.dumps(e) + "\n")
            out.flush()
            gotm = summarize(epsm)
            ok0b = (got is not None
                    and gotm["mean_ticks"] == got["mean_ticks"]
                    and gotm["mean_return"] == got["mean_return"]
                    and gotm["crush_rate"] == got["crush_rate"]
                    and gotm["success"] == got["success"])
            print("%s/%d W1m: ticks=%.3f -> 0b %s"
                  % (arm, seed, gotm["mean_ticks"],
                     "MATCH" if ok0b else "MISMATCH"), flush=True)
        else:
            prevm = load_agent_world(arm, seed, "W1m")
            gotm = summarize(prevm) if prevm else None
            ok0b = (got is not None and gotm is not None
                    and gotm["mean_ticks"] == got["mean_ticks"]
                    and gotm["mean_return"] == got["mean_return"]
                    and gotm["crush_rate"] == got["crush_rate"]
                    and gotm["success"] == got["success"])
            print("%s/%d W1m: already done, 0b rechecked -> %s"
                  % (arm, seed, "MATCH" if ok0b else "MISMATCH"),
                  flush=True)
        if not (ok0c and ok0b):
            print("%s/%d: CHECK FAILED, W2/W5 skipped (reported, not "
                  "evaluated)" % (arm, seed), flush=True)
            continue
        for wname, gaps in WORLDS.items():
            if (arm, seed, wname) in done:
                print("%s/%d %s: already done, skipping"
                      % (arm, seed, wname), flush=True)
                continue
            eps = evaluate(agent, encoder, period, open_len, gaps,
                           wname)
            for e in eps:
                e.update({"arm": arm, "seed": seed, "period": period,
                          "open_len": open_len})
                out.write(json.dumps(e) + "\n")
            out.flush()
            s = summarize(eps)
            print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f"
                  % (arm, seed, wname, s["success"],
                     s["mean_ticks"] if s["mean_ticks"] is not None
                     else float("nan"), s["crush_rate"]), flush=True)
    out.close()


if __name__ == "__main__":
    main()

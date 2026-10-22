#!/usr/bin/env python3
"""WO-TBM-P-Da runner: Step 1 retraining + Step 2 evaluation.

Design note: nn/decisions_pda_design.md (committed Step 0, 0679d19).
Work order: ~/workspace/your_files/WO-TBM-P-Da.md (issued, owner
ruled proceed on the current substrate, no spike).

For each (arm, seed, protocol) in the design note:
  - Training world W5 {(5,1),(5,3),(5,5),(5,7),(5,9)}, all doors on
    the arm's training schedule, timing uniform. Fresh from scratch
    via nn/train_pda.py trainers ONLY. No fine-tuning, no edits to
    frozen files, no hyperparameter changes.
  - Wrapped arm (stateful ScheduleModel): FRESH training per eval
    world (5 wrapped agents x 5 worlds = 25 wrapped trainings).
    That training's encoder is used ONLY for that world's eval.
  - Non-wrapped arms (stateless encoders): one training, then
    eval sequentially across worlds, as in doors/P-Db/Luck.
  - Training records flushed per training job to
    training_pda.jsonl (summary + per-door entry counts, design
    note §5 definition). A restart loses at most one training:
    training records are flushed immediately after training, and
    a re-training on relaunch (deterministic, same weights) does
    NOT duplicate its training record (key already present).
  - Greedy eval: one deterministic episode per start phase, all
    phases (20 for 20/10, 30 for 30/14), cap 200 ticks, on
    PdaWorld (world_pda.py) for every eval world, including E0.
    Evaluation precedent: doors_run.py / luck_run.py (greedy
    policy per encoder, stall_signature, staging, arrival, steps).
  - Records flushed per agent-world pair to results_pda.jsonl.
    Resume-safe: on startup, jobs with a full phase count are
    skipped; jobs with a partial count (>0, <period) have their
    partial rows removed and are re-run deterministically, so no
    duplicates and no partials remain at the end. A job with
    MORE rows than its phase count is an error: the runner stops
    before training, so a duplicate is never silently pooled.
  - Wrapped eval logs f101/f102/f103 per step. An episode with
    any step f103 == 0.0 is flagged blank_feature=True in its
    record. Blank (drifted) rows are reported and must be
    excluded from analysis, never pooled with clean rows.
  - Evaluation only: if retrained agents fail everywhere, that
    IS the result. No fine-tuning, no hyperparameter change, no
    single-door weight substitution, for any reason.

Expected totals: 50 trainings, 150 eval jobs, 3750 eval episodes
(20/10: 15 agents x 5 worlds x 20 = 1500; 30/14: 15 x 5 x 30 = 2250).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step
from doors_run import AGENTS, segments, stall_signature
from train_pda import TRAINERS
from world_pda import PdaWorld, W5

OUT = os.path.join(_HERE, "results_pda.jsonl")
TRAIN_OUT = os.path.join(_HERE, "training_pda.jsonl")

# (world label, gaps) in evaluation order.
WORLDS = [
    ("E0", ((5, 5),)),
    ("E1", W5),
    ("E2", ((5, 1), (5, 3), (5, 7), (5, 9))),
    ("E3", ((5, 8),)),
    ("E4", ((5, 1), (5, 3), (5, 5), (5, 7), (5, 8), (5, 9))),
]
WORLD_GAPS = dict(WORLDS)


def gap_key(g):
    return "%d,%d" % (g[0], g[1])


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


def evaluate(agent, encoder, period, open_len, gaps, world_kind,
             log_features=False):
    """Greedy eval over all start phases on one P-Da world."""
    episodes = []
    for start in range(period):
        w = PdaWorld(period, open_len, gaps)
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
            rec = {"t": w.tick - 1, "x": px, "y": py,
                   "action": ACTION_NAMES[a],
                   "door": bool(door), "crushed": bool(crushed)}
            if log_features and len(x) >= 104:
                rec.update({"f101": float(x[101]),
                            "f102": float(x[102]),
                            "f103": float(x[103])})
            steps.append(rec)
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
        # Staging cells in first-occupation order (design note §4):
        # {(4,y) for each gap} PLUS (4,5) always.
        stage_cells = set((4, y) for (_x, y) in gaps)
        stage_cells.add((4, 5))
        staging_order = []
        for p in positions:
            if p in stage_cells and p not in staging_order:
                staging_order.append(p)
        arrival = staging_order[0] if staging_order else None
        # Post-entry behaviour (design note §4, P-Dc connection):
        # positions AFTER the first entry position.
        if entries:
            first_idx = positions.index(entries[0])
            post_positions = positions[first_idx + 1:]
            post_stall, post_stall_kind = stall_signature(
                post_positions, gaps)
        else:
            first_idx = None
            post_positions = []
            post_stall, post_stall_kind = False, None
        blank = bool(log_features and any(
            st.get("f103") == 0.0 for st in steps))
        episodes.append({
            "kind": "episode", "world": world_kind,
            "start_phase": start, "ticks": ticks,
            "crushes": crushes, "reached": reached,
            "return": total, "first_door": (list(entries[0])
                                            if entries else None),
            "door_entries": [list(p) for p in entries],
            "switches": switches, "stall": stall,
            "stall_kind": stall_kind, "revisits": revisits,
            "staging_order": [list(p) for p in staging_order],
            "arrival": list(arrival) if arrival else None,
            "first_entry_index": first_idx,
            "post_entry_ticks": len(post_positions),
            "post_entry_stall": post_stall,
            "post_entry_stall_kind": post_stall_kind,
            "blank_feature": blank,
            "steps": steps})
    return episodes


def load_done_counts():
    """(arm, seed, world) -> episode count in results_pda.jsonl."""
    counts = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if r.get("kind") == "episode":
                k = (r["arm"], r["seed"], r["world"])
                counts[k] = counts.get(k, 0) + 1
    return counts


def load_training_keys():
    keys = set()
    if os.path.exists(TRAIN_OUT):
        for line in open(TRAIN_OUT):
            r = json.loads(line)
            if r.get("kind") == "training":
                keys.add((r["arm"], r["seed"], r["eval_world"]))
    return keys


def write_training_record(out, keys, arm, seed, period, open_len,
                          eval_world, stats, dec, ewe):
    key = (arm, seed, eval_world)
    if key in keys:
        print("%s/%d training record for %s already present; "
              "re-training is deterministic, not duplicating"
              % (arm, seed, eval_world), flush=True)
        return
    reached = [s for s in stats if s[1]]
    mean_loss_vals = [s[3] for s in stats if s[3] is not None]
    rec = {
        "kind": "training", "arm": arm, "seed": seed,
        "period": period, "open_len": open_len,
        "training_world": "W5", "eval_world": eval_world,
        "episodes": len(stats),
        "reached": len(reached),
        "crushes": sum(s[2] for s in stats),
        "mean_ticks": (sum(s[0] for s in reached) / len(reached)
                       if reached else None),
        "mean_ticks_all": sum(s[0] for s in stats) / len(stats),
        "mean_loss": (sum(mean_loss_vals) / len(mean_loss_vals)
                      if mean_loss_vals else None),
        "door_entry_counts": {gap_key(g): dec[g] for g in dec},
        "episodes_with_entry": {gap_key(g): ewe[g] for g in ewe},
    }
    out.write(json.dumps(rec) + "\n")
    out.flush()
    keys.add(key)


def main():
    counts = load_done_counts()
    # Duplicate check: a job with MORE rows than its phase count
    # must never be silently pooled or skipped over.
    for (arm, seed, world), n in sorted(counts.items()):
        period = 20 if seed < 100 else 30
        if n > period:
            print("ERROR: duplicate job %s/%d %s has %d rows, "
                  "expected %d; stopping before training"
                  % (arm, seed, world, n, period), flush=True)
            sys.exit(1)
    # Partial jobs: remove their rows, re-run deterministically.
    partials = set()
    for (arm, seed, world), n in sorted(counts.items()):
        period = 20 if seed < 100 else 30
        if 0 < n < period:
            partials.add((arm, seed, world))
    if partials:
        print("Partial jobs found, removing their rows for "
              "deterministic re-run: %s" % sorted(partials),
              flush=True)
        kept = []
        for line in open(OUT):
            r = json.loads(line)
            if (r.get("arm"), r.get("seed"), r.get("world")) \
                    not in partials:
                kept.append(line)
        with open(OUT, "w") as f:
            f.writelines(kept)
        for k in partials:
            counts.pop(k, None)
    train_keys = load_training_keys()
    out = open(OUT, "a")
    tout = open(TRAIN_OUT, "a")
    for arm, seed, period, open_len in AGENTS:
        need = [w for w, _g in WORLDS
                if (arm, seed, w) not in counts]
        if not need:
            print("%s/%d: all P-Da worlds done, skipping"
                  % (arm, seed), flush=True)
            continue
        if arm == "wrapped":
            for wname in need:
                gaps = WORLD_GAPS[wname]
                print("%s/%d %s: fresh training..."
                      % (arm, seed, wname), flush=True)
                agent, stats, encoder, dec, ewe = TRAINERS[arm](
                    seed, period, open_len, W5,
                    hidden=32, episodes=2000, lr=1e-3)
                write_training_record(tout, train_keys, arm, seed,
                                      period, open_len, wname,
                                      stats, dec, ewe)
                eps = evaluate(agent, encoder, period, open_len,
                               gaps, wname, log_features=True)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                counts[(arm, seed, wname)] = len(eps)
                s = summarize(eps)
                entered = sum(1 for e in eps if e["first_door"])
                n_blank = sum(1 for e in eps if e["blank_feature"])
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "entered=%d blank_eps=%d"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], entered,
                         n_blank), flush=True)
        else:
            print("%s/%d %d/%d: training..."
                  % (arm, seed, period, open_len), flush=True)
            agent, stats, encoder, dec, ewe = TRAINERS[arm](
                seed, period, open_len, W5,
                hidden=32, episodes=2000, lr=1e-3)
            write_training_record(tout, train_keys, arm, seed,
                                  period, open_len, "base",
                                  stats, dec, ewe)
            for wname in need:
                gaps = WORLD_GAPS[wname]
                eps = evaluate(agent, encoder, period, open_len,
                               gaps, wname, log_features=False)
                for e in eps:
                    e.update({"arm": arm, "seed": seed,
                              "period": period, "open_len": open_len})
                    out.write(json.dumps(e) + "\n")
                out.flush()
                counts[(arm, seed, wname)] = len(eps)
                s = summarize(eps)
                entered = sum(1 for e in eps if e["first_door"])
                print("%s/%d %s: succ=%.2f ticks=%.2f crush=%.3f "
                      "entered=%d"
                      % (arm, seed, wname, s["success"],
                         s["mean_ticks"] if s["mean_ticks"] is not None
                         else float("nan"), s["crush_rate"], entered),
                      flush=True)
    out.close()
    tout.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""WO-TBM-FT1 runner: Step 1 fine-tuning + Step 2 evaluation (DRAFT).

Design note: nn/decisions_ft1_design.md (Step 0 b5cc464, gate
ruling ffa5582 spike-first, second ruling 6c9ed08: PROCEED on
the current substrate, no substrate change).
Work order: ~/workspace/your_files/WO-TBM-FT1.md (issued Step 0
only, then second ruling at the gate).

Phase A (Step 1, fine-tuning): all 22 kept agents ONLY, 200
episodes each on (5,8) alone on the arm's training schedule
(world_pdb2.f3_schedules, reused unchanged), design-note
protocol exactly: flat eps 0.05, fresh replay buffer,
start-tick RNG seed+2000, target + Adam moments carried,
agent RNG streams carried, wrapped encoder carried
(ScheduleModel and all; period/open_len recorded at
fine-tune start and end). Checkpoints: the 6 Step 0 sanity
snapshots are loaded with a double-load audit
(train_ft1.assert_load_equal); the remaining 16 are
realised fresh with train_pdb2.py trainers and VERIFIED
against their P-Db2 training records exactly BEFORE
fine-tuning. Any checkpoint mismatch: that agent is NOT
fine-tuned, the mismatch is reported, never hidden.
Sanity and spike fine-tuned weights are ALWAYS discarded:
Step 1 fine-tunes FRESH from the checkpoints even where
sanity/spike used the same agent. One fine-tuned snapshot
per agent is saved (nn/checkpoints_ft1_finetuned/) for
Step 2 loading; loads are not trainings. Records flushed
per agent to training_ft1.jsonl (P-Db2 format + checkpoint
identity + fine-tune choices). No set-aside agents, no
dose other than 200, no hyperparameter changes.

Phase B (Step 2, evaluation): greedy only, one
deterministic episode per start phase, all phases (20 for
20/10, 30 for 30/14), worlds F0-F4 (reused from P-Db2, no
new code) + G1 (world_ft1, eval only). Fresh encoder per
(agent, world) around the fine-tuned weights (design-note
Step 2 rule), persisting across that world's episodes.
Wrapped eval logs f101/f102/f103 per step; drift =
persistent whole-episode blank with schedule-blind
behaviour (reported, audited, EXCLUDED, never pooled);
transient spawn warm-up is NOT drift - the audit, not
the flag alone, decides. Conflict analysis for F1/F2
(same generalised definition as P-Db2 §4); preferred-
door-closed counts are computed in Step 3 analysis from
the recorded statuses, with preferred door = modal F1
first door of the P-Db2 agent (NOT recomputed from
post-fine-tuning F1). Records flushed per agent-world
pair to results_ft1.jsonl.

Resume-safe: Phase A skips agents with both a training
record AND a fine-tuned snapshot present; a snapshot
without a record (or vice versa) is re-run without
duplicating the record. Phase B skips complete jobs,
removes partial jobs for deterministic re-run, and
stops on an overfull job. A restart loses at most one
agent (Phase A) or one job (Phase B).

Expected totals: 22 fine-tune summaries, 132 eval jobs,
3240 eval episodes (12x6x20 + 10x6x30).
"""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step
from doors_run import segments, stall_signature
from train_pdb2 import TRAINERS
from train_ft1 import (FINE_TUNE_EPISODES, FINE_TUNE_EPS,
                       assert_load_equal, finetune_encoder,
                       finetune_neural, fresh_encoder,
                       load_checkpoint, save_checkpoint)
from world_doors_sched import SchedDoorWorld, door_open_at
from world_ft1 import g1_schedules
from world_pdb2 import (TRAIN_20, TRAIN_30, Pdb2AbsentWorld,
                        f0_schedules, f3_schedules, f4_schedules,
                        familiar_schedule, rival_schedules)

TRAIN_OUT = os.path.join(_HERE, "training_ft1.jsonl")
OUT = os.path.join(_HERE, "results_ft1.jsonl")
SNAP_DIR = os.path.join(_HERE, "checkpoints_ft1")
FT_SNAP_DIR = os.path.join(_HERE, "checkpoints_ft1_finetuned")
PDB2_TRAIN = os.path.join(_HERE, "training_pdb2.jsonl")

KEPT = ([("wrapped", s, 20, 10) for s in range(5)]
        + [("plain", s, 20, 10) for s in (0, 2, 3)]
        + [("memory", s, 20, 10) for s in (0, 2, 3, 4)]
        + [("oracle_ind", s, 30, 14) for s in (100, 101, 102, 103)]
        + [("oracle_new", s, 30, 14) for s in (100, 101, 102, 103)]
        + [("phase_only_new", s, 30, 14) for s in (102, 104)])

SANITY_SNAPS = {("wrapped", 0), ("plain", 0), ("memory", 0),
                ("oracle_ind", 100), ("oracle_new", 100),
                ("phase_only_new", 102)}

WORLD_NAMES = ["F0", "F1", "F2", "F3", "F4", "G1"]


def gap_key(g):
    return "%d,%d" % (g[0], g[1])


def training_summary(stats, dec, ewe, frac):
    reached = [s for s in stats if s[1]]
    mean_loss_vals = [s[3] for s in stats if s[3] is not None]
    return {
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
        "open_fractions": {gap_key(g): frac[g] for g in frac},
    }


def verify_checkpoint(summary, pdb2_recs):
    """Exact-match check vs P-Db2 record(s); [] = match."""
    mismatches = []
    fields_int = ("episodes", "reached", "crushes")
    fields_float = ("mean_ticks", "mean_ticks_all", "mean_loss")
    fields_dict = ("door_entry_counts", "episodes_with_entry",
                   "open_fractions")
    ref = pdb2_recs[0]
    for other in pdb2_recs[1:]:
        for k in fields_int + fields_float + fields_dict:
            if ref[k] != other[k]:
                mismatches.append("P-Db2 copies disagree on %s" % k)
    for k in fields_int:
        if summary[k] != ref[k]:
            mismatches.append("%s: realised %r vs P-Db2 %r"
                              % (k, summary[k], ref[k]))
    for k in fields_float:
        a, b = summary[k], ref[k]
        if (a is None) != (b is None):
            mismatches.append("%s: realised %r vs P-Db2 %r" % (k, a, b))
        elif a is not None and abs(a - b) > 1e-9:
            mismatches.append("%s: realised %r vs P-Db2 %r (diff %.3g)"
                              % (k, a, b, abs(a - b)))
    for k in fields_dict:
        if summary[k] != ref[k]:
            mismatches.append("%s: realised %r vs P-Db2 %r"
                              % (k, summary[k], ref[k]))
    return mismatches


def world_for(train, wname):
    """(make_world, gaps, schedules, period, open_len) for an eval world."""
    period, open_len = familiar_schedule(train)
    if wname == "F0":
        sched = f0_schedules(train)
        return (lambda: SchedDoorWorld(sched),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F1":
        sched = train
        return (lambda: SchedDoorWorld(sched),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F2":
        sched = rival_schedules(train)
        return (lambda: Pdb2AbsentWorld(sched, period, open_len),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F3":
        sched = f3_schedules(train)
        return (lambda: Pdb2AbsentWorld(sched, period, open_len),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "F4":
        sched = f4_schedules(train)
        return (lambda: SchedDoorWorld(sched),
                tuple(sorted(sched)), sched, period, open_len)
    if wname == "G1":
        sched = g1_schedules(train)
        return (lambda: Pdb2AbsentWorld(sched, period, open_len),
                tuple(sorted(sched)), sched, period, open_len)
    raise ValueError("unknown world %r" % (wname,))


def evaluate(agent, encoder, period, open_len, wname, train,
             log_features=False):
    """Greedy eval over all start phases on one FT1 eval world."""
    make_world, gaps, sched, _cp, _co = world_for(train, wname)
    gaps_set = set(gaps)
    stage_cells = set((4, y) for (_x, y) in gaps)
    is_conflict_world = wname in ("F1", "F2")
    episodes = []
    for start in range(period):
        w = make_world()
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
        entry_ticks = []
        for s in steps:
            p = (s["x"], s["y"])
            if p in gaps_set and (not entries or entries[-1] != p):
                entries.append(p)
                entry_ticks.append(s["t"])
        switches = sum(1 for i in range(1, len(entries))
                       if entries[i] != entries[i - 1])
        segs = segments(positions)
        seen = set()
        revisits = 0
        for c, _k in segs:
            if c in seen:
                revisits += 1
            seen.add(c)
        stall, stall_kind = stall_signature(positions, gaps_set)
        staging_order = []
        for p in positions:
            if p in stage_cells and p not in staging_order:
                staging_order.append(p)
        arrival_step = None
        for s in steps:
            if (s["x"], s["y"]) in stage_cells:
                arrival_step = s
                break
        if arrival_step is not None:
            at = arrival_step["t"]
            arr_cell = (arrival_step["x"], arrival_step["y"])
            arr_door = (5, arr_cell[1])
            status_arr = {gap_key(g): door_open_at(at, *sched[g])
                          for g in gaps}
            clock_open = door_open_at(at, period, open_len, 0)
            arrival = {"tick": at, "cell": list(arr_cell),
                       "arrival_door": list(arr_door),
                       "clock_open_at_arrival": bool(clock_open)}
        else:
            at = None
            arr_door = None
            status_arr = None
            arrival = None
        if entries:
            et = entry_ticks[0]
            status_entry = {gap_key(g): door_open_at(et, *sched[g])
                            for g in gaps}
            wait = (et - at) if at is not None else None
        else:
            status_entry = None
            wait = None
        if not is_conflict_world or arrival is None:
            conflict = None
            outcome = None
        else:
            arr_closed = not door_open_at(at, *sched[arr_door])
            others_open = [g for g in gaps if g != arr_door
                           and door_open_at(at, *sched[g])]
            conflict = bool(arr_closed and others_open)
            if not conflict:
                outcome = None
            elif not entries:
                outcome = "other"
            elif entries[0] == arr_door:
                outcome = "waits-at-closed-door"
            elif entries[0] in others_open:
                outcome = "takes-open-door"
            else:
                outcome = "other"
        if entries:
            first_idx = positions.index(entries[0])
            post_positions = positions[first_idx + 1:]
            post_stall, post_stall_kind = stall_signature(
                post_positions, gaps_set)
        else:
            first_idx = None
            post_positions = []
            post_stall, post_stall_kind = False, None
        blank = bool(log_features and any(
            st.get("f103") == 0.0 for st in steps))
        episodes.append({
            "kind": "episode", "world": wname,
            "start_phase": start, "ticks": ticks,
            "crushes": crushes, "reached": reached,
            "return": total, "first_door": (list(entries[0])
                                            if entries else None),
            "door_entries": [list(p) for p in entries],
            "switches": switches, "stall": stall,
            "stall_kind": stall_kind, "revisits": revisits,
            "staging_order": [list(p) for p in staging_order],
            "arrival": arrival,
            "first_entry_index": first_idx,
            "post_entry_ticks": len(post_positions),
            "post_entry_stall": post_stall,
            "post_entry_stall_kind": post_stall_kind,
            "blank_feature": blank,
            "conflict": conflict, "conflict_outcome": outcome,
            "staging_wait": wait,
            "door_status_at_arrival": status_arr,
            "door_status_at_entry": status_entry,
            "schedules": {gap_key(g): list(sched[g]) for g in gaps},
            "steps": steps})
    return episodes


def load_pdb2_train():
    recs = {}
    with open(PDB2_TRAIN) as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault((r["arm"], r["seed"]), []).append(r)
    return recs


def load_done_ft():
    done = {}
    if os.path.exists(TRAIN_OUT):
        for line in open(TRAIN_OUT):
            r = json.loads(line)
            if r.get("kind") == "training_ft1":
                done[(r["arm"], r["seed"])] = r
    return done


def load_done_counts():
    counts = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if r.get("kind") == "episode":
                k = (r["arm"], r["seed"], r["world"])
                counts[k] = counts.get(k, 0) + 1
    return counts


def ft_snap_path(arm, seed):
    return os.path.join(FT_SNAP_DIR, "%s_%d.pkl" % (arm, seed))


def main():
    pdb2 = load_pdb2_train()
    done_ft = load_done_ft()
    counts = load_done_counts()
    for (arm, seed, world), n in sorted(counts.items()):
        period = 20 if seed < 100 else 30
        if n > period:
            print("ERROR: duplicate job %s/%d %s has %d rows, "
                  "expected %d; stopping before fine-tuning"
                  % (arm, seed, world, n, period), flush=True)
            sys.exit(1)
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
    os.makedirs(FT_SNAP_DIR, exist_ok=True)
    tout = open(TRAIN_OUT, "a")
    out = open(OUT, "a")
    deviations = []

    # ---- Phase A: fine-tuning ----
    for arm, seed, period, open_len in KEPT:
        train = TRAIN_20 if period == 20 else TRAIN_30
        ft_path = ft_snap_path(arm, seed)
        have_record = (arm, seed) in done_ft
        have_snap = os.path.exists(ft_path)
        if have_record and have_snap:
            print("%s/%d: fine-tune record + snapshot present, "
                  "skipping" % (arm, seed), flush=True)
            continue
        if (arm, seed) in SANITY_SNAPS:
            snap = os.path.join(SNAP_DIR, "%s_%d.pkl" % (arm, seed))
            print("%s/%d: loading Step 0 snapshot %s ..."
                  % (arm, seed, snap), flush=True)
            agent, encoder, _b = load_checkpoint(
                snap, arm, period, open_len)
            agent2, encoder2, _b2 = load_checkpoint(
                snap, arm, period, open_len)
            assert_load_equal(agent, agent2)
            if arm == "wrapped":
                assert encoder2.model.period == encoder.model.period \
                    and encoder2.model.open_len == \
                    encoder.model.open_len, \
                    "wrapped ScheduleModel not preserved by snapshot"
            print("%s/%d: snapshot load-audit PASS"
                  % (arm, seed), flush=True)
            ckpt_id = {"arm": arm, "seed": seed,
                       "protocol": "%d/%d" % (period, open_len),
                       "training_world": "F1",
                       "source": "snapshot",
                       "snapshot_file": snap}
        else:
            print("%s/%d: realising P-Db2 checkpoint (2000 episodes)..."
                  % (arm, seed), flush=True)
            agent, stats, encoder, dec, ewe, frac = TRAINERS[arm](
                seed, train, hidden=32, episodes=2000, lr=1e-3)
            summary = training_summary(stats, dec, ewe, frac)
            mismatches = verify_checkpoint(summary, pdb2[(arm, seed)])
            if mismatches:
                print("%s/%d: CHECKPOINT DEVIATION, NOT fine-tuning:"
                      % (arm, seed), flush=True)
                for m in mismatches:
                    print("  %s" % m, flush=True)
                deviations.append((arm, seed, mismatches))
                continue
            print("%s/%d: checkpoint verified exact vs P-Db2 "
                  "(reached=%d, crushes=%d)"
                  % (arm, seed, summary["reached"],
                     summary["crushes"]), flush=True)
            ckpt_id = {"arm": arm, "seed": seed,
                       "protocol": "%d/%d" % (period, open_len),
                       "training_world": "F1",
                       "source": "realised",
                       "realised_summary": summary}
        enc_ft = finetune_encoder(arm, encoder, period, open_len)
        model_start = None
        if arm == "wrapped":
            model_start = (enc_ft.model.period, enc_ft.model.open_len)
        stats_ft, dec_ft, ewe_ft, frac_ft = finetune_neural(
            agent, enc_ft, f3_schedules(train), seed,
            episodes=FINE_TUNE_EPISODES, eps=FINE_TUNE_EPS)
        model_end = None
        if arm == "wrapped":
            model_end = (enc_ft.model.period, enc_ft.model.open_len)
        reached_ft = [s for s in stats_ft if s[1]]
        mean_loss_ft = [s[3] for s in stats_ft if s[3] is not None]
        nan_loss = any(isinstance(v, float) and math.isnan(v)
                       for v in mean_loss_ft)
        if len(stats_ft) != FINE_TUNE_EPISODES or nan_loss:
            print("%s/%d: FINE-TUNE FAILURE (episodes=%d, nan=%s), "
                  "NOT recording; agent excluded from eval"
                  % (arm, seed, len(stats_ft), nan_loss), flush=True)
            deviations.append((arm, seed, ["fine-tune failure"]))
            continue
        save_checkpoint(agent, enc_ft, arm, seed, ft_path)
        print("%s/%d: fine-tuned snapshot saved (%s)"
              % (arm, seed, ft_path), flush=True)
        if not have_record:
            rec = {
                "kind": "training_ft1", "arm": arm, "seed": seed,
                "period": period, "open_len": open_len,
                "training_world": "F1",
                "fine_tune_world": "(5,8)",
                "checkpoint": ckpt_id,
                "fine_tune_choices": {
                    "episodes": FINE_TUNE_EPISODES,
                    "eps": FINE_TUNE_EPS,
                    "replay_buffer": "fresh",
                    "start_rng": "seed+2000"},
                "schedule_model_start": model_start,
                "schedule_model_end": model_end,
                "episodes": len(stats_ft),
                "reached": len(reached_ft),
                "crushes": sum(s[2] for s in stats_ft),
                "mean_ticks": (sum(s[0] for s in reached_ft)
                              / len(reached_ft) if reached_ft else None),
                "mean_ticks_all": sum(s[0] for s in stats_ft)
                / len(stats_ft),
                "mean_loss": (sum(mean_loss_ft) / len(mean_loss_ft)
                              if mean_loss_ft else None),
                "door_entry_counts": {gap_key(g): dec_ft[g]
                                      for g in dec_ft},
                "episodes_with_entry": {gap_key(g): ewe_ft[g]
                                        for g in ewe_ft},
                "open_fractions": {gap_key(g): frac_ft[g]
                                   for g in frac_ft},
            }
            tout.write(json.dumps(rec) + "\n")
            tout.flush()
            done_ft[(arm, seed)] = rec
        print("%s/%d: fine-tune done (reached=%d, crushes=%d, "
              "(5,8) entries=%d)"
              % (arm, seed, len(reached_ft),
                 sum(s[2] for s in stats_ft), dec_ft.get((5, 8), 0)),
              flush=True)

    # ---- Phase B: evaluation ----
    for arm, seed, period, open_len in KEPT:
        if (arm, seed) not in done_ft:
            print("%s/%d: NO fine-tune record, skipping eval "
                  "(excluded, see deviations)" % (arm, seed), flush=True)
            continue
        ft_path = ft_snap_path(arm, seed)
        if not os.path.exists(ft_path):
            print("%s/%d: NO fine-tuned snapshot, skipping eval"
                  % (arm, seed), flush=True)
            continue
        train = TRAIN_20 if period == 20 else TRAIN_30
        need = [w for w in WORLD_NAMES
                if (arm, seed, w) not in counts]
        if not need:
            print("%s/%d: all FT1 worlds done, skipping"
                  % (arm, seed), flush=True)
            continue
        agent, _enc, _b = load_checkpoint(ft_path, arm, period,
                                          open_len)
        for wname in need:
            encoder = fresh_encoder(arm, period, open_len)
            eps = evaluate(agent, encoder, period, open_len,
                           wname, train,
                           log_features=(arm == "wrapped"))
            for e in eps:
                e.update({"arm": arm, "seed": seed,
                          "period": period, "open_len": open_len})
                out.write(json.dumps(e) + "\n")
            out.flush()
            counts[(arm, seed, wname)] = len(eps)
            ok = [e for e in eps if e["reached"]]
            entered = sum(1 for e in eps if e["first_door"])
            n_blank = sum(1 for e in eps if e["blank_feature"])
            print("%s/%d %s: succ=%.2f ticks=%s crush=%d "
                  "entered=%d blank_eps=%d"
                  % (arm, seed, wname, len(ok) / len(eps),
                     ("%.2f" % (sum(e["ticks"] for e in ok) / len(ok))
                      if ok else "nan"),
                     sum(e["crushes"] for e in eps), entered,
                     n_blank), flush=True)
    out.close()
    tout.close()
    if deviations:
        print("ft1_run: DEVIATIONS (agents excluded from eval): %s"
              % deviations, flush=True)


if __name__ == "__main__":
    main()

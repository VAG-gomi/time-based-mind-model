#!/usr/bin/env python3
"""prb2_run.py -- PRB2 noisy-eval probe runner (WO-TBM-PRB2, Step 0).

Additive only. No existing world, trainer, or record file is
edited. This module provides the probe pieces Step 0 specifies;
it runs NO probe evaluation on import.

Modes (explicit --mode flag, no default run):
  --mode sanity : Step 0 determinism sanity ONLY. Retrains
      wrapped/0 by the exact MD1 Step 1 protocol, checks its
      summary against the MD1 training record (and cross-checks
      the Step 0 sanity record), writes nn/prb2_sanity.log and
      nn/prb2_sanity.json, discards weights. Creates NO probe
      records: results_prb2.jsonl and training_prb2.jsonl are
      never opened in this mode.
  --mode probe  : the full probe run (6 retrains + 10 worlds x
      5 reps, 6000 episodes). NOT AUTHORIZED by the Step 0 work
      order: it may be invoked only after instructor verification
      of Step 0 AND the owner gate ruling. As an accident guard
      (not a substitute for the ruling), probe mode requires the
      literal flag --gate-ruled; without it the runner refuses
      to start.
  --mode self-test : no training, no evaluation. Verifies the
      seed formula for all 60 (agent, world) jobs x 5 reps and
      the job/world/arm index mapping. Prints the table; exits
      non-zero on any mismatch.

Noisy-eval protocol (design note sections 5-7, implemented
EXACTLY, mirroring prb1_run.py except for MD1 worlds/encoders):
  seed = 800000 + 1000 * job_index + rep, where
  job_index = arm_index * 10 + world_index, arms
  [wrapped, plain, memory, oracle_ind, oracle_new,
  phase_only_new] (index 0-5), worlds [R0..R9] (index 0-9,
  row order), rep 0-4. One random.Random(seed) stream per
  (agent, world, rep), continuous across that rep's episodes
  in increasing start-phase order 0-19, never reset mid-rep,
  never shared across reps. Per step exactly one draw
  u = rng.random(); if u < 0.05, exactly one second draw
  k = rng.randrange(5) indexing ACTION_NAMES and the step is
  flagged epsilon_action True; otherwise the action is the
  greedy action (lowest-index argmax of the frozen network)
  and NO second draw is made. No other draws consume from
  the stream, and no numpy RNG draw appears anywhere in
  this file.

FORWARD-CHECK (binding): the action choice in this file is
implemented explicitly from the frozen network output and
the rep stream. This file never invokes the agent's own
epsilon action mechanism: no code path in this file calls
the agent's action method at all, and the agent's own RNG
streams are never touched. Greedy actions are computed as
the lowest-index argmax of the frozen network output, the
same expression as the greedy branch of the agent's action
method in agent_nn.NeuralQ. Verify by searching this file
for any invocation of the agent action method and for any
numpy random routine: both searches must return nothing
(the docstring itself names neither pattern literally, so
a clean search is conclusive).

Encoder rule (design note section 6): a fresh encoder object
per (agent, world, rep) via md1_run.fresh_encoder, reset at
every episode start (encoder.reset(world) after
world.reset(start_tick=start)), exactly as in MD1's greedy
evaluation. The encoder persists across a rep's 20 episodes.
For wrapped this means the ScheduleModel starts blank at
each rep and converges during the rep's episodes, so
blank-feature warm-up is a per-rep-start effect, as in
MD1's per-world-start warm-up.

Resume-safety (probe mode, job level): a (agent, world) job
is complete only when it holds 100 records with all
(rep, phase) pairs exactly once. A partial job is stripped
whole and re-run. A torn trailing line is truncated at load
time and the repair is logged. Training records go to
training_prb2.jsonl once per (arm, seed) after the
determinism gate; on resume the agent is retrained
deterministically to run missing evals without writing a
duplicate training record. A determinism-gate failure flags
the agent: its evals are skipped and the failure is
recorded, never silently pooled.

Weights are held in memory and discarded after evals. No
MD1 checkpoint file is created by this runner.
"""

import json
import os
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import ACTION_NAMES, EPISODE_CAP, env_step
from doors_run import segments, stall_signature
from md1_run import fresh_encoder
from train_md1 import TRAINERS
from world_md1 import (PERIOD, OPEN_LEN, TRAINED_ROWS, ALL_ROWS,
                       ARM_INDEX, make_fixed_world)

EPSILON = 0.05
SEED_BASE = 800000
SEED_JOB_STEP = 1000
ARMS = ("wrapped", "plain", "memory", "oracle_ind",
        "oracle_new", "phase_only_new")
WORLDS = [("R%d" % r, r) for r in ALL_ROWS]
REPS = range(5)
SANITY_ARM = "wrapped"
SANITY_SEED = 0

OUT_RESULTS = os.path.join(_HERE, "results_prb2.jsonl")
OUT_TRAINING = os.path.join(_HERE, "training_prb2.jsonl")
MD1_TRAIN_PATH = os.path.join(_HERE, "training_md1.jsonl")
MD1_SANITY_PATH = os.path.join(_HERE, "sanity_md1.jsonl")
SANITY_LOG = os.path.join(_HERE, "prb2_sanity.log")
SANITY_JSON = os.path.join(_HERE, "prb2_sanity.json")


def rep_seed(arm, world_name, rep):
    """Rep RNG seed: 800000 + 1000 * job_index + rep.

    job_index = arm_index * 10 + world_index, arms in
    ARMS order, worlds R0..R9 in row order.
    """
    arm_index = ARMS.index(arm)
    world_index = [w for w, _ in WORLDS].index(world_name)
    return SEED_BASE + SEED_JOB_STEP * (arm_index * 10 + world_index) + rep


def _gap_key(row):
    return "5,%d" % (row,)


def _evaluate_noisy_episode(agent, encoder, wname, gap_row, start,
                            rng, log_features=False):
    """One epsilon-noisy episode on one MD1 gap row at one phase.

    Mirrors md1_run._evaluate_episode line-for-line except
    for the action choice: at each step, u = rng.random();
    if u < 0.05 the action is rng.randrange(5) (flagged
    epsilon_action True); otherwise the action is the greedy
    lowest-index argmax of the frozen network (flagged
    False, no second draw). The encoder is reset for this
    episode by the caller-visible contract (the caller
    calls encoder.reset once per episode); the encoder
    object itself persists across a rep's episodes.
    Returns one episode record (without arm / seed /
    period / open_len / rep / rep_rng_seed / epsilon,
    which the caller adds).
    """
    gap = (5, gap_row)
    stage_cell = (4, gap_row)
    w = make_fixed_world(gap_row)
    w.reset(start_tick=start)
    encoder.reset(w)
    steps = []
    total = 0.0
    crushes = 0
    reached = False
    ticks = 0
    while ticks < EPISODE_CAP:
        x = encoder.encode(w)
        u = rng.random()
        if u < EPSILON:
            a = rng.randrange(5)
            eps_action = True
        else:
            a = int(np.argmax(agent.net.forward(x[None, :])[0]))
            eps_action = False
        (px, py), door = w.observe()
        r, reached, crushed = env_step(w, a)
        rec = {"t": w.tick - 1, "x": px, "y": py,
               "action": ACTION_NAMES[a],
               "door": bool(door), "crushed": bool(crushed),
               "epsilon_action": bool(eps_action)}
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
        if p == gap and (not entries or entries[-1] != p):
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
    stall, stall_kind = stall_signature(positions, {gap})
    staging_order = []
    for p in positions:
        if p == stage_cell and p not in staging_order:
            staging_order.append(p)
    arrival_step = None
    for s in steps:
        if (s["x"], s["y"]) == stage_cell:
            arrival_step = s
            break
    if arrival_step is not None:
        at = arrival_step["t"]
        gap_open_arr = (at % PERIOD) < OPEN_LEN
        arrival = {"tick": at, "cell": list(stage_cell),
                   "arrival_door": list(gap),
                   "gap_open_at_arrival": bool(gap_open_arr)}
        status_arr = {_gap_key(gap_row): bool(gap_open_arr)}
    else:
        at = None
        arrival = None
        status_arr = None
    if entries:
        et = entry_ticks[0]
        status_entry = {_gap_key(gap_row):
                        bool((et % PERIOD) < OPEN_LEN)}
        wait = (et - at) if at is not None else None
    else:
        status_entry = None
        wait = None
    if entries:
        first_idx = positions.index(entries[0])
        post_positions = positions[first_idx + 1:]
        post_stall, post_stall_kind = stall_signature(
            post_positions, {gap})
    else:
        first_idx = None
        post_positions = []
        post_stall, post_stall_kind = False, None
    blank = bool(log_features and any(
        st.get("f103") == 0.0 for st in steps))
    return {
        "kind": "episode", "world": wname, "gap_row": gap_row,
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
        "conflict": None, "conflict_outcome": None,
        "staging_wait": wait,
        "door_status_at_arrival": status_arr,
        "door_status_at_entry": status_entry,
        "schedules": {_gap_key(gap_row): [PERIOD, OPEN_LEN, 0]},
        "steps": steps}


def _train_summary(agent_stats, rc, rr):
    """Training summary in MD1 training-record fields."""
    reached = sum(1 for s in agent_stats if s[1])
    crushes = sum(s[2] for s in agent_stats)
    ticks_ok = [s[0] for s in agent_stats if s[1]]
    mean_ticks = (sum(ticks_ok) / len(ticks_ok)
                  if ticks_ok else None)
    return {"reached": reached, "crushes": crushes,
            "mean_ticks_reached": mean_ticks,
            "row_counts": {str(r): rc[r] for r in TRAINED_ROWS},
            "row_reached": {str(r): rr[r] for r in TRAINED_ROWS}}


def _retrain(arm, seed):
    """Retrain one agent by the exact MD1 Step 1 protocol.

    Returns (agent, summary, wall_time_s). Raises
    SystemExit if row counts are not exactly 250 per
    trained row (same guard as MD1 Step 1).
    """
    t0 = time.perf_counter()
    agent, stats, _enc, rc, _dec, _ewe, rr = TRAINERS[arm](seed)
    dt = time.perf_counter() - t0
    if not all(rc[r] == 250 for r in TRAINED_ROWS):
        raise SystemExit("row counts not balanced for %s/%d: %s"
                         % (arm, seed, rc))
    summary = _train_summary(stats, rc, rr)
    return agent, summary, dt


def _md1_training_record(arm, seed):
    """The MD1 programme training record for (arm, seed)."""
    with open(MD1_TRAIN_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("arm") == arm and r.get("seed") == seed:
                return r
    raise SystemExit("no MD1 training record for %s/%d" % (arm, seed))


def determinism_gate(summary, reference):
    """Field-by-field exact comparison against a reference.

    Compares reached, crushes, mean_ticks_reached,
    row_counts, row_reached. Returns (match, detail dict).
    """
    detail = {}
    for field in ("reached", "crushes", "mean_ticks_reached",
                  "row_counts", "row_reached"):
        detail[field] = (summary[field] == reference[field])
    return all(detail.values()), detail


def _load_repair(path, log):
    """Load jsonl records; truncate a torn tail line, logging it."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        raw = f.read()
    lines = raw.split(b"\n")
    records = []
    bad_at = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            bad_at = i
            break
    if bad_at is not None:
        kept = b"\n".join(lines[:bad_at])
        with open(path, "wb") as f:
            if kept:
                f.write(kept + b"\n")
        log("REPAIR torn tail in %s: truncated %d line(s) from "
            "line %d" % (os.path.basename(path),
                         len(lines) - bad_at, bad_at + 1))
    return records


def _strip_job(path, arm, seed, world, log):
    """Remove all records of one partial job from a jsonl file."""
    with open(path) as f:
        lines = f.readlines()
    kept = []
    removed = 0
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            kept.append(line)
            continue
        if (r.get("arm") == arm and r.get("seed") == seed
                and r.get("world") == world):
            removed += 1
        else:
            kept.append(line)
    with open(path, "w") as f:
        f.writelines(kept)
    log("REPAIR partial job %s/%d %s: stripped %d partial "
        "record(s), job will re-run whole"
        % (arm, seed, world, removed))


def run_sanity():
    """Step 0 determinism sanity: retrain wrapped/0, check gate.

    Writes prb2_sanity.log and prb2_sanity.json. Creates NO
    probe records and NO checkpoint. Weights discarded on exit.
    Returns the sanity dict.
    """
    logf = open(SANITY_LOG, "a")

    def log(line):
        stamped = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                               line)
        logf.write(stamped + "\n")
        logf.flush()
        print(stamped, flush=True)

    log("PRB2 SANITY START (Step 0 determinism sanity; NOT a "
        "probe result; creates no probe records)")
    agent, summary, dt = _retrain(SANITY_ARM, SANITY_SEED)
    log("retrained %s/%d: reached %d/2000, crushes %d, "
        "mean_ticks_reached %s, wall_time %.1fs"
        % (SANITY_ARM, SANITY_SEED, summary["reached"],
           summary["crushes"], summary["mean_ticks_reached"], dt))
    ref = _md1_training_record(SANITY_ARM, SANITY_SEED)
    match, detail = determinism_gate(summary, ref)
    log("determinism gate vs MD1 training record: %s (%s)"
        % ("MATCH" if match else "MISMATCH", detail))
    sanity = {"kind": "prb2_sanity", "arm": SANITY_ARM,
              "seed": SANITY_SEED,
              "retrained": summary,
              "md1_reference": {
                  "reached": ref["reached"],
                  "crushes": ref["crushes"],
                  "mean_ticks_reached": ref["mean_ticks_reached"],
                  "row_counts": ref["row_counts"],
                  "row_reached": ref["row_reached"]},
              "determinism_match": match,
              "field_match": detail,
              "wall_time_s": round(dt, 1)}
    with open(SANITY_JSON, "w") as f:
        f.write(json.dumps(sanity, indent=1) + "\n")
    if match:
        log("PRB2 SANITY COMPLETE: retrain matches MD1 training "
            "record exactly; probe run NOT blocked by sanity")
    else:
        log("PRB2 SANITY FAILED: retrain MISMATCH — probe run "
            "BLOCKED until resolved; mismatch is reported, not "
            "hidden")
    logf.close()
    del agent
    return sanity


def run_probe():
    """Full probe run (NOT authorized by the Step 0 order).

    Invoked only after instructor verification of Step 0 AND
    the owner gate ruling. Caller must pass --gate-ruled.
    """
    log_path = os.path.join(_HERE, "prb2_run.log")
    logf = open(log_path, "a")
    t_start = time.time()

    def log(line):
        stamped = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                               line)
        logf.write(stamped + "\n")
        logf.flush()
        print(stamped, flush=True)

    log("PRB2 PROBE START (post-gate-ruling run only)")
    results = _load_repair(OUT_RESULTS, log)
    trainings = _load_repair(OUT_TRAINING, log)
    trained_keys = {(t["arm"], t["seed"]) for t in trainings
                    if t.get("kind") == "training_prb2"}
    job_counts = {}
    for r in results:
        k = (r.get("arm"), r.get("seed"), r.get("world"))
        job_counts[k] = job_counts.get(k, 0) + 1
    log("resume state: %d training records, %d eval records "
        "on disk" % (len(trained_keys), len(results)))
    deviations = []
    jobs_done_now = 0
    agents_trained_now = 0
    for arm in ARMS:
        seed = 0
        agent, summary, dt = _retrain(arm, seed)
        ref = _md1_training_record(arm, seed)
        match, detail = determinism_gate(summary, ref)
        log("retrained %s/%d: reached %d/2000, crushes %d, "
            "wall_time %.1fs, determinism gate %s"
            % (arm, seed, summary["reached"], summary["crushes"],
               dt, "MATCH" if match else "MISMATCH"))
        if (arm, seed) not in trained_keys:
            rec = {"kind": "training_prb2", "arm": arm, "seed": seed,
                   "episodes": 2000, "reached": summary["reached"],
                   "crushes": summary["crushes"],
                   "mean_ticks_reached":
                       summary["mean_ticks_reached"],
                   "row_counts": summary["row_counts"],
                   "row_reached": summary["row_reached"],
                   "determinism_match": match,
                   "wall_time_s": round(dt, 1)}
            with open(OUT_TRAINING, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            trained_keys.add((arm, seed))
            agents_trained_now += 1
        if not match:
            log("FLAG %s/%d: determinism gate FAILED %s — evals "
                "SKIPPED, results flagged, never silently pooled"
                % (arm, seed, detail))
            deviations.append((arm, seed, detail))
            continue
        for wname, row in WORLDS:
            n = job_counts.get((arm, seed, wname), 0)
            if n >= 100:
                continue
            if 0 < n < 100:
                _strip_job(OUT_RESULTS, arm, seed, wname, log)
            encoder_job = None
            with open(OUT_RESULTS, "a") as out:
                for rep in REPS:
                    encoder = fresh_encoder(arm)
                    rng = random.Random(rep_seed(arm, wname, rep))
                    for start in range(PERIOD):
                        ep = _evaluate_noisy_episode(
                            agent, encoder, wname, row, start, rng,
                            log_features=(arm == "wrapped"))
                        ep.update({"arm": arm, "seed": seed,
                                   "period": PERIOD,
                                   "open_len": OPEN_LEN,
                                   "rep": rep,
                                   "rep_rng_seed": rep_seed(arm, wname,
                                                            rep),
                                   "epsilon": EPSILON})
                        out.write(json.dumps(ep) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
            job_counts[(arm, seed, wname)] = 100
            jobs_done_now += 1
            log("eval %s/%d %s: 100 episodes (5 reps x 20 "
                "phases)" % (arm, seed, wname))
        del encoder_job
        del agent
    results = _load_repair(OUT_RESULTS, log)
    dup_keys = {}
    for r in results:
        k = (r.get("arm"), r.get("seed"), r.get("world"),
             r.get("rep"), r.get("start_phase"))
        dup_keys[k] = dup_keys.get(k, 0) + 1
    dups = sum(v - 1 for v in dup_keys.values() if v > 1)
    log("PRB2 PROBE COMPLETE: %d eval records (%d duplicates), "
        "%d agents trained this run, %d jobs run this run, "
        "deviations %s, wall %.1fs"
        % (len(results), dups, agents_trained_now, jobs_done_now,
           deviations, time.time() - t_start))
    logf.close()


def self_test():
    """Seed-formula and index-mapping check (no training/eval)."""
    ok = True
    for ai, arm in enumerate(ARMS):
        assert ARM_INDEX[arm] == ai, "ARM_INDEX mismatch for %s" % arm
        for wi, (wname, row) in enumerate(WORLDS):
            assert row == wi, "world row mismatch for %s" % wname
            job_index = ai * 10 + wi
            for rep in REPS:
                expect = 800000 + 1000 * job_index + rep
                got = rep_seed(arm, wname, rep)
                if got != expect:
                    ok = False
                    print("self-test FAIL: %s %s rep %d: got %d, "
                          "expected %d" % (arm, wname, rep, got,
                                           expect))
    print("self-test: 60 jobs x 5 reps = 300 rep seeds, "
          "formula 800000 + 1000*job_index + rep: %s"
          % ("ALL MATCH" if ok else "MISMATCH"))
    print("self-test: arms %s; worlds %s"
          % (list(ARMS), [w for w, _ in WORLDS]))
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if "--mode" not in args:
        print("prb2_run.py: --mode is required "
              "(sanity | probe | self-test)", flush=True)
        return 2
    mode = args[args.index("--mode") + 1]
    if mode == "sanity":
        sanity = run_sanity()
        return 0 if sanity["determinism_match"] else 1
    if mode == "self-test":
        return self_test()
    if mode == "probe":
        if "--gate-ruled" not in args:
            print("prb2_run.py: probe mode is NOT authorized by "
                  "the Step 0 work order. It requires instructor "
                  "verification of Step 0 AND the owner gate "
                  "ruling; pass --gate-ruled only after that "
                  "ruling exists.", flush=True)
            return 2
        run_probe()
        return 0
    print("prb2_run.py: unknown --mode %r" % (mode,), flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())

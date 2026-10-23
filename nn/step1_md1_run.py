#!/usr/bin/env python3
"""step1_md1_run.py -- MD1 Steps 1 programme runner (WO-TBM-MD1).

Additive only. Trains 30 programme agents (6 arms x seeds 0-4)
at the exact Step 0 protocol via train_md1.TRAINERS and evaluates
each on 11 worlds (10 fixed R0-R9 + M3, 20 phases each) via
md1_run.run_eval_job / run_eval_job_m3 (fresh encoder per job).

Resume-safety (0 duplicates):
- Training records go to training_md1.jsonl (kind "training_md1",
  same fields as sanity_md1.jsonl plus door_entry_counts and
  episodes_with_entry). A training record is written only once,
  after that agent's training completes; if a relaunch finds the
  record, the agent is NOT retrained for record purposes -- but
  if its evals are incomplete, it IS retrained deterministically
  (same seed/protocol reproduces identical weights, as the spike
  determinism cross-check showed) purely to run the missing evals,
  and no second training record is written.
- Eval jobs are job-level resume-safe in md1_run (20 existing
  records skip the job). A partial job (0 < n < 20, left by a
  crash mid-append) is repaired by stripping its partial records
  and re-running the whole job, so no job ever holds > 20 records.
- Torn trailing lines (crash mid-write) are truncated at load
  time and the repair is logged.
- An agent is skipped entirely only when its training record
  exists AND all 11 eval jobs hold 20 records each.

Progress lines are counts only. No analysis, no verdict here:
Step 2/3 read the record files separately.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_md1 import TRAINERS
from world_md1 import TRAINED_ROWS, ALL_ROWS
from md1_run import run_eval_job, run_eval_job_m3

ARMS = ("wrapped", "plain", "memory", "oracle_ind",
        "oracle_new", "phase_only_new")
SEEDS = (0, 1, 2, 3, 4)
FIXED = [("R%d" % r, r) for r in ALL_ROWS]
LOG_PATH = os.path.join(_HERE, "step1_md1.log")
TRAIN_PATH = os.path.join(_HERE, "training_md1.jsonl")
RES_PATH = os.path.join(_HERE, "results_md1.jsonl")


def _load_repair(path, log):
    """Load jsonl records; truncate a torn tail line, logging it.

    Returns the list of records. A file that does not exist yet
    yields [].
    """
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
        log("REPAIR torn tail in %s: truncated %d line(s) from line %d"
            % (os.path.basename(path), len(lines) - bad_at, bad_at + 1))
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
    log("REPAIR partial job %s/%d %s: stripped %d partial record(s), "
        "job will re-run whole" % (arm, seed, world, removed))


def main():
    logf = open(LOG_PATH, "a")
    t_start = time.time()

    def log(line):
        stamped = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), line)
        logf.write(stamped + "\n")
        logf.flush()
        print(stamped, flush=True)

    log("STEP1 MD1 START (resume-safe; existing records will be kept, "
        "partial jobs repaired, complete agents skipped)")
    trainings = _load_repair(TRAIN_PATH, log)
    results = _load_repair(RES_PATH, log)
    trained_keys = {(t["arm"], t["seed"]) for t in trainings
                    if t.get("kind") == "training_md1"}
    eval_counts = {}
    for r in results:
        k = (r.get("arm"), r.get("seed"), r.get("world"))
        eval_counts[k] = eval_counts.get(k, 0) + 1
    log("resume state: %d training records, %d eval records on disk"
        % (len(trained_keys), len(results)))

    agents_done = 0
    agents_trained_now = 0
    jobs_done_now = 0
    jobs_skipped = 0
    for arm in ARMS:
        for seed in SEEDS:
            worlds = FIXED + [("M3", None)]
            complete = ((arm, seed) in trained_keys and all(
                eval_counts.get((arm, seed, w), 0) >= 20
                for w, _ in worlds))
            if complete:
                agents_done += 1
                log("skip %s/%d: training + 11/11 eval jobs already "
                    "complete" % (arm, seed))
                continue
            # (Re)train deterministically; record only if absent.
            t0 = time.perf_counter()
            agent, stats, _enc, rc, dec, ewe, rr = TRAINERS[arm](seed)
            dt = time.perf_counter() - t0
            reached = sum(1 for s in stats if s[1])
            crushes = sum(s[2] for s in stats)
            ticks_ok = [s[0] for s in stats if s[1]]
            mean_ticks = (sum(ticks_ok) / len(ticks_ok)
                          if ticks_ok else None)
            balanced = all(rc[r] == 250 for r in TRAINED_ROWS)
            log("trained %s/%d: reached %d/2000, crushes %d, "
                "mean_ticks_reached %s, row_counts_balanced %s, "
                "wall_time %.1fs"
                % (arm, seed, reached, crushes, mean_ticks, balanced,
                   dt))
            if not balanced:
                raise SystemExit(
                    "row counts not balanced for %s/%d: %s"
                    % (arm, seed, rc))
            if (arm, seed) not in trained_keys:
                rec = {
                    "kind": "training_md1", "arm": arm, "seed": seed,
                    "episodes": 2000, "reached": reached,
                    "crushes": crushes,
                    "mean_ticks_reached": mean_ticks,
                    "row_counts": {str(r): rc[r] for r in TRAINED_ROWS},
                    "row_reached": {str(r): rr[r] for r in TRAINED_ROWS},
                    "door_entry_counts": {str(r): dec[r]
                                          for r in TRAINED_ROWS},
                    "episodes_with_entry": {str(r): ewe[r]
                                            for r in TRAINED_ROWS},
                    "wall_time_s": round(dt, 1)}
                with open(TRAIN_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                trained_keys.add((arm, seed))
                agents_trained_now += 1
            else:
                log("note %s/%d: training record already on disk; "
                    "retrain used only to run missing evals, no "
                    "duplicate record written" % (arm, seed))
            # Evals: repair partial jobs, skip complete jobs.
            for wname, row in FIXED:
                n = eval_counts.get((arm, seed, wname), 0)
                if n >= 20:
                    jobs_skipped += 1
                    continue
                if 0 < n < 20:
                    _strip_job(RES_PATH, arm, seed, wname, log)
                got, _sk = run_eval_job(agent, arm, seed, wname, row,
                                        RES_PATH)
                eval_counts[(arm, seed, wname)] = got
                jobs_done_now += 1
                log("eval %s/%d %s: %d episodes (job %d/11)"
                    % (arm, seed, wname, got,
                       FIXED.index((wname, row)) + 1))
            n = eval_counts.get((arm, seed, "M3"), 0)
            if n >= 20:
                jobs_skipped += 1
            else:
                if 0 < n < 20:
                    _strip_job(RES_PATH, arm, seed, "M3", log)
                got, _sk = run_eval_job_m3(agent, arm, seed, RES_PATH)
                eval_counts[(arm, seed, "M3")] = got
                jobs_done_now += 1
                log("eval %s/%d M3: %d episodes (job 11/11)"
                    % (arm, seed, got))
            agents_done += 1
            log("agent %s/%d complete (%d/30 agents)"
                % (arm, seed, agents_done))

    # Final accounting (counts only; analysis is Step 2).
    trainings = _load_repair(TRAIN_PATH, log)
    results = _load_repair(RES_PATH, log)
    dup_keys = {}
    for r in results:
        k = (r.get("arm"), r.get("seed"), r.get("world"),
             r.get("start_phase"))
        dup_keys[k] = dup_keys.get(k, 0) + 1
    dups = sum(v - 1 for v in dup_keys.values() if v > 1)
    train_keys = [(t["arm"], t["seed"]) for t in trainings
                  if t.get("kind") == "training_md1"]
    dup_train = len(train_keys) - len(set(train_keys))
    log("STEP1 MD1 COMPLETE: %d training records (%d duplicates), "
        "%d eval records (%d duplicates), %d agents trained this run, "
        "%d eval jobs run this run, %d eval jobs skipped, wall %.1fs"
        % (len(train_keys), dup_train, len(results), dups,
           agents_trained_now, jobs_done_now, jobs_skipped,
           time.time() - t_start))
    logf.close()


if __name__ == "__main__":
    main()

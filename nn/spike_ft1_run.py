#!/usr/bin/env python3
"""FT1 benchmark spike (MEASUREMENT ONLY, SPIKE-labelled).

Owner gate ruling (design-note commit ffa5582): benchmark spike
first. This script measures, it does not run Step 1:
- 6 timed fine-tunes (200 episodes on (5,8), one per arm) from the
  Step 0 sanity checkpoint snapshots, timed with perf_counter
  around the fine-tune call only. Spike fine-tuned weights are
  DISCARDED (never saved as checkpoints, never reused); records
  are SPIKE-labelled in nn/spike_ft1.jsonl and are NEVER Step 1
  records and must never be pooled with Step 1.
- A 12-job greedy eval sample (6 spike fine-tuned agents x 2
  worlds: F3, the fine-tuning world, and F1, the multi-door
  training world), fresh encoder per (agent, world) per the
  design-note Step 2 rule, timed per job. Reason for the
  choice: one one-door world + one multi-door world bounds
  eval pace across the world shapes Step 2 will use; the
  sample only needs to bound pace, not to measure outcomes.
- Load audit before every fine-tune: the snapshot is loaded
  twice and train_ft1.assert_load_equal must pass (plus the
  wrapped ScheduleModel period/open_len check). A failed
  audit stops that agent's measurement and is reported; the
  agent is never fine-tuned from an unaudited load.

Resume-safe: agents already present in nn/spike_ft1.jsonl are
skipped. A resumed spike measurement is still labelled SPIKE.
No existing file is edited. No commit, no push.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import EPISODE_CAP, env_step
from train_ft1 import (FINE_TUNE_EPISODES, FINE_TUNE_EPS,
                       assert_load_equal, finetune_encoder,
                       finetune_neural, fresh_encoder,
                       load_checkpoint)
from world_doors_sched import SchedDoorWorld
from world_pdb2 import (TRAIN_20, TRAIN_30, Pdb2AbsentWorld,
                        f3_schedules)

OUT = os.path.join(_HERE, "spike_ft1.jsonl")
SNAP_DIR = os.path.join(_HERE, "checkpoints_ft1")

SPIKE_AGENTS = [
    ("wrapped", 0, 20, 10),
    ("plain", 0, 20, 10),
    ("memory", 0, 20, 10),
    ("oracle_ind", 100, 30, 14),
    ("oracle_new", 100, 30, 14),
    ("phase_only_new", 102, 30, 14),
]


def gap_key(g):
    return "%d,%d" % (g[0], g[1])


def greedy_eval(agent, encoder, wname, train, period, open_len):
    """One greedy job: one deterministic episode per start phase."""
    if wname == "F3":
        sched = f3_schedules(train)
        make = lambda: Pdb2AbsentWorld(sched, period, open_len)
    elif wname == "F1":
        make = lambda: SchedDoorWorld(train)
    else:
        raise ValueError("unknown spike eval world %r" % (wname,))
    reached = 0
    ticks_total = 0
    episodes = 0
    for start in range(period):
        w = make()
        w.reset(start_tick=start)
        encoder.reset(w)
        x = encoder.encode(w)
        ticks = 0
        done = False
        while ticks < EPISODE_CAP:
            a = agent.act(x, 0.0)
            _r, done, _crushed = env_step(w, a)
            x = encoder.encode(w)
            ticks += 1
            if done:
                break
        reached += int(done)
        ticks_total += ticks
        episodes += 1
    return {"episodes": episodes, "reached": reached,
            "mean_ticks_all": ticks_total / episodes}


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["arm"], r["seed"]))
    out = open(OUT, "a")
    n_fail = 0
    for arm, seed, period, open_len in SPIKE_AGENTS:
        if (arm, seed) in done:
            print("%s/%d: spike record present, skipping"
                  % (arm, seed), flush=True)
            continue
        train = TRAIN_20 if period == 20 else TRAIN_30
        snap = os.path.join(SNAP_DIR, "%s_%d.pkl" % (arm, seed))
        # Determinism/load audit: load twice, must be identical.
        agent, encoder, _b1 = load_checkpoint(
            snap, arm, period, open_len)
        agent2, encoder2, _b2 = load_checkpoint(
            snap, arm, period, open_len)
        try:
            assert_load_equal(agent, agent2)
            if arm == "wrapped":
                assert (encoder2.model.period
                        == encoder.model.period
                        and encoder2.model.open_len
                        == encoder.model.open_len), \
                    "wrapped ScheduleModel not preserved by load"
            audit = "pass"
        except AssertionError as e:
            print("%s/%d: LOAD AUDIT FAILED (%s), NOT fine-tuning"
                  % (arm, seed, e), flush=True)
            out.write(json.dumps({
                "kind": "spike_ft1", "arm": arm, "seed": seed,
                "period": period, "open_len": open_len,
                "snapshot": snap, "load_audit": "fail",
                "load_audit_error": str(e)}) + "\n")
            out.flush()
            n_fail += 1
            continue
        print("%s/%d: load audit PASS (%s)"
              % (arm, seed, snap), flush=True)
        enc_ft = finetune_encoder(arm, encoder, period, open_len)
        t0 = time.perf_counter()
        stats, dec, ewe, frac = finetune_neural(
            agent, enc_ft, f3_schedules(train), seed,
            episodes=FINE_TUNE_EPISODES, eps=FINE_TUNE_EPS)
        seconds = time.perf_counter() - t0
        reached_ft = [s for s in stats if s[1]]
        mean_loss = [s[3] for s in stats if s[3] is not None]
        print("%s/%d: SPIKE fine-tune %.1fs (200 episodes, "
              "reached=%d, crushes=%d, (5,8) entries=%d)"
              % (arm, seed, seconds, len(reached_ft),
                 sum(s[2] for s in stats), dec.get((5, 8), 0)),
              flush=True)
        # Eval sample on the spike fine-tuned (in-memory) agent.
        # Weights are used here only, then discarded.
        evals = {}
        for wname in ("F3", "F1"):
            enc_ev = fresh_encoder(arm, period, open_len)
            t1 = time.perf_counter()
            ev = greedy_eval(agent, enc_ev, wname, train,
                             period, open_len)
            ev_seconds = time.perf_counter() - t1
            ev["seconds"] = ev_seconds
            evals[wname] = ev
            print("%s/%d: SPIKE eval %s %.3fs (%d episodes, "
                  "reached=%d)"
                  % (arm, seed, wname, ev_seconds,
                     ev["episodes"], ev["reached"]), flush=True)
        rec = {
            "kind": "spike_ft1", "arm": arm, "seed": seed,
            "period": period, "open_len": open_len,
            "snapshot": snap, "load_audit": audit,
            "fine_tune": {
                "episodes": len(stats),
                "seconds": seconds,
                "reached": len(reached_ft),
                "crushes": sum(s[2] for s in stats),
                "mean_ticks": (sum(s[0] for s in reached_ft)
                              / len(reached_ft) if reached_ft
                              else None),
                "mean_ticks_all": sum(s[0] for s in stats)
                / len(stats),
                "mean_loss": (sum(mean_loss) / len(mean_loss)
                              if mean_loss else None),
                "door_entry_counts": {gap_key(g): dec[g]
                                      for g in dec},
                "episodes_with_entry": {gap_key(g): ewe[g]
                                        for g in ewe},
                "open_fractions": {gap_key(g): frac[g]
                                   for g in frac},
                "eps": FINE_TUNE_EPS,
                "start_rng": "seed+2000",
            },
            "eval_sample": evals,
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        # Spike weights discarded here: `agent` goes out of scope
        # per agent; nothing is saved.
    out.close()
    if n_fail:
        print("spike_ft1: %d LOAD AUDIT FAILURE(S), see above"
              % n_fail, flush=True)
        sys.exit(1)
    print("spike_ft1: all 6 agents measured (SPIKE only)", flush=True)


if __name__ == "__main__":
    main()

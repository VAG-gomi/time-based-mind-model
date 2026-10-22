#!/usr/bin/env python3
"""SPIKE ONLY - P-Db2 benchmark spike (measurement only, owner gate ruling).

NOT Step 1 records, NOT Step 1 trainings, NOT Step 2 evaluation.
These spike trainings are timed reruns of the Step 0 sanity agents
(1 per arm, 2000 episodes each, train_pdb2.py trainers, F1
schedule-varied training world) for wall-clock measurement only.
They must NOT be reused, quoted, or pooled as Step 1 findings.

Resume-safe: each arm's timed training and timed F1 eval are
flushed to stdout (tee'd to nn/spike_pdb2.log) immediately. On
relaunch, arms with both markers present are skipped; an arm
with a training marker but no eval marker is retrained UNTIMED
(deterministic, labelled) to recover its agent, then its eval
is timed. A restart therefore loses at most one arm's work.

Outputs are printed to stdout; the caller redirects/tee to
nn/spike_pdb2.log. No existing file is edited by this script.
"""

import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_nn import EPISODE_CAP, env_step
from train_pdb2 import TRAINERS
from world_doors_sched import SchedDoorWorld
from world_pdb2 import (TRAIN_20, TRAIN_30, Pdb2AbsentWorld,
                        f3_schedules, familiar_schedule,
                        rival_schedules)

SPIKE_AGENTS = [("wrapped", 0, TRAIN_20),
                ("plain", 0, TRAIN_20),
                ("memory", 0, TRAIN_20),
                ("oracle_ind", 100, TRAIN_30),
                ("oracle_new", 100, TRAIN_30),
                ("phase_only_new", 100, TRAIN_30)]

LOG_PATH = os.path.join(_HERE, "spike_pdb2.log")


def logged_markers():
    marks = set()
    if os.path.exists(LOG_PATH):
        for line in open(LOG_PATH):
            m = re.match(r"SPIKE TRAIN (\S+)/(\d+) ", line)
            if m:
                marks.add(("train", m.group(1), int(m.group(2))))
            m = re.match(r"SPIKE EVAL (\S+)/(\d+) (F1|F3) ", line)
            if m:
                marks.add(("eval", m.group(1), int(m.group(2)), m.group(3)))
    return marks


def make_world(kind, train, period, open_len):
    if kind == "F1":
        return SchedDoorWorld(train)
    if kind == "F3":
        return Pdb2AbsentWorld(f3_schedules(train), period, open_len)
    raise ValueError(kind)


def eval_job(agent, encoder, kind, train, period):
    period_f, open_len = familiar_schedule(train)
    t0 = time.perf_counter()
    total_ticks = 0
    reached = 0
    for start in range(period):
        w = make_world(kind, train, period_f, open_len)
        w.reset(start_tick=start)
        encoder.reset(w)
        ticks = 0
        done = False
        while ticks < EPISODE_CAP and not done:
            x = encoder.encode(w)
            a = agent.act(x, 0.0)
            _r, done, _crushed = env_step(w, a)
            ticks += 1
        total_ticks += ticks
        reached += int(done)
    return period, time.perf_counter() - t0, total_ticks, reached


def main():
    marks = logged_markers()
    print("SPIKE P-Db2 benchmark - measurement only, NOT Step 1/Step 2",
          flush=True)
    for arm, seed, train in SPIKE_AGENTS:
        period = 20 if seed < 100 else 30
        train_done = ("train", arm, seed) in marks
        eval_done = ("eval", arm, seed, "F1") in marks
        if train_done and eval_done:
            print("SPIKE SKIP %s/%d: train+eval already logged"
                  % (arm, seed), flush=True)
            continue
        if train_done:
            print("SPIKE RETRAIN-UNTIMED %s/%d: recovering agent for "
                  "eval (training already timed)" % (arm, seed),
                  flush=True)
            agent, _stats, enc, _dec, _ewe, _frac = TRAINERS[arm](
                seed, train, hidden=32, episodes=2000, lr=1e-3)
        else:
            t0 = time.perf_counter()
            agent, stats, enc, dec, _ewe, _frac = TRAINERS[arm](
                seed, train, hidden=32, episodes=2000, lr=1e-3)
            dt = time.perf_counter() - t0
            reached = sum(1 for s in stats if s[1])
            print("SPIKE TRAIN %s/%d wall=%.1fs episodes=%d reached=%d "
                  "total_entries=%d"
                  % (arm, seed, dt, len(stats), reached,
                     sum(dec.values())), flush=True)
        n, dt, ticks, reached = eval_job(agent, enc, "F1", train, period)
        print("SPIKE EVAL %s/%d F1 episodes=%d wall=%.2fs "
              "total_ticks=%d reached=%d"
              % (arm, seed, n, dt, ticks, reached), flush=True)
        if arm == "wrapped" and ("eval", "wrapped", 0, "F3") not in marks:
            n, dt, ticks, reached = eval_job(agent, enc, "F3", train, 20)
            print("SPIKE EVAL wrapped/0 F3 episodes=%d wall=%.2fs "
                  "total_ticks=%d reached=%d" % (n, dt, ticks, reached),
                  flush=True)
    print("SPIKE DONE - timed trainings and eval jobs logged above",
          flush=True)


if __name__ == "__main__":
    main()

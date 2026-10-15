#!/usr/bin/env python3
"""run_arms.py -- A5/B2: arms and the results runner.

Trains the plain arm (agent_nn.py), the one-step-memory arm
(agent_mem.py), the wrapped arm (agent_wrapped.py: same learner plus
ScheduleModel time features), and the raw-clock control arm
(agent_wrapped.py: tick input, no schedule model) on seeds 0-9 with
identical hyperparameters and no per-arm tuning, evaluates each policy
twice (pure greedy and epsilon = 0.05), and compares them against three
hand-coded yardsticks (naive, time-aware, delay-line) on the same
protocol.

Primary metric: mean episode return = undiscounted sum of rewards
(step -0.01, goal +1.0, crush -1.0), with failure to reach the goal
included. Also reported: crush rate (mean crushes/episode), success
rate, mean ticks to goal over successes.

Results are appended to a JSONL file (one record per arm/seed) as each
seed finishes, so an interrupted run loses no completed work; re-running
resumes from the existing file. The human-readable summary
(output_a5.txt) is rewritten after every completed seed.

The Sequence B comparison rule is written into the summary here, in A5,
before the wrapper exists (pre-registration).

world.py is untouched. No per-seed tuning, no hyperparameter tuning in
this step (any future tuning uses separate seeds 100+, same budget for
every arm, per the A4 ruling).
"""

import argparse
import json
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from world import World, ACTIONS, AGENT_START, GOAL_POS, DOOR_CELL
from agent_nn import (PERIOD, OPEN_LEN, EPISODE_CAP, R_STEP, R_GOAL, R_CRUSH,
                      GAMMA, ACTION_NAMES, N_ACTIONS, PlainEncoder,
                      train_neural, naive_policy, time_aware_policy,
                      env_step)
from agent_mem import MemoryEncoder, train_memory
from agent_wrapped import (train_wrapped, WrappedEncoder,
                           train_clock, ClockEncoder)

HIDDEN = 32
LR = 1e-3
EPISODES = 2000
EPS_EVAL = 0.05
EPS_REPS = 10  # episodes per start phase for the epsilon evaluation

COMPARISON_RULE = """\
SEQUENCE B COMPARISON RULE (adopted in A5-R, before the wrapper exists;
replaces the median-based rule of A5. Reason: the return ceiling is
small -- time-aware 0.787, naive 0.737, delay-line 0.725 -- so at most
~0.05 separates the best from the naive plateau, while seed-to-seed
spread (std 0.49 in the memory arm) dwarfs it; and greedy returns
cluster on plateaus (0.737 naive, 0.787 ideal), so a median mostly
records which plateau most seeds landed on. Demanding 10/10 perfect
seeds also punishes the learner's greedy "stay"-loop property rather
than the wrapper.)
A seed is "clean and fast" iff its greedy run has zero crushes,
success >= 0.9, and mean return >= 0.75. (With 0 crushes and full
success, return >= 0.75 is mean ticks <= 25: strictly better than the
naive plateau at 0.737, short of the tick-knowing ideal at 0.787.)
A5 baselines: plain 1/10 clean-and-fast (seed 2), memory 1/10
(seed 8); counts are recomputed in the per-arm aggregates below.
The wrapped arm trains on the same seeds 0-9, with the same budget
(2000 episodes), the same hyperparameters, and the same evaluation
(greedy + eps = 0.05). The wrapped arm WINS iff at least 7 of its 10
seeds are clean and fast. Against either baseline's 1/10, 7/10 is
p ~= 0.01 by Fisher's exact test (one-sided). The eps = 0.05
evaluation stays a robustness report, not a gate: a wrapper that wins
on greedy but collapses under eps = 0.05 noise must say so.
"""


def _clean_fast(rec):
    """A seed is clean and fast iff its greedy run has zero crushes,
    success >= 0.9, and mean return >= 0.75."""
    g = rec["greedy"]
    return (g["crush_rate"] == 0.0 and g["success"] >= 0.9
            and g["mean_return"] >= 0.75)


# ---------------------------------------------------------------- policies

class LearnedPolicy(object):
    """Wraps (agent, encoder): reset(world) + act(world, eps) -> idx.

    Uses the agent's own RNG stream for epsilon-greedy randomness, so
    evaluation draws continue deterministically from training. The
    pipeline order (train -> greedy eval -> eps eval) is fixed, which
    makes the whole run exactly reproducible.
    """

    def __init__(self, agent, encoder):
        self.agent = agent
        self.encoder = encoder

    def reset(self, world):
        self.encoder.reset(world)

    def act(self, world, eps, rng=None):
        return self.agent.act(self.encoder.encode(world), eps)


class FuncPolicy(object):
    """Wraps a stateless world -> action-name function (yardsticks)."""

    def __init__(self, fn):
        self.fn = fn

    def reset(self, world):
        pass

    def act(self, world, eps, rng=None):
        if eps > 0:
            assert rng is not None, "eps eval needs an rng"
            if rng.random() < eps:
                return rng.randrange(N_ACTIONS)
        return ACTION_NAMES.index(self.fn(world))


def _step_toward(target, x, y):
    """One action name moving toward target (x-first, then y)."""
    tx, ty = target
    if x < tx:
        return "right"
    if x > tx:
        return "left"
    if y < ty:
        return "down"
    if y > ty:
        return "up"
    return "stay"


class DelayLinePolicy(object):
    """Hand-coded delay-line yardstick.

    Waits in holding cell (0,5) while the door is open; on seeing the
    door closed, approaches staging cell (4,5); at staging, crosses
    only on a fresh opening (door open now AND closed at the previous
    observation). Past the door, walks to the goal.

    The fresh-opening rule makes zero crushes hold by construction:
    entering on the first open tick of a period leaves the full 10-tick
    open window, and the agent steps off the door cell on the next tick.
    It is deliberately conservative: arriving at staging mid-open-period
    means waiting out the rest of that opening plus the next closed
    period. Its measured numbers (not its intent) say how much that
    conservatism costs.

    After a crush (possible only under eps-greedy noise, when a random
    action overrides the policy), the agent is back at (0,0); the phase
    is reset to hold so the policy re-enters its holding pattern instead
    of acting on a stale phase.
    """

    HOLD = (0, 5)
    STAGE = (4, 5)

    def __init__(self):
        self.prev_door = 0
        self.phase = "hold"

    def reset(self, world):
        _pos, door = world.observe()
        self.prev_door = 1 if door else 0
        self.phase = "hold"

    def act(self, world, eps, rng=None):
        (x, y), door = world.observe()
        door_bit = 1 if door else 0
        if (x, y) == AGENT_START and self.phase in ("stage", "escape"):
            self.phase = "hold"  # crushed/reset under eps noise: re-hold
        if (x, y) == DOOR_CELL or x > 5:
            if x < 9:
                a = "right"
            elif y < 9:
                a = "down"
            else:
                a = "stay"
            self.phase = "escape"
        elif self.phase == "hold":
            if door:
                a = _step_toward(self.HOLD, x, y)
            else:
                self.phase = "approach"
                a = _step_toward(self.STAGE, x, y)
        elif self.phase == "approach":
            if (x, y) == self.STAGE:
                self.phase = "stage"
                a = ("right" if (door and self.prev_door == 0) else "stay")
            else:
                a = _step_toward(self.STAGE, x, y)
        else:  # stage
            if door and self.prev_door == 0:
                a = "right"
                self.phase = "escape"
            else:
                a = "stay"
        if eps > 0:
            assert rng is not None, "eps eval needs an rng"
            if rng.random() < eps:
                a = ACTION_NAMES[rng.randrange(N_ACTIONS)]
        self.prev_door = door_bit
        return ACTION_NAMES.index(a)


# ---------------------------------------------------------------- evaluation

def run_episode(policy, start_phase, eps, rng, cap=EPISODE_CAP):
    """One episode from start_phase. Returns (return, ticks, crushes, reached)."""
    w = World(period=PERIOD, open_len=OPEN_LEN)
    w.reset(start_tick=start_phase)
    policy.reset(w)
    total = 0.0
    ticks = 0
    crushes = 0
    reached = False
    while ticks < cap:
        a = policy.act(w, eps, rng)
        r, reached, crushed = env_step(w, a)
        total += r
        crushes += crushed
        ticks += 1
        if reached:
            break
    return total, ticks, crushes, reached


def evaluate(policy, eps, reps, rng):
    """All 20 start phases x reps episodes. Returns a list of rows."""
    rows = []
    for start in range(PERIOD):
        for _ in range(reps):
            rows.append(run_episode(policy, start, eps, rng))
    return rows


def summarize_rows(rows):
    """(mean return, crushes/episode, frac episodes with >=1 crush,
    success rate, mean ticks to goal over successes, n)."""
    n = len(rows)
    mean_ret = sum(r[0] for r in rows) / n
    crush_rate = sum(r[2] for r in rows) / n
    frac_crushed = sum(1 for r in rows if r[2] > 0) / n
    success = sum(1 for r in rows if r[3]) / n
    ticks_ok = [r[1] for r in rows if r[3]]
    mean_ticks = sum(ticks_ok) / len(ticks_ok) if ticks_ok else None
    return mean_ret, crush_rate, frac_crushed, success, mean_ticks, n


def _pack(metrics):
    mean_ret, crush_rate, frac_crushed, success, mean_ticks, n = metrics
    return {"mean_return": mean_ret, "crush_rate": crush_rate,
            "frac_crushed": frac_crushed, "success": success,
            "mean_ticks": mean_ticks, "n": n}


# ---------------------------------------------------------------- running

HYPERPARAMS = {"hidden": HIDDEN, "lr": LR, "episodes": EPISODES,
               "gamma": GAMMA, "r_step": R_STEP, "r_goal": R_GOAL,
               "r_crush": R_CRUSH, "episode_cap": EPISODE_CAP,
               "period": PERIOD, "open_len": OPEN_LEN,
               "eps_eval": EPS_EVAL, "eps_reps_per_phase": EPS_REPS}


def run_arm_seed(arm, seed, episodes):
    """Train one (arm, seed) and evaluate it twice. Returns the record."""
    extra = {}
    if arm == "plain":
        agent, stats = train_neural(seed, hidden=HIDDEN,
                                    episodes=episodes, lr=LR)
        policy = LearnedPolicy(agent, PlainEncoder())
    elif arm == "memory":
        agent, stats = train_memory(seed, hidden=HIDDEN,
                                    episodes=episodes, lr=LR)
        policy = LearnedPolicy(agent, MemoryEncoder())
    elif arm == "wrapped":
        agent, stats, encoder = train_wrapped(seed, hidden=HIDDEN,
                                              episodes=episodes, lr=LR)
        # Eval reuses the SAME encoder, so the ScheduleModel's knowledge
        # persists from training into both evaluations.
        policy = LearnedPolicy(agent, encoder)
        extra = {"frac_ticks_known": encoder.frac_known(),
                 "first_converged_episode": encoder.converged_episode,
                 "first_converged_tick": encoder.converged_tick,
                 "model_contradictions": encoder.model.contradictions}
    elif arm == "clock":
        agent, stats = train_clock(seed, hidden=HIDDEN,
                                   episodes=episodes, lr=LR)
        policy = LearnedPolicy(agent, ClockEncoder())
    else:
        raise ValueError("unknown arm %r" % (arm,))
    greedy = _pack(summarize_rows(evaluate(policy, 0.0, 1, None)))
    eps = _pack(summarize_rows(
        evaluate(policy, EPS_EVAL, EPS_REPS, None)))
    last = stats[-100:]
    train = {"last100_success": sum(s[1] for s in last) / len(last),
             "last100_mean_ticks": sum(s[0] for s in last) / len(last),
             "last100_mean_crushes": sum(s[2] for s in last) / len(last)}
    train.update(extra)
    return {"kind": "arm", "arm": arm, "seed": seed,
            "hyperparams": dict(HYPERPARAMS, episodes=episodes),
            "greedy": greedy, "eps005": eps, "train": train}


YARDSTICKS = [("naive", FuncPolicy(naive_policy)),
              ("time-aware", FuncPolicy(time_aware_policy)),
              ("delay-line", DelayLinePolicy())]


def run_yardstick(name, policy, idx, episodes):
    """Evaluate one yardstick twice. idx fixes its eps-eval RNG."""
    greedy = _pack(summarize_rows(evaluate(policy, 0.0, 1, None)))
    rng = random.Random(20261014 + idx)
    eps = _pack(summarize_rows(
        evaluate(policy, EPS_EVAL, EPS_REPS, rng)))
    return {"kind": "yardstick", "name": name, "seed": None,
            "hyperparams": dict(HYPERPARAMS, episodes=episodes),
            "greedy": greedy, "eps005": eps}


# ---------------------------------------------------------------- persistence

def load_records(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_record(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


# ---------------------------------------------------------------- summary

def _fmt(x, spec="%.3f"):
    return spec % x if x is not None else "n/a"


def _eval_table(title, records, key):
    """Per-seed table for one arm (or yardstick set) and one eval key."""
    lines = ["%s (%s)" % (title, key), "seed | mean_return | crush/ep | "
             "success | mean_ticks"]
    for rec in records:
        m = rec[key]
        label = str(rec["seed"]) if rec["seed"] is not None \
            else rec.get("name", "?")
        lines.append("%4s | %11s | %8s | %7s | %10s"
                     % (label, _fmt(m["mean_return"]), _fmt(m["crush_rate"]),
                        _fmt(m["success"], "%.2f"),
                        _fmt(m["mean_ticks"], "%.1f")))
    return "\n".join(lines)


def _arm_aggregates(arm, records):
    """Median/spread/fractions for one arm over its seeds (greedy primary)."""
    by_seed = sorted([r for r in records if r.get("arm") == arm],
                     key=lambda r: r["seed"])
    out = ["%s arm aggregates (%d seeds)" % (arm, len(by_seed))]
    for key in ("greedy", "eps005"):
        rets = [r[key]["mean_return"] for r in by_seed]
        zero = sum(1 for r in by_seed if r[key]["crush_rate"] == 0.0)
        good = sum(1 for r in by_seed if r[key]["success"] >= 0.9)
        out.append("  %s: median return %s | min %s | max %s | std %s | "
                   "zero-crush seeds %d/%d | success>=0.9 seeds %d/%d"
                   % (key, _fmt(statistics.median(rets)),
                      _fmt(min(rets)), _fmt(max(rets)),
                      _fmt(statistics.pstdev(rets)),
                      zero, len(by_seed), good, len(by_seed)))
    cf = [r["seed"] for r in by_seed if _clean_fast(r)]
    out.append("  clean-and-fast seeds (greedy: 0 crushes, success>=0.9, "
               "return>=0.75): %d/%d %s"
               % (len(cf), len(by_seed), cf))
    if arm == "wrapped":
        fracs = [r["train"].get("frac_ticks_known") for r in by_seed
                 if r["train"].get("frac_ticks_known") is not None]
        conv = sorted(set(r["train"].get("first_converged_episode")
                          for r in by_seed))
        contra = sorted(set(r["train"].get("model_contradictions")
                            for r in by_seed))
        out.append("  schedule model (train): frac ticks with features "
                   "known median %s (min %s) | first converged episode %s | "
                   "contradictions %s"
                   % (_fmt(statistics.median(fracs)), _fmt(min(fracs)),
                      conv, contra))
    return "\n".join(out)


def build_summary(records):
    present = set(r["arm"] for r in records if r["kind"] == "arm")
    arms = [a for a in ("plain", "memory", "wrapped", "clock")
            if a in present] + \
        sorted(present - {"plain", "memory", "wrapped", "clock"})
    yards = [r for r in records if r["kind"] == "yardstick"]
    yards.sort(key=lambda r: r["name"])
    hp = records[0]["hyperparams"] if records else HYPERPARAMS
    parts = []
    parts.append("A5/B2: baseline arms, wrapped arm, and clock control")
    parts.append("python %s | numpy %s"
                 % (sys.version.split()[0], np.__version__))
    parts.append("hyperparameters: hidden=%d, lr=%s, episodes=%d, gamma=%s, "
                 "rewards step=%s goal=%s crush=%s, episode cap=%d, "
                 "period=%d open_len=%d"
                 % (hp["hidden"], hp["lr"], hp["episodes"], hp["gamma"],
                    hp["r_step"], hp["r_goal"], hp["r_crush"],
                    hp["episode_cap"], hp["period"], hp["open_len"]))
    parts.append("protocol: each (arm, seed) trains once, then two "
                 "evaluations over all 20 start phases: pure greedy "
                 "(1 episode/phase) and eps = 0.05 (10 episodes/phase, "
                 "agent's own RNG stream, fixed pipeline order). "
                 "Primary metric: mean episode return (undiscounted sum "
                 "of rewards, failure included).")
    parts.append("")
    parts.append(COMPARISON_RULE)
    for arm in arms:
        by_seed = sorted([r for r in records if r.get("arm") == arm],
                         key=lambda r: r["seed"])
        parts.append("")
        parts.append("##### %s arm: per-seed table #####" % arm)
        parts.append(_eval_table("per-seed", by_seed, "greedy"))
        parts.append(_eval_table("per-seed", by_seed, "eps005"))
        parts.append(_arm_aggregates(arm, records))
    if yards:
        parts.append("")
        parts.append("##### yardsticks (same protocol) #####")
        parts.append(_eval_table("yardsticks", yards, "greedy"))
        parts.append(_eval_table("yardsticks", yards, "eps005"))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- CLI

def parse_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",") if s.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="A5/B2 arms runner")
    ap.add_argument("--arms", default="plain,memory")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out", default=os.path.join(_HERE, "results_a5.jsonl"))
    ap.add_argument("--summary", default=os.path.join(_HERE, "output_a5.txt"))
    ap.add_argument("--overwrite", action="store_true",
                    help="truncate the results file instead of resuming")
    ap.add_argument("--skip-yardsticks", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)
    records = load_records(args.out)
    done = set()
    for r in records:
        if r["kind"] == "arm":
            done.add(("arm", r["arm"], r["seed"]))
        else:
            done.add(("yardstick", r["name"], None))
        hp = r.get("hyperparams", {})
        if hp.get("episodes") != args.episodes:
            raise SystemExit(
                "existing %s has episodes=%r, requested %r: use --overwrite "
                "to start fresh" % (args.out, hp.get("episodes"),
                                    args.episodes))

    def save_summary():
        with open(args.summary, "w") as f:
            f.write(build_summary(load_records(args.out)))

    if args.summarize_only:
        save_summary()
        print("summary written to %s from %s" % (args.summary, args.out))
        return

    if not args.skip_yardsticks:
        for idx, (name, policy) in enumerate(YARDSTICKS):
            if ("yardstick", name, None) in done:
                print("yardstick %s: already in %s, skipping"
                      % (name, args.out), flush=True)
                continue
            print("yardstick %s: evaluating..." % name, flush=True)
            rec = run_yardstick(name, policy, idx, args.episodes)
            append_record(args.out, rec)
            print("yardstick %s: greedy return %.3f, crush/ep %.3f, "
                  "success %.2f" % (name, rec["greedy"]["mean_return"],
                                    rec["greedy"]["crush_rate"],
                                    rec["greedy"]["success"]), flush=True)
            save_summary()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = parse_seeds(args.seeds)
    for arm in arms:
        for seed in seeds:
            if ("arm", arm, seed) in done:
                print("%s seed %d: already in %s, skipping"
                      % (arm, seed, args.out), flush=True)
                continue
            print("%s seed %d: training (%d episodes)..."
                  % (arm, seed, args.episodes), flush=True)
            rec = run_arm_seed(arm, seed, args.episodes)
            append_record(args.out, rec)
            print("%s seed %d: greedy return %.3f, crush/ep %.3f, "
                  "success %.2f | eps005 return %.3f, crush/ep %.3f, "
                  "success %.2f"
                  % (arm, seed, rec["greedy"]["mean_return"],
                     rec["greedy"]["crush_rate"], rec["greedy"]["success"],
                     rec["eps005"]["mean_return"],
                     rec["eps005"]["crush_rate"],
                     rec["eps005"]["success"]), flush=True)
            save_summary()
    print("done: %s, summary %s" % (args.out, args.summary))


if __name__ == "__main__":
    main()

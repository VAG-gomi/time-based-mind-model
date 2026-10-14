#!/usr/bin/env python3
"""A4 verification for agent_nn.py. Prints PASS/FAIL per check."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from agent_nn import (train_tabular, train_neural, make_greedy_policy,
                      evaluate_policy, summarize, naive_policy,
                      time_aware_policy, ACTION_NAMES, PERIOD)

EPISODES = 2000
HIDDEN = 32
LR = 1e-3  # Adam learning rate (SGD tried and failed: see agent_nn.py)
REPEATS = 3


def action_map(agent, kind, door):
    """10x10 map of greedy actions (for diagnostics)."""
    from agent_nn import encode_obs, obs_index, World
    chars = {"up": "^", "down": "v", "left": "<", "right": ">", "stay": "."}
    rows = []
    for y in range(10):
        row = ""
        for x in range(10):
            w = World(period=PERIOD, open_len=10)
            w.reset(start_tick=0 if door else 10)
            w.agent = (x, y)
            if kind == "tabular":
                a = agent.act(obs_index(w), 0.0)
            else:
                a = agent.act(encode_obs(w), 0.0)
            row += chars[ACTION_NAMES[a]]
        rows.append(row)
    return rows


def main():
    all_ok = True

    # ---- check 1: tabular Q-learner proves the world learnable ----
    print("##### check 1: tabular Q-learner (learnability proof) #####")
    tab_succ = []
    for seed in range(3):
        agent, stats = train_tabular(seed, episodes=EPISODES)
        rows = evaluate_policy(make_greedy_policy(agent, "tabular"),
                               repeats=REPEATS)
        cr, fc, succ, mt = summarize(rows)
        tab_succ.append(succ)
        print("seed %d: success %.2f | crush/ep %.3f | mean ticks %s"
              % (seed, succ, cr, "%.1f" % mt if mt is not None else "n/a"))
    ok = all(s >= 0.9 for s in tab_succ)
    print("TABULAR LEARNABILITY: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 2: neural net learns to reach the goal (5 seeds) ----
    print("##### check 2: neural Q-learner, %d hidden, 5 seeds #####"
          % HIDDEN)
    agents = []
    nn_succ = []
    for seed in range(5):
        agent, stats = train_neural(seed, hidden=HIDDEN,
                                    episodes=EPISODES, lr=LR)
        agents.append(agent)
        # training diagnostics: success per 500-episode window
        wins = []
        for w0 in range(0, EPISODES, 500):
            win = stats[w0:w0 + 500]
            wins.append(sum(s[1] for s in win) / len(win))
        print("seed %d: train success/500eps %s"
              % (seed, ["%.2f" % v for v in wins]))
        rows = evaluate_policy(make_greedy_policy(agent, "neural"),
                               repeats=REPEATS)
        cr, fc, succ, mt = summarize(rows)
        nn_succ.append(succ)
        print("seed %d eval: success %.2f | crush/ep %.3f | mean ticks %s"
              % (seed, succ, cr, "%.1f" % mt if mt is not None else "n/a"))
    ok = all(s >= 0.9 for s in nn_succ)
    print("NEURAL LEARNING: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 3: greedy evaluation, 20 phases x repeats ----
    print("##### check 3: greedy eval over all 20 start phases #####")
    for i, agent in enumerate(agents):
        rows = evaluate_policy(make_greedy_policy(agent, "neural"),
                               repeats=REPEATS)
        # repeats must be identical (deterministic greedy policy)
        per_phase = {}
        for (s, c, r, t) in rows:
            per_phase.setdefault(s, []).append((c, r, t))
        identical = all(len(set(v)) == 1 for v in per_phase.values())
        cr, fc, succ, mt = summarize(rows)
        print("seed %d: crush/ep %.3f | episodes-with-crush %.2f | "
              "success %.2f | mean ticks %s | repeats identical: %s"
              % (i, cr, fc, succ,
                 "%.1f" % mt if mt is not None else "n/a", identical))
    print("(no PASS/FAIL: numbers feed the Sequence B comparison)")

    # ---- check 4: hand-coded yardsticks, re-derived ----
    print("##### check 4: yardstick policies (re-derived) #####")
    for name, fn in (("naive", naive_policy),
                     ("time-aware", time_aware_policy)):
        rows = evaluate_policy(lambda w, f=fn: ACTION_NAMES.index(f(w)),
                               repeats=REPEATS)
        cr, fc, succ, mt = summarize(rows)
        bad = sorted(set(s for (s, c, r, t) in rows if c > 0))
        print("%s: success %.2f | crush/ep %.3f | mean ticks %s | "
              "crushed phases: %s"
              % (name, succ, cr,
                 "%.1f" % mt if mt is not None else "n/a", bad))

    # ---- diagnostics: what does the learned policy do? ----
    print("##### diagnostics: greedy action map (seed 0, door open) #####")
    for row in action_map(agents[0], "neural", door=True):
        print(row)
    print("##### diagnostics: greedy action map (seed 0, door closed) #####")
    for row in action_map(agents[0], "neural", door=False):
        print(row)

    print("=== OVERALL: %s ===" % ("PASS" if all_ok else "FAIL"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

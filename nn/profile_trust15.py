#!/usr/bin/env python3
"""profile_trust15.py -- WO-TBM-L15-diag Part A profile.

Same method as the Step-0 profile (nn/profile_trust.py): cProfile,
seed 200, 200 episodes, all arms; same category definitions so the
fractions are comparable to the Step-0 table. Adds:
  - the trust15 arm, with fine buffer categories (sampling incl.
    prefix search / priority updates / insertion path repair /
    IS-weight computation);
  - total env steps per arm, for per-step rates.
Diagnostic only; no timing claim is made from profiled runs.
"""

import cProfile
import os
import pstats
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent_nn import train_neural, PlainEncoder
from layer1_trust_train import train_trust
from layer15_trust_train import train_trust15

SEED = 200
EPISODES = 200

BUFFER_FUNCS = {("agent_nn.py", "sample"),
                ("trust_replay.py", "sample"),
                ("trust_replay.py", "update_trust"),
                ("trust_replay_fast.py", "sample"),
                ("trust_replay_fast.py", "update_trust"),
                ("trust_replay_fast.py", "add"),
                ("layer15_trust_train.py", "is_weights")}
NETWORK_FUNCS = {("nnet.py", "forward"), ("nnet.py", "backward"),
                 ("nnet.py", "step"), ("nnet.py", "masked_mse_grad")}
ENV_FUNCS = {("agent_nn.py", "env_step"), ("agent_nn.py", "encode"),
             ("agent_nn.py", "reset"),
             ("world.py", "step"), ("world.py", "reset"),
             ("world.py", "observe")}
FINE = {("trust_replay_fast.py", "sample"): "seg-sampling",
        ("trust_replay_fast.py", "update_trust"): "seg-updates",
        ("trust_replay_fast.py", "add"): "seg-add(path repair)",
        ("layer15_trust_train.py", "is_weights"): "IS weights",
        ("trust_replay_fast.py", "_find"): "  of which _find",
        ("trust_replay_fast.py", "_set_leaves"): "  memo _set_leaves"}


def run_profile(train):
    holder = {}

    def run():
        _agent, stats = train(SEED, episodes=EPISODES,
                              encoder=PlainEncoder())
        holder["steps"] = sum(s[0] for s in stats)

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    return pstats.Stats(pr), holder["steps"]


def categorize(stats):
    cats = {"buffer": 0.0, "network": 0.0, "env": 0.0}
    fine = {}
    total = 0.0
    for (fn, _ln, name), (_cc, _nc, tt, ct, _callers) in stats.items():
        total += tt
        key = (os.path.basename(fn), name)
        if key in BUFFER_FUNCS:
            cats["buffer"] += ct
        elif key in NETWORK_FUNCS:
            cats["network"] += ct
        elif key in ENV_FUNCS:
            cats["env"] += ct
        if key in FINE:
            fine[FINE[key]] = ct
    cats["glue"] = total - cats["buffer"] - cats["network"] - cats["env"]
    return total, cats, fine


def main():
    arms = (("uniform", train_neural), ("trust-naive", train_trust),
            ("trust15", train_trust15))
    for arm, train in arms:
        st, steps = run_profile(train)
        total, cats, fine = categorize(st.stats)
        print("=== %s (seed %d, %d episodes, %d env steps) ==="
              % (arm, SEED, EPISODES, steps))
        print("  total %8.2fs | per-step %.4fs" % (total, total / steps))
        for k in ("buffer", "network", "env", "glue"):
            print("  %-10s %8.2fs  %5.1f%%  per-step %.4fs"
                  % (k, cats[k], 100.0 * cats[k] / total,
                     cats[k] / steps))
        for k, v in fine.items():
            print("  %-22s %8.2fs  per-step %.4fs" % (k, v, v / steps))


if __name__ == "__main__":
    main()

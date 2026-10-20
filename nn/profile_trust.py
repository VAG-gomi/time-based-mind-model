#!/usr/bin/env python3
"""profile_trust.py -- WO-TBM-L15 Step 0 profiling spike.

Profiles the CURRENT Layer-1 code (uniform arm: agent_nn.train_neural;
trust arm: layer1_trust_train.train_trust) with cProfile, same seed
(200), same episode count (200). The replay buffer reaches its
10000 capacity at episode ~52 (measured from
results_trust_rerun.jsonl ticks), so ~3/4 of steps run at full
buffer -- representative of the 1000-episode re-run regime.

Categories (cumulative time of disjoint subtrees, so no double
counting):
  (a) buffer sampling + trust updates:
      ReplayBuffer.sample / TrustReplayBuffer.sample +
      TrustReplayBuffer.update_trust
  (b) network forward/backward/optimizer:
      MLP.forward, MLP.backward, Adam.step, masked_mse_grad
  (c) environment + encoding:
      env_step, World.step/reset/observe, PlainEncoder.encode/reset
  (d) remaining Python glue: total - (a) - (b) - (c)

Amdahl ceiling for the Step-1 fix pair: post-fix trust time cannot
go below the trust arm's measured non-buffer time (a-segment-tree
removes buffer cost, adds O(log N) work back -- the ceiling assumes
buffer cost -> 0, i.e. it is a best case). If that floor does not
clear parity with the uniform arm's total, the L15 gate FAILS.
"""

import cProfile
import io
import os
import pstats
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent_nn import train_neural, PlainEncoder
from layer1_trust_train import train_trust

SEED = 200
EPISODES = 200

BUFFER_FUNCS = {("agent_nn.py", "sample"),
                ("trust_replay.py", "sample"),
                ("trust_replay.py", "update_trust")}
NETWORK_FUNCS = {("nnet.py", "forward"), ("nnet.py", "backward"),
                 ("nnet.py", "step"), ("nnet.py", "masked_mse_grad")}
ENV_FUNCS = {("agent_nn.py", "env_step"), ("agent_nn.py", "encode"),
             ("agent_nn.py", "reset"),
             ("world.py", "step"), ("world.py", "reset"),
             ("world.py", "observe")}


def categorize(stats):
    cats = {"buffer": 0.0, "network": 0.0, "env": 0.0}
    total = 0.0
    for (fn, _ln, name), (_cc, _nc, tt, ct, _callers) in stats.items():
        total += tt
        base = os.path.basename(fn)
        key = (base, name)
        if key in BUFFER_FUNCS:
            cats["buffer"] += ct
        elif key in NETWORK_FUNCS:
            cats["network"] += ct
        elif key in ENV_FUNCS:
            cats["env"] += ct
    cats["glue"] = total - cats["buffer"] - cats["network"] - cats["env"]
    return total, cats


def profile(arm):
    train = train_neural if arm == "uniform" else train_trust
    holder = {}

    def run():
        agent, _stats = train(SEED, episodes=EPISODES,
                              encoder=PlainEncoder())
        holder["buflen"] = len(agent.buffer)

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    st = pstats.Stats(pr)
    total, cats = categorize(st.stats)
    return total, cats, holder["buflen"]


def main():
    results = {}
    for arm in ("uniform", "trust"):
        total, cats, buflen = profile(arm)
        results[arm] = (total, cats)
        print("=== %s (seed %d, %d episodes, buffer len %d) ==="
              % (arm, SEED, EPISODES, buflen))
        print("  total (profiled)         %8.2fs" % total)
        for k in ("buffer", "network", "env", "glue"):
            print("  %-24s %8.2fs  %5.1f%%"
                  % ("(" + k + ")", cats[k], 100.0 * cats[k] / total))
    tu, cu = results["uniform"]
    tt, ct = results["trust"]
    floor = tt - ct["buffer"]
    print("=== Amdahl ceiling (trust-1.5) ===")
    print("  trust total %.2fs; buffer %.2fs; non-buffer floor %.2fs"
          % (tt, ct["buffer"], floor))
    print("  uniform total %.2fs" % tu)
    print("  best-case trust-1.5 = %.2fs vs uniform %.2fs -> %s"
          % (floor, tu,
             "CLEARS parity" if floor <= tu else "DOES NOT clear parity"))
    print("  (profiled times; cProfile inflates Python-level costs"
          " roughly uniformly across arms)")


if __name__ == "__main__":
    main()

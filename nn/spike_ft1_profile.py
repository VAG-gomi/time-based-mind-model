#!/usr/bin/env python3
"""FT1 spike cProfile (MEASUREMENT ONLY): one wrapped/0 spike
fine-tune (200 episodes on (5,8)) under cProfile.

Loads the Step 0 sanity snapshot, audits the load (double-load
+ assert_load_equal), then profiles finetune_neural only.
Profile time is profiler accounting, NOT wall time: the
report uses it only for shares, never as a wall-time claim.
Weights are discarded; nothing is saved. No existing file
is edited.
"""

import cProfile
import os
import pstats
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_ft1 import (FINE_TUNE_EPISODES, FINE_TUNE_EPS,
                       assert_load_equal, finetune_encoder,
                       finetune_neural, load_checkpoint)
from world_pdb2 import TRAIN_20, f3_schedules

SNAP = os.path.join(_HERE, "checkpoints_ft1", "wrapped_0.pkl")
OUT = os.path.join(_HERE, "spike_ft1_profile.txt")


def main():
    agent, encoder, _b = load_checkpoint(SNAP, "wrapped", 20, 10)
    agent2, _e2, _b2 = load_checkpoint(SNAP, "wrapped", 20, 10)
    assert_load_equal(agent, agent2)
    enc_ft = finetune_encoder("wrapped", encoder, 20, 10)
    pr = cProfile.Profile()
    pr.enable()
    finetune_neural(agent, enc_ft, f3_schedules(TRAIN_20), 0,
                    episodes=FINE_TUNE_EPISODES, eps=FINE_TUNE_EPS)
    pr.disable()
    with open(OUT, "w") as f:
        f.write("FT1 spike cProfile: wrapped/0, 200 fine-tune "
                "episodes on (5,8), SPIKE only\n")
        f.write("NOTE: cProfile time is profiler accounting, "
                "NOT wall time; shares only.\n\n")
        st = pstats.Stats(pr, stream=f)
        st.sort_stats("cumulative")
        st.print_stats(25)
    print("profile written to %s" % OUT)


if __name__ == "__main__":
    main()

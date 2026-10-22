#!/usr/bin/env python3
"""FT1 Step 0 sanity (WO-TBM-FT1, Step 0b): checkpoint realisation
verification + fine-tune loop validity, ONE kept agent per arm.

Sanity agents (all kept agents): wrapped/0, plain/0, memory/0
(20/10); oracle_ind/100, oracle_new/100, phase_only_new/102
(30/14). Scope rationale, stated before running: one seed per
arm validates the checkpoint realisation + fine-tune loop for
every encoder type (including the stateful wrapped encoder and
the period-parameterised oracle/phase encoders) without doing
Step 1.

Per sanity agent, in order:
  (1) Realise the P-Db2 checkpoint by deterministic re-training
      with frozen code (train_pdb2.TRAINERS, 2000 episodes, F1
      training world). This is a CHECK realisation, not a Step 1
      record.
  (2) VERIFY the realised training summary against the P-Db2
      training record(s) for that (arm, seed) (for wrapped, all
      5 eval-world copies must agree, and the realised summary
      must match them): reached, crushes, mean_ticks,
      mean_ticks_all, mean_loss, door_entry_counts,
      episodes_with_entry. Integers exact; floats to 1e-9
      (deterministic same code must agree to floating-point
      noise at most). ANY mismatch is a DEVIATION: the agent
      is NOT fine-tuned, the mismatch is recorded, and the
      script exits nonzero at the end (other sanity agents
      are still checked). A mismatch is never hidden by
      substituting a weight or adjusting anything.
  (3) Save the checkpoint snapshot (train_ft1.save_checkpoint)
      and audit it by a save/load round-trip
      (train_ft1.assert_load_equal, bitwise). The snapshot is
      what Step 1 reuses for this agent (design note); the
      snapshot file is NEW and additive, never a P-Db2 file.
  (4) Fine-tune 200 episodes on (5,8) from the checkpoint
      (train_ft1.finetune_neural, flat eps 0.05, fresh replay
      buffer, carried weights/target/Adam/RNG, carried
      ScheduleModel for wrapped). Sanity criteria (loop
      validity ONLY, not experiment outcomes): 200 episodes
      complete without crash; no NaN in mean_loss values;
      per-door counts recorded. Reached count is NOT a
      criterion here: the fine-tuning world is novel, and
      zero reached after 200 episodes is a legitimate
      possible outcome, not a loop failure. Requiring
      reached > 0 would prejudge acquisition, which is the
      experiment's question.

Sanity outputs are CHECK outputs only (nn/sanity_ft1.log,
nn/sanity_ft1.jsonl, nn/checkpoints_ft1/*.pkl). They are NOT
Step 1 records: Step 1 fine-tunes fresh from the checkpoint
snapshots (for these 6 agents, from the snapshots saved
here; for the other 16, from checkpoints realised at
Step 1 start and verified the same way), so Step 1 records
are clean and auditable on their own. Sanity fine-tuned
weights are discarded (only the checkpoint snapshots are
kept).

Resume-safe: on startup, agents with a sanity record
already in sanity_ft1.jsonl are skipped (their snapshot
and fine-tune record already exist); a restart loses at
most the in-progress agent.
"""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_pdb2 import TRAINERS, TRAIN_BY_PERIOD
from train_ft1 import (FINE_TUNE_EPISODES, FINE_TUNE_EPS,
                       assert_load_equal, finetune_encoder,
                       finetune_neural, load_checkpoint,
                       save_checkpoint)
from world_pdb2 import TRAIN_20, TRAIN_30, f3_schedules

OUT = os.path.join(_HERE, "sanity_ft1.jsonl")
SNAP_DIR = os.path.join(_HERE, "checkpoints_ft1")
PDB2_TRAIN = os.path.join(_HERE, "training_pdb2.jsonl")

SANITY_AGENTS = [
    ("wrapped", 0, 20, 10),
    ("plain", 0, 20, 10),
    ("memory", 0, 20, 10),
    ("oracle_ind", 100, 30, 14),
    ("oracle_new", 100, 30, 14),
    ("phase_only_new", 102, 30, 14),
]


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


def load_pdb2_records():
    recs = {}
    with open(PDB2_TRAIN) as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault((r["arm"], r["seed"]), []).append(r)
    return recs


def verify_checkpoint(arm, seed, summary, pdb2_recs):
    """Compare realised summary against P-Db2 record(s).

    Returns a list of mismatch strings (empty = exact match
    within tolerance). For wrapped, the 5 P-Db2 copies must
    also agree with each other (checked first).
    """
    mismatches = []
    fields_int = ("episodes", "reached", "crushes")
    fields_float = ("mean_ticks", "mean_ticks_all", "mean_loss")
    fields_dict = ("door_entry_counts", "episodes_with_entry",
                   "open_fractions")
    # All P-Db2 copies for this agent must agree with each other.
    ref = pdb2_recs[0]
    for other in pdb2_recs[1:]:
        for k in fields_int + fields_float + fields_dict:
            if ref[k] != other[k]:
                mismatches.append(
                    "P-Db2 copies disagree on %s: %r vs %r"
                    % (k, ref[k], other[k]))
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


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["arm"], r["seed"]))
    pdb2 = load_pdb2_records()
    out = open(OUT, "a")
    n_deviation = 0
    n_ft_fail = 0
    for arm, seed, period, open_len in SANITY_AGENTS:
        if (arm, seed) in done:
            print("%s/%d: sanity record present, skipping"
                  % (arm, seed), flush=True)
            continue
        train = TRAIN_20 if period == 20 else TRAIN_30
        print("%s/%d: realising P-Db2 checkpoint (2000 episodes)..."
              % (arm, seed), flush=True)
        agent, stats, encoder, dec, ewe, frac = TRAINERS[arm](
            seed, train, hidden=32, episodes=2000, lr=1e-3)
        summary = training_summary(stats, dec, ewe, frac)
        mismatches = verify_checkpoint(arm, seed, summary,
                                       pdb2[(arm, seed)])
        if mismatches:
            n_deviation += 1
            print("%s/%d: CHECKPOINT DEVIATION, NOT fine-tuning:"
                  % (arm, seed), flush=True)
            for m in mismatches:
                print("  %s" % m, flush=True)
            rec = {"kind": "sanity_ft1", "arm": arm, "seed": seed,
                   "status": "checkpoint_mismatch",
                   "mismatches": mismatches,
                   "realised_summary": summary}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            continue
        print("%s/%d: checkpoint verified exact vs P-Db2 "
              "(reached=%d, crushes=%d)"
              % (arm, seed, summary["reached"], summary["crushes"]),
              flush=True)
        snap = os.path.join(SNAP_DIR, "%s_%d.pkl" % (arm, seed))
        save_checkpoint(agent, encoder, arm, seed, snap)
        agent2, encoder2, _bundle = load_checkpoint(
            snap, arm, period, open_len)
        assert_load_equal(agent, agent2)
        if arm == "wrapped":
            assert encoder2.model.period == encoder.model.period \
                and encoder2.model.open_len == encoder.model.open_len, \
                "wrapped ScheduleModel not preserved by snapshot"
        print("%s/%d: snapshot saved + load-audit PASS (%s)"
              % (arm, seed, snap), flush=True)
        # Fine-tune from the checkpoint agent (sanity agent is
        # discarded afterwards; only the snapshot is kept).
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
        # NOTE (fixed 2026-10-22): the first run of this script
        # tested "(5,8)" here, but gap_key() returns "5,8",
        # so every fine-tune was wrongly marked FAIL even
        # though all criteria were met. The key below is the
        # correct gap_key output.
        ok = (len(stats_ft) == FINE_TUNE_EPISODES and not nan_loss
              and "5,8" in {gap_key(g) for g in dec_ft})
        print("%s/%d: fine-tune sanity %s (200 episodes, "
              "reached=%d, crushes=%d, (5,8) entries=%d, "
              "model %s -> %s)"
              % (arm, seed, "PASS" if ok else "FAIL",
                 len(reached_ft), sum(s[2] for s in stats_ft),
                 dec_ft.get((5, 8), 0), model_start, model_end),
              flush=True)
        rec = {
            "kind": "sanity_ft1", "arm": arm, "seed": seed,
            "period": period, "open_len": open_len,
            "status": "pass" if ok else "fine_tune_fail",
            "checkpoint": summary,
            "snapshot": snap,
            "fine_tune": {
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
                "eps": FINE_TUNE_EPS,
                "start_rng": "seed+2000",
                "schedule_model_start": model_start,
                "schedule_model_end": model_end,
            },
        }
        if not ok:
            n_ft_fail += 1
        out.write(json.dumps(rec) + "\n")
        out.flush()
    out.close()
    if n_deviation:
        print("sanity_ft1: %d CHECKPOINT DEVIATION(S), see above"
              % n_deviation, flush=True)
        sys.exit(1)
    if n_ft_fail:
        # NOTE (fixed 2026-10-22): the first run exited 0
        # despite fine_tune_fail records, because only
        # checkpoint deviations set the exit code. A
        # fine-tune failure must also fail the run.
        print("sanity_ft1: %d FINE-TUNE FAILURE(S), see above"
              % n_ft_fail, flush=True)
        sys.exit(1)
    print("sanity_ft1: all 6 agents PASS (or skipped as present)",
          flush=True)


if __name__ == "__main__":
    main()

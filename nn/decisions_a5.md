# A5 — decisions and deviations

Work order: A5, baseline arms and the results runner. Same standing rules:
deviate where something better was found and say why, list anything
skipped, world.py untouched. Claude's A4 re-verification and its rulings
are adopted (see "Adopted rulings").

## What was built

- `nn/agent_mem.py` (new): one-step-memory arm. Same learner, training
  loop, replay sampler, and hyperparameters as the plain arm; the only
  change is the input, 101 -> 102 (previous tick's door bit appended).
- `nn/run_arms.py` (new): trains plain + memory on seeds 0-9, evaluates
  each policy twice (pure greedy, eps = 0.05), evaluates the three
  yardsticks on the same protocol, appends one JSON record per
  arm/seed to `results_a5.jsonl` as each seed finishes, rewrites
  `output_a5.txt` after every seed. Supports `--resume` (default),
  `--overwrite`, `--summarize-only`, `--seeds`, `--episodes`, `--arms`
  for independent subset re-runs.
- `nn/agent_nn.py` (changed, minimally): input construction is now an
  encoder object (`reset`/`encode`/`n_inputs`); `NeuralQ` takes
  `n_inputs`; `train_neural` takes `encoder` (default `PlainEncoder`,
  i.e. exactly the A4 path). Nothing else in the learner, training
  loop, sampler, or hyperparameters changed.
- `nn/results_a5.jsonl` (new): 23 records (20 arm/seed + 3 yardsticks).
- `nn/output_a5.txt` (new): per-seed tables, aggregates, yardsticks,
  and the pre-registered Sequence B comparison rule.

## Adopted rulings (Claude's A4 analysis + A5 order)

- Report all seeds, failures visible, no per-seed retuning. No
  hyperparameter tuning in this step; any future tuning uses separate
  seeds 100+, same budget for every arm.
- Primary metric: mean episode return (undiscounted sum of rewards,
  failure to reach the goal included). Crush rate, success rate, and
  mean ticks to goal also reported.
- Two evaluations per policy: pure greedy, and eps = 0.05 with a fixed
  pipeline (agent's own RNG stream for learned arms, dedicated seeded
  RNGs for yardsticks).
- The plain arm is not memoryless: seed 2's delay-line behaviour (body
  as memory) is real. Hence the delay-line yardstick, and the
  wrapper's bar is "zero crushes reliably AND near the 21.3-tick
  ideal", not "the baseline can't avoid crushes".
- Comparison rule for Sequence B is written into `output_a5.txt`
  before the wrapper exists (pre-registration). With A5's numbers the
  bar is concrete: best baseline median greedy return is the memory
  arm's 0.736, so the wrapper WINS iff median greedy return >= 0.756
  AND 10/10 seeds zero crushes AND 10/10 seeds success >= 0.9.

## Decisions

- **Memory encoder, prev-door at reset = current door.** Signals "no
  change observed yet" on the first step. The alternative (prev = 0)
  was rejected: on a reset tick with the door open it would fabricate
  a closed->open edge the agent never observed.
- **Delay-line yardstick:** holding cell (0,5), staging cell (4,5),
  cross only on a fresh opening (door open now AND closed at the
  previous observation). Zero crushes hold by construction: entering
  on the first open tick leaves the full 10-tick window and the agent
  steps off the door cell next tick. Deliberately conservative:
  arriving at staging mid-open-period waits out the rest of that
  opening plus the next closed period. Measured: 0.725 return, 0
  crushes, 27.5 ticks (greedy) — slower than the naive 21.3 ticks,
  which is the cost of the conservative rule. Under eps = 0.05 noise
  it degrades to 0.399 / 0.010 crushes / 0.91 success, because random
  actions disrupt the holding pattern.
- **Greedy eval: 1 episode per phase** (A4 used 3 repeats). Greedy
  policies are deterministic, so repeats are identical; A4 verified
  this. The eps eval uses 10 episodes per phase (200 total).
- **Spread = min/max + population std** over the 10 seeds; median is
  the primary aggregate. Fractions (zero-crush seeds, success>=0.9
  seeds) are computed on the greedy eval, and also shown for the
  eps eval.
- **Return accounting** uses `env_step` from agent_nn.py, so reward
  accounting is identical to A4. A crush costs -1 and the episode
  continues; a failed episode (cap 200, no goal) contributes
  -0.01 * ticks - crushes.
- **Resume guard:** re-running against an existing results file with
  a different `--episodes` aborts unless `--overwrite` is given, so
  records with different budgets can never mix silently.

## Deliberate non-changes (deviations from permitted options)

- **Faster replay sampler: NOT taken.** Claude permitted one, with
  the plain arm re-run. I kept the A4-verified training loop
  byte-for-byte instead: changing the sampler would invalidate A4's
  independently verified numbers and `test_agent_nn.py`'s
  expectations, replacing a known-good loop with a statistically
  equivalent but numerically different one. The cost (~52 min for
  20 seeds) was handled the way the order anticipated: background
  execution plus per-seed incremental saves. Both A5 arms use the
  identical A4-verified loop, so the comparison is unconfounded.
- **No separate `test_mem.py`.** Not ordered. Coverage instead:
  `agent_mem.py` has a no-training `__main__` structural check of
  the prev-door bookkeeping (PASS); the full run was verified by
  exact re-run (below) and by the A4 cross-check (below).
- **No per-seed or hyperparameter tuning.** Forbidden by the ruling.

## Verification (all before delivery)

- **A4 cross-check:** plain seeds 0-4 greedy crush + success match
  A4 exactly (0.05/1.00, 0.00/0.30, 0.00/1.00, 0.05/1.00,
  0.05/1.00); return-implied mean ticks match A4 (seed 3's true
  value is 22.65; A4 printed 22.6 under `%.1f` rounding). Naive and
  time-aware yardsticks match A4 exactly (0.737/0.050/21.3 and
  0.787/0.000/21.3). This proves the `agent_nn.py` refactor preserved
  the A4 training path exactly.
- **Independent recompute:** medians, min/max, std, and fractions
  recomputed from `results_a5.jsonl` with a separate script match
  `output_a5.txt` exactly.
- **Determinism:** re-ran memory seed 0 (plus all yardsticks) via
  `run_arms.py --seeds 0 --arms memory`; the record is identical to
  the full-run record, full JSON equality including all floats.
  (The first attempt at this check was wiped by a runtime service
  restart; it was redone cleanly. The main run had already finished
  and was verified unaffected.)
- **`--summarize-only`** regenerates `output_a5.txt` from the JSONL
  byte-identically.
- **Smoke tests:** 20-episode run on seed 0 for both arms plus all
  yardsticks passed end-to-end before the full run; the delay-line
  yardstick showed 0 crushes in smoke and in the full run.

## Results (headline)

- Plain greedy: median return 0.712, min -6.235, max 0.757;
  zero-crush seeds 4/10, success>=0.9 seeds 6/10.
- Memory greedy: median return 0.736, min -0.531, max 0.787;
  zero-crush seeds 2/10, success>=0.9 seeds 8/10.
- Notable seeds: plain 1/5/6/8 are greedy-fragile (loops; seed 6 is
  a crush-loop at 5.45 crushes/episode, return -6.235) while eps =
  0.05 rescues them (success 0.57-0.99) — the two-eval protocol
  earns its keep. Memory seed 8 hits the time-aware ideal exactly
  (0.787, 0 crushes, 21.3 ticks): one-step memory *can* reach the
  ideal, in 1/10 seeds. The wrapper must do it in 10/10, at
  median >= 0.756.
- `world.py` untouched. `test_agent_nn.py` untouched (its A4
  expectations still hold: the training path is unchanged).
- Skipped: nothing else. No step of the order was left undone.

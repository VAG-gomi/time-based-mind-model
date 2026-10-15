# Time-Based Mind Model

A plain-Python, fixed-seed agent experiment. The question: does explicit
time structure help a neural network, or is it enough to just remember?

**The philosophy.** The classical neural network (thousands of neurons) is
the substrate — an empty world before the big bang. The time-based mind
model is the *meta-reality* developed on top of it. Sequence A builds the
substrate; Sequence B builds the mind model as a wrapper on it. The final
comparison is an ablation: identical substrate across arms, only the
wrapper differs.

## Structure

```
world.py            # the environment (Sequence A foundation)
output.txt          # world's check output (all checks PASS)
nn/
  bench.py          # A1: environment + speed check
  nnet.py           # A2/A3: hand-written MLP (numpy only, no frameworks)
  test_nnet.py      # A2b/A3: verification suite (gradient checks, XOR, ...)
  output*.txt       # recorded outputs of each stage
```

### The world (`world.py`)

10×10 grid, wall on x=5 with a door gap at (5,5). The door opens on a
periodic schedule (`tick % period < open_len`). A **crush rule** makes
timing matter: an agent caught at (5,5) when the door closes is crushed
back to the start. Episodes begin at a random start phase, so no fixed
trajectory can leak the schedule.

### The substrate (`nn/nnet.py`)

Fully connected MLP written by hand on numpy: tanh hidden layers, linear
output, float64, forward/backward/SGD, Xavier / zero / scaled init, and a
masked MSE loss for Q-learning. Verified by central finite differences
(80/80 cases over 20 seeds), a negative control, determinism checks, an
"empty world" zero-init check, and XOR (10/10 seeds solve; zero init pins
at loss exactly 0.125; init scales ≤1e-3 never escape the saddle).

## Status

- [x] World: Instructions 1–5 complete, all checks PASS
- [x] A1: environment + speed check
- [x] A2/A2b: network + gradient verification
- [x] A3: XOR
- [x] A4: neural Q-learner in the world (`agent_nn.py`) — honest
      partial failure: 4/5 seeds reach 1.00 greedy success; seed 1
      reaches 0.30 (greedy stay-loop on a near-tied Q-value). See
      `nn/output_a4.txt`.
- [x] A5: baseline results (`run_arms.py`, seeds 0–9): plain 1/10
      clean-and-fast, one-step-memory 1/10. See `nn/output_a5.txt`.
- [x] R: comparison rule replaced pre-wrapper ("clean and fast":
      0 crushes, success ≥ 0.9, return ≥ 0.75; wrapper wins iff
      ≥ 7/10 seeds; p ≈ 0.01 Fisher exact vs 1/10).
- [x] B1: schedule-estimating feature wrapper (`nn/wrapper.py`) —
      all 6 checks PASS (`nn/test_wrapper.py`, `nn/output_b1.txt`).
- [x] B2: wrapped arm (`nn/agent_wrapped.py`) + raw-clock control,
      seeds 0–9. Result: wrapped **5/10** clean-and-fast — does NOT
      win under its own 7/10 criterion; clock 0/10. Honest negative
      with much improved safety/consistency (8/10 zero-crush,
      10/10 success). See `nn/decisions_b2.md`.

Note on test suites: `nn/test_agent_nn.py` still reports FAIL on its
older five-seed all-must-pass gate (seed 1, 0.30 success) — that is
the A4 honest partial failure, recorded as-is. The A5/B2 full
comparison (`run_arms.py`) is a different, later protocol that
intentionally reports all seeds heterogeneously instead of gating
on all-pass.

## Running

```bash
python3 world.py > output.txt        # world checks
cd nn
python3 bench.py > output.txt       # A1
python3 test_nnet.py > output_a2b_a3.txt   # A2b + A3
```

numpy is the only dependency. Everything is seeded and reproducible.

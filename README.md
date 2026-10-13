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
- [ ] A4: neural Q-learner in the world (`agent_nn.py`)
- [ ] A5: baseline results (`run_arms.py`)
- [ ] Sequence B: schedule-estimating feature wrapper (`wrapper.py`),
      wrapped arm, full comparison

## Running

```bash
python3 world.py > output.txt        # world checks
cd nn
python3 bench.py > output.txt       # A1
python3 test_nnet.py > output_a2b_a3.txt   # A2b + A3
```

numpy is the only dependency. Everything is seeded and reproducible.

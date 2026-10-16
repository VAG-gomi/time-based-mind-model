# Preregistration: matched controls (EBI priority 2)

## Research question

Is the wrapped arm's advantage caused specifically by the inferred
schedule representation, or merely by giving the network additional
temporal information and inputs?

## Arms (all: same learner, hidden=32, lr=1e-3, 2000 episodes,
same rewards/gamma, period 20, open_len 10)

- `oracle`: 104 inputs. TRUE phase (tick%20)/20, TRUE remaining/10,
  known=1.0 always. Perfect schedule knowledge, no inference.
- `phase_only`: 102 inputs. True phase only.
- `remaining_only`: 102 inputs. True remaining time only.
- `param_matched`: 104 inputs. Plain 101 + 3 constant-zero dummies.
  Same input dim and parameter count as wrapped.
- `memory10`: 111 inputs. Last 10 door bits. Stronger memory baseline.

Reference arms (already run, seeds 10-49): plain, memory, wrapped.

## Seeds

10-49 (same 40 seeds as the replication), enabling paired comparison
against the existing wrapped/plain/memory records.

## Protocol

Identical to the replication: train once per (arm, seed), evaluate
greedy (1 ep/phase) and eps=0.05 (10 eps/phase), all 20 start phases.
Output: nn/results_controls.jsonl (separate file).

## Primary analysis (paired, same seeds)

1. Clean-and-fast proportions with Clopper-Pearson 95% CIs, per arm.
2. Paired wrapped-minus-control differences with McNemar exact
   p-values and paired 95% CIs, for each control.
3. Median greedy-return differences with 10,000-seed bootstrap 95%
   CIs (seed=0), wrapped vs each control.
4. Zero-crush and success>=0.9 reported separately (descriptive).

## Interpretation rules (pre-registered)

- If `oracle` >> `wrapped`: the inferred representation is lossy;
  perfect knowledge helps more.
- If `oracle` ~= `wrapped`: the ScheduleModel already extracts
  everything the task needs from observations.
- If `phase_only` or `remaining_only` ~= `wrapped`: that single
  feature explains the gain; the other is redundant.
- If `param_matched` ~= `wrapped`: the gain was just parameters.
  (Expected: no, given the 96-param difference.)
- If `memory10` ~= `wrapped`: a long history window suffices; no
  explicit schedule representation needed.
- If `wrapped` > all controls: the inferred schedule representation
  itself is the cause.

No binary win/lose gate. Estimates and intervals only.

## Missing-data rule

Same as replication: if a seed is missing in either compared arm,
exclude that incomplete pair from the paired analysis; retain
available records descriptively.

## Code

- nn/agent_controls.py (encoders + trainers; structural checks PASS)
- nn/run_arms.py extended with the five new arms
- Frozen for this experiment at the commit containing this file.

Approved: _________________  Date: ___________

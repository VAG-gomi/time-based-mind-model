# R + B1 — decisions and deviations

## R: comparison rule replaced (before the wrapper exists)

Adopted the "clean and fast" rule: a seed is clean-and-fast iff its
greedy run has zero crushes, success >= 0.9, and mean return >= 0.75.
The wrapper wins iff >= 7/10 seeds are clean and fast; eps = 0.05
stays a robustness report. Reason (in the rule text): the return
ceiling is small (time-aware 0.787, naive 0.737, delay-line 0.725),
so <= ~0.05 separates the best from the naive plateau while
seed-to-seed spread (std 0.49, memory arm) dwarfs it; greedy returns
cluster on plateaus, so a median records which plateau most seeds
landed on; and 10/10 perfect seeds punishes the learner's greedy
"stay"-loop property, not the wrapper.

Verified, not hand-typed: Fisher's exact test for 7/10 vs 1/10
(one-sided) is P(X>=7) = [C(8,7)*C(12,3) + C(8,8)*C(12,2)] / C(20,10)
= (1760 + 66)/184756 ~= 0.0099, i.e. p ~= 0.01 as claimed. The
baseline counts are recomputed by run_arms.py into the summary
(plain 1/10 [seed 2], memory 1/10 [seed 8]); the 0.75 bar sits
strictly between the naive plateau (0.737) and the ideal (0.787),
i.e. mean ticks <= 25 with 0 crushes and full success. Adopting now,
with no wrapper in existence, is not moving the goalposts.

## B1: wrapper.py (ScheduleModel)

Pure stdlib, no world/numpy import. Design decisions:

- **Transition-only events.** An opening is recorded only on an
  observed closed->open transition between consecutive ticks. The
  first state at an episode start therefore *cannot* be treated as a
  real opening, structurally. `new_episode()` clears the pending
  previous observation (belt and braces); a backward/duplicate tick
  auto-detects a boundary the same way.
- **Check-first ordering.** Every observation is consistency-checked
  *before* it can form a transition. A set_door change is therefore
  caught at the first diverging tick (measured latency 4 ticks on
  20/10 -> 10/4 at tick 60), and the suspect transition spanning the
  change is never recorded -- no garbage re-estimates.
- **Clear on contradiction, honest None meanwhile.** Stale events are
  dropped; estimates become None until new transitions arrive. The
  model never reports a confident wrong schedule.
- **Clean/dirty span tracking (found by testing, not in the first
  draft).** The first version trusted every opening gap; the
  cross-episode test exposed the flaw: an opening missed across an
  episode boundary makes a dirty gap (a multiple of the true period),
  and the model false-fired a contradiction on it. Now the model
  tracks whether each opening-to-opening / opening-to-closing span
  was observed without gaps or boundaries, and estimates use clean
  spans only: period = most recent clean gap, open_len = most recent
  clean pair. A dirty gap is never trusted, so a missed opening can
  never become a wrong period.
- **Periodic projection for features.** ticks_since_opening and
  open_ticks_remaining use modular arithmetic from the last observed
  opening, so a model built over earlier episodes predicts a new
  episode's phase from tick 0 (verified: tick 123 -> since 3,
  remaining 7, exact).
- **Persists across episodes** by construction: new_episode() touches
  only the pending observation and the gap flag.

## Verification (test_wrapper.py, output_b1.txt, OVERALL PASS)

1. Exact estimates on 20/10, 10/4, 12/8, 6/2 from 8 random start
   phases each (worst-case ticks to exact: 40/21/24/13; means
   31.5/16.8/18.9/9.8); last_opening verified a real transition;
   zero contradictions on stable schedules.
2. set_door 20/10 -> 10/4 at tick 60: contradiction at 64
   (latency 4, bound 20), exact 10/4 re-convergence 21 ticks after
   the change, then stable (exactly 1 contradiction total).
3. Features vs ground truth over full cycles on 20/10 and 12/8.
4. Cross-episode: after 3 episodes, new episode at tick 123
   predicted from tick 0 (since 3, remaining 7); first observation
   not recorded as an opening.
5. Mid-open start (tick 3): first state not an opening, features
   unknown until real transitions; then exact.

Deviations found while testing (fixed, not hidden): the first
wrapper draft lacked clean/dirty tracking and failed check 4 with a
false contradiction; the check-5 test initially asserted the *first*
opening where the *latest* (40) is correct -- the test was wrong, the
model was right.

## The two flags

- **GitHub:** fresh-clone verified the repo tree identical to local
  at 1a686ac *before* any R/B1 change; the repo is now PUBLIC
  (https://github.com/VAG-gomi/time-based-mind-model) per the
  request, so it is readable. This commit pushes the R + B1 files.
- **Truncated paste:** the repo's agent_mem.py is complete (114
  lines, ends with the `__main__` structural check) -- verified in
  the fresh clone. The truncation was a paste artifact, not the
  repo; review from the repo/attachments, not pastes.

## Skipped

Nothing in the R + B1 order was left undone. world.py untouched.
`test_agent_nn.py` untouched (A4 training path unchanged by R; the
rule change is text + counts in the summary).

#!/usr/bin/env bash
# run_repro.sh -- reproducibility record for the time-based mind model.
#
# Runs the verification suites and records the environment. Does NOT
# retrain arms (see run_arms.py for the training protocol).
#
# Note: nn/test_agent_nn.py is EXPECTED to report NEURAL LEARNING: FAIL
# on its older five-seed all-must-pass gate (seed 1, 0.30 success).
# That is the recorded A4 honest partial failure, not a regression.
# The A5/B2 full comparison (run_arms.py) uses a different protocol.
#
# Output: a timestamped directory ./repro_<UTC-timestamp>/ with
# environment info and all test outputs.

set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/repro_$STAMP"
mkdir -p "$OUT"

{
  echo "commit: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch: $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "python: $(python3 --version 2>&1)"
  echo "numpy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>&1)"
  echo "scipy: $(python3 -c 'import scipy; print(scipy.__version__)' 2>&1)"
  echo "os: $(uname -srm)"
  echo "date_utc: $STAMP"
} > "$OUT/environment.txt"
cat "$OUT/environment.txt"

run_one() {
  # $1 = label, $2... = command (run from repo root or nn/)
  local label="$1"; shift
  echo "=== $label ==="
  "$@" > "$OUT/$label.log" 2>&1
  local code=$?
  echo "$label: exit $code -> $OUT/$label.log"
  return 0  # keep going; FAILs are recorded, not hidden
}

cd "$ROOT"
run_one world python3 world.py
cd "$ROOT/nn"
run_one test_nnet python3 test_nnet.py
run_one test_wrapper python3 test_wrapper.py
run_one agent_wrapped_checks python3 agent_wrapped.py
run_one test_agent_nn python3 test_agent_nn.py || true
run_one analyze_replication python3 analyze_replication.py
run_one analyze_expa python3 analyze_expa.py

echo "done: $OUT"
ls "$OUT"

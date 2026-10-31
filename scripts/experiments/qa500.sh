#!/usr/bin/env bash
# WP22 QA500 zero-choice local execution. The human authorizes the exact phase;
# the agent does not choose datasets, models, layers, budgets, or fallback data.
set -Eeuo pipefail
IFS=$'\n\t'

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PHASE="${1:-full}"
shift || true
case "$PHASE" in
  preflight|prepare|pilot|core|effects|native|audit|stage|full) ;;
  *) echo "ERROR: unsupported frozen phase '$PHASE'" >&2; exit 2 ;;
esac

APPROVE_GPU=0
APPROVE_NATIVE=0
while (($#)); do
  case "$1" in
    --allow-local-gpu) APPROVE_GPU=1 ;;
    --allow-native-compute) APPROVE_NATIVE=1 ;;
    *) echo "ERROR: unknown execution parameter '$1'" >&2; exit 2 ;;
  esac
  shift
done

case "$PHASE" in
  pilot|core|effects|full)
    if ((APPROVE_GPU != 1)); then
      echo "ERROR: explicit --allow-local-gpu required for real model inference" >&2
      exit 3
    fi ;;
  native)
    if ((APPROVE_NATIVE != 1)); then
      echo "ERROR: explicit --allow-native-compute required for native circuits/train" >&2
      exit 3
    fi ;;
esac

if [[ "$(git branch --show-current)" != "feat/qa-evaluation-500-v1" ]]; then
  echo "ERROR: checkout the frozen QA500 feature branch before execution" >&2
  exit 4
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked working tree has uncommitted modifications" >&2
  exit 4
fi

mkdir -p experiment/qa500/logs
RECEIPT="experiment/qa500/logs/driver_${PHASE}_$(date -u +%Y%m%dT%H%M%SZ).log"
if [[ -e "$RECEIPT" ]]; then
  echo "ERROR: QA500 driver log already exists" >&2
  exit 4
fi
exec > >(tee -a "$RECEIPT") 2>&1
echo "QA500 phase=$PHASE commit=$(git rev-parse HEAD) UTC=$(date -u +%FT%TZ)"
echo "driver=$RECEIPT"
python - <<'PY'
import shutil, os
from pathlib import Path
root=Path("experiment/qa500")
free=shutil.disk_usage(root).free
print("free_disk_bytes", free)
if free < 20 * 1024**3:
    raise SystemExit("ERROR: QA500 requires at least 20 GiB unallocated disk before start")
PY

if [[ ! -x ".venv-qa500/bin/python" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv-qa500
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv-qa500
  else
    echo "ERROR: Python 3.11 or 3.12 required" >&2
    exit 3
  fi
fi
# shellcheck disable=SC1091
source .venv-qa500/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -e '.[analysis,dev,qa-eval]'
python -m pip check

run_preflight() {
  python -m pytest -q \
    experiments/qa500/tests/test_cohort.py \
    experiments/qa500/tests/test_execution.py \
    harness/github_work_test/test_wp22_architecture.py
  python -m harness.contracts.registry check \
    --plan harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md
  python -m experiments.qa500.runner plan > experiment/qa500/logs/frozen_plan.json
  python -m experiments.qa500.capture sites > experiment/qa500/logs/frozen_sites.json
}

run_preflight
PARTIAL=0
run_scoped() {
  if "$@"; then
    return 0
  else
    result=$?
    PARTIAL=1
    echo "QA500 scoped stage returned $result: $*"
    echo "The frozen protocol remains unchanged; inspect the stage receipts."
    return 0
  fi
}
case "$PHASE" in
  preflight)
    echo "Preflight accepted; no remote datasets or model weights loaded."
    ;;
  prepare)
    python -m experiments.qa500.runner prepare
    python -m experiments.qa500.runner verify
    ;;
  pilot)
    python -m experiments.qa500.runner verify
    python -m experiments.qa500.runner pilot --model all --device cuda
    ;;
  core)
    python -m experiments.qa500.runner verify
    python -m experiments.qa500.runner core --model all --device cuda
    ;;
  effects)
    python -m experiments.qa500.runner verify
    python -m experiments.qa500.runner effects --model all --device cuda
    ;;
  native)
    run_scoped python -m experiments.qa500.native circuits --approved
    if python -m experiments.qa500.native prepare-train; then
      run_scoped python -m experiments.qa500.native train --approved
    else
      PARTIAL=1
      echo "QA500 native training preparation blocked; circuits receipt is preserved"
    fi
    ;;
  audit)
    python -m experiments.qa500.report audit
    ;;
  stage)
    python -m experiments.qa500.report audit
    python -m experiments.qa500.report stage
    ;;
  full)
    # Scope failures are independent; do not let gated GPQA prevent nine
    # other real datasets from running on the approved local machine.
    run_scoped python -m experiments.qa500.runner prepare
    run_scoped python -m experiments.qa500.runner verify
    run_scoped python -m experiments.qa500.runner pilot --model all --device cuda
    run_scoped python -m experiments.qa500.runner core --model all --device cuda
    run_scoped python -m experiments.qa500.runner effects --model all --device cuda
    run_scoped python -m experiments.qa500.report audit
    if ((PARTIAL)); then
      # Local-only partial exports are clearly marked and cannot be published.
      run_scoped python -m experiments.qa500.report stage --allow-partial
    else
      run_scoped python -m experiments.qa500.report stage
    fi
    ;;
esac
echo "QA500 phase=$PHASE finished UTC=$(date -u +%FT%TZ) partial=$PARTIAL"
echo "No Hugging Face publication was performed."
if ((PARTIAL)); then
  echo "QA500 completed with blocked/partial scoped units; return receipt for review" >&2
  exit 5
fi

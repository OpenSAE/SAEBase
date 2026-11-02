#!/usr/bin/env bash
# Independent WP23 entrypoint. No action is permitted without an explicit stage.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  printf '%s\n' \
    "Usage: bash scripts/workplan/wp023.sh <stage> [--run-id ID] [--allow-compute] [--allow-gpu]" \
    "Stages: inspect inventory dataset model sae compatibility task audit" \
    "inspect is read-only; inventory writes a new immutable target manifest." \
    "Native operations require explicit compute/GPU permissions and prepared assets." >&2
  exit 2
}

case "${1:-}" in
  inspect|inventory|dataset|model|sae|compatibility|task|audit) ;;
  *) usage ;;
esac

# Run as a source-root module, independently of the invoking shell's PYTHONPATH.
# Runtime permissions remain owned by the Python task and Workplan approval.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m scripts.workplan.wp023_orchestrator "$@"

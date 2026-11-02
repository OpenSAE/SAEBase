#!/usr/bin/env bash
# Independent Precheck preparation, intentionally not a GPU/full-execution launcher.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/workplan/wp023.sh inspect | inventory --run-id ID" >&2
  exit 2
fi
exec python scripts/workplan/wp023_orchestrator.py "$@"

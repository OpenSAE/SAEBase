#!/usr/bin/env bash
# Offline metric bridges + optional real checkpoint acceptance (no hidden downloads).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pytest -q
SAR_DIR="${SAR_SOURCE:-}"
RSS_DIR="${RSS_SOURCE:-}"
if [[ -n "$SAR_DIR" && -n "$RSS_DIR" ]]; then
  SAEBASE_SAR_SOURCE="$SAR_DIR" SAEBASE_RSS_SOURCE="$RSS_DIR" \
    python -m pytest -q tests/test_phase_c_reference_contract.py
  python examples/07_sar_via_saebase.py --paper-dir "$SAR_DIR"
  python examples/08_rss_via_saebase.py --paper-dir "$RSS_DIR"
else
  echo "[phase-c] Set SAR_SOURCE and RSS_SOURCE to the two original paper checkouts for offline metric parity."
fi
if [[ "${RUN_REAL_MODELS:-0}" == "1" ]]; then
  DEVICE="${DEVICE:-cpu}"
  python examples/09_second_model_smoke.py --model pythia70m --device "$DEVICE"
  python examples/09_second_model_smoke.py --model gemma270m --device "$DEVICE"
  if [[ -n "$SAR_DIR" && -n "$RSS_DIR" ]]; then
    python examples/07_sar_via_saebase.py --real --paper-dir "$SAR_DIR" --device "$DEVICE"
    python examples/08_rss_via_saebase.py --real --paper-dir "$RSS_DIR" --device "$DEVICE"
  fi
else
  echo "[phase-c] Real checkpoints require RUN_REAL_MODELS=1 (downloads weights)."
fi

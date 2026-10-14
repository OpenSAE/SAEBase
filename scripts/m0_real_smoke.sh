#!/usr/bin/env bash
# Opt-in REAL Pythia-70M/SAE M0 acceptance. Requires online public weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cpu}"
DATA_DIR="${DATA_DIR:-$ROOT/m0-artifacts/data}"
OUT_DIR="${OUT_DIR:-$ROOT/m0-artifacts}"
N_GENERATE="${N_GENERATE:-120}"
MIN_KEPT="${MIN_KEPT:-5}"
N_PROMPTS="${N_PROMPTS:-3}"
SEED="${SEED:-0}"

saebase-check-env --mode analysis
python scripts/m0_reference_smoke.py \
  --device "$DEVICE" --data-dir "$DATA_DIR" \
  --output "$OUT_DIR/m0_reference.json" \
  --seed "$SEED" --n-generate "$N_GENERATE" \
  --min-kept "$MIN_KEPT" --n-prompts "$N_PROMPTS"

# Verify the same prompts also work through the original Phase A task and CLI.
saebase --task sae --model pythia70m --sae pythia70m_resid_sm \
  --prompt-source SimpleCopying --layer 3 --device "$DEVICE" \
  --data-dir "$DATA_DIR" --n-prompts "$N_PROMPTS" \
  --run-dir "$OUT_DIR/activation_health"

python - "$OUT_DIR/m0_reference.json" "$OUT_DIR/activation_health/sae_activation_summary.json" <<'PY'
import json, sys
from pathlib import Path
ref, health = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
assert ref['real_checkpoint_validated'] is True
assert ref['status'] == 'passed' and ref['sample_count'] > 0
assert health['summary']['n_ok'] == ref['sample_count']
assert health['summary']['n_fail'] == 0
assert health['sae']['hook_name'] == ref['site']['hook']
assert all(row['codes_l0'] > 0 for row in ref['samples'])
print('[M0] Real checkpoint reference AND activation-health CLI passed')
PY

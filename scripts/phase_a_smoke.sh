#!/usr/bin/env bash
# Real checkpoint acceptance run. Needs weights, SAE Lens and TransformerLens.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cpu}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
RUN_DIR="${RUN_DIR:-$ROOT/experiment/phase_a_${DEVICE}}"
python scripts/check_env.py --mode analysis
python scripts/sae_prepare_prompts.py \
    --model-key pythia70m --behavior SimpleCopying --device "$DEVICE" \
    --n-generate 300 --min-kept 20 --seed 0 --data-dir "$DATA_DIR"
python main.py --task sae --model pythia70m --sae pythia70m_resid_sm \
    --prompt-source SimpleCopying --layer 3 --n-prompts 5 --device "$DEVICE" \
    --data-dir "$DATA_DIR" --run-dir "$RUN_DIR"
python - "$DATA_DIR" "$RUN_DIR" <<'PY2'
import json,sys
from pathlib import Path
data,output=map(Path,sys.argv[1:])
assert (data/'SimpleCopying'/'pythia70m'/'v1'/'prompts.jsonl').is_file()
assert (data/'SimpleCopying'/'pythia70m'/'v1'/'manifest.json').is_file()
p=output/'sae_activation_summary.json'
result=json.loads(p.read_text())
assert result['summary']['n_ok'] > 0 and result['summary']['n_fail'] == 0
assert result['layer'] == 3
assert result['sae']['hook_name'] == 'blocks.3.hook_resid_post'
print('[phase-a] real-checkpoint acceptance passed:', p)
PY2

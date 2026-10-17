#!/usr/bin/env bash
# End-to-end activation_health smoke (needs CUDA, model+SAE weights, prepared prompts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 main.py \
  --task sae \
  --model pythia70m \
  --prompt-source SimpleCopying \
  --sae pythia70m_resid_sm \
  --layer 3 \
  --n-prompts 5

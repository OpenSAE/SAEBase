# Six local workflows

These commands run on a machine with the recorded weights. The public Research Lab pages show the measured cards and do not execute the models.

```bash
# G2 — 16-token greedy steering. Prompt files stay local.
python scripts/wp15_steer.py \
  --prompt-file /path/to/prompt-a.txt \
  --prompt-file /path/to/prompt-b.txt \
  --sae-key gemma270m_scope_resid --layer 5 --device cuda \
  --max-new-tokens 16 --scale 2.0 \
  --output /tmp/generation.json

# G3 — pinned SAEBench Core and Sparse Probing
# upstream commit 8042bb3828c6340da8d12062324e92b2077c571c
JAXTYPING_DISABLE=1 python scripts/m3_native_saebench_smoke.py

# G5 — Sparsify on a local Pythia snapshot and a text dataset directory
# TrainPlan: layers.3, topk, d_in 512, d_sae 1024, k 32, 50 steps
# Then SparseCoder.load_from_disk and encode one real activation.

# G6 — local Qwen2.5-0.5B-Instruct snapshot
python scripts/wp12_autointerpret.py --help
```

G4 uses the original `circuit-tracer attribute` command with `--offload cpu` on Llama-3.2-1B or Gemma-2-2B and a matching official transcoder. G1 capture and edit stay on the authenticated private worker. Prompt text, explanations, and weights are not part of the public cards.

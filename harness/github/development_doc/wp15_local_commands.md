# Six local workflows

These commands run on a machine with the recorded weights. The public Research Lab pages show the measured cards and do not execute the models.

```bash
# G1 — capture genuine SAE codes for a new local prompt (never auto-publishes)
python saebase/tools/wp12_p2_capture.py \
  --prompt-file /path/to/authored-prompt.txt \
  --sae-key gemma270m_scope_resid --layer 5 --device cuda \
  --sample-id local-example-001 --group-id local-group-001 \
  --output /tmp/local-feature-trace.json

# G2 — 16-token greedy steering. Prompt files stay local.
python saebase/tools/wp15_steer.py \
  --prompt-file /path/to/prompt-a.txt \
  --prompt-file /path/to/prompt-b.txt \
  --sae-key gemma270m_scope_resid --layer 5 --device cuda \
  --max-new-tokens 16 --scale 2.0 \
  --output /tmp/generation.json

# G3 — pinned SAEBench Core and Sparse Probing
# upstream commit 8042bb3828c6340da8d12062324e92b2077c571c
JAXTYPING_DISABLE=1 python saebase/tools/m3_native_saebench_smoke.py

# G5 — Sparsify on a local Pythia snapshot and a text dataset directory
# TrainPlan: layers.3, topk, d_in 512, d_sae 1024, k 32, 50 steps
# Then SparseCoder.load_from_disk and encode one real activation.

# G6 — local Qwen2.5-0.5B-Instruct snapshot
python saebase/tools/wp12_autointerpret.py --help
```

G4 uses the original `circuit-tracer attribute` command with `--offload cpu` on a supported Gemma-2-2B or Llama-3.2-1B and matching genuine transcoder. After that real native command produces the frontend-v1 graph JSON, use the **portable** importer below, pointing to the *same* exact checkpoints that were used for tracing:

```bash
python saebase/tools/wp15_import_gemma_graph.py \
  --native /path/to/native-graph.json \
  --model-dir /path/to/exact-model-snapshot \
  --transcoder-dir /path/to/exact-transcoder-snapshot \
  --tool-commit 255d989e93be1299d53d67b59e9fd76a0e274b14 \
  --card-output /path/to/new-staging/wp15/circuit.json \
  --summary-output /path/to/private/G4-import-summary.json \
  --public-demo-prompt
```

Use `--public-demo-prompt` **only** when the original prompt is author-approved. The produced card starts with `rights_reviewed:false` until a human has reviewed redistribution. The importer refuses to guess model/transcoder weights by searching unrelated cached directories. Full graph tensors remain local.

A public CPU Lab may display safe measured summaries, but **does not** perform model inference. For reviewed published text, use a separate immutable `$OPENSAE_LAB_DATA/wp15/` bundle. A reviewed `generation.json` may contain `public_example` only when the source prompt's SHA256 matches and all publication review flags are true. A reviewed `explanation.json` may display a human-approved description; an unreviewed candidate never exposes its text.

G1 capture and edit stay on the local Python API or authenticated owner-only worker. For full commands to stage a reviewed G1 trace, see [the WP15 pre-release handoff](../github_work_plan/15_pre_release_remaining_handoff.md).

# Phase C: research integrations

SAEBase's public API supplies activations, feature edits, logit readouts, and
checkpoint provenance. SAR and RSS retain their published algorithms in their
independent GitHub repositories. **Their estimator files are not copied or
modified here.**

## SAR

From a checkout containing the original `sar.py`:

```bash
python examples/07_sar_via_saebase.py --paper-dir /path/to/SAR-repo
```

This captures one prompt/site, selects the K largest absolute SAE codes,
samples subsets through the original `sar.coalitions`, removes only excluded
selected SAE contributions, patches the model and calls the unchanged
`sar.sar` estimator. All remaining SAE codes and the original reconstruction
residual are retained. `h + decode(z_new) - decode(z)` requires a linear/affine
SAE decoder for SAR's additive-coalition interpretation. A pairwise and full-subset
additivity check catches common mismatches; it is not a proof of linearity
for arbitrary nonlinear decoders. `readout` maps `[vocab]` logits to a scalar.

**Demo settings**: K=3, M=80, 55 training coalitions, 25 held out. These are
small demonstration settings, not the paper's reported experiment protocol.
The current collection is sequential for clarity and correctness; large
coalition experiments will need optimized batching.

## RSS

From a checkout containing the original `rss.py`:

```bash
python examples/08_rss_via_saebase.py --paper-dir /path/to/RSS-repo
```

The bridge captures clean, selection and held-out variants separately.
Feature selection uses only `selection_changes` passed to `rss.rss`, then
`rss.held_out_rate` evaluates the chosen support on unseen variants.
Input strings must be distinct across the three groups. Explicit
`selection_positions` and `held_out_positions` support matching content at
different token positions; otherwise the requested site position (default
last token) is applied independently to each variant. **The caller validates
semantic alignment of the clean and transformed prompts.** The demo's toy
prompts and the optional Pythia prefixes demonstrate code paths, not a
replication of the paper's transformation or safety-readout experiments.

## SAE/model coverage

| Configuration | Level established in Phase C |
|---|---|
| Synthetic TransformerLens-like `resid_post` | Offline capture + SAR/RSS + patch verified |
| Synthetic TransformerLens-like `mlp_out` | Offline capture + SAR/RSS + patch verified |
| Synthetic HFHooks layer-output | Offline capture + SAR/RSS + patch verified |
| Pythia-70M + `pythia70m_resid_sm` layer 3 | Registered; real checkpoint run pending |
| Gemma-3 270M + `gemma270m_scope_resid` layer 5 | Registered; second real checkpoint run pending |

Real checkpoint commands (downloads require the requested model and SAE
sources and any required access approval):

```bash
python examples/09_second_model_smoke.py --model pythia70m --device cpu
python examples/09_second_model_smoke.py --model gemma270m --device cpu
```

The default SAE and layer can be overridden only with a registered and
compatible combination. Large models require appropriate memory.

## Validation

The standalone tests run without the papers or checkpoints. For an extra
check against **the actual untouched** paper modules, provide the source paths:

```bash
SAEBASE_SAR_SOURCE=/path/to/SAR-repo \
SAEBASE_RSS_SOURCE=/path/to/RSS-repo \
python -m pytest -q tests/test_phase_c_reference_contract.py
```

For all offline checks and optional real checkpoints:

```bash
SAR_SOURCE=/path/to/SAR-repo RSS_SOURCE=/path/to/RSS-repo \
  bash scripts/phase_c_smoke.sh

# Opt into external model downloads and real model evaluations:
RUN_REAL_MODELS=1 DEVICE=cpu SAR_SOURCE=/path/to/SAR-repo \
RSS_SOURCE=/path/to/RSS-repo bash scripts/phase_c_smoke.sh
```

## Advanced generation and NNsight

- vLLM in-process sampling and HTTP dispatch have CPU-only contract tests with
  fake backends. Actual GPU generation remains an integration acceptance item.
- SAEBase's generator cache now tracks model ID, dtype, memory utilization and
  maximum context length. Sampling output counts are checked.
- `stop_vllm()` releases only the in-process generator and HTTP server
  **started by this Python process**. Other users' vLLM servers are preserved.
- The NNsight worker's `smoke`/`probe`/`dump_resid` commands remain available
  for GPU acceptance. Its protocol-specific `generate` mode references
  archived task modules unavailable in this standalone checkout and now raises
  an explicit dependency error before GPU resources are modified.
- The streaming worker enforces a wall-clock timeout even if a child is silent.

**No actual Pythia/Gemma weights, real GPU vLLM engine or NNsight model
intervention were run in the current offline environment.** The fake model
numbers in examples are implementation checks, not scientific results.

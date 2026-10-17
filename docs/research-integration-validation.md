# Research Integration Validation

This page preserves detailed technical contracts and verification context. For current product-facing coverage, see [Validation and Reproducibility](validation.md) and [Documentation Home](index.md).

## Scope

Baseline: `SAEBase_Phase_B_Ready.zip`. research integrations adds two research integrations
while leaving the original SAR and RSS estimator code in their own repositories.
It also adds a second registered model/SAE acceptance path and improves the
safety and consistency of advanced generation resource management.

### Implemented

1. `saebase/integrations/sar.py` — SAR coalitions via the public API capture/patch
   interface, using the original `sar.coalitions` and `sar.sar` functions.
   Preserves unselected SAE codes and reconstruction residual, checks identity
   replays and selected decoded contribution additivity.
2. `saebase/integrations/rss.py` — clean/selection/held-out SAE response capture,
   explicit token positions and strict split-disjointness. Calls original
   `rss.rss` only on selection changes, then `rss.held_out_rate` on held-out
   changes; no metric rewriting.
3. `saebase/integrations/sources.py` — direct loading of user-supplied `sar.py`
   and `rss.py` from the original checkouts; no automatic downloads or copies.
4. `examples/07_*`, `08_*` — deterministic standalone offline demonstrations
   and explicit opt-in Pythia acceptance mode.
5. `examples/quickstart/model_checkpoint_probe.py` — registered Pythia-70M layer 3 and
   Gemma-3 270M layer 5 checkpoint acceptance commands. The latter is an
   additional configured pairing, **not yet a successful downloaded run**.
6. `saebase/generate` — vLLM cache keys include all engine settings; returned
   prompt counts are validated; generation-only text decoding uses generated
   IDs. Server shutdown terminates only subprocesses owned by the caller;
   silent streaming workers respect wall-clock timeouts; archived NNsight
   protocol-generation tasks are detected before allocating GPU resources.

### Executed tests

```text
python -m pytest -q
81 passed, 3 skipped
```

The skipped cases include the pre-existing Hydra composition check (Hydra
unavailable locally), and two optional reference-module contract tests that
need paths to the SAR and RSS repositories.

With original source paths explicitly set:

```text
SAEBASE_SAR_SOURCE=<SAR checkout> SAEBASE_RSS_SOURCE=<RSS checkout> \
python -m pytest -q
83 passed, 1 skipped
```

The two additional original-implementation tests passed against the unchanged
`SAR/sar.py` and `RSS/rss.py` from the user's supplied project archives.
The small synthetic results were:

- SAR: `sar1≈0.21565`, `sar2≈0`, `rho2≈1` for a 3-feature, 80-subset toy;
  values matched calls to the original SAR estimator.
- RSS: features `[3, 0, 1, 4, 5]`, directions all positive, held-out
  rates all 1 for a controlled synthetic example; values matched the original
  RSS implementation.

Additional checks:

- `bash scripts/phase_c_smoke.sh` executed with both paper source locations;
  it ran the original-source tests and both offline example scripts.
- 181 Python files parsed without syntax errors.
- `python -m saebase.generate.nnsight_worker --help` works without loading
  GPU dependencies.
- Fake vLLM, HTTP, model patching, runtime ownership, and silent worker timeout
  tests passed.

### Earlier external acceptance status (historical)

The local environment lacks `transformer_lens`, `sae_lens`, `hydra-core`,
`transformers`, `vllm` and `nnsight`. This phase did not execute downloaded
Pythia or Gemma checkpoints, real CUDA generation, or live NNsight steering.
No synthetic metric or simulated vLLM output represents a paper result.

To run the real acceptance in a prepared environment:

```bash
RUN_REAL_MODELS=1 DEVICE=cuda SAR_SOURCE=/path/to/SAR-repo \
  RSS_SOURCE=/path/to/RSS-repo bash scripts/phase_c_smoke.sh
```

The NNsight worker's `generate` mode references experimental task sources
omitted from this standalone SAEBase archive. The explicit preflight protects
against late failures; those sources must be ported and validated before this
mode is advertised as available.

### Integration contract

- Python artifact schema and public API public API continue to work.
- There is no embedded or modified copy of the paper estimators.
- Both integrations expose raw sampled measurements and site metadata for
  audit; documentation distinguishes toy exploration from publication results.

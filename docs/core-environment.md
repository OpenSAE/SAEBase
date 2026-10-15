# Core Setup and Installation Validation

This page preserves detailed technical contracts and verification context. For current product-facing coverage, see [Validation and Reproducibility](validation.md) and [Documentation Home](index.md).

## Implemented

- Restored `tasks.plugins` registration with the bundled `SAETask` and its `activation_health` evaluation.
- The CLI checks arguments and only the dependencies needed for activation analysis, after parsing `--help`.
- Unbundled internal research task names now produce clear errors instead of implied support.
- Added `scripts/sae_prepare_prompts.py` for Pythia + SimpleCopying with seed, tokenizer IDs,
  logit-difference filtering, artifact manifest, configurable directory and CPU/CUDA selection.
- Routed `paths.data_dir` consistently through preflight, prompt loading and evaluation.
- Added an optional `--run-dir` that resolves against the caller's directory, then passes to Hydra.
- Made vLLM/NNsight dependencies optional through `requirements-generation.txt` (one Python interpreter).
- Removed the missing protocol-04 import from the legacy generation wrapper. Its sampling parameters are explicit.
- The absent BatchTopK research loader is called out clearly; checkpoint inference is not approximated.
- Replaced the internal protocol-heavy README with a standalone README. The supplied
  source ZIP remains the unmodified private baseline.

## Exact real-model acceptance procedure

```bash
python -m pip install -r requirements.txt
python scripts/check_env.py --mode analysis
python scripts/sae_prepare_prompts.py --model-key pythia70m --behavior SimpleCopying --device cpu --seed 0 --n-generate 300 --min-kept 20 --data-dir ./data
python main.py --task sae --model pythia70m --sae pythia70m_resid_sm --prompt-source SimpleCopying --layer 3 --n-prompts 5 --device cpu --data-dir ./data --run-dir ./experiment/phase_a_cpu
```

You can also run `bash scripts/phase_a_smoke.sh`; it executes both commands and
asserts the final outputs. Verify `prompts.jsonl`, `manifest.json` and
`sae_activation_summary.json` exist
at those exact locations, `n_ok > 0`, `n_fail == 0`, and the reported profile
has layer 3 and the expected hook `blocks.3.hook_resid_post`.

## Evidence and limitations

The unit/integration suite validates registry, prompt identity and explicit paths,
plus a complete activation-health cycle using deterministic small model/SAE doubles.
It does not establish published checkpoint load compatibility.

The supplied execution environment has no Internet-accessible Python package index
and lacks `hydra-core`, `transformer_lens`, `sae_lens`, and the pretrained weights.
Therefore the published Pythia + SAE end-to-end acceptance command requires a run
in an environment that can install these packages and download the checkpoints.
Actual checkpoint compatibility, device runtime, and large-model inference remain
unverified and are **not** marked as passed.

## File ownership

The source ZIP is the unchanged baseline. Existing registry entries, checkpoint
IDs, evaluation metric definitions, and archived protocol YAMLs are preserved.
core setup repairs the main path and adds tests/entry scripts. The scope does not
extend to implementing missing research-specific tasks or the high-performance
NNsight-worker pathways.

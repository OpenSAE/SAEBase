# SAEBase × OpenResearch code integration

This note describes the SAEBase v0.6.0 integration. SAEBase v0.5.0 remains the backward-compatibility baseline; v0.6.0 includes OpenResearch execution and agent harness components in the source and software distributions.

## Merge and ownership rules

1. Preserve SAEBase's existing `saebase/` code, registry adapters, CLI, `apps/`, original models and `harness/github_development_doc/`. Do not replace the v0.5.0 API with the smaller OpenResearch subset.
2. Copy OpenResearch `datasets/`, `saebase/runtime/`, the Gemma LLM adapter, seven LLM task families, `harness/`, `.cursor/`, `doc/`, and nonconflicting Hydra YAML groups. These are additive paths.
3. Where module names have different meanings, keep both: `models/plugins.py` remains SAEBase's legacy delegation; `models/dlbase_plugins.py` is the DLBase registration implementation. The same rule applies to `tasks/plugins.py` and `tasks/dlbase_plugins.py`.
4. Keep `conf/config.yaml` for the released SAEBase workflow; `conf/sae_research.yaml` is the DLBase entrypoint. Keep original `conf/model/gemma2_2b_base.yaml`; choose `conf/model/openresearch_gemma2_2b_base.yaml` for DLBase's lazy model configuration.
5. Keep `saebase/data/` and original prompt builders. The new `datasets/` modules carry OpenResearch's separate prompt artifact layout. Do not silently migrate experiment datasets, checksums, or benchmarks.
6. OpenResearch's prior Code/Knowledge JSON and Core graph were copied for reference and require SAEBase-specific evidence review. Public Atlas snapshots must be regenerated from SAEBase before they can represent this repository's verified state.
7. Temporary OpenResearch work plans, demo ledgers and privileged GitHub-only notes are excluded. Governance code and Cursor approval gates are included.

## Checkpoints

```bash
python .cursor/hooks/validate_repo.py --full
python -m pytest -q tests/test_openresearch_integration.py
python -m saebase.runtime.train --cfg job
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py
```

These commands are also covered by the release CI where applicable. Bounded real-GPU execution is recorded in the 2026-10-27 GPU report, with scientific limits and remaining blockers stated there.

## Explain generator

`python -m saebase.runtime.train task=explain` cannot put a Python callable in YAML. With no generator it refuses to invent text. Two supported entries:

- `execute(cfg, generator=callable)` from `tasks.explain.operation`.
- `task.generator=saebase_generate` with `task.approved=true` and `task.generator_model` set to a cached model id or directory. That name calls `saebase.generate.generate` (vLLM). Gemma-2 needs `generator_dtype=bfloat16`. Any other string is rejected and is not evaluated. The main environment on this machine does not have a complete vLLM install; use an isolated environment that already imports vLLM.

`heldout_score` measures whether a supplied predictor agrees with the SAE feature being active. It is not a semantic accuracy for the generated sentence.

## Identified compatibility issue

The `datasets/` package uses its own Python name, avoiding a collision with Hugging Face's `datasets` package. Research Lab and the legacy dataset readers retain their original imports.

The v0.6.0 distribution includes DLBase, the Hydra resources and registered NLP components, in addition to the original public API and model/task shims. The immutable GitHub release is governed by the documented exact-source CI workflow; no PyPI upload is implied.

# SAEBase × OpenResearch code integration

This note describes the additive integration branch. The public SAEBase 0.5.0 package remains the compatibility baseline, while OpenResearch provides the research execution layout and agent harness.

## Merge and ownership rules

1. Preserve SAEBase's existing `saebase/` code, registry adapters, CLI, `apps/`, original models and `docs/`. Do not replace the v0.5.0 API with the smaller OpenResearch subset.
2. Copy OpenResearch `datasets/`, `dlbase/`, the Vision and Gemma adapters, seven LLM task families, `harness/`, `.cursor/`, `doc/`, and nonconflicting Hydra YAML groups. These are additive paths.
3. Where module names have different meanings, keep both: `models/plugins.py` remains SAEBase's legacy delegation; `models/dlbase_plugins.py` is the DLBase registration implementation. The same rule applies to `tasks/plugins.py` and `tasks/dlbase_plugins.py`.
4. Keep `conf/config.yaml` for the released SAEBase workflow; `conf/openresearch.yaml` is the DLBase entrypoint. Keep original `conf/model/gemma2_2b_base.yaml`; choose `conf/model/openresearch_gemma2_2b_base.yaml` for DLBase's lazy model configuration.
5. Keep `saebase/data/` and original prompt builders. The new `datasets/NLP/` modules carry OpenResearch's separate prompt artifact layout. Do not silently migrate experiment datasets, checksums, or benchmarks.
6. OpenResearch's prior Code/Knowledge JSON and Core graph were copied for reference and require SAEBase-specific evidence review. Public Atlas snapshots must be regenerated from SAEBase before they can represent this repository's verified state.
7. Temporary OpenResearch work plans, demo ledgers and privileged GitHub-only notes are excluded. Governance code and Cursor approval gates are included.

## Checkpoints

```bash
python .cursor/hooks/validate_repo.py --full
python -m pytest -q tests/test_openresearch_integration.py
python -m dlbase.train --cfg job
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py
```

Commands shown here are verification targets, not claims of having run them. GPU checkpoints, NNsight and vLLM backend compatibility, research reproducibility, browser interaction, full SAEBase test matrix and packages/wheels require follow-up evidence.

## Identified compatibility issue

The new top-level `datasets/` Python package can shadow Hugging Face's third-party `datasets` package. OpenResearch provides `models.hf_compat.import_hf_datasets` and a subprocess-based adapter for supported data-loader calls. The original SAEBase research-lab applications may use the Hugging Face dataset object API directly and must be audited before this integration becomes a release.

The original Python distribution currently packages its public API and existing model/task shims; the merged source workspace is not yet declared a new installable release. Do not publish PyPI artifacts or merge to main until compatibility tests and a release review pass.

# SAEBase and OpenResearch architecture

SAEBase v0.6.0 introduced the research Harness and seven research operations.
This development branch unifies the component layout without changing the
published v0.6.0 tag or production Research Lab deployment.

## Component ownership

- `datasets/` contains the canonical prompt sources and dataset plugins.
  `saebase/data/` retains narrowly scoped compatibility imports.
  The Python import namespace is `saebase.datasets` so Hugging Face
  `datasets` remains importable.
- `models/` contains model adapters and registrations, preserving the verified
  SAEBase model and checkpoint catalog.
- `tasks/` contains capture, intervene, generate, circuits, explain,
  benchmark and train. The historical `sae` task remains an API compatibility
  entry, not a separate Registry.
- `saebase/registry.py` is the central registry and
  `saebase/plugin_bootstrap.py` loads the model, dataset and task plugins.
- `saebase/runtime/` hosts shared configuration, training and experiment
  execution. Agent edits to this protected layer require explicit review.
- `saebase/research_lab/` contains the Research Lab service and its static
  assets. The publication and evidence-review gates remain unchanged.
- `harness/github_development_doc/` contains developer documentation and
  `harness/github_work_plan/` retains historical work plans. Their old
  experiment records are not automatically new release evidence.

## Verification

For changes on the refactor branch, run Python 3.11 and 3.12 offline tests,
quality checks, Hydra composition, the installed-wheel test, the Research Lab
offline checks, the Harness source and Atlas checks, and pinned research
integration tests. Run bounded GPU smoke tests only for affected model/task
interfaces. Do not merge while required checks fail.

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

The v0.6.0 distribution includes SAEBase runtime, the Hydra resources and registered NLP components, in addition to the original public API and model/task shims. The immutable GitHub release is governed by the documented exact-source CI workflow; no PyPI upload is implied.

# SAEBase Experiment Directories and Provenance

Hydra and `saebase/runtime/experiment_paths.py` manage experimental output directories. The Harness validates configuration identity, provenance, and saved artifacts.

```text
experiment/
└── IOI/
    └── gemma2_2b_base/
        └── capture/
            ├── _tmp/
            └── basic_42/
                ├── .hydra/
                ├── checkpoints/
                ├── metrics/
                ├── tables/
                ├── figures/
                ├── logs/
                ├── metadata/
                └── reports/
```

The directory is an organizational example, not a record of completed training. Actual experiments must record the model, SAE checkpoint, input fingerprint, task, random seed, configuration hash, and environment. Existing run directories are protected from accidental overwrite.

Independent analysis can be organized under `analysis/<purpose>/<subpurpose>/`, with stable links to existing SAE Knowledge IDs in `doc/knowledge/context_graph.json`.

```python
from saebase.runtime.experiment_paths import create_analysis_group

group = create_analysis_group(
    cfg, cwd, "feature_study/ablation",
    context_ids=["g.sae.feature_edit"],
    description="Compare feature interventions under a fixed input.",
)
```

GPU experiments require confirmed model and data access, resource budgets, and output paths. Static Atlas views cannot substitute for actual scientific acceptance evidence.

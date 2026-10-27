# SAEBase 实验目录与可追溯性

运行目录由 `saebase/runtime/experiment_paths.py` 和 Hydra 管理，Harness 仅验证来源、配置与结果文件。

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

目录示例仅说明文件组织，模型权重与实际指标由真实执行生成。每次运行记录模型、SAE、输入数据指纹、任务、随机种子、配置哈希及环境快照。默认拒绝覆盖现有实验目录。

`analysis/<purpose>/<subpurpose>/` 可用于独立分析，例如特征消融；研究关联使用 `doc/knowledge/context_graph.json` 中已存在的 SAE Knowledge ID。

```python
from saebase.runtime.experiment_paths import create_analysis_group
group = create_analysis_group(
    cfg, cwd, "feature_study/ablation",
    context_ids=["g.sae.feature_edit"],
    description="以固定输入比较 SAE 特征干预及其对模型输出的影响。",
)
```

真实训练与 GPU 实验需事先确认数据、模型、预算及保存路径，不得用静态视图替代科研验收。

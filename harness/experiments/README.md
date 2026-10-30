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

目录示例仅说明文件组织，模型权重与实际指标由真实执行生成。单数 `experiment/` 是当前 Runtime 的正式路径；治理准备过程为真实工作流考试预建独立工作树的输出根。默认拒绝覆盖已有实验产物。

当前 `metadata/run.json` 记录 dataset/model/task 名称、任务层级、sub_name、seed、resolved YAML 的配置哈希、类型目录及分析组。它本身并未完整记录输入内容、模型和 SAE checkpoint 哈希、源码身份或环境快照。真实工作流另外提供 evidence manifest，逐项绑定这些来源、实际命令、退出码及产物哈希，要求见 [真实工作流协议](../contracts/REAL_WORKFLOW_PROTOCOL.md#9-真实证据与独立验收)。原 Runtime 配置哈希保留文本哈希语义，canonical 配置身份如使用则另列。

`analysis/<purpose>/<subpurpose>/` 可用于独立分析，例如特征消融；研究关联使用 `doc/knowledge/context_graph.json` 中已存在的 SAE Knowledge ID。

```python
from saebase.runtime.experiment_paths import create_analysis_group
group = create_analysis_group(
    cfg, cwd, "feature_study/ablation",
    context_ids=["g.sae.feature_edit"],
    description="目标：以固定输入比较 SAE 特征干预及其对模型输出的影响。",
)
```

`python harness/experiments/validate.py <run-leaf>` 核对分析组的 Context 引用、目录和 manifest 一致性。它的通过只说明这些结构检查通过，完整实验验收还需独立验证实际输入、checkpoint、数值输出与文件完整性。

真实训练与 GPU 实验依据已指派题目的数据、模型、预算及保存路径执行。缺失条件按实际阻塞记录，静态视图和配置检查分别保留自身证据等级。

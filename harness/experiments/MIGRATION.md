# 旧实验产物与新目录协议的迁移说明

这个版本是对实验文件布局的**有意变更**。不自动搬运已有数据或 checkpoint，也不对旧目录执行删除操作。正式训练入口仍是 `python -m dlbase.train`，配置发生以下变化：

| 旧规则 | 新规则 |
| --- | --- |
| `experiment/<dataset>/<model>/<task>/<sub_name>_<seed>/` | `experiment/<dataset>/<model>/<task.path>/<sub_name>_<seed>/` |
| `<run>/best.ckpt` | `<run>/checkpoints/best.ckpt` |
| `<run>/summary.log` | `<run>/reports/summary.log` |
| `<run>/tensorboard/` | `<run>/logs/tensorboard/` |
| `experiment/tmp/` | `experiment/<dataset>/<model>/<task.path>/_tmp/<session>/` |
| 无统一产物说明 | `<run>/metadata/run.json`（配置 hash、层级、类型目录） |

SAE Task 可分别使用 `capture`、`intervene`、`generate`、`circuits`、`explain`、`benchmark` 或 `train` 作为 `task.path`；目录按选择的 Task 独立组织。如果原来的 `basic_42` 已有产物，新的 `guard_existing_run` 会明确拒绝覆盖，需更换 sub_name/seed 或由用户显式设置 `experiment.overwrite=true`。不要为了迁移而自动启用覆盖。

每次实验根目录还有 `analysis/<purpose>/<subpurpose>/`，每层分析叶子包含类型化目录以及 `metadata/context.json`，引用 `doc/knowledge/context_graph.json` 中的 Knowledge ID。运行 `python harness/experiments/validate.py <run-leaf>` 核对引用与产物结构。添加分组的 Python 函数是 `dlbase.experiment_paths.create_analysis_group`，而不是 Harness 中的运行时替代实现。

**迁移建议：** 保留旧运行目录为只读历史；下次训练使用新实验标识；分析脚本若寻找 `best.ckpt`、`summary.log`，明确改为新路径。正式大规模数据/Lightning/GPU 训练尚未在当前 GitHub CI 环境完成；现有 CI 覆盖的是无 GPU 的目录、结构、配置和可视化契约。

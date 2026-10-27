# SAEBase Task Registry

正式 SAE 操作包括 `capture`、`intervene`、`generate`、`circuits`、`explain`、`benchmark` 和 `train`。

七类操作在 `tasks/plugins.py` 中通过 `saebase/registry.py` 统一注册，由
`saebase/runner.py` 根据 Hydra `task.name` 分派。
`python -m saebase.runtime.train --cfg job` 仅展示合成配置，不会加载模型或启动计算。
真实实验应先通过任务准入、权重来源、身份、数据划分及执行授权校验。

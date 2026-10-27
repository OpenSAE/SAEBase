# SAEBase Task Registry

当前统一任务包括 `capture`、`intervene`、`generate`、`circuits`、`explain`、`benchmark` 和 `train`。

七类研究操作通过 `tasks/dlbase_plugins.py` 注册，由 `dlbase.train` 根据 Hydra 配置分发。真实模型、权重、数据来源与授权条件需由每个任务验证。默认 `execute: false` 仅用于检查配置，不启动模型计算。

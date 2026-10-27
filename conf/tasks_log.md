# Task 注册

公共任务接口是 `dlbase.registry.TaskModule.name/run(cfg)`，由 `tasks/plugins.py` 登记、`tasks/runner.py` 调度。

Vision：`tasks/vision/classification/` 提供通用的图像分类训练。

LLM：`tasks/LLM/` 下分别为 `capture`、`intervene`、`generate`、`circuits`、`explain`、`benchmark`、`train`。七个 Task 共享 `tasks/LLM/sae_runtime.py` 骨架，但每个保留独立入口和 `conf/task/<task>.yaml`；真实运算调用 SAEBase 的对应后端。

所有 Task 以能力命名，具体 `dataset` 与 `model` 通过组合配置选择。LLM 默认 `execute: false` 仅检查配置；真实运算需要明确执行选项及相关输入/授权。Vision 原有训练配置和日志职责保持原状。

# SAEBase 与外部工程管理

## 目的和主要功能

说明科研组件、共同运行支持与 ResearchHarness 的实际分工，并连接现行开发入口。

## 适用范围

用于理解 SAEBase 与外部工程管理的关系。组件维护采用当前项目工作模式，科研核心独立安装运行。

## 当前维护入口

[当前架构](current-architecture.md)集中维护组件职责、公共运行和结果接口。[开发模式](development-modes.md)说明密集开发与常规管理的具体选择。数据集、模型和八类 Task 保留所属实现，公共能力由实际调用者复用。

ResearchHarness 提供明确选择的外部管理。正式计划、中央验证和 Atlas 各按相应任务调用。修改 Runtime 时先定位共同支持接口及直接调用者，检查范围依据实际影响选择。

## Explain generator

`python -m saebase.runtime.train task=explain` cannot put a Python callable in YAML. With no generator it refuses to invent text. Two supported entries:

- `execute(cfg, generator=callable)` from `tasks.explain.operation`.
- `task.generator=saebase_generate` with `task.approved=true` and `task.generator_model` set to a cached model id or directory. That name calls `tasks.generate.backend.generate` (vLLM). Gemma-2 needs `generator_dtype=bfloat16`. Any other string is rejected and is not evaluated. The main environment on this machine does not have a complete vLLM install; use an isolated environment that already imports vLLM.

`heldout_score` measures whether a supplied predictor agrees with the SAE feature being active. It is not a semantic accuracy for the generated sentence.

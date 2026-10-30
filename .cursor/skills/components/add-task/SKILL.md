---
name: add-task
description: 当需要新增训练或评测任务，包括 TaskModule、pipeline checks、任务配置与指标 adapter 时使用。
---

# 接入 Task

1. 阅读 saebase.registry.TaskModule、saebase/runner.py、tasks/plugins.py、已有 Task、当前 Dataset 和 Model 的门控。
2. 明确输入、输出、loss、评测指标、split 假设和 checkpoint 选择原则；区分科研协议与程序实现。
3. 在 tasks/<name>/ 实现 TaskModule 的 name 与 run(cfg)、训练逻辑和 pipeline_checks.py 或等效检查。
4. 在 tasks/plugins.py 使用 register_task 注册；添加 conf/task/<name>.yaml；按需更新 conf/tasks_log.md。
5. Pipeline 先执行 Dataset policy 与 Model compatibility，再执行任务自身的 shape、dtype 与语义断言。
6. 复用 saebase/runtime/training 的可配置 facets；Task 实现指标 adapter 并满足当前 saebase.runtime.summary_log 的 train/test 运行时契约，不能伪造缺失的 split。
7. 验证 Hydra 合成、注册、最小 dry-run 和针对 loss/metric 的回归；运行 python .cursor/hooks/validate_repo.py --full。
8. 输出新增 Task 的适用组件、假设、验证结果与未覆盖场景。

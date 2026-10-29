# WP20 · 集成考试工作区

[Work Plan 20](../../github_work_plan/20_integrated_multistage_harness_exams.md) 定义 X01–X20 共 20 个较大的多阶段集成任务。当前状态为 `SPEC_REVIEWED`，已有冻结的主题、功能要求、Contract 绑定、模块来源、正常/异常场景和独立 Oracle 设计。**正式的独立行为 verifier 尚未实现，这些 Case 尚不属于 READY；考试 Agent 不应提前运行或自行编写官方评分器。**

后续可信治理方将逐题实施参考行为与故障注入、Good/Bad 校准，才允许受控开考。Agent 的新文件范围计划为 `harness/tests/wp20/runs/<run_id>/Xxx/`。必须按任务的实际 API 实现多个阶段，不允许只返回一个布尔条件，也不允许将原任务偷换成检查一个 JSON 字段。

所有读源文件只用于核对既有接口，**生产代码、Core、Contract、CI、Cursor Hooks、历史 WP18/WP19、独立 acceptance 和正式 Atlas 全部不可由考试 Agent 修改**。输出独立文件 SHA、真实运行日志、candidate AST 和未解决项；人工 approve 仍独立完成。真实候选代码的执行需低权限、可丢弃环境，仅靠 Hooks/timeout 不是 OS 安全隔离。

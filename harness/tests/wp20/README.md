# WP20 · X01–X20 CPU 多阶段行为考试

全部 20 题状态为 `READY`。正式计划见 [WP20](../../github_work_plan/20_integrated_multistage_harness_exams.md)，机器规范在 [wp20_cases.json](../../contracts/wp20_cases.json)，正反场景在 [wp20_scenarios.json](../../contracts/wp20_scenarios.json)，可信评分器为 [wp20_verify.py](../../contracts/wp20_verify.py)。

执行 Agent 逐题在新建的 `harness/tests/wp20/runs/<run_id>/Xnn/` 完成，入口 `implementation.py` 提供 `run(payload: dict, workdir: pathlib.Path) -> dict`，可通过 `src/` 中多个模块实现。合法输入必须计算真实输出，非法输入明确抛 ValueError；一些题目要求实际在临时目录写文件、创建归档或 Git 提交。输出会与独立参考的完整对象和文件 SHA 对照。

每题交付 `tests/`、`fixtures/`、`output_manifest.json`、`contract_receipt.json`、`run_trace.json`、`review_candidate.json`（全部 unchecked）与 `report.md`。公开验证命令为 `python -m harness.contracts.wp20_verify --check-spec` 和 `python -m harness.contracts.wp20_verify --case X01 --run-dir harness/tests/wp20/runs/<run_id>/X01`。每批 2–4 题；公开自测不等于独立审核。

严禁 Agent 改 Contract、Reference、Scenario、Verifier、CI、Rules、Hooks、历史 WP18/WP19、权威 Atlas、正式产品文件或独立审阅记录。只可新增当前 Run 的 case 文件。Reviewer 从固定可信治理提交使用不同数据复验，正式批准与合并始终独立。

**安全限制**：候选代码只有在低权限、无网络、无真实凭据、可丢弃的 OS 隔离环境中才能由独立评分器执行。Cursor Hooks、GitHub Scope CI 和 subprocess timeout 都不是 OS 安全沙箱。
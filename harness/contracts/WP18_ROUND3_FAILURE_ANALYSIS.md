## WP18 第二批 H07–H26：独立核验与 Harness 通用修复（2026-10-29）

- 原始考试提交：[PR #87](https://github.com/OpenSAE/SAEBase/pull/87)，head `8e4c37f6660a4fee90a3f02ea54c6ca4a6f73bf2`。20 题及公开 86 检查据 Agent 报告均通过，仓库 CI 绿色；不等同独立验收。
- 独立 reviewer：[PR #89](https://github.com/OpenSAE/SAEBase/pull/89) 基于原 Agent head 运行新的反例，5 项均触发了已知约束遗漏。**不合并 reviewer PR，不修改 Agent 分支**。
- H07：Windows 盘符绝对路径 `C:/...` 未被拒绝；应由共用跨平台相对路径校验负责。
- H09：嵌套 `privateKey`/`sessionToken` 与普通字符串中的 `token=...` 仍可泄漏；应复用已有 `invariants.redact_structure` 而不是只接受固定的字段枚举。
- H13：中文占位 `中文翻译不可用` 未被识别；`invariants.localization_defects` 已具备覆盖能力，应进入独立实际产物验收。
- H19：合法有限浮点输入可能计算出 Infinity；检查计算后的 JSON 结果也要符合有限性约束。
- Harness 处理：通用 `harness/contracts/compliance.py` 将上述来源、隐私、双语及非有限数值验收组合在**独立 reviewer 实际采集的 artifact** 上，配套 `test_compliance_regression.py`，不因旧考生失误反向改变 WP18 公开 86 个样例或历史 verdict。
- 定性：Contract 与 Selection 正确，主要为 `IMPLEMENTATION_ERROR`，而公开测试未检出为 `ENFORCEMENT_GAP`。跨平台路径、不同形式凭据和数值有限性是已有 Contract 的延伸，而非为考试随意增加全新要求。独立验收的具体数据和证据仍需由 reviewer 冻结、保留。
- WP19 路线：P0 H27–H46 有源参考、精确输入输出和公开可执行检查，仍需后续不同数据的 Mutation/metamorphic 及 reviewer 核验；H47–H226 尚未 READY，不可让考试 Agent 编写、改写或替代它们的评分器。

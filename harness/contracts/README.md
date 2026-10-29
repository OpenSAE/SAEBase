# Universal Task Contracts · SAEBase Harness v1

Harness 的通用 Contract **在任务出现之前**定义于 [catalog.json](catalog.json)，由治理维护者版本化；任务 Agent 和 Work Plan 只能选择已有 ID，不能在任务中临时修改、弱化或重新定义合同。它不是针对 WP18 六个例子的解题脚本，也不保证预见世上每种算法的专属正确性条件；它覆盖任意任务都需要考虑的**跨任务约束类别**，具体算法的确定性输入输出仍由 Work Plan 声明。

## 合同层级和原则

1. **Always-on**：`CT-BASE`、`CT-SCOPE`、`CT-TRUTH`、`CT-CHECK`、`CT-TRACE`。任何任务都继承，Work Plan 不能取消。
2. **Conditional**：证据 `CT-EVIDENCE`、隐私 `CT-PRIVACY`、树结构 `CT-STRUCT`、审阅 `CT-REVIEW`、双语 `CT-LOCALE`、知识 `CT-CONTEXT`、AST `CT-AST`、环境 `CT-ENV`、确定性 `CT-DETERMINISM`、接口 `CT-INTERFACE`、错误语义 `CT-ERROR`。
3. **Work Plan binding**：每个 Work Plan 明确引用固定 ID；每个 CASE 再声明 task tags 和对应合同 ID。解析器拒绝未知 ID、缺少覆盖、重复 ID、越权 Work Plan 路径。合同列表由机器生成并用于实施前检查与独立验收，不能让 Agent 自己改变裁判标准。
4. **不可放宽**：局部要求可以比目录更严格，不能把 Always-on 或已选 Contract 改弱。如发现目录缺陷，提交治理建议，只有可信维护者修改 catalog/validator 和对应独立测试。
5. **未知新任务**：人类仍可直接说“测 Harness 的 X”。Agent 先从**现成 tags**选择合同，落在 `harness/tests/` 做隔离测试并记录选择；这不是正式产品授权。需正式 Work Plan 的任务，治理方再把 tag/ID 写入受保护的 Work Plan 文件。无合适的 Contract 时报告 `CONTRACT_GAP`，不能默默自创或跳过。

## 输入合同和实际验证

`catalog.json` 是规范化注册表。只读选择器：

```bash
python -m harness.contracts.registry list
python -m harness.contracts.registry check --plan harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md
python -m harness.contracts.registry resolve --plan harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md --case H04
```

对于自定义测试，按 `catalog.json` 的固定 tags 选择 `CT-` ID；不能改变目录。独立工具 `harness/contracts/invariants.py` 提供可复用来源验证、完整结构脱敏、Tree 身份和中英完整性检查的参考实现，属于**治理层**，不让被测 Agent 为通过测验而编辑。

`harness/contracts/trace.py` 可检查每个 Run 的 `contract_receipt.json` 中 ID、版本、文件范围与动作记录的结构完整性。Receipt 不会自动证明命令真的运行、文件真的被读取、Agent 看过所有内容或人类已批准；验收者须另外核对 Git diff、真实 AST、CI 日志及审批事件。

## Work Plan 声明（放在具体 Work Plan 中）

下面是格式示例，**不是另一个合同目录**：

```harness-contracts-json
{
  "registry": "saebase.harness.contract_registry.v1",
  "plan": ["CT-ERROR"],
  "cases": {
    "H01": {
      "tags": ["evidence", "interface"],
      "contracts": ["CT-EVIDENCE", "CT-INTERFACE"]
    }
  }
}
```

选项 `plan` 是全计划共同附加合同，`cases` 为逐任务清单。无论省略与否，5 个 always-on 始终由解析器补齐。读入顺序：全局常驻 Rule → Work Plan 声明 → 此 CASE 的 `resolve` 输出 → 最小必要业务 Context → 代码及测试。不要递归加载整个 Core、Knowledge、Glossary。

## Execution Trace / Failure Attribution

每个测试 Run 写 `contract_receipt.json`，至少包含 schema、contract_ids、registry_version、case_id、context_paths、edited_paths、commands（argv+exit_code）、unresolved。可选记录 source SHA、Context ID、工具拒绝/失败原因，但不要求或保存 Agent 私有推理。独立验收记录：

- `CONTRACT_GAP`：确有适用的约束，现有合同目录没有表达。
- `SELECTION_GAP`：目录已定义但 Work Plan 未引用/Agent 未加载。
- `IMPLEMENTATION_ERROR`：规范明确且可见，Agent 实现仍违反。
- `ENFORCEMENT_GAP`：Agent 产出错误但自动校验未阻止或标记。
- `EVALUATOR_SCOPE_GAP`：评审新加的检查超出原冻结合同。
- `ENV_BLOCKED`：权限、依赖、设备或文件系统限制使任务不能执行。

修复责任在 Harness 治理维护者；测试 Agent 只按当前规则执行并提交可观察证据。验收阶段保持原始 Agent 分支不变，独立评审在单独分支验证。下一轮使用新的同类任务检查改进的泛化能力，而非仅要求旧失败变绿。

## 轻量 Task IO 审计及独立 Acceptance

`selection.audit_selection(contract_ids, task_signals)` 只核对明确声明的输入、输出和操作信号（例如 `source_reference`、`sensitive_output`、`tree_nodes`、`bilingual_fields`），不猜测自然语言的隐含意思。未知信号停下报告 `CONTRACT_GAP`，遗漏适用合同标记 `SELECTION_GAP`。Work Plan 原有 Contract binding 不变，历史 Work Plan 不要求迁移。

治理层 `python -m harness.contracts.acceptance --plan <plan> --case <case> --receipt <run/contract_receipt.json> --evaluator <acceptance/evaluation.json>` 读取独立评审者写在受保护 `acceptance/` 的记录，核对真实输出、负面测试状态、source_verified 和既有 Invariants。该 gate 不运行 agent 代码，不相信 agent 自评，不提供审批；成功仅表示技术检查完成、**仍需人类审阅**。评审者须在隔离环境自己运行测试、检查 Git diff 和 AST，才能写上述记录。Hooks 不是 OS 沙箱。

## WP18 H07–H26 · 先验规范与可执行 Verification

2026-10-29 治理更新：`catalog.json` v1.1 新增六个跨任务 Contract：`CT-DATA`、`CT-CONFIG`、`CT-RESOURCE`、`CT-RELEASE`、`CT-RECOVERY`、`CT-NAMESPACE`，历史 Work Plan 的兼容性保持不变。新 20 题不是 20 个临时 Contract，而是引用固定的现有/新增通用 ID。

- `wp18_cases.json`：20 个任务的目标、规则、选择 tags、signals、测试数据与预期结果，当前冻结 86 个正/反验证用例。
- `wp18_verify.py`：按 Work Plan 解析并验证每题所选 Contract，复核实际 `solve(payload)` 的结果/拒绝、输入不变性、稳定输出和超时；并在考试 PR 中进行真实 diff 的治理范围审计。
- `wp18_worker.py`：隔离子进程接口，输出结构化结果，不打印候选异常中的敏感值；子进程不是安全沙箱。
- `harness/github_work_test/test_wp18_h07_h26_contracts.py`：独立回归，确保 20 个任务、全部合同绑定和验证器存在且有效。

Agent 可 **读取** 上述全部资料。只可新增自己的 `harness/tests/wp18/runs/<run_id>/<case>/`，不能编辑任何规范、验证或治理源码。独立 CI 从实际 PR diff 检查修改范围，并实际执行新 HW 的可见验证；另由独立评审运行不同输入与人工审阅。失败先归因，不给任务 Agent 放宽验证标准的权限。

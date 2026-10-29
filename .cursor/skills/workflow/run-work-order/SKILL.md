---
name: run-work-order
description: 用户指定 OpenResearch Work Order，或需要选择 Core、科研 Knowledge、Glossary 和 Code Tree 等 Harness 工具时使用。
---

# 执行 Work Order v3

**通用 Contract 是先验约束，不是执行者根据代码临时补写的测试答案。** 正式任务应在 Work Plan 声明并绑定 Contract ID，使用 `harness/contracts/registry.py` 的只读解析器核对；新 Work Order 可携带 `contract_binding` 与选择的 ID。原有不包含绑定的旧 Work Order 保持兼容，但标注为 legacy，不能声称其 Contract 选择已经验证。


1. 只读取用户点名的 `.cursor/work_orders/<ID>.yaml`，运行 `python .cursor/skills/workflow/run-work-order/scripts/validate_work_order.py <ID>`。未成功验证，不按该工单执行。
2. 区分两个不同的结构：`core_ids` 描述**项目架构/研发管理**，通过 `python harness/core/select.py <IDs>` 获取；`knowledge_ids` 描述 **LLM SAE 研究问题与源码依据**，通过 `harness/knowledge/select.py <IDs>` 按需读取。
3. 读取 `context.doc.required`；`optional` 文件仅在当前任务需要时读取。涉及科学概念必须服从 `doc/knowledge/glossary.json` 和 `foundations/`；不得重新发明同义术语。
4. 仅执行显式 `activation.<module>=true` 的功能：`core`、`code`、`knowledge`、`experiments`、`transfer`、`environment`。否则不主动遍历该目录。参考区 `harness/github_work_plan/` 只能按 `context.harness.read_only` 中的具体文件只读参考。
5. `activation.code=true` 且涉及修改代码时，先运行 `harness/code/build.py` 保存 Before，记录 Expected（Core ID 与 Knowledge ID 各自准确），完成后重新 AST 构建并使用 `harness/code/diff.py` 比较。不能拿 Expected 冒充 Observed。
6. `activation.knowledge=true` 允许提出 Knowledge 或 Glossary 的修改提案，但不默认覆写 `doc/knowledge/`。若发现科学知识缺口，向 GitHub 管理方报告“哪个术语、定义、证据与关联目标”。
7. `activation.core=true` 允许读指定项目设计目标并报告状态变化提案；不要把 Core 的工程里程碑标记为科研成果。
8. 实验运行遵守 `saebase/runtime/experiment_paths.py`，成果报告参考已有 Knowledge ID；Environment 仅采集本机真实信息，Transfer 默认 dry-run，均不得擅自超出任务预算。
9. 最终报告列出：已激活模块、使用的 Core/Knowledge/Term ID、实际阅读的文件、修改前后 Code Tree 差异、已执行的测试与尚未验证的科研结论。GitHub commit 本身不会自动唤醒本地 Cursor。

## Harness 测验的最简入口

在写测试实现之前，先按 WP18 为当前 CASE 解析已有合同（全局 Always-on + 本题 Conditional），从中派生输入约束与正反测试。解析输出是固定规则，Agent **不能**修改 Work Plan/Registry/治理验证器。不要求人类输入任何命令。
运行产生 `contract_receipt.json`，记录选择的 IDs、实际阅读文件、编辑路径、测试命令/退出码和 unresolved。它属于自述证据，独立验收者再与仓库及执行日志对照。


当人类在 Cursor Agent 模式里明确输入“测试 Harness H01”、“测试 Harness 的审阅队列”等自然语言指令，或“测 WP18 H01”，Agent 可直接在 `harness/tests/` 的本次独立测试空间中编写模拟功能、测试、候选 Tree 和报告，并运行仅针对该空间的 pytest。**不需要**人类输入 Bash、检查路径白名单或创建 Agent Flow。可以自由接受人类指定的任何 Harness 测试主题，不局限于 WP18 六题；未知主题先根据职责选一个可读的短名称。

只能修改 `harness/tests/`，不得借此修改 `harness/core/`、`harness/visualization/`、`.cursor/`、`.github/`、Runtime 或正式 Core/Knowledge/Code/Context 权威图。预览的 Tree 是待审提案，独立验收方从实际源码/fixture 另行验证。

如任务必须修改受保护的正式实现，停止在测试 sandbox，输出清晰的治理变更建议，交由独立维护者审批；人类如需批准正式修改，走正式 Agent Flow，不可把聊天中的普通测验指令伪作 `authorize`。

## Reviewed Agent Flow (外部 GitHub Work Plan)

- 对用户明确指定的 GitHub Work Plan，先制作用于 `harness/core/agent_flow.py intake` 的结构化更改清单。不得运行标记为 DEFERRED / NOT_AUTHORIZED 的存档计划。
- 预览使用 Atlas 本身的 `knowledge.html?review=<PlanID>`（本地以 `serve` 启动），红色是计划目标，不是已运行结果。先取得用户对特定文件范围的直接授权。
- 编码前运行 `scope`，更新后运行 `stage --changed ... --evidence ...`；实际实验结果通过 `observe` 登记证据 SHA，再请人类逐 Node 作出决定。
- 绿色只来自有结果证据的人类审批；保留黄色表示待下一次实验；回退红色表示更换实现；删除是人类主动废弃该提案。
- Agent 不能阅读/修改 Core 治理 Tree 或 Harness 规则；非当前任务关联的改动必须另行提出 scope 授权。

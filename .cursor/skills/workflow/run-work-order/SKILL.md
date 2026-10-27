---
name: run-work-order
description: 用户指定 OpenResearch Work Order，或需要选择 Core、科研 Knowledge、Glossary 和 Code Tree 等 Harness 工具时使用。
---

# 执行 Work Order v3

1. 只读取用户点名的 `.cursor/work_orders/<ID>.yaml`，运行 `python .cursor/skills/workflow/run-work-order/scripts/validate_work_order.py <ID>`。未成功验证，不按该工单执行。
2. 区分两个不同的结构：`core_ids` 描述**项目架构/研发管理**，通过 `python harness/core/select.py <IDs>` 获取；`knowledge_ids` 描述**CUB-200 ResNet 科研问题与目标**，通过 `python harness/knowledge/select.py <IDs>` 获取；`term_ids` 用 `python harness/knowledge/terms.py <IDs>` 获取统一术语定义。只读指定 ID，避免上下文污染。
3. 读取 `context.doc.required`；`optional` 文件仅在当前任务需要时读取。涉及科学概念必须服从 `doc/knowledge/glossary.json` 和 `foundations/`；不得重新发明同义术语。
4. 仅执行显式 `activation.<module>=true` 的功能：`core`、`code`、`knowledge`、`experiments`、`transfer`、`environment`。否则不主动遍历该目录。参考区 `harness/github_work_plan/` 只能按 `context.harness.read_only` 中的具体文件只读参考。
5. `activation.code=true` 且涉及修改代码时，先运行 `harness/code/build.py` 保存 Before，记录 Expected（Core ID 与 Knowledge ID 各自准确），完成后重新 AST 构建并使用 `harness/code/diff.py` 比较。不能拿 Expected 冒充 Observed。
6. `activation.knowledge=true` 允许提出 Knowledge 或 Glossary 的修改提案，但不默认覆写 `doc/knowledge/`。若发现科学知识缺口，向 GitHub 管理方报告“哪个术语、定义、证据与关联目标”。
7. `activation.core=true` 允许读指定项目设计目标并报告状态变化提案；不要把 Core 的工程里程碑标记为科研成果。
8. 实验运行遵守 `dlbase/experiment_paths.py`，成果报告参考已有 Knowledge ID；Environment 仅采集本机真实信息，Transfer 默认 dry-run，均不得擅自超出任务预算。
9. 最终报告列出：已激活模块、使用的 Core/Knowledge/Term ID、实际阅读的文件、修改前后 Code Tree 差异、已执行的测试与尚未验证的科研结论。GitHub commit 本身不会自动唤醒本地 Cursor。

## Reviewed Agent Flow (外部 GitHub Work Plan)

- 对用户明确指定的 GitHub Work Plan，先制作用于 `harness/core/agent_flow.py intake` 的结构化更改清单。不得运行标记为 DEFERRED / NOT_AUTHORIZED 的存档计划。
- 预览使用 Atlas 本身的 `knowledge.html?review=<PlanID>`（本地以 `serve` 启动），红色是计划目标，不是已运行结果。先取得用户对特定文件范围的直接授权。
- 编码前运行 `scope`，更新后运行 `stage --changed ... --evidence ...`；实际实验结果通过 `observe` 登记证据 SHA，再请人类逐 Node 作出决定。
- 绿色只来自有结果证据的人类审批；保留黄色表示待下一次实验；回退红色表示更换实现；删除是人类主动废弃该提案。
- Agent 不能阅读/修改 Core 治理 Tree 或 Harness 规则；非当前任务关联的改动必须另行提出 scope 授权。

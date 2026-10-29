# WP18 测验前置缺陷、流程修正与授权操作手册

状态：**OPEN / PRE-FLIGHT BLOCKED** · 2026-10-28 · 由独立规划侧核对当前 SAEBase main 后建立。  
关联：[WP18](18_harness_local_agent_examination_and_independent_acceptance.md)。本文件记录**尚待修复的设计/治理缺口**，不是 H01–H06 的执行授权，不表示任何考试已经开始或通过。

## 1. 本地 Agent 反馈核查结论

Agent 关于「WP18 仍为 PLANNED、Ask 模式不能编辑、Hooks 需要已经授权的 Agent Flow、普通任务 Agent 不能自行修改治理文件、不能自行验收、六题应该隔离」的描述**基本符合现有源码**。这些是合理的门禁与职责约束，并非应当移除的障碍。

但原 WP18 **确实缺少可操作的人类授权入口和逐题派发材料**。这属于规划设计缺口，不能把开考失败归咎于 Agent。未完成下面 P0 项时应统一登记 `PRE-FLIGHT BLOCKED`，不对 Agent 扣分。

## 2. 缺陷登记和拟议改动

| ID | 优先级 | 核实的事实 / 设计缺陷 | 拟修复与验收条件 | 负责人 |
| --- | --- | --- | --- | --- |
| GATE-01 | **P0** | `agent_flow.authorize(obj, actor, human=True, channel=...)` 存在，但 `python -m harness.core.agent_flow` CLI 只有 intake/inspect/browse/preview/scope/stage/observe/finalize/serve，**没有可信的 preflight authorize 操作**。 | 增加由真实人类触发、绑定会话/身份和 `preview_sha256` 的预审入口；不可由 Agent 调用；审批后 `phase=authorized`，错误/伪造请求与过期 SHA 必须拒绝。 | 治理维护者 |
| GATE-02 | **P0** | `agent_review_server.py` 的 POST 只有 `/api/review/decision`，处理节点审阅，**不是开考前的人类文件/命令授权**；`github_review.py` 支持 GitHub 审批回执，但当前 `.github/workflows/*.yml` 未找到调用处理器或 `pull_request_review` 事件。 | 二选一：接入受信任的本地预审界面，或安全接通 GitHub 一文件授权回执工作流（核对 reviewer、SHA、scope、revision）。跑真实一轮批准/拒绝流程和伪造拒绝测试，再称为「可开门」。 | 治理维护者 |
| WO-01 | **P0** | `validate_work_order.py` 的 `harness.read_only` 前缀误写 `harness/github_harness/github_work_plan/`，存在性检查却使用 `harness/github_work_plan/`；Skill 的脚本路径误指向 `run-work-order/saebase/tools/...`。 | 修正为现有真实路径 `.cursor/skills/workflow/run-work-order/scripts/validate_work_order.py` 与 `harness/github_work_plan/`，增加合法引用/越权/路径逃逸用例。修复前不能把 Work Order 校验成功当作授权依据。 | 治理维护者 |
| GATE-03 | **P0** | WP18 只有泛化路径和题目描述，缺少可直接 `intake` 的**逐题 v1 manifest**、可检查的 `changes` 与精确允许命令；当前 `agent_flow.intake` 要求每题有 1..300 个合法、source-backed 的 Code/Knowledge/Context/Environment 变更声明。 | 独立预审者为 H01–H06 各制备单独 manifest、合法新 Code `node_id`、实际 `source_path`、`after`、`files`、`allowed_paths`、`authorized_commands`；在干净测试分支预演 intake→preview，不批准之前不允许 Agent 改文件。 | 考试协调者 |
| PLAN-01 | P1 | 「六题一次执行」与 WP18「每题从冻结基线独立工作树、独立授权」冲突。单个 Agent 会话可以逐题继续，但**不能共用同一个工作树/授权/测试缓存**。 | 默认逐题 H01→独立验收→H02；若人类以后批准批量执行，也必须使用六个隔离 worktree、六份精确授权、六份报告，禁止用一个白名单覆盖全部。 | 考试协调者 |
| PLAN-02 | P1 | 候选 `review_candidate.json` 若要求填写**其自身最终 commit 的 `candidate_head_sha`**，会发生提交 SHA 自指循环：写入 SHA 又改变提交。 | 候选中记录 `source_content_sha256`（代码内容摘要）、`base_sha`，`head_sha` 由提交后独立验收者从 Git 记录，不要求 Agent 在同一提交中的文件写入自身 SHA。 | WP18 规划侧 |
| PLAN-03 | P1 | WP18 的「Agent 生成可供人看的树」容易被理解为允许编辑权威 Atlas/Code Tree。但 `guard_paths.py` 硬保护 `harness/`、`.cursor/`、`.github/`、`saebase/runtime/`，正常的 AST/Atlas 生成命令也可能被 `guard_shell.py` 拦截。 | Agent **只提交候选 Tree、Expected Diff、自己写的源码及自测**；独立验收者运行 Code/Atlas 构建器生成 Observed，测试独立来源，不放宽治理权限。 | 考试协调者 |
| PLAN-04 | P1 | 「保存供人审阅的 Tree」和「最终 Approve/绿灯」是不同层级。正式源码 AST 节点可以真实存在；候选审阅决定仍是 `unchecked`。 | 将候选节点的“人类决定状态”和源码事实分开；Agent Flow 不得由 Agent 人为伪造 `human=True`，不得把批准空节点当成功能/科研验证。 | 评审者 |
| DOC-01 | P2 | 历史文档中的 `tests/test_harness_migration.py` 等路径已经过时，实际位于 `harness/github_work_test/`；部分文档仍混用旧 OpenResearch 路径。 | 治理方逐一校正引用并增加路径存在性检查，避免 Agent 因文档失效被误判。 | 文档维护者 |
| EVAL-01 | P1 | 原计划要求隐藏反例、独立验收，但没有给出可复用的**外部验收者 runner**及带 SHA 的评分产物。 | 创建隔离 evaluator fixture/命令/报告 schema，只能由独立验收者运行；Agent 不能编辑、执行以自批、改变判分逻辑或宣称最终 PASS。 | 独立验收者 |
| EVAL-02 | P1 | 若基线门禁或 Work Order 本身失败，Agent 无法实现。将这个状态算成 H 题失败是不公正的。 | 区分 `BASELINE_DEFECT`、`BLOCKED_BY_AUTH`、`NOT_RUN`、`FAIL`；缺乏可用的预审入口不能消耗考试次数或得分。 | 独立验收者 |

**保持不变**：禁止 Agent 编辑治理/权威 Tree、拒绝未授权路径、同一任务最小 Context、禁止自动绿色结果、保留 Hooks、零 SAE 训练/GPU/模型权重/科研实验、独立评分，以及不自动推送生产分支。

## 3. 当前正确的开门步骤

**现在没有已经验证的完整开门路径。** 不能只凭聊天里的“授权六题”使 `guard_paths.py` 放行；也不能只设置环境变量、切换 Agent 模式，或运行一条把 `human=True` 硬编码进调用的临时 Python 代码来伪造独立审批。

### 第一步：你可以立即做的事（不解除 Harness 门禁）

1. 在本地 Cursor 的模式选择器中将 **Ask → Agent**；这只允许 Agent 执行编辑操作，**不**自动授予仓库治理写权限。
2. 请本地 Agent 仅报告 `git status --short`、`git rev-parse HEAD`，确认干净工作区和准确仓库版本。不要直接运行六道题。
3. 明确派题 H01，`RUN_ID=wp18-h01`；目前先作为意向，不当作激活的 `OPENRESEARCH_ACTIVE_AGENT_FLOW`。

### 第二步：由可信治理维护者完成一次性门禁修复

1. 在**单独治理 PR** 中处理 GATE-01、GATE-02、WO-01，保留并增强现有拒绝条件；不由参加考试的任务 Agent 修改。
2. 给预审入口增加明确的人类身份、计划摘要展示、固定 `allowed_paths` / `authorized_commands` 展示、一次确认/拒绝、revision/哈希校验、执行收据。
3. 外部运行测试：合法人类批准通过；匿名/Agent 调用、篡改文件、错误 SHA、未列文件、超过命令范围、过期授权全被拒绝。通过后合并治理修复并冻结新的基线。
4. 由考试协调者为 H01 写一份单独 v1 manifest 和相应 Work Order（如用 v3）；确保**每一个**要生成的文件有精确路径，命令逐字符串列出，没有通配目录。入场前先验证 Work Order 和 manifest。

### 第三步：门禁修复后的单题操作（命令供可信协调者使用）

在干净的 H01 工作树执行：

```bash
git status --short
git rev-parse HEAD
python .cursor/skills/workflow/run-work-order/scripts/validate_work_order.py WP18-H01
python -m harness.core.agent_flow intake tests/harness_exam/manifests/wp18_h01.json --source harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md
python -m harness.core.agent_flow preview WP18-H01
```

其中 `WP18-H01.yaml` 和 `wp18_h01.json` **尚未创建**，这些命令是经过治理修复、协调者建立文件以后才能执行的顺序，不是当前已经可以复制执行的现成命令。通过 **新补的可信预审入口**，由你逐条核对并批准 `preview_sha256`、文件清单和自测命令。批准后再由协调者确认：

```bash
python -m harness.core.agent_flow inspect WP18-H01
python -m harness.core.agent_flow scope WP18-H01 saebase/tools/harness_trial_evidence.py tests/harness_exam/test_h01.py
```

须在输出中实际看到 `phase: authorized`、`authorization.preview_sha256 == preview_sha256`，并确认已批准精确命令与路径。此时才允许在**该任务 Agent 的本地进程**选择对应账本：

- 环境变量：`OPENRESEARCH_ACTIVE_AGENT_FLOW=WP18-H01`。
- 在 Cursor Agent 模式执行且仅运行该题获准的命令；授权的范围不自动延伸到别题。
- 如被 Hook 拒绝，原样报告拒绝原因，不修改 Hook 或放宽 `policy.json`。
- Agent 提交源码、自测、候选审阅 Tree / Expected Diff 后停止，外部验收者再做实际 AST/Atlas 和评分。

在 Windows PowerShell 可由可信操作人员在启动 Cursor 所用的进程环境设置 `$env:OPENRESEARCH_ACTIVE_AGENT_FLOW = "WP18-H01"`；在 bash/zsh 中可用 `export OPENRESEARCH_ACTIVE_AGENT_FLOW=WP18-H01`。**环境变量仅引用已经授权的账本，不是授权本身**；如果 Cursor 进程没有继承它，Hook 仍会拒绝，应检查进程继承而非绕开 Gate。

### 第四步：正式验收

Agent **不**运行外部隐藏测试、不批准自己的候选 Tree；独立评审者在单独工作树从源码重建 Code Tree 和 Atlas，核对 Expected/Observed 与权限记录，做出逐节点人类审阅及最终评分。隔离工作树的本地模拟结果不得当作生产功能或科学效果。

## 4. 推荐的第一张正式试卷

`H01` 文件证据索引：这是最小、最适合验证门禁和闭环的第一题。协调者应精确允许如下五个文件（需固定的 CASE/RUN_ID），禁止目录级通配符：

```text
saebase/tools/harness_trial_evidence.py
tests/harness_exam/test_h01.py
tests/harness_exam/review_packets/wp18-h01/H01/review_candidate.json
tests/harness_exam/review_packets/wp18-h01/H01/expected_code_diff.json
tests/harness_exam/review_packets/wp18-h01/H01/selfcheck.json
```

`authorized_commands` 最少应覆盖对应 `pytest` 单测、必要的最小范围 Git 检查；提交代码/PR 的方式需要协调者预先核准。评审者的权威 Tree 生成和隐藏负面用例不放入上述 Agent 授权范围。

**开考判据**：门禁入口已经实际通过授权/拒绝测试，Work Order 路径修复，H01 manifest、单题隔离 worktree、批准记录、命令范围均存在。未达到该条件，记录 `PRE-FLIGHT BLOCKED`，不应让 Agent 将人为阻塞当作考试不及格。

## 5. 追踪与后续实施原则

本文件记录问题和建议，**没有修改任何 `.cursor` / `harness/core` 治理代码，没有创建工作单、没有执行预审，也没有启动 H01–H06**。治理代码修复需要单独 PR、独立代码审阅和人类明确同意，不能由任务 Agent 自行修复。每处理完一个 ID，更新此登记的状态、修复提交、自动测试结果及真正的审批凭证，不可以只凭文字宣称问题消失。

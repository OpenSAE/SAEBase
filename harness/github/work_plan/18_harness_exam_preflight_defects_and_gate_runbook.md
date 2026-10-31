# WP18 — 简化开考、测试门禁与历史缺陷登记

状态：**测试工作区与 Contract Registry 已合并；新版独立验收门禁待 CI 验证后使用**。这是一份面向用户和本地 Agent 的操作说明；它**不是**生产 Harness 权威代码写入授权。

## 1. 你应该打什么

在 Cursor **Agent 模式**（不是 Ask）里直接输入：

- `测 Harness H01`
- `测 WP18 全部六题`
- `测试 Harness 的树节点去重规则`
- `停止 WP18`
- `查看 WP18 结果`

前面三种都属于 **Harness 测验请求**。人类不需复制 Bash、填写 YAML、抄 SHA、逐一确认允许的文件、设置环境变量或打开 GitHub 审批页面。Agent 收到后在 `harness/tests/wp18/runs/<run_id>/<case>/` 中准备隔离实现与单测，自动运行本题 pytest，输出候选审阅 Tree 和报告，等待外部独立验收。自定义命题也可以按此方式实施。

**Harness 测验 sandbox 的门已通过路径级别小范围例外预置，不等于把所有 Harness 治理门禁解锁。**

## 2. 为什么普通 Harness 测验不再要求 Agent Flow 授权

原 WP18 设计把“改生产 Harness”所需的严格 Agent Flow 误套到了“只在 sandbox 写测试代码”上，导致还没开始就要求人类逐题准备 manifest、白名单、SHA 和手动 CLI 授权。这不符合方便地进行测验的目标。

现调整为两个等级：

| 场景 | 允许范围 | 用户操作 | Agent 限制 |
| --- | --- | --- | --- |
| 仅测 Harness | `harness/tests/` 的本次 Run | 在 Agent 模式说“测 Harness …” | 可写实现、fixture、自测、候选 Tree 和报告；不准改正式权威图、受保护代码或独立评分 |
| 真正修改 Harness/Core/Rules/CI/Runtime | 受保护的生产目录 | 单独提出“请修正式 Harness 的 …” | 测试 Agent 仅复现、形成治理修改建议；需可信治理维护者独立授权、审查和提交 |
| 真实 SAE 实验 | 生产运行路径及已授权结果目录 | 单独提出明确实验任务与预算 | WP18 不启动 SAE、GPU、权重或产品发布 |

**没有通用开关可以安全地把整个 `harness/` 设为 Agent 可写。** 如生产治理修复需要执行，应由可信维护者通过独立 PR/审阅方式实施，而不是让本地考试 Agent 修改 `guard_paths.py`、调用 `human=True` 或关闭 Hooks。

## 3. 文件具体在哪里

- 计划：`harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md`。
- 测试执行规则：`harness/tests/README.md`，WP18 专属提示：`harness/tests/wp18/README.md`。
- 单题实现/自测/fixture/候选树/报告：`harness/tests/wp18/runs/<run_id>/<case>/`。
- 人类与独立验收方记录：`harness/tests/wp18/acceptance/<run_id>/<case>/`，禁止考试 Agent 写入。
- 受保护的回归测试：`harness/github_work_test/test_wp18_sandbox_gate.py`，由 CI/治理维护者验证，考生不修改。

无归属的、必须新建的 Python 测试代码可以按另行授权使用根目录 `tests/`；真实 SAE 实验输出使用约定 `experiments/`，但 WP18 不触碰这些。无论何种指令，Agent 不得任意创建其他项目根一级目录；已有具体模块代码依归属子路径组织。**Cursor Hooks 是工具门禁，不等于 OS 级权限隔离；绝对隔离需单独容器/低权限工作树。**

## 4. 旧缺陷的处理记录

| 记录 | 旧状况 | 现在的安排 |
| --- | --- | --- |
| GATE-01 | Agent Flow 只有代码层 `authorize`，CLI 没有安全的人类授权入口 | **仅生产治理仍需独立审查**；WP18 沙盒测验不再调用这个入口，也不假称已有可信的通用解锁命令 |
| GATE-02 | 本地审阅服务只有 `/api/review/decision`，GitHub 回执处理器无已验证的 Workflow 连接 | 仍是生产审批能力的已知缺口，不阻塞测试沙盒；不得把审阅 POST 误当生产授权 |
| WO-01 | Work Order Skill 脚本路径错误，校验器 `harness.read_only` 前缀错误 | 在本次治理改造中更正，另加独立回归；测试 sandbox 不依赖 Work Order |
| PLAN-01 | 六题共用一次审批/工作树与逐题隔离冲突 | 每题单独 Run 子目录和结果，允许一个自然语言命令顺序完成六题；独立评审从冻结源码和每题目录复现 |
| PLAN-02 | 候选 Tree 在自身 commit 中写 `candidate_head_sha` 会自引用 | Agent 写源内容哈希，最终 Git SHA 由独立验收者记录 |
| PLAN-03 | “生成可审阅的树”被理解为改正式 Atlas | Agent 只写 sandbox 中的 `review_candidate.json`；真实 Code/Atlas 由独立验收方从 AST 重建 |
| DOC-01 | 历史文档指向旧 `tests/test_harness_migration.py` 等路径 | WP18 的新路径指向 `harness/tests/`；历史独立 CI 测试仍在 `harness/github_work_test/`，不伪称已全部清理旧文档 |
| EVAL-01 | 独立评估器和评分收据尚未真实执行 | 评分条件已写在 WP18，后续考试时独立生成并保存在受保护的 `acceptance/`，不能称考试已经通过 |

## 5. 开始前与结束后

**本次治理 PR 合并且 CI 的测试门禁检查通过后**，人类切换到 Cursor Agent 模式，只需发“测 Harness H01”。Agent 应自主完成测验；如想测其他 Harness 规则，用自然语言直接说明。完成时只需向人类返回本次 `run_id`、具体文件目录、可见自测状态和待人审内容。

如果 User 想真正“让 Agent 修改正式 Harness”，那是与本测验不同的高权限开发请求。此时人类只需说希望修改的功能，**不应由人类手工准备 Bash**；由可信治理方据此生成 PR / 预审资料，再由人类审阅。当前没有可声称已经验证的“一句话解除所有治理权限”机制。

任何考试 Agent 不得自行给自己最终 PASS，不得删除失败、覆盖上一 Run、写入独立验收记录或自动合并生产分支。

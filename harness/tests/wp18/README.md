# WP18 · 本地 Agent Harness 测验工作区

权威考题与验收细则：[`harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md`](../../github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md)。本目录的测验运行不改变生产 Harness。

## 一句话开考

你可以在 Cursor Agent 模式输入：

- **“测 Harness H01”**：只跑一个小题。
- **“测 WP18 H01–H06”**：六题顺序执行，每题独立目录，不在当前代码上相互覆盖。
- **“测试 Harness 的 X 功能”**：用户定定义新题，Agent 定己归纳一个具体测试目标和可核查的预期行为；仍只能修改本目录。
- **“停止 WP18”**：不再写文件，汇报当前结果和剩余工作。
- **“查看 WP18 交付”**：列出已有 Run 和报告路径。

不需要人类输入任何 Bash、手动设置 `OPENRESEARCH_ACTIVE_AGENT_FLOW`、核验哈希或逐一批准测试文件。测试空间是刻意收窄的预授权区域；人类明确说“测试……”就是选择测试主题。**如果要修改真正的 Harness 规则或权威代码，应独立发起治理请求，不能用这条简单入口绕开。**

## 每次运行结构

**新增必选步骤**：WP18 已在 Work Plan 中声明 H01–H06 的固定 Contract ID。执行 CASE 前由 Agent 解析 `harness/contracts/catalog.json` 的 Always-on 与当前任务选择；写 `contract_receipt.json` 到该 CASE 目录。不要让人类查 Bash 或 YAML；缺少合同/违反合同只能报告，不得改治理目录。自定义测试只可从同一目录选择适用 tags，并在收据注明选择依据。



每次新 Run 使用 `runs/<run_id>/<case>/`。短 `run_id` 可以来自日期加流水号，也可使用用户指定名称；须避免覆盖现有 Run。CASE 使用 `H01`–`H06` 或其他简短稳定名称；每题内部必须有 `implementation.py`、`test_implementation.py`、`review_candidate.json`、`expected_code_diff.json`、`selfcheck.json`、`contract_receipt.json`、`report.md`。必要的小型 fixture 放 `fixtures/`；不把每次运行都放到项目根目录。

- Agent 可直接实现测试功能并运行仅针对本 Run 的 `python -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/<run_id>/<case>/test_implementation.py`。
- Agent 必须在候选树中使用真实源文件、源码 ID、行号与签名。未确认的 Context 关联留空或标 unresolved；状态保留 unchecked，不能宣称树已通过人工审阅。
- Agent 提供清楚的完成摘要：做了什么、测试命令结果、哪些文件、哪些未通过、人工应审核什么。报告等于提交测验证据，不是独立验收。
- 验收者对照不同 Run、真实 AST、Rules、Context、翻译、权限及 UI，另行写 `acceptance/`，考生对它只有只读权限。

所有工具和文件路径沿用 `harness/tests/README.md`，不得替代正式 `.cursor` Hooks 或制作同名治理替身。CI / hidden evaluator 可以补充测试，但考生不可删除或更改外部测试来使自己通过。

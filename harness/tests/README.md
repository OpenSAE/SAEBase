# Harness Tests · 统一文件及交付约定

这个目录仅用于 **Harness 的设计/执行/规则/AST/Tree/Atlas/双语/审核流程** 等测试，不测试 SAE 的模型、算法、GPU 或 Research Lab 产品功能。

## 通用 Contract

测试任务必须遵守预先定义在 `harness/contracts/catalog.json` 的通用合同，Work Plan 指明各 CASE 使用的 Conditional IDs。Agent 不得为通过 H03–H06 而修改规则，只在新 Run 中交可观察的 `contract_receipt.json`，独立验收用真实源码、日志及新任务确认改进能泛化。

## 测试入口

人在 Cursor **Agent** 模式直接用自然语言发出测试指令，例如“测试 Harness H01”、“测试 Harness 的 Context 引用规则”、“运行 WP18 全部六题”。不需要确认 Bash、SHA、逐文件权限表或手动生成 Work Order。Agent 自行安排、测试并把交付位置告诉人类。

**可自动写入的唯一 Harness 子树：`harness/tests/`。** 在此可以建立代码、fixture、自测、临时报告与候选树；不能借此修改 `harness/core/`、`harness/code/`、`harness/visualization/`、`.cursor/`、`.github/`、正式权威树或 SAE Runtime。无法在测试空间实现的正式功能修改需要另立治理变更，由可信维护者处理。

## 标准位置

- `harness/tests/wp18/README.md`：WP18 专用规范，受保护，不由考生改写。
- `harness/tests/wp18/runs/<run_id>/<case>/implementation.py`：本次功能实现，小型纯函数，按题可拆成多个明确命名的模块，但不要新建项目根一级目录。
- `harness/tests/wp18/runs/<run_id>/<case>/test_implementation.py`：可重复的本地单元测试。
- `harness/tests/wp18/runs/<run_id>/<case>/fixtures/`：只包含小型合成测试数据，须标记 synthetic。
- `harness/tests/wp18/runs/<run_id>/<case>/review_candidate.json`：供人审阅的 Code/Context/Environment 候选 Tree，provisional、unchecked。
- `harness/tests/wp18/runs/<run_id>/<case>/expected_code_diff.json`：Agent 对新增/修改/删除 AST 的预测。
- `harness/tests/wp18/runs/<run_id>/<case>/contract_receipt.json`：实际选择的预声明合同及 Context/编辑/命令证据，不是最终评分。
- `harness/tests/wp18/runs/<run_id>/<case>/selfcheck.json`：命令、退出码、真实测试状态和失败。
- `harness/tests/wp18/runs/<run_id>/<case>/report.md`：一页式总结、证据链接、未验证项。
- `harness/tests/wp18/acceptance/<run_id>/<case>/`：独立验收者写入的 observed diff、scope audit、隐藏测试、审阅结论。**考生不可修改。**

不是 WP18 的 Harness 测试可新建 `harness/tests/<topic>/...`，沿用相同结构与命名规则。原有 `harness/github_work_test/` 为历史/CI 测试位置，保留但不作为新 WP18 考生输出位置。根目录 `tests/` 只存真正无法归属到现有模块的通用测试；优先将 Harness 测试放本目录。

## 根目录治理

**Agent 不得创建其他项目根一级目录。** 唯一例外：`experiments/`（经过授权的真实 SAE 实验结果，包含运行标识与 provenance）和 `tests/`（没有合理模块归属、但必须新增的通用测试 Python 文件）。两个例外也不自动授权生产实验或任意写入。

对于已有的 `datasets/`、`models/`、`tasks/`、`saebase/`、`doc/`、`harness/` 等目录，必须放在明确归属的子目录内。不要创造根目录的 `tmp/`、`results/`、`scripts/`、`utils/`、`output/`、`test_run/` 等通用文件夹。旧文档中的 `experiment/` 单数为历史 Runtime 路径；不要由测试 Agent 新建。若将来需改变 Runtime 默认路径，须走独立治理改动，不能靠 WP18 偷换。

文件名使用 snake_case 与稳定 `run_id` / `case`；不写大文件、凭据、个人主机记录或模型权重。源树节点必须来自真实 AST，Expected 与 Observed 分开，审阅候选不得宣称最终通过。

## 谁能验收

Agent 负责实现、执行可见自测、报告事实，不能宣布最终验收 PASS，也不能改 `acceptance/` 或由自己批准节点。独立验收者重新生成真实 Tree、运行隐藏反例、逐项审批并签署最终结论。Hooks 只是工具级保护，不是 OS 级沙箱；需要绝对防护时，使用独立容器或低权限工作树加只读治理层。

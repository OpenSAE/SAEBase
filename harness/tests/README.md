# Harness Tests · 统一文件及交付约定

WP18–WP20 的冻结考试用于 **Harness 的设计、执行、规则、AST、Tree、Atlas、双语及审核流程**。这些历史考试的允许范围、原始成绩与证据等级保持原样。WP21 在单独分派的真实产品工作树中测试 SAEBase；本目录保存它的规范、题目及运行证据，具体边界见本文末节与 [真实工作流协议](../contracts/REAL_WORKFLOW_PROTOCOL.md)。

## 通用 Contract

测试任务必须遵守预先定义在 `harness/contracts/catalog.json` 的通用合同，Work Plan 指明各 CASE 使用的 Conditional IDs。Agent 不得为通过 H03–H06 而修改规则，只在新 Run 中交可观察的 `contract_receipt.json`，独立验收用真实源码、日志及新任务确认改进能泛化。

## 测试入口

人在 Cursor **Agent** 模式直接用自然语言发出测试指令，例如“测试 Harness H01”、“测试 Harness 的 Context 引用规则”、“运行 WP18 全部六题”。不需要确认 Bash、SHA、逐文件权限表或手动生成 Work Order。Agent 自行安排、测试并把交付位置告诉人类。

**普通 Harness 合成测验可自动写入的唯一 Harness 子树：`harness/tests/`。** 在此可以建立代码、fixture、自测、临时报告与候选树；不能借此修改 `harness/core/`、`harness/code/`、`harness/visualization/`、`.cursor/`、`.github/`、正式权威树或 SAE Runtime。无法在测试空间实现的正式功能修改需要另立治理变更，由可信维护者处理。

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

对于已有的 `datasets/`、`models/`、`tasks/`、`saebase/`、`doc/`、`harness/` 等目录，必须放在明确归属的子目录内。不要创造根目录的 `tmp/`、`results/`、`scripts/`、`utils/`、`output/`、`test_run/` 等通用文件夹。当前 Runtime 的正式路径为单数 `experiment/`。WP18–WP20 的合成测验不因此获得真实实验权限；WP21 由治理准备过程预建独立工作树的受控 `experiment/` 根，再由被明确指派的题目使用现有 Runtime 规则写入其授权叶子。`experiments/` 的可用名称不会替代 Runtime 的实际路径。修改 Runtime 默认路径属于另行明确的治理范围。

文件名使用 snake_case 与稳定 `run_id` / `case`；不写大文件、凭据、个人主机记录或模型权重。源树节点必须来自真实 AST，Expected 与 Observed 分开，审阅候选不得宣称最终通过。

## 谁能验收

Agent 负责实现、执行可见自测、报告事实，不能宣布最终验收 PASS，也不能改 `acceptance/` 或由自己批准节点。独立验收者重新生成真实 Tree、运行隐藏反例、逐项审批并签署最终结论。Hooks 只是工具级保护，不是 OS 级沙箱；需要绝对防护时，使用独立容器或低权限工作树加只读治理层。

新版 WP18 使用已有 Contract Registry、`harness/contracts/selection.py` 和治理方单独运行的 `harness/contracts/acceptance.py`。Agent 仅声明已选合同与结构化任务 IO；独立验收方从实际输入输出重新确认信号，运行隐藏正反例，提交 `acceptance/.../evaluation.json` 并执行 Gate。考试不得修改独立验收器或复用旧 Run。

## WP19 考试授权

WP19 的题目范围、冻结状态与可执行入口以 [WP19 专用规范](wp19/README.md) 和 [WP19 Work Plan](../github_work_plan/19_harness_large_scale_examination_and_verifier_assurance.md) 为准。考试 Agent 只在新建的 `harness/tests/wp19/runs/<run_id>/Hxx/` 下交付实现、自测和报告；原有规范、验收、Contracts、治理文件和历史 Run 保持只读。运行不可信候选代码仍需低权限 reviewer 环境。


## WP21 真实 SAEBase 工作流

WP21 使用 [真实工作流协议](../contracts/REAL_WORKFLOW_PROTOCOL.md) 和逐题冻结范围。它是本地准备的新执行类型，未重新定义 WP18–WP20 的授权或历史结论。

- 每道题对应一个独立工作树，生产代码改在所归属的 `datasets/`、`models/`、`tasks/`、`conf/` 或 `saebase/` 文件中；题目回归放在指定 `tests/` 路径。`harness/tests/` 用于证据和报告，不作为真实算法的替代实现目录。
- 任务包的初始状态为 `prepared_not_assigned`。用户明确指派具体题目后，agent 依据真实指令和冻结 scope 执行；读取包或通过预检不产生人类批准。治理、裁判、历史记录和受保护 Runtime 语义保持只读，现有 Hook 的拒绝由维护者处理。
- 准备过程预建 `experiment/` 根；实际调用使用 SAEBase 路径函数与 Hydra。证据须区分配置检查、合成输入的真实接口执行和真实 checkpoint 执行。
- reviewer 的评分和验收目录位于考生工作树外；本地 Hook 同时增加当前 WP21 指派 profile，并将 acceptance 保护推广到各主题。考生自测和独立验收分别记录，WP21 receipt 使用独立 schema，保留旧考试验证器的沙箱范围。
- 新增与删除组件均使用题目给定的独立起点；保持其他组件可运行及历史实验可追溯。具体缺口及本地更新见 [WP21 规范审计](wp21/AUDIT.md)。

WP21 两轮真实考试已有 [round01 与 round02 结果及阶段结项](../github_work_plan/21_harness_phase_closeout_20261030.md)。所有历史判定保持不变；新测试须创建新 run_id 并独立验收。

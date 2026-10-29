# WP19 · 200 道 CPU 合成不变量 Exam（H27–H226）

全部 200 题已经预先冻结公开输入、输出、有效/无效样例及只读评分入口，合计 800 个检查。H47–H226 实现的是对应主题的**合成约束检查**，不是对真实产品科研/浏览器/多 Agent 场景的端到端认证。

参见 [WP19](../../github_work_plan/19_harness_large_scale_examination_and_verifier_assurance.md)、[全部规范](../../contracts/wp19_cases.json)、[初始参考](../../contracts/wp19_reference.py)、[新增共享不变量参考](../../contracts/wp19_reference_ext.py)。

考试 Agent 从最新 main 创建唯一 Run，按 10–20 题分批实现每题的 `solve(payload: dict) -> dict`。合法输入返回冻结预期输出，非法输入抛出 `ValueError`，输入对象不被修改。逐题在 `harness/tests/wp19/runs/<run_id>/Hxx/` 提交实现、pytest、fixtures、Contract Receipt、selfcheck、候选 unchecked Tree、expected diff 和报告。

允许只读使用 `python -m harness.contracts.wp19_verify --check-spec` 和 `python -m harness.contracts.wp19_verify --case H47 --run-dir harness/tests/wp19/runs/<run_id>/H47`。**禁止**修改 `harness/contracts/`、`.cursor/`、`.github/`、评分器、冻结样例、历史 Run、WP18、独立 reviewer 记录或正式源码。Agent 只能自测，不得自批或自行合并。独立审查者使用可信治理基线和不同输入复验，运行候选代码需低权限沙箱；Hooks / subprocess / CI 不能代替 OS 隔离。

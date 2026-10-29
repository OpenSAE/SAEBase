# WP19 · H27–H226 考试工作区

当前 P0 只开放 **H27–H46**；其余 180 题状态 `PROPOSED`。请阅读 [WP19](../../github_work_plan/19_harness_large_scale_examination_and_verifier_assurance.md) 和治理层 [WP19 case catalog](../../contracts/wp19_cases.json)。

考生读取但不修改 `harness/contracts/wp19_reference.py`、`wp19_verify.py`、`wp19_cases.json`、`catalog.json` 或独立评分；不修改 WP18 的历史考试。每题只在新的 `harness/tests/wp19/runs/<run_id>/Hxx/` 中写 `implementation.py`、`test_implementation.py`、小型 fixtures、`contract_receipt.json`、`selfcheck.json`、`review_candidate.json`、`expected_code_diff.json`、`report.md`。所有候选 Tree 节点仍为 unchecked，测试成功不构成批准。

P0 接口为 `solve(payload: dict) -> dict`；非法输入抛 `ValueError`，输入对象不修改。例子可读，可用 `python -m harness.contracts.wp19_verify --case H27 --run-dir harness/tests/wp19/runs/<run_id>/H27` 检验。合法的 `--check-spec` 不运行候选实现。独立 reviewer 仍需用**不同数据**验证，并从可信治理 commit 调用评分器，不接受候选分支中的同名评审程序。

这套 Hooks 是工具级防护，不是任意代码运行的 OS 沙箱。考试分支只提交 sandbox 新增文件，不自合并。真正环境、模型或 GPU 任务必须单独授权。

# H24 Checkpoint 恢复身份核对

本文件是自测记录。它不是独立验收结论，也没有把候选节点标成已审阅。

## 实现

`implementation.py` 提供 `solve(payload) -> dict`。合法输入返回 JSON 对象；非法输入抛出 `ValueError` 的子类 `InputError`，并且不改调用方传入的对象。额外样本在 `fixtures/extra.json`，与冻结用例的取值不同。

规则：Resume does not mutate state or silently merge unrelated runs.

## 合同

解析得到的 `contract_ids`：["CT-BASE", "CT-SCOPE", "CT-TRUTH", "CT-CHECK", "CT-TRACE", "CT-ERROR", "CT-RECOVERY", "CT-CONFIG", "CT-INTERFACE"]。

任务信号：["resume_receipt", "config_metadata"]。`selection_audit` 为空，没有 `SELECTION_GAP`。

## 实际观察

- 合同解析退出码 0。
- 单题 pytest：6 passed in 0.01s，退出码 0。其中包含冻结检查和两道额外自测。
- `wp18_verify --case H24`：passed 4，failed 0，`technical_only` 为 true，`human_approved` 为 false，退出码 0。
- 二十题目录一次收集时 pytest 退出码 2，原因是每题都按目录合同使用同名 `test_implementation.py`。随后按单题命令重跑，单题退出码都是 0。这个失败留在 `selfcheck.json`，没有通过改名绕开。

## 预期、自测和独立验收

冻结用例的预期输出以 `harness/contracts/wp18_cases.json` 为准。上面的 verifier 结果只说明这些公开检查在本工作区技术上通过。`expected_code_diff.json` 仍是预测，没有用 `harness/code/build.py` 重建 Observed。候选节点状态是 `unchecked`。U01–U12 和 `acceptance/` 没有跑。不能把自测通过写成独立验收通过。

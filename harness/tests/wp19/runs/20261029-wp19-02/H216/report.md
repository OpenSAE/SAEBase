# H216 发布后回归

本文件是自测记录。它不是独立验收结论，候选节点仍是 unchecked。

## 实现

`implementation.py` 提供 `solve(payload) -> dict`。合法输入返回冻结预期对象；非法输入抛出 `ValueError` 的子类 `InputError`，并且不改调用方对象。 本题 operator 为 `provenance`。实现没有 import `harness.contracts.wp19_reference` 或 `wp19_reference_ext`。

要求：dispatch green 不是部署实证

## 合同

解析得到的 `contract_ids`：["CT-BASE", "CT-SCOPE", "CT-TRUTH", "CT-CHECK", "CT-TRACE", "CT-ERROR", "CT-INTERFACE"]。

任务信号：[]。`selection_audit` 为空。

## 实际观察

- 合同解析退出码 0。
- 单题 pytest：7 passed in 0.03s，退出码 0。
- `wp19_verify --case H216`：passed 4，failed 0，`human_approval` 为 false，status PASS，退出码 0。
- 整目录一次 pytest 收集退出码 2，原因是每题都使用同名 `test_implementation.py`。该失败留在 `selfcheck.json`。

## 预期、自测和独立验收

冻结预期以 `harness/contracts/wp19_cases.json` 为准。公开验证通过只说明这些公开检查在本工作区技术上通过。`expected_code_diff.json` 仍是预测。不能把自测通过写成独立验收通过。

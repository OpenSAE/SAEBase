# H35 多计划相互冲突

本文件是自测记录。它不是独立验收结论，候选节点仍是 unchecked。H47–H226 仍是 PROPOSED，本题没有补写官方评分器。

## 实现

`implementation.py` 提供 `solve(payload) -> dict`。合法输入返回 JSON 对象；非法输入抛出 `ValueError` 的子类 `InputError`，并且不改调用方传入的对象。实现没有 import `harness.contracts.wp19_reference`。额外样本在 `fixtures/extra.json`。

要求：权限矛盾时停下且报告冲突

## 合同

解析得到的 `contract_ids`：["CT-BASE", "CT-SCOPE", "CT-TRUTH", "CT-CHECK", "CT-TRACE", "CT-ERROR", "CT-INTERFACE"]。

任务信号：[]。`selection_audit` 为空。

## 实际观察

- 合同解析退出码 0。
- 单题 pytest：7 passed in 0.01s，退出码 0。
- `wp19_verify --case H35`：passed 4，failed 0，`human_approval` 为 false，整体 status PASS，退出码 0。
- 二十题目录一次收集时 pytest 退出码 2，原因是每题都按目录合同使用同名 `test_implementation.py`。该失败留在 `selfcheck.json`。

## 预期、自测和独立验收

冻结用例的预期输出以 `harness/contracts/wp19_cases.json` 为准。上面的公开验证只说明这些公开检查在本工作区技术上通过。`expected_code_diff.json` 仍是预测。不能把自测通过写成独立验收通过。

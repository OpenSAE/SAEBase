# X20 长时程交接、需求变更、中断恢复及终局审阅

本文件是考生自测记录，不是独立验收结论。`review_candidate.json` 全部为 unchecked，verifier `human_approved` 为 false。

## 实现

`implementation.py` 提供 `run(payload, workdir) -> dict`，计算位于 `src/schema.py` 与 `src/pipeline.py`。合法输入返回多阶段对象并在需要时写入 `workdir`；非法输入抛出 `InputError`。未 import 官方 WP20 reference。

目标：以 8–10 个预先定义的事件组成完整 workflow，包含多 Agent 交接、需求修订、两次失败注入和最后的候选 Tree 审阅。

## 合同

解析得到的 `contract_ids`：["CT-BASE", "CT-SCOPE", "CT-TRUTH", "CT-CHECK", "CT-TRACE", "CT-ERROR", "CT-REVIEW", "CT-RECOVERY", "CT-EVIDENCE", "CT-STRUCT", "CT-DETERMINISM"]。

## 实际观察

- 合同解析退出码 0。
- 单题 pytest 退出码 0。
- `wp20_verify --case X20` 退出码 0。输出摘要：
```
{
  "spec": {
    "cases": 20,
    "scenarios": 80,
    "status": "VERIFIER_QUALIFIED",
    "human_approval": false
  },
  "cases": [
    {
      "case": "X20",
      "checks": [
        {
          "scenario": "positive_a",
          "passed": true
        },
        {
          "scenario": "positive_b",
          "passed": true
        },
        {
          "scenario": "negative_a",
          "passed": true
        },
        {
          "scenario": "negative_b",
          "passed": true
        }
      ],
      "passed": 4,
      "failed": 0,
      "technical_only": true,
      "human_approved": false
    }
  ],
  "status": "PASS"
}
```
- 全套 `harness/tests/wp20/runs/20261029-wp20-01` pytest 使用互不冲突的测试模块名，并用 importlib 隔离 `implementation`/`src`。公开自测不等于独立验收。

## 未验证

- Observed AST 未由独立 reviewer 重建
- 隐藏场景未运行
- 产品级集成、真实模型、外网发布未覆盖
- 不能把公开 verifier PASS 写成人类批准或独立验收通过

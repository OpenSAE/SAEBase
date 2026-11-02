# WPXX · Research delivery plan

## 1. Research overview and Atlas role
Explain the actual scientific question, prior concepts, why the planned measurement matters, and macro objective. Existing Core/Knowledge/Context IDs are primary; new Atlas nodes require separate review.

## 2. Execution principles
Use the smallest scientifically meaningful sample first; maximize useful evidence without weakening validity. Independent experiments continue when unrelated branches are blocked. Scientific parameter, license, privacy or resource changes require the predeclared alternative or a new human decision.

## 3. Experimental plan
### E1 · [Actual scientific objective]
- Paper-level support:
- Existing Dataset/Model/SAE/Task implementation and exact configuration reference:
- Inputs, algorithm steps, minimal pilot, full population, dependencies:
- Predeclared positive/negative/ambiguous result scenarios:
- Exact metric definition, expected scope, output table:
- Allowed fallback and explicit block threshold:

## 4. Deliverables and result tables
| Experiment | Metric | Expected | Actual | Status | Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | metric | population | NOT_RUN | NOT_RUN | | |

The executor returns a **fully filled copy of this Work Plan** as the primary artifact. Keep immutable frozen source and separate raw receipts. Every result has a notes field; exceptions from the prescribed steps must be explained.

## 5. Setup and runtime budget
Real environment probe, source and dependency identity, representative dataset/model/SAE/task smoke, throughput and startup costs, peak memory/disk, estimated full work, bottlenecks, time budget and decision thresholds. Default setup estimates cost rather than changing algorithms.

## 6. Audit, blocking and independent acceptance
Source/checkpoint/metric integrity; expected failures; local versus global blocks, warnings, human-decision thresholds; original logs and immutable receipts; required independent replay and review.

## 7. Result location and B2 delivery
Use `experiment/` Runtime paths, study-specific `reports/<run_id>/workplan_completed.md`, run manifest and audit. B2 destinations and upload scope must be authorized in the Work Plan. rclone copy/check never implies a public release.

## 8. Terminology and supporting sources
Use existing Context definitions, reference exact Code/Knowledge paths and avoid new synonyms.


## Workplan 起草后的双重 Human Review（模板附属要求）

在提交 Draft 的首次审阅或实质修订后，生成两份**不写入任何 Node 正文/性质**的伴随材料：

- `harness/github/work_plan/reviews/<WP-ID>/summary.md`：`## 概览` 以及与本模板正式 WP 的每个编号章节一一对应的 `## N.` 摘要。概览 40–100、每节 25–115、总体 400–950 个汉字，均须说明具体决定/输出/待决事项，先展示给用户。
- `harness/github/work_plan/reviews/<WP-ID>/node_impact.json`：分别审计 Code（功能及展示源码）、Knowledge、Environment、Core、Harness、Context，明确 `NONE/ADD/CHANGE/REMOVE/DEFER` 和实际改变的字段/文件；不影响既有功能则保留节点，具体 Workplan 只做独立关联。

使用 `python -m harness.github.work_plan.review_gate --plan ... --summary ... --impact ...` 检验格式；再按审计结论判断是否需要真正的 Node Human Check，源码写入授权独立判断。详见 [编写标准 §6A](WRITING_STANDARD.md)。

## 9. Structured bindings
The fields below are references and predefined output slots, **not a duplicate copy of experiment parameters**. `status: approved` alone does not grant execution or publish permission.

```workplan-json
{
  "schema": "saebase.workplan.v1",
  "plan_id": "WP23",
  "status": "draft",
  "from_human": "harness/github/work_plan/from_human/WP23.md",
  "atlas": {
    "core_ids": ["g.pipeline"],
    "knowledge_ids": ["q.sae.methods"],
    "context_term_ids": ["term.sae.reproducible"],
    "code_refs": ["saebase/runner.py"]
  },
  "experiment": {
    "study_id": "example_study",
    "config": "conf/experiment/example.yaml"
  },
  "steps": [
    {"id": "setup", "needs": [], "permission": "local", "goal": "Check actual environment", "evidence": "setup receipt", "command": ["python", "-m", "saebase.setup", "--help"]},
    {"id": "run", "needs": ["setup"], "permissions": ["network", "gpu"], "goal": "Real bounded experiment", "evidence": "run receipt", "command": ["bash", "scripts/experiments/example.sh", "full"]},
    {"id": "audit", "needs": ["run"], "permission": "local", "goal": "Independent result audit", "evidence": "audit report", "command": ["python", "-m", "saebase.runtime.audit", "--help"]}
  ],
  "deliverables": [
    {"experiment": "E1", "metric": "example metric", "expected": "fixed population"}
  ]
}
```

```harness-contracts-json
{
  "registry": "saebase.harness.contract_registry.v1",
  "plan": ["CT-ENV", "CT-CONFIG", "CT-EVIDENCE", "CT-RESOURCE"],
  "cases": {
    "E1": {"tags": ["environment", "config"], "contracts": ["CT-ENV", "CT-CONFIG"]}
  }
}
```

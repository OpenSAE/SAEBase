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

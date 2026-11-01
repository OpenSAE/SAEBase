# Work Plan execution deliverables · v1

**Primary result:** one complete `workplan_completed.md` containing the original reviewed plan plus filled result rows for *every* declared experiment, including explicitly NOT_RUN/BLOCKED results. A separate free-form summary never replaces the filled plan.

A run has a unique ID and exact source commit. Store files at:
`experiment/_studies/<study_id>/reports/<run_id>/workplan_completed.md`;
`experiment/_studies/<study_id>/reports/<run_id>/audit.json`;
`experiment/_studies/<study_id>/metadata/<run_id>/execution_receipt.json`;
`experiment/_studies/<study_id>/metadata/<run_id>/artifact_manifest.json`.
Scientific task raw outputs remain in existing Dataset/Model/Task run leaves. Paths are recommendations for a user-approved study, not permission to overwrite existing runs.

## Observed receipt format

```json
{
  "schema": "saebase.workplan.run.v1",
  "plan_id": "WP23",
  "run_id": "run_001",
  "source_commit": "record the actual full Git SHA",
  "results": {
    "E1": {
      "value": null,
      "status": "NOT_RUN",
      "notes": "Reason for deviation, if any",
      "evidence": ""
    }
  },
  "environment": {},
  "timing_seconds": {},
  "resource_observations": {},
  "unresolved": []
}
```

Allowed statuses: PASS, FAIL, BLOCKED, PARTIAL, NOT_RUN, WARNING, HUMAN_DECISION. Never label a result PASS without an actual value and evidence path. Include actual input population, measured time, estimated time and confidence, resource/cost policy, errors and retries in the receipt. Missing observations remain unknown, not zero.

Generate the filled copy using `python -m harness.github.work_plan.lifecycle complete --plan ... --receipt ... --output ...`; it uses exclusive creation, preserving the reviewed original and earlier reports. The separate generic Runtime Audit verifies file identities and integrity, while task-specific evaluators verify scientific meaning. Harness Acceptance and human approval remain separate.

B2 may be the default authorized archive transport. Attach remote prefix, transfer configuration, per-file digest manifest, copy/check receipts and any privacy/license exclusions. Neither a successful upload nor an Atlas status automatically means scientific acceptance.

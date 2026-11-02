# WP23 · Agent Ready implementation audit

**Current decision: BLOCKED — not Agent Ready.** The merged product CI is valid software-regression evidence, but it is not an end-to-end verification of this Workplan. This report is a pre-handoff engineering gate, not a new Workplan, an Atlas node review, or approval to use any external resources.

## Verified source facts

- The independent entrypoint lives at `scripts/workplan/wp023.sh` and `scripts/workplan/wp023_orchestrator.py`. The shell previously displayed only two preparation stages, while the Python entrypoint declares further component and audit stages. The dedicated handoff-fix PR makes stage selection explicit and runs the Python module with a stable source-root import path.
- `conf/experiment/wp23_precheck.yaml` explicitly says Draft and enumerates default objects/limits, but the orchestrator does not currently read these values to make actual check selection, token limits, or per-worker wall-time decisions. Its content digest is currently checked for source consistency, **not execution equivalence**.
- `conf/task/precheck.yaml` retains a different, general-purpose default (`mode: sae`, `max_tokens: 32`, `max_seconds: 120`). A complete standalone handoff must reconcile actual configured behavior with its declared Workplan defaults without changing unrelated product behavior.
- The full WP23 sequence calls for separate **Setup** and optional authorized **Asset Prepare**. The Python CLI currently provides `inspect/inventory/dataset/model/sae/compatibility/task/audit` but no dedicated setup or asset-preparation subcommand; no step command should be invented.
- The Task stage explicitly returns `BLOCKED_CONFIGURATION` for the default Benchmark case; a successful native Benchmark smoke is not yet established. This must either be implemented with genuine task-owned bounded inputs and tests or reviewed as an explicitly blocked non-default capability, rather than asserted as PASS.
- A component with a stored `BLOCKED_AUTH` receipt is treated as already recorded on resume. A later authorization does **not** currently cause it to rerun; configuration/asset/permission hashes are not comprehensively validated per receipt. This conflicts with the strict resumability and identity requirements of the plan.
- The existing lightweight Product CI executes `tests/precheck/` but does **not** by itself execute the dedicated `tests/workplan/wp23/` integration tests or a full staged Run / Resume / Audit round trip.
- The source-impact review is `DEFER_NODE_DECISION`. Implemented code must be re-audited against actual AST; any real durable Atlas change requires its own Human Check. Approved source paths and permissions do not automatically authorize additional paths or scientific workloads.

## Required closure before changing status to agent_ready

| Gate | What constitutes actual evidence | Current |
| --- | --- | --- |
| G1 · Entry | Every step has a real CLI command, fail-closed flag/permission checks, consistent run ID and source-root imports | PARTIAL |
| G2 · Defaults | The active runner reads WP23's own checked config and validates all default selectors/limits | BLOCKED |
| G3 · Real workers | Dataset, Model, SAE, valid Pair and four default Tasks each reach genuine bounded Worker paths; missing assets yield truthful blockers | PARTIAL |
| G4 · Setup | A real, separately permissioned Setup and asset-preparation entry without hidden network/GPU work | BLOCKED |
| G5 · Resume | Identical source, data, model, SAE, site, parameters and authorization permit safe reuse; changed permissions/resources cause safe independent retry | BLOCKED |
| G6 · Audit | Missing, altered, extra, crashed and timed-out receipts are handled, with four complete tables and immutable filled Workplan | NOT_INTEGRATION_VERIFIED |
| G7 · Tests | Bounded negative and end-to-end smoke under exact-source CI, including `tests/workplan/wp23/`; no real-weight claim from mock | NOT_INTEGRATION_VERIFIED |
| G8 · Review | Updated six-domain source impact, explicit Human Check disposition or verified URL, human-authorized exact implementation scope | PENDING |
| G9 · Handoff | Exact commit SHA and **actual** offline/integration/CI evidence attached to `readiness`; every `steps[].command` runnable | BLOCKED |

No G1–G9 gate can be satisfied merely by writing a plausible evidence string or turning a Draft into `agent_ready`. Hardware availability, network, gated assets, paid GPU and publishing remain separately human-authorized.

## Next implementation sequence

1. Resolve G2/G4 and G1 with source-owned config parsing and precise standalone stage commands; preserve source path owners.
2. Fix G5 with evidence-bound resume and explicit safe retries, then G3 with real bounded task input selection (including Benchmark) and independent receipt checks.
3. Prove G6/G7 by replaying exact-file negative cases, interrupted runs, parameter drift, blocked resources and complete audit; inspect CI logs and independent source SHA.
4. Re-audit G8 and finalize G9 **only** after all preceding gates have concrete successful evidence. Leave Workplan status `draft` or `reviewed` until then.

This report intentionally contains no references to other Workplans and grants no compute authority.

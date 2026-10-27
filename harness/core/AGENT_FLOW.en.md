# Auditable Agent Execution and Atlas Human Review

Human proposal review occurs in the existing Atlas Tree interface. Core and Harness governance cannot be modified by an ordinary task agent.

## Reviewing proposed nodes

**Unchecked** nodes are purple and await a human decision. **Approve** marks an accepted placeholder red (Not Implemented): the node may exist in the research plan, but no implementation or scientific result has been established. **Deny** records a gray decision without inserting the proposal into the authoritative Knowledge graph. **Unchecked** can restore a previously approved or denied draft to the undecided state, subject to source and revision validation.

Review selectors span All, Tree, Parent, and Node. The queue follows DFS order from authoritative edges rather than the Work Plan array. The interface displays state counters, previous/next navigation, and keyboard shortcuts.

From the repository root, with a previously created and authorized plan ID:

```bash
python -m harness.core.agent_flow inspect EXISTING_PLAN_ID
python -m harness.core.agent_flow serve EXISTING_PLAN_ID --port 8766
```

Open the full address printed by the local service. A session token in the URL fragment must never be committed or shared publicly.

Authenticated decisions are written to `harness/github_work_plan/agent_flow/<ID>.json`. Eligible, approved empty Knowledge nodes can also update the authoritative Knowledge graph and glossary. Revision and source-hash checks reject stale writes. Saving a decision does not execute a model.

## Scientific acceptance is separate

Approving a proposal does not demonstrate a working algorithm. The scientific execution workflow requires a bounded plan, exact path and command authorization, source review, an actual experiment, independent result auditing, and human acceptance of node status. Partial implementations remain separately indicated; verified scientific status needs reproducible evidence and review.

Normal task agents cannot silently extend approval to other files, nodes, authority graphs, or governance rules.

## Local and public operation

The local review server listens only on `127.0.0.1` and checks session identity, origin, and revision. This is an application-level control, not an operating-system security sandbox.

GitHub Pages serves static HTML. Its review controls modify a local browser draft and support JSON export; they do not authenticate a GitHub repository write. Publishing a decision requires the permitted local workflow or an independently reviewed GitHub PR.

## Verification

```bash
python -m pytest -q harness/github_work_test/test_harness_migration.py harness/github_work_test/test_openresearch_integration.py
python harness/core/validate.py
python harness/knowledge/validate.py
node --check harness/visualization/app.js
python harness/visualization/build_pages.py --check
```

These checks cannot replace a real browser interaction test or scientific experiment. Node status belongs to that node and does not automatically propagate from children to parents.

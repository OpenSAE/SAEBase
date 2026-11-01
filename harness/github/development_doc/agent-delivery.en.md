# SAEBase conversational planning and GitHub collaboration

This guide is for a conversational GPT or other planning assistant. Its entry point
is [SAEBase Delivery Skill](../../../.github/skills/saebase-delivery/SKILL.md).
Its main purpose is to define outcomes, inspect relevant source, decompose work,
and coordinate implementation and review. Planning does not require a local
coding agent, Cursor Skill, precheck script, GPU, or access to the execution machine.

The existing architecture boundaries, Work Plan writing standard, Universal
Contracts, Work Order v3, Agent Flow and independent acceptance remain authoritative.

## Human Check (default human review before GitHub edits)

Human Check is **on by default**. Before changing source, configuration, docs, governance files or submitting a PR, the conversational assistant prepares a source- and Context-grounded written change report, alongside Atlas Before / Proposed After nodes. The report identifies exact source SHA, Core / Knowledge / Code / Environment / Context impact, file ownership, reason, risks, and checks. Proposed new Code nodes are not represented as implemented AST.

**Each node is decided independently: there is no Approve Plan.** Approve / Deny / Unchecked affects only that item. An unrelated approved item may proceed when another is denied. Overlapping file ownership with a rejected or unchecked item must be split or revised rather than bypassed with broad file approval. Approval of a proposal is not approval of implementation, GPU execution, scientific findings, publication, or release.

Proposal reports and decisions live outside tracked source at `experiment/_human_check/<id>/`. The authenticated local Atlas review updates only this proposal ledger, never Core/Knowledge/Code authority. Public static Atlas browser decisions are drafts, not authorization. A human may explicitly turn Human Check off using the local Atlas control, with an audit record; default remains on and all other governance/permission requirements remain in force. Source drift or revised scope requires refreshed node review.

See [Human Check guide](../HUMAN_CHECK.md) for the manifest, local Atlas command and per-path `check` gate; attach the reviewed scope and provenance to the eventual PR without exposing local review keys.

---

## Desk precheck

Inspect only necessary source and available evidence. For a docs/UI task, look
at generated assets, links and site packaging. For software, inspect ownership,
callers, registries, config, compatibility and related tests. For a specific
study, distinguish reusable Dataset/Model/Task implementations from experiment
orchestration, and specify eventual pilot and evidence needs. For governance
and release, inspect frozen requirements, approval and exact-source SHA gates.

Separate `VERIFIED_FROM_SOURCE`, `INFERRED`, `OPEN` and
`REQUIRES_EXECUTION`. Do not claim local installation, test success, real
checkpoint loading or scientific acceptance based on static inspection.

## Task decomposition

Make the smallest directed acyclic graph (DAG) of independently meaningful
deliverables. For each task specify owner, expected paths, dependencies,
acceptance evidence and human permissions. Prefer a single coherent PR;
split only across genuine ownership, permission or evidence boundaries.
If multiple branches are needed, specify base and integration order.
Avoid parallel work on the same files.

Keep Dataset, Model, Task, Runtime and concrete Experiment ownership separate.
Do not invent a parallel registry, new top-level study source package or a
second governance framework. Build cheap planning checks before expensive work.

## Handoff and approval

By default, give a concise plan in the conversation. Do not create a Work
Order or PR just to discuss a proposal. When the user explicitly requests
repository changes, create a focused branch and reviewable PR, then report
actual changes and actual tests. A subsequent implementer may be a human,
GitHub agent or an existing local agent; no local agent is required.

Do not change frozen Contracts, release provenance, protected governance,
scientific measurements, GPU permissions or publication approval.

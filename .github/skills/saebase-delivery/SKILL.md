---
name: saebase-delivery
description: Plan and deliver SAEBase changes through the existing Harness, Work Plan, Work Order, independent review, and local agent handoff. Use for feature, experiment, integration, CI, architecture, or release requests.
---

# SAEBase delivery orchestration

This is a **routing skill**, not a new registry, approval system, or scientific acceptance standard.
Authoritative rules: `AGENTS.md`, `harness/core/BOUNDARIES.md`,
`harness/github/work_plan/WRITING_STANDARD.md`, `harness/contracts/catalog.json`,
`harness/core/AGENT_FLOW.md` and `harness/github/development_doc/agent-delivery.md`.
Read only the relevant additional context; do not bulk load all trees.

## Start with an outcome, then choose the smallest plan

1. State the requested observable result, its owner, scientific/engineering boundary,
   existing reusable implementation, deliverables, and exact acceptance evidence.
2. Identify whether this is an ordinary change, a protected governance change, a new
   architecture entity, or a real experiment. Path permission is not architecture approval.
3. Reuse existing Work Plans and their frozen Contracts. For new plans, follow the
   existing writing standard; resolve applicable predeclared Contract IDs. A gap is
   `CONTRACT_GAP`, not permission to invent or weaken a contract.
4. Assign each new or moved file to an existing owner. Study-specific orchestration
   belongs in `scripts/experiments/<study>/` with configs in `conf/experiment/<study>/`.
   Reusable Dataset, Model, Task, and Runtime implementations retain their own owners.
5. Split by **independent ownership, authorization, or evidence boundary**, not by every
   implementation step. Prefer one focused PR with ordered stages. Parallel PRs need
   an explicit dependency DAG and a single integration owner; do not create circular
   stacked branches or merge a Draft experimental branch into main accidentally.
6. Freeze the base SHA, target ref, allowed paths, expected evidence, required
   environments, compute/download budget, and stopping conditions before execution.
   No unspecified write, download, GPU experiment, deployment, publication, or release.

## Classify prechecks before implementation

Use `python scripts/workplan/delivery_precheck.py --mode plan --profile auto
--changed <paths>` on an existing checkout for a non-mutating route report.
Use `--profile` if no files exist yet. This report chooses checks, **not permissions**.
Run the same command in local mode with `--expected-branch`, `--expected-sha` and
the named `--work-order` (when required); `--run-safe` performs only explicitly
listed, local, read-only checks. Never install, download, execute a GPU workload,
publish or edit governance as a precheck.

- Docs-only: Git identity, scope and relevant links.
- Source / registry: architecture and syntax, dependency availability, smallest
  related import/test. Test a new or changed parameter's actual propagation.
- Experiment / model / dataset: dependencies, pinned source/weights, licenses,
  resources, a bounded synthetic/CPU pilot, then a separately authorized real pilot.
- Governance / release: separate maintainer approval and exact-source evidence;
  an agent's local PASS is never equivalent to independent review.

Classify `PASS`, `FAIL`, `BLOCKED`, and `NOT_RUN` accurately. The planner
must not claim that checks ran in a machine it cannot access.

## Produce a handoff instead of guessing the local machine

Record in the Work Plan: exact branch/base SHA, dependency DAG, scope, selected
Contracts, Precheck commands, local environment/pilot/full commands, authorization
boundaries, expected artifacts, recovery, and independent acceptance. Bind an
existing v3 Work Order when production Agent Flow requires one; do not modify the
v3 schema or forge approval.

Return the smallest self-contained local-agent message, specifying the Work Plan,
Work Order (if authorized), branch, exact SHA, first precheck command and stop
conditions. The local agent must actually measure its environment and report logs,
exit codes, source identity, changes and remaining blockers. GitHub-side reviewer
replays relevant tests independently before merge.

## Delivery optimization

Batch related file writes into normal Git commits; do not create a commit per file.
Keep research evidence and frozen history immutable. Check generated Atlas assets
and release SHA **before** merge, rather than relying on a bot-generated post-merge
commit. Preserve the existing exact-main-SHA release gate. Use CI results from the
actual head SHA; never equate PR creation, green lint, or a configured-only runtime
with successful real execution.

Finish with: what is ready, what was truly executed, what's blocked, the branch/PR,
the next local-agent instruction, and the human approval still needed.

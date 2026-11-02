# SAEBase agent instructions

Before changing files read [the Harness overview](harness/README.en.md), [repository architecture](harness/core/BOUNDARIES.en.md), [GitHub workspace](harness/github/README.en.md), [Agent Flow](harness/core/AGENT_FLOW.en.md), and relevant [universal contracts](harness/contracts/README.md).

1. Search existing modules, registries and configurations before adding files.
2. Separate general framework capabilities from particular Dataset, Model, Task or Experiment instances. Shared modules must not depend on a particular experiment.
3. Keep configuration in `conf/`, launch scripts under `scripts/`, product tests under `tests/`, and runtime outputs under `experiment/`.
4. Prefer scripts/tests grouped by Work Plan or Dataset/Model/Task/Sub-name. These are extensible conventions with ownership justification.
5. Declare new/moved files, owner, dependencies, reuse alternatives, tests and evidence in the PR. New repository root modules and changes to shared architecture require separate approval.
6. A path-edit authorization is not permission to change architecture or protected governance. Historical results and acceptance remain unchanged without independent evidence.

Offline CI proves software checks only, not real GPU or scientific acceptance.

Before implementing a new Work Plan, read [the Work Plan authoring standard](harness/github/work_plan/WRITING_STANDARD.en.md). Do not assume every remote branch is locally fetched; do not run a compute-intensive phase before environment, dependency and small-pilot checks succeed.

## Process-first correction for Harness / GitHub / local Agents

When a Harness, GitHub maintenance, or local-Agent execution workflow fails, behaves inconsistently, or is reported to violate project rules, **repair the process before patching the isolated symptom**. The governing written contracts take precedence over an ad-hoc instruction, a convenient workaround, or the urge to finish a specific Work Plan.

1. **Stop and preserve evidence** of the contested action (source SHA, observed behavior, logs, proposal/review state). Do not automatically rewrite nodes, change approved scopes, restart GPU work, or relax a gate.
2. **Read the applicable authorities first**: this file, `harness/core/BOUNDARIES.md`, `harness/github/HUMAN_CHECK.md`, `harness/github/work_plan/WRITING_STANDARD.md`, Atlas authoring standards and the active Work Order/Contracts as relevant. Identify the violated clause, conflicting implementations and missing enforcement.
3. **If the human explicitly identifies a process violation**, first present a concrete, feasible **process-standardization change proposal**: root cause, normative change, exact owner/files, backward compatibility, negative tests, and approval/rollback implications. Discuss or confirm the governance solution with the human before returning to the original technical fix. Existing protected-governance/Human Check permissions still apply.
4. **Then fix the specific problem together with the human** using the updated or confirmed process, scoped review decisions, and actual tests. Report what changed in the process separately from what was fixed in the task.

A Work Plan may reference Core/Knowledge/Code/Environment/Context without changing them. Every new or materially revised Work Plan must first produce the regulated human-facing section summary and a source-backed six-domain node-impact audit described in [Work Plan authoring](harness/github/work_plan/WRITING_STANDARD.md). If the durable node responsibility is unchanged, **do not edit it** or create a synthetic Human Check change to justify project-specific tasks.

## Generalizable failure prevention for Harness and GitHub workflows

**Mandatory on every Harness or GitHub execution error, failed check, or workflow defect, even when the immediate symptom has an easy fix:** determine whether the root cause can recur in another Work Plan, PR, agent session, CI job, repository path, machine, or task. Do not treat an isolated passing rerun or a one-off patch as a durable correction.

1. **Classify and preserve evidence.** Capture the exact source SHA, command, input, failure output, affected owner and permission boundary. Distinguish an isolated data/configuration problem from a reusable failure in a shared rule, default, contract, validator, script, CI, agent instruction, or workflow. If generalizability is uncertain, say so; do not label a guess as established fact.
2. **Propose a prevention mechanism.** For a potentially generalizable failure, describe its root cause, other affected scenarios, and at least one smallest-scoped **systemic prevention option** (such as a shared validation rule, preflight gate, standardized default, safer API, test fixture, regression test, or actionable process instruction). State the owner, exact proposed paths, compatibility impact, acceptance evidence and rollback. A local symptom fix by itself does **not** close this obligation.
3. **When the mechanism and root cause are well supported:** if the change is within already authorized paths, roles and budgets, implement the reusable prevention together with the immediate repair; add a targeted **negative regression test** that reproduces the failure, and validate both the original case and at least one distinct applicable scenario. Report the general fix separately from the local patch, including exact commit/CI evidence. Do not wait for another human answer solely because the technical diagnosis is clear and the modification is already authorized.
4. **When uncertain, risky, or outside the granted scope:** present the evidence, competing explanations, concrete prevention alternatives, risks and requested approval to the human **before** making the shared or protected change. A high-confidence technical diagnosis never overrides Human Check, protected governance, source-path authorizations, frozen research parameters, security boundaries, or GPU/network/publishing permissions.
5. **Verify durable closure.** Record which recurrence is now blocked, where the preventive rule/check lives, and which tests prove it; otherwise explicitly say `PREVENTION_NOT_YET_VERIFIED` and keep the general problem open. A clean CI run on one commit does not prove every real environment or scientific workload.

This rule is in addition to the process-first requirement above. Treat formatting errors, test-status classification mismatches, missing dependency checks, inconsistent command entrypoints, incorrect defaults and CI routing failures as candidates for generalization rather than automatically categorizing them as unrelated local incidents.

## Python formatting before any local commit or GitHub API write

Ruff is a deterministic format/lint gate, not a substitute for tests. The exact CI version is `ruff==0.15.22`; see `pyproject.toml` and `.github/workflows/ci.yml`. Do **not** discover formatting mistakes by waiting for a failed PR CI.

- Local Agent first stages **only human-authorized** new Python files, then runs `python saebase/tools/check_new_python_quality.py --fix`. Inspect the diff and re-stage only those files before committing. Existing modified Python files may be included *only* with explicit `--include-modified`, because Ruff rewrites entire files.
- Once per fresh checkout, install `python -m pip install "ruff==0.15.22" "pre-commit>=4,<5"` and `pre-commit install` to run the same bounded formatter automatically during local commits. Missing tools or version mismatch fail closed with an installation hint, not a silent skip.
- **GPT Agent with GitHub API / connector (including a fresh chat):** before EVERY Python `create_file` or `update_file`, take the complete proposed file text, then run the repository-owned stdin entrypoint (no Git working tree or staging required):

  ```bash
  python saebase/tools/check_new_python_quality.py --stdin \
    --filename tasks/precheck/worker.py \
    < candidate.py > candidate.formatted.py
  ```

  Replace `--filename` with the exact intended repository path. This reads only the proposed bytes and outputs **only** the formatted, lint-clean source on successful exit; all diagnostics go to stderr. Send `candidate.formatted.py`'s exact text, not `candidate.py`, to the GitHub API. Before updating an existing file, inspect the resulting full-file diff and do not accept unrelated formatting changes without scoped approval. This command does not grant source edit permission or execute the source.
- This is a required **pre-write step**, not optional advice: use pinned Ruff `0.15.22`; if the GPT environment cannot execute it, stop the Python GitHub write and report `BLOCKED_FORMAT_TOOL` rather than bypassing it or pretending the new file passed. Local pre-commit cannot intercept a GitHub API call. The independent CI check stays read-only and remains mandatory; do not enable a CI bot that pushes unreviewed changes.
- This formatting gate neither expands editable paths nor grants protected governance/GPU/network/publish permission. Never auto-format frozen WP21 fixtures, historical results, or unrelated Python files merely to make a check green.

## Mandatory Workplan Human Check disposition

Every new or materially revised Workplan review must explicitly present a verified Atlas review link for an actual Node ADD/CHANGE/REMOVE, or a concrete reason why Node Human Check is not required. Deferred Code/AST changes require a no-review-now explanation and post-implementation audit, not permanent clearance. Store the disposition in the six-domain impact artifact and validate its structure. Source code approvals remain separate, and imaginary Atlas links are forbidden.

## Human Check before GitHub changes

Before source edits or a PR, follow [Human Check](harness/github/HUMAN_CHECK.md) and the [Delivery Skill](.github/skills/saebase-delivery/SKILL.md). First distinguish **durable Atlas semantic changes** from a source-only modification. For genuine Node ADD/CHANGE/REMOVE, Human Check remains enabled and requires the existing Atlas overlay with independent Approve / Deny / Unchecked decisions (only humans can opt out through the authenticated local UI). For **NO_NODE_CHANGE**, do not fabricate a Node proposal: provide the source-backed impact audit and obtain explicit human review of the exact changed paths and PR diff before implementation/merge, with protected governance authorization retained. Neither route bypasses Contracts, Work Orders, GPU/network/publishing gates or independent verification.


## Conversational task planning

For planning, task decomposition, ownership review or multi-PR coordination,
use [SAEBase Delivery](.github/skills/saebase-delivery/SKILL.md) and the
[GitHub collaboration principles](harness/github/development_doc/agent-delivery.en.md).
A conversational assistant can apply the skill using repository source context;
it does not need a Cursor Skill, local precheck runner, Work Order, or GPU.
Actual source edits and experimental execution still obey existing Harness
permissions, Work Plans, Contracts, and independent acceptance.

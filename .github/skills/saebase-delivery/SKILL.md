---
name: saebase-delivery
description: Use when planning, decomposing, prioritizing, reviewing, or coordinating SAEBase research and engineering tasks with a conversational AI. Produce a grounded, minimal Work Plan and conditional precheck recommendations without requiring a local coding agent or GPU.
---

# SAEBase planning and task decomposition

This is a **planning skill for a conversational assistant**, not a new software framework, Runner, local-agent workflow, verification tool, or authorization mechanism. By default, **analyze and propose only**. Editing GitHub, committing, merging, executing experiments, downloading assets, and publishing require separate user authorization. Never represent proposed checks as completed.

Source hierarchy: the user's current goal and authorization; the actual repository and relevant existing Work Plan; `AGENTS.md`; `harness/core/BOUNDARIES.md`; `harness/github/work_plan/WRITING_STANDARD.md`; existing Universal Contracts and Agent Flow; and `harness/github/development_doc/agent-delivery.md`. Reuse those sources instead of inventing parallel governance. For source-dependent claims, inspect the **current** branch and relevant paths when repository access is available; otherwise clearly mark assumptions. Read narrowly, not whole trees by default.

## 1. Begin with the outcome

Write a one-sentence target result, relevant scientific/engineering value, actual inputs and outputs, and what counts as success. Separate **required deliverables** from optional improvements, real scientific evidence from software-only checks, and what is out of scope.

Identify whether the change is:
- a local fix or documentation change;
- a reusable Dataset / Model / Task / runtime capability;
- a particular experiment or study configuration;
- Harness, architecture, CI, release, or research-governance work;
- a combination that needs explicit dependencies.

For each new concept, ask whether an existing implementation can serve it. Apply the simplest compliant design. A specific experiment is **not** a second Dataset API, framework, registry, or top-level source package.

## 2. Perform a **desk precheck** only when it changes the plan

Desk precheck means reasoning and source inspection by the conversational assistant. It does **not** require Cursor, a shell, a GPU, a server, or real model weights.

Choose the minimum relevant questions:
- **Always:** identify current source version if available, existing owners/interfaces, protected paths, compatibility constraints, relevant frozen contracts and outstanding blockers.
- **Docs or UI:** links, generator/build ownership, generated assets, localization, deployment route; no environment checklist unless needed.
- **Code / components:** Registry, signatures, configuration, imports, callers, backward compatibility and existing tests; highlight possible dependency conflicts based on declarations.
- **Experiments:** distinguish study orchestration from shared components, check protocol, dataset and checkpoint identities, sample/compute budgets and evidence requirements **as a plan**. Only specify eventual runtime checks if actual execution is needed.
- **Governance / release:** frozen tests, authorization, exact-SHA release or independent acceptance requirements.

Mark each item **VERIFIED_FROM_SOURCE**, **INFERRED**, **OPEN**, or **REQUIRES_EXECUTION**. Never claim dependency resolution, tests, weight loading, GPU availability, CI success, or scientific validity based solely on reading files. Do not write or run a precheck script.

If a material ambiguity remains, state a bounded assumption and proceed with a best-effort plan; ask the user only about decisions they must make.

## 3. Decompose into the smallest meaningful task DAG

Decompose by **deliverable and independent ownership**, not by number of files or execution commands. For every task specify:
- ID and concrete observable result;
- existing owner and expected touched paths (with new/moved/deleted paths distinguished);
- dependencies and which tasks can truly proceed in parallel;
- acceptance evidence and remaining risks;
- who can perform it and any required human authorization.

Reuse existing `datasets/<Dataset>/`, `models/<Model>/`, `tasks/<Task>/`, `saebase/`, `conf/`, `scripts/`, `tests/`, and `harness/` ownership. Study-specific code belongs under `scripts/experiments/<study>/`; its config belongs under `conf/experiment/<study>/`; run outputs follow the existing singular `experiment/` runtime. New top-level roots, parallel registries and cross-component abstractions need separate architecture approval.

Prefer **one coherent PR** with ordered tasks. Split PRs only if independent review/permission, non-overlapping file ownership, genuine parallel development, or a separate merge gate makes this worthwhile. If split, show dependency edges, base branches, integration order, and a single merge owner. Avoid circular stacked PRs, unnecessary merge-sync PRs, and many one-file commits.

## 4. Define phase gates, not busywork

For each task decide whether it needs only a source review, a targeted code test, a packaging/dependency check, independent Harness review, or eventually an actual dataset/model experiment. The conversational assistant specifies the required check and **who will perform it**; no local agent is required to generate the plan.

Use a short pipeline: design acceptance -> ownership and dependency check -> implementation -> relevant tests -> integration review -> merge -> opt-in real scientific acceptance / release as appropriate. Put cheap failure detection **before** expensive implementation. Do not make all project tasks wait for the availability of a real GPU when only planning or static code work is needed.

Do not weaken the existing Work Order v3, Contracts, Hooks or Agent Flow when they are applicable to **later** execution. Planning alone does not activate them, freeze user permission or imply approval. Treat PASS/FAIL/BLOCKED/NOT_RUN as execution results only, with actual evidence.

## 5. Output a decision-ready plan

Default response order:
1. **Recommendation**: smallest approach and key trade-off.
2. **Desk precheck findings**: source-verified vs uncertain.
3. **Task DAG**: task, owner/path, dependencies, deliverable, acceptance.
4. **GitHub support changes**: only necessary files, docs, workflow/PR changes; separate optional CI tuning.
5. **Execution handoff**: a short actionable instruction or commands for whichever implementer the user chooses (human, GitHub agent, or a local agent), **without requiring a local agent**.
6. **Completion criteria**: merge dependencies, independent checks, deferred real experiments, and permissions.

When the user requests a formal Work Plan, follow `harness/github/work_plan/WRITING_STANDARD.md` and reuse Contract IDs selected from existing governance. Otherwise, a concise plan in the conversation is sufficient; do not create a Work Order or PR merely to discuss a proposal.

If the user asks to make the change, use a focused branch and one reviewable PR. State exactly what was changed, what was tested, what remains unverified, and whether `main` was affected. Preserve historical scientific results, release identities and the existing approval process.

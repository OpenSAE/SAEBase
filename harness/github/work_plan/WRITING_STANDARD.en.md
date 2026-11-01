# Work Plan authoring standard

Applies to new or materially updated Work Plans; preserve frozen historical exams and evidence.

1. **Goal and ownership first.** State success criteria, non-goals, input/output, and whether the change is a reusable Dataset/Model/Task/Runtime capability or a study instance. Inventory every created/moved/deleted path, identify its owner and existing reusable alternative, and justify new entities. Study glue belongs in existing `conf/experiment/<study>/` and `scripts/experiments/<study>/`, product tests in `tests/workplan/<id>/` or an experiment-owned hierarchy. Do not invent a top-level framework or registry for a study.
2. **Reproducible repository checkout.** Name the exact branch and commit. An origin remote may fetch main only; give an explicit one-off refspec `git fetch origin refs/heads/<branch>:refs/remotes/origin/<branch>` and verify HEAD without altering local remote settings.
3. **Dependency preflight before computation.** Declare compatible OS/Python, torch/CUDA, dependencies/extras, transitive pins, disk, and access requirements. Actually resolve/install on the target configuration; run `pip check`, crucial imports and bounded tests. A dependency conflict is a source defect, not a reason to silently replace a scientific method.
4. **Small first.** Source -> dependency -> import/contracts -> frozen configs -> CPU negatives -> resource/access -> tiny real pilot -> explicitly authorized full run -> independent audit. Give commands, expected status/exit codes, evidence files, budgets, stopping and resumption rules for each step.
5. **Artifact hygiene.** Treat virtualenvs, `*.egg-info/`, wheel build intermediates, weights, caches, and results as generated artifacts rather than new source owners. Keep them under an ignored Runtime/workspace location where feasible. Never delete preexisting user files without approval.
6. **Independent acceptance.** Bind existing universal Contract IDs, state real model/SAE identities, distinguish CODE_READY / SELF_TESTED / VERIFIED / BLOCKED / PUBLISHED, and require separate permission for downloads, GPU usage, native training and publication.
7. **Required sections.** Goal, ownership table, dependencies and environment, exact branch/commands, small-pilot budget, risks and rollback, contracts and independent acceptance, artifact paths, authorizations.

Repository authority: [architecture boundaries](../../core/BOUNDARIES.en.md).

## Intake, Atlas relationships, execution and delivery (new plans)

Before drafting, retain confirmed human intent, scientific motivation, already-known Atlas IDs, resource and permission decisions, and open questions in [from_human/](from_human/README.md). Use existing Core and Knowledge nodes and Context term IDs; introduce new nodes only after independent review. During drafting use [WORKPLAN_TEMPLATE.md](WORKPLAN_TEMPLATE.md), binding one canonical experiment configuration, typed `workplan-json` references, the existing Contract IDs, a dependency DAG and predetermined result tables. After drafting, validate the plan with `python -m harness.github.work_plan.lifecycle validate --plan <path>` and the existing Contract registry, then review Atlas and source-path relationships.

A shared Setup estimates environment, reproducibility and runtime using real observations, without silently changing scientific parameters. A reusable Audit checks provenance, integrity, failure scope and tolerance, while the owning Task checks metric meaning. Preserve unrelated work when one branch is blocked. Authorization to execute, use the GPU, transfer to B2, or publish must be explicit and separate.

Follow [DELIVERABLE_STANDARD.md](DELIVERABLE_STANDARD.md). The primary deliverable is a fully completed copy of the original Work Plan, with each predeclared result, actual timing, notes, path and evidence. Keep the approved source immutable, generate unique runtime receipts/manifests under `experiment/`, use B2 only within an approved scope, and distinguish self-check from independent acceptance.

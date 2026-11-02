# GitHub engineering workspace

This folder groups [Work Plans](work_plan/README.md), [independent tests](work_test/), [development documentation](development_doc/README.md), and [assets](assets/). GitHub-native settings remain under `.github/`.

Before modifying files read the [Harness overview](../README.en.md), [repository architecture](../core/BOUNDARIES.en.md), [task contracts](../contracts/README.md), and the assigned Work Plan. Identify the existing owner first. Shared components must not depend on a single experiment.

Prefer `scripts/workplan/<id>.sh` or `scripts/<dataset>/<model>/<task>/<sub_name>/`, and equivalent `tests/` layouts. Study-based grouping is also acceptable. These layouts are flexible with documented ownership. A new root module, framework or changed ownership requires separate approval. Include new/moved paths, dependencies, tests and evidence in the PR.

## Work Plan standard

All new development plans follow the [authoring standard](work_plan/WRITING_STANDARD.en.md), including real dependency resolution, owned file paths, exact commands and independent acceptance.

## Generalizable Harness / GitHub failure prevention

When a Harness, CI, GitHub, or agent workflow fails, determine whether the underlying error could recur across PRs, Work Plans, machines, tasks or sessions, even if a local patch is straightforward. Preserve the actual SHA, command, inputs, logs and owner. Propose a minimal systemic prevention option with concrete affected paths, risks, negative regression tests and rollback; fixing only the immediate symptom is not durable closure.

If the root cause and prevention mechanism are well supported **and the exact changes are already authorized**, implement the reusable fix and original repair together, test the original failure and at least one other applicable case, and report both. If technical confidence is insufficient, effects are uncertain or permission is missing, present the alternatives and risks to the human and request approval **before** changing shared or protected behavior. Technical confidence never supersedes Human Check, architecture, frozen research configurations or execution/network/publishing permissions.

Report where the prevention mechanism is enforced and what evidence verifies it; otherwise mark the preventive work `PREVENTION_NOT_YET_VERIFIED`. See the authoritative `AGENTS.md` governance section.

## Workplan isolation and Agent Ready

Phase 3 can draft or document historical evidence; Phase 7 implements and validates the actual code, defaults, entrypoints and tests. Only an independently verified Agent Ready local-Agent plan is executable. Cross-WP references, protocol or configuration inheritance and execution dependencies require explicit per-WP, per-purpose human approval in the current from_human record. Reading past records grants no dependency permission. Frozen records remain unchanged.


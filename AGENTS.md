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

## Human Check before GitHub changes

For conversational GitHub maintenance, use [Human Check](harness/github/HUMAN_CHECK.md) and the [Delivery Skill](.github/skills/saebase-delivery/SKILL.md) **before source edits or a PR**. Default is enabled. Generate one Context-grounded written report plus a five-tree Atlas proposed-change overlay; every node has an **independent** Approve / Deny / Unchecked decision, without a whole-plan approval. Denied/unreviewed node scopes do not authorize code changes; unrelated approved scopes may proceed. Only a human may explicitly opt out via the local Atlas review. This does not bypass protected changes, contracts, GPU/publish permissions, or independent verification.


## Conversational task planning

For planning, task decomposition, ownership review or multi-PR coordination,
use [SAEBase Delivery](.github/skills/saebase-delivery/SKILL.md) and the
[GitHub collaboration principles](harness/github/development_doc/agent-delivery.en.md).
A conversational assistant can apply the skill using repository source context;
it does not need a Cursor Skill, local precheck runner, Work Order, or GPU.
Actual source edits and experimental execution still obey existing Harness
permissions, Work Plans, Contracts, and independent acceptance.

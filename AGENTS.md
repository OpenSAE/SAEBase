# SAEBase agent instructions

Before changing files read [the Harness overview](harness/README.en.md), [repository architecture](harness/core/BOUNDARIES.en.md), [GitHub workspace](harness/github/README.en.md), [Agent Flow](harness/core/AGENT_FLOW.en.md), and relevant [universal contracts](harness/contracts/README.md).

1. Search existing modules, registries and configurations before adding files.
2. Separate general framework capabilities from particular Dataset, Model, Task or Experiment instances. Shared modules must not depend on a particular experiment.
3. Keep configuration in `conf/`, launch scripts under `scripts/`, product tests under `tests/`, and runtime outputs under `experiment/`.
4. Prefer scripts/tests grouped by Work Plan or Dataset/Model/Task/Sub-name. These are extensible conventions with ownership justification.
5. Declare new/moved files, owner, dependencies, reuse alternatives, tests and evidence in the PR. New repository root modules and changes to shared architecture require separate approval.
6. A path-edit authorization is not permission to change architecture or protected governance. Historical results and acceptance remain unchanged without independent evidence.

Offline CI proves software checks only, not real GPU or scientific acceptance.

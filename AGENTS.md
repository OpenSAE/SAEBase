# SAEBase Working Instructions

## Purpose and Main Functions

Explain the responsibilities and development practices of the research software and help assistants complete user tasks.

## Scope

These instructions apply to reading, developing, and maintaining SAEBase. Start with the project README, dependency declarations, and official commands, then read the source code and local instructions relevant to the task. Shared skills are available after connecting the project to EverResearch.

## Task Authorization

- Perform local operations under the authorization granted for the current task and the identity actually running them. Reading and discussion are limited to reading and drafting. Development and maintenance follow the scope set by the user. Project code reviews, formal plans, and collaboration approvals follow the project's GitHub workflow.
- Model downloads, full training runs, real data transfers, remote pushes, releases, and replacement of the production Atlas require the user's authorization for the respective operation.

## Work Toward the Intended Outcome

- Base engineering changes on this repository's actual entry points, configuration, and checks. After connecting to EverResearch, use the guide reported by `python -IS .everresearch/maintain.py status` for shared conventions on layout, prechecks, and delivery of results.

- Establish the outcome the user needs and the conditions for completion. Locate the existing implementation and its direct callers, then investigate, make engineering decisions, and act within the existing authorization. Implement straightforward tasks directly. For complex tasks, describe the dependencies and files involved in the actual deliverable. When a human decision is needed, present the facts established, the specific choices, and their effects.
- Carry authorized routine implementation through to completion. Choose local engineering approaches based on the agreed goals, practical benefits, and maintenance costs. When a new decision concerns the meaning of a metric, a research use, sample selection, or compute investment, explain the concrete differences, practical effects, and recommendation in light of existing decisions. Agree on the choice with the user before proceeding with the affected work.
- For substantial changes in how research results are used or what a feature offers, show representative questions, feature comparisons, or report excerpts early so the user can assess their practical value. Use existing results or authorized small-scale experiments, then adjust the implementation and presentation in response to feedback. Conduct additional experiments within the current task's authorization.
- When designing a new feature or a change, first identify how its result will be used and what a specific caller or downstream task needs to accomplish with it. Then determine the checks, return values, and storage. For information that needs to persist, also design how it will be found and read, how artifacts will be located, and how the information will be processed later. Implement models, data, tasks, and shared support according to their respective purposes. Prefer existing records and interfaces.
- Completion includes the outcome in actual use. On success, users can obtain and use the required content. When a failure, blockage, or partial completion is relevant to the feature, downstream work can continue from the recorded facts, effects, and suggested actions. If the cause is still under investigation, retain the known facts and the next diagnostic method.
- Read the material relevant to the current question. Distinguish documented definitions, actual code behavior, observed runtime behavior, and inferences. Record sources and versions as needed for the task. Record the respective sources of historical results and synthetic samples.
- Fix the actual failure first. Base new preventive measures on a concrete path in the current code, a practical consequence, and a proportionate cost.
- Assign each kind of fact to a module responsible for establishing it, then reuse the result in downstream callers. Concentrate routine checks at key points such as inputs and shared interfaces.
- Incorporate solutions found through temporary investigation into the production code responsible for the behavior. Remove temporary checks whose purpose is complete and redundant decisions. Report the practical effects and follow-up actions.
- Deliver fixes through the entry points users normally use. Define the scope of completion from the current task.
- For ongoing development and maintenance, keep an execution record at `doc/code/maintenance/<maintenance-topic>/execution.md` and prefer appending to the existing record for the same topic. Put dates and execution identifiers in the body and register the entry in the [maintenance index](doc/maintenance.md). Start the record with the background, goals, and overall approach. Append important actions and decisions in chronological order. At the end, summarize the effects, the AI assistant's assessment, and follow-up recommendations. When pausing, record where to resume. Reference result files through their existing paths.

## Directories and Responsibilities

- `datasets/`, `models/`, and `tasks/` hold research components. `base/application/registry.py` and `base/application/runner.py` handle shared registration and dispatch.
- `base/interfaces/` provides common model and SAE operations. `base/runtime/` maintains shared configuration, run paths, and records. Reuse existing interfaces and design cross-component capabilities around actual callers' needs.
- `scripts/examples/` holds complete runnable demonstrations, `tests/fixtures/` holds synthetic components, and `doc/code/documentation/source/development/` holds tutorials.
- `conf/` holds configuration. `scripts/` holds entry points for software releases, distribution, documentation, and code checks. `tests/` holds product tests. Put application logic in the module that owns it and have scripts and tools call the official implementation.
- `doc/knowledge/` holds research knowledge and terminology. `doc/code/` holds architecture and source code material, `doc/code/documentation/source/development/` holds development documentation, and `doc/code/maintenance/` holds maintenance execution records. `experiment/` is dedicated to research experiments and their execution records.
- `AGENTS.md` holds engineering rules, directory responsibilities, and the maintenance mode entry point.
- EverResearch provides shared skills, local resource bindings, and registration of structural artifacts. SAEBase owns its research source code, runtime entry points, and experiment results.
- Handle `saebase/research_lab/` under a new maintenance task explicitly requested by the user. Routine Dataset, Model, and Task development focuses on the components currently in use.

## Configuration Maintenance

Configuration paths under `conf/` follow the first two levels of the responsible source code's relative directory path. Keep a single file for each configuration branch and represent deeper responsibilities as named sections in that file. The project root configuration `conf/config.yaml` is the exception. Additions, edits, migrations, removals, and deployments of configuration must pass `tests/test_configuration.py`. The commit workflow automatically runs `python -m scripts.documentation conf --write` to synchronize `doc/conf.md`. Continuous integration and release steps verify the synchronized result. Update the corresponding documentation when parameter meanings, behavior, or conditions of use change.

## Checks and Research Results

Concentrate routine checks at a few key points and reuse their guarantees in downstream operations. Use temporary checks for specific problems. Once the cause is understood, incorporate the solution into the production code so similar problems are handled automatically, then remove temporary checks whose purpose is complete.

Handle new test scripts and supporting code according to their ongoing use. Incorporate material with lasting value into the production implementation or long-term tests. Remove material whose temporary purpose is complete. Clean up temporary inputs and test caches generated for this task. Retained checks should let downstream work use existing guarantees directly and reduce repeated decisions and manual intervention.

Keep the scope of checks proportionate to the change. For a local change, select checks that verify the functional outcome. Use existing results when they answer the question. Add or rerun tests to address the specific problem at hand. Use project-wide checks for existing continuous integration, release preparation, or tasks that explicitly require a project-wide review.

Python uses Ruff `0.15.22`. Before committing, stage only the changes for the current task. Use `python scripts/quality.py --fix` for new files. For existing files, use `--include-modified` within an explicit scope. When writing Python through the GitHub API, use the same tool with `--stdin --filename <exact-path>` to check the actual content. Continuous integration performs read-only checks.

Keep the provenance of model and SAE identities, weights, data splits, metrics, numerical precision, and historical results clear. Explain any changes to their meaning and perform the relevant checks. Record the distinct facts and evidence established by software checks, real experiments, and independent acceptance reviews.

## Writing and Communication

In online maintenance mode, use the maintenance mode entry point below to find the shared EverResearch maintenance guidance.

Name SAEBase features, stages, errors, and reports according to their practical purpose. Preserve the original meaning of model names, mathematical notation, and identifiers in historical formats. Use Chinese for Markdown and user-facing instructions, and preserve the original spelling of code and standard technical terms.



## Documentation Synchronization

Use the [project documentation](doc/README.md) to find the pages affected by the current change. Update the documentation when commands, public parameters, dependencies, result interfaces, or metrics change. Refer to the configuration and existing artifacts for the relevant run in experiment reports. When reviewing a feature, check the actual entry points, configuration, and results described in the documentation. At delivery, explain what changed, what was checked, and the effect on existing usage.

<!-- researchharness:search -->
## Locate projects and files

Start with known registrations and explicit paths when locating a project, command, or configuration. When you need a broader search, list the immediate directories first and pick specific paths. Limit routine recursive searches to active source material. Search local workspace output, historical snapshots, experiment results, or caches by explicitly naming the required files or directories and limiting the output.
<!-- /researchharness:search -->

<!-- researchharness:mode -->
## Maintain this checkout

Use the project instructions above for ordinary work and read-only inspection. EverResearch is an optional maintenance tool.

- **First use.** If `.everresearch/mode.toml` is missing and the user has authorized EverResearch setup, run `rh projects setup-maintenance . --online` for online maintenance or `rh projects setup-maintenance .` for a standalone checkout. An older `mode.json` requires a current mode selection as well.
- **Offline maintenance.** With `enabled = false`, use the [offline guide](.everresearch/offline/operations.md).
- **Online maintenance.** With `enabled = true`, run `python -IS .everresearch/maintain.py status` and follow the relevant section of the reported guide. If the connection fails, report the error and pause work that needs the central guide.

For an authorized structure check, run `python -IS .everresearch/maintain.py check` from the project root. Use the project's own entry points to install, run tasks, and read their results.
<!-- /researchharness:mode -->

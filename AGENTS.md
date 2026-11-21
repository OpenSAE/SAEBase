# SAEBase Job Description

## Purpose

Description of the duties and methods of development of scientific software to assist smart assistants in their user assignments.

## Scope

Read, develop and maintain for SAEBase. Read the source code and local description on a mission, starting with the project README, relying on a statement and a formal command.

## Mandate

- This operation is carried out according to current mandate and operational identity. Reading and discussion is limited to reading and drafting, and development and maintenance is carried out within a user-defined range. Project code review, formal planning and collaborative authorization follows the project GitHub process.
- Download models, formal training, real data transmission, remote transfer, release and Atlas Official replacements are carried out on the basis of the user ' s delegation of authority for the corresponding operation.

## Fulfilling the mission around its purpose

- The project changes are based on the physical access, configuration and inspection of the warehouse. Access to ResearchHarness allows the central engineering agreement to check the shared layout, pre-check and delivery agreements, and from `src/project/saebase/README.md` the corresponding realization of these agreements in the project.

- First, the results and conditions of completion required by the user are identified, the actual and direct callers are identified, and investigations, judgements and processing are done on their own initiative and within the delegated authority. Simple tasks are carried out directly; complex tasks are based on actual delivery and involve documents. When human decisions are needed, the facts, specific options and implications identified are provided.
- Accomplished routines are carried out by assistants on a continuous basis, and local engineering options are selected on the basis of defined objectives, actual benefits and maintenance costs.
- When it comes to larger adjustments in the use or functional value of research results, the user is assisted to judge the actual use as early as possible with representative topics, features or reporting clips; small-scale experiments with available or approved results are used, and follow-up and presentation is adjusted on the basis of feedback. Additional experiments are carried out in accordance with current mandates.
- When new or modified functions are developed, the actual use of the results is selected, the caller or subsequent mission is identified, and the inspection steps, the return of content and the way they are stored are determined. Information needs to be kept for long periods of time, while the search, reading, product location and subsequent processing is designed. Models, data, tasks and common support are implemented according to their respective uses; records and interfaces are available for priority reuse.
- The conditions for completion include the actual results of use: the required content is available and used when normal completion occurs; and the follow-up task can continue on the basis of the specific facts, implications and recommendations kept. The reasons for this remain known and the next way to locate.
- Read around the current problem. Distinguishing between document definition, actual code behaviour, operational observations and extrapolations; sources and versions are recorded as required by the mission. Historical results and synthetic samples are recorded as their respective sources.
- The additional precautions should be specific to the current code’s path of occurrence, the actual consequences and the corresponding costs.
- Each fact is determined by a clear module, followed by a reuse result. Regular checks focus on key locations such as input and shared interface.
- The solution obtained by the temporary check goes into the official code responsible for the act, cleans up the provisional check and duplicates the judgement of the completed use. The results show the actual impact and follow-up.
- The scope of the restoration, which is to be completed on the basis of this task, is to be determined through the normal user-used access delivery.
- Continuous development and maintenance missions leave an implementation record at `doc/code/maintenance/<maintenance_topic>/execution.md`, giving priority to the continuation of existing records on the same subject. Dates and implementation markers are included in the body of the text and are entered in the [maintenance index](doc/maintenance.md)]. The records begin with a background, objectives and holistic approach, add key actions and judgements over time, and conclude with results, AI comments and follow-up recommendations.

## Contents and duties

- `datasets/`, `models/`, `tasks/` for the preservation of scientific components; `base/application/registry.py` and `base/application/runner.py` for the uniform registration and distribution.
- `base/interfaces/` provides common models and SAE operations; `base/runtime/` maintains co-configuration, run paths and records. Re-use existing interfaces, and cross-component capabilities are designed on the basis of actual call needs.
- `scripts/examples/` Saves a full runable demonstration, `tests/fixtures/` Saves a composite component, with a teaching note at `doc/code/documentation/source/development/`.
- `conf/` Save Configuration, `scripts/` Save software release, distribution, document and code check entrance, `tests/` Save product testing. Business logic goes to the responsible module, scripts and tools are formally implemented.
- `doc/knowledge/` Preservation of scientific knowledge and terminology, `doc/code/` Preservation of architecture and source code information, `doc/code/documentation/source/development/` Preservation of development notes, `doc/code/maintenance/` Preservation and maintenance of implementation records. `experiment/` is dedicated to scientific experiments and their implementation records.
- `AGENTS.md` Preservation of project rules and catalogue duties, `.agents/skills/` Preservation of project access and maintenance skills.
- The SAEBase maintains the scientific source code, the operational entrance and the results of the experiment.
- `saebase/research_lab/` Processes according to the user's renewed maintenance task, daily Dataset, Model, Task develops components that focus on current uses.

## Configure maintenance

The configuration path under `conf/` is organized along the first two levels of the responsible source relative directory. Each configuration branch retains only one file, with the in-depth responsibilities expressed in the name section of the document. The exception is that the root of the project is configured `conf/config.yaml`. All new, modified, moved, removed or deployed configurations must be `tests/test_configuration.py`. Automatically run `python -m scripts.documentation conf --write` sync `doc/conf.md`, CI and release steps to verify the synchronized results. Synchronize the text when parameters, behaviour and conditions of use change.

## Inspection and scientific findings

Regular inspections are concentrated in a few key locations, and follow-up processes are re-enacted.

This new test script and support code is processed on a continuous basis: there is a continuing use for inclusion in formal realization or long-term testing; temporary mission deletion has been completed. The temporary input and test cache generated this time follows task clean-up.

The scope of the check corresponds to the actual changes. The selected partial modifications confirm the relevant inspection of the functional results; the results are sufficient to be used directly to answer the question.

Python use Ruff `0.15.22`.Save only this change while the new document is used before submission `python scripts/quality.py --fix`;Existing documents are used in a clear range `--include-modified`.passed GitHub Interface Writing Python , using the same tool `--stdin --filename <Accurate Path>` Check the actual contents. CI Read-only check..

The model maintains a clear source with SAE identity, weight, data disaggregation, indicators, numerical precision and historical results.

## Writing and communication

When you access ResearchHarness, you can use its root AGENTS.md for co-maintenance.

The functions, phases, errors and reports of SAEBase are named for their actual use; model names, mathematical logs and historical format identifiers are retained in their original meaning. Markdown and user-oriented descriptions are in Chinese, and codes and generic technical names are retained in their actual writings.

<!-- researchharness:projects -->
## ResearchHarness optional access

The project README, the relying list and the public entry points provide an independent installation, configuration and operation mode.
The item in this access log is marked as `saebase`,The central entrance is `../ResearchHarness`,Shared skills are on. Relative paths are analysed from the root of the project. `src/project/maintained/saebase/README.md`,Access to the project ' s engineering experience.
Engineering maintenance follows the physical entry and call relationship of the project.
The project code reviews, plans and authorizes the continuation of the project ' s GitHub process and current session authorization.
Humans provide research purposes, experience feedback and input preferences, and assistants take on engineering judgement and continuous processing within their delegated authority.
When `maintain-knowledge` is enabled, it can be used to organize research and reflections that are worth long-term preservation.
The project access is queryed by `rh projects status saebase`, and the user switch command is applied by `rh projects set`.
This machine data and Python selects the query by `rh projects resources saebase`.

The project access description is at `.agents/skills/researchharness/SKILL.md`.

### Daily maintenance and light authentication

When a maintenance request is received, skills are selected in accordance with the actual responsibilities in the public entry point. Read the corresponding `SKILL.md` and implement the applicable steps before editing, running or configuration changes are made.

| Maintain content | First-reading skills |
| --- | --- |
| Project-wide access, inspection or cross-component maintenance | [`maintain-project`](.agents/skills/maintain-project/SKILL.md) |
| Access, configuration synthesis, distribution of components or common operation | [`maintain-base`](.agents/skills/maintain-base/SKILL.md) |
| Dataset, sample reading, labeling, sorting or cache | [`maintain-dataset`](.agents/skills/maintain-dataset/SKILL.md) |
| Model, variant, construction or weight load | [`maintain-model`](.agents/skills/maintain-model/SKILL.md) |
| Task calculation, pre-check, indicator or outcome interface | [`maintain-task`](.agents/skills/maintain-task/SKILL.md) |
| Open Parameters, Defaults, Configure Presets or Overwrite Orders | [`maintain-conf`](.agents/skills/maintain-conf/SKILL.md) |
| Configure, run, restore or read the results of a single experiment | [`maintain-experiment`](.agents/skills/maintain-experiment/SKILL.md) |
| Independent script, test, document or external file | [`maintain-optional`](.agents/skills/maintain-optional/SKILL.md) |
| Code review, duplicate realization or functional gap | [`maintain-code`](.agents/skills/maintain-code/SKILL.md) |
| One-time clean-up and temporary engineering work | [`maintain-tmp`](.agents/skills/maintain-tmp/SKILL.md) |
| Assessment and maintenance skills per se | [`maintain-skill`](.agents/skills/maintain-skill/SKILL.md) |
| Collating and preserving research knowledge and thinking | [`maintain-knowledge`](.agents/skills/maintain-knowledge/SKILL.md) |

The light verification is enabled by ** after the change has been completed. ** runs the minimum inspection associated with the change from the actual entry. The structural, naming or document attribution changes are repeated at `rh projects layout check`, the whole project is accepted at `rh projects project check`, the experimental and precheck evidence is used at `rh projects experiment check` and `rh projects precheck check`. Behavior change prioritizes the light testing, configuration preview or precheck already available on the project, using the resource and boundary input available on the machine. Downloading, installation, official scientific operations and large-scale calculations are carried out according to mandate.

When delivered, describe the entry point, the behaviour covered and the results of the physical inspection.
<!-- /researchharness:projects -->

## Document Sync

From [Item documents](doc/README.md) Commands, open parameters, dependability, outcome interfaces, and indicator changes are updated with the description. The experimental report refers to the current configuration and existing product. The function reviews the actual entry, configuration and results in the document; the function changes are stated at the time of delivery, the basis of the check and the impact on the existing use..

# SAEBase Contribution and Development

## Purpose

Description of the maintenance of codes, documents and scientific interfaces, as well as the contribution to GitHub and the release portal.

## Scope

Applies to SAEBASe development and external contributions. Smart assistant works according to the root AGENTS.md 's current working pattern and user tasks; public contribution uses the GitHub process described below.

## Complete one change

When developing the functional programme, indicate how the caller or follow-up study will use the results to arrange for the input, inspection, return of information and information to be stored; maintain and read together with design, where the failure provides the specific cause, impact and follow-up. Maintain the model, SAE, data disaggregation and results source.

Select an existing check that will confirm the result of this function, using the minimum input when supplemented. The secondary code is incorporated into the formal realization or long-term test as a continuous use, and the code that completes the temporary use is cleared as task.

Development environment needs-based installation `python -m pip install -e ".[dev,quality]"`.Local development of running tests, project-level validation is a continuous integration, release preparation or task-specific arrangement. Document modification checks the content and links of actual changes. `python scripts/documentation.py --check`.

## Public contributions

Questionnaires are used to report problems and discuss needs and to extract requests for the submission of codes that can be consolidated by the maintainer.

Use new public interface `saebase.api` with `saebase.cli`.Synchronizes the actual caller and describes the impact when interfaces or lasting changes in format.[Packing instructions](../doc/code/documentation/source/development/packaging.md).

## Scientific research and software results

Synthetic input is used to check software behaviour; true weights, data and scientific agreement conclusions are based on the results of the actual operation. Third-party source code retains its source and licence.

## Dissemination and quality tool

The software release is carried out in accordance with a clear release assignment, see [the software release process](../doc/code/documentation/source/development/release-process.md)]. Versions, release notes and actual release products are consistent.

[Code Quality Description](../doc/code/documentation/source/development/code-quality.md)Description of local format and type check;[Ongoing integration statement](../doc/code/documentation/source/development/ci-maintenance.md)Description of the purpose of day-to-day inspections, central access and release of certification.

## Document & Code Review

The functional description directly explains the treatment of objects, input, calculate, output and use methods, and the necessary terminology is interpreted in conjunction with the actual object. The current operation is centrally maintained in the corresponding module, which other documents need to be connected to by reading; the README introduction capability, the mission statement preservation protocol, and the process of record-keeping once maintained.

Checks the real call, save and read after, cause uncompleted and follow-up, the interface changes to process the caller in a synchronized fashion. CODEOWNERS assigns review responsibility for the path, and the warehouse protection settings decide on the request to merge; the maintainer reads the results of the current branch and the application of the continuous integration and then processes the contribution.


## Attribution of scientific research and engineering operations

Data are prepared using `saebase-prepare-prompts` for environmental queries `saebase-check-env`. Scientific processes and inspections enter the corresponding Task; numerical references use `python -m saebase.api.reference`, and real pair checks use `python -m tasks.precheck.sae_checks`. Short appliance protocols are selected by `saebase-benchmark --pair-protocol cpu_pair --sae-key <name> --real --device cpu` or `--pair-protocol reference_pair --model <name> --real` and parameters are saved at `conf/tasks/precheck/task.yaml`. These short protocols save their own input quantities, budgets and statistics.

Project maintenance uses the following four entry points:

- `python scripts/quality.py`: Format and static check for selected codes; `select --base <commit> --head <commit>` Select to continue integration check.
- `python scripts/documentation.py --check`: Document link; `support --check` or `support --write` to read the assessment evidence and check or update the supporting form.
- `python scripts/distribution.py contents|install|roundtrip <directory>`: Separately check distribution, installation outside the warehouse and re-establishment of the source package.
- `python scripts/release.py --help`: Version source, conditions for publication and asset register.

Use of saved assessment evidence `python -m tasks.benchmark.results.archive --help`;Use of specific operational indicators and identification checks `python -m tasks.benchmark.results.evidence --output <Run Directory>`.Source Query Usage `python -m tasks.benchmark.sites.catalog sources --output <File>`.Central Query Use explicitly bound `rh` commands;a specified acceptance and integration test for central engine identity.

Lab interpretation, private execution, public package preparation and circuit display access is provided at their respective backends, using the help of the corresponding modules `python -m saebase.research_lab.backend.auto_interp_local`, `private_worker`, `publish bundle` and `published_records` in turn.

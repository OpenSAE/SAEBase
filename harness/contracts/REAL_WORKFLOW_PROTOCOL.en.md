# SAEBase real workflow execution protocol

Version `1.0`, prepared locally on `2026-10-30`. The full normative text is [REAL_WORKFLOW_PROTOCOL.md](REAL_WORKFLOW_PROTOCOL.md). This English guide preserves its operational requirements. The existing [catalog.json](catalog.json) remains version `1.1.0` with 22 reusable contracts. Each work plan selects those IDs before freezing its task scope, inputs, budget and acceptance criteria.

Preparing the package establishes neither candidate execution nor independent acceptance. Those stages require their own evidence.

## Workspaces and authority

Each candidate uses an isolated worktree from a pinned source baseline plus declared, hashed setup material. Twenty WP21 agents therefore have twenty independent writable worktrees, output roots and Python process states.

A direct user assignment of a specific case provides the basis for the case's bounded local work. Reading a work order, setting its status to ready or passing preflight does not create human approval. Governance, frozen criteria, evaluators, historical records, `.cursor/`, `.github/` and runtime governance remain read-only to candidates. Publishing, GitHub mutations and authoritative Atlas decisions retain their separate permissions.

Keep reviewer output outside candidate worktrees. The local `wp21_scope.py` hook profile checks a dispatcher-owned record of an existing user assignment and frozen specification hashes. Preparation remains `prepared_not_assigned`; acceptance-directory protection covers every topic. Hook rejections must be recorded and resolved by governance; candidates may not disable hooks, manufacture approval or weaken checks. Hooks are tool-level controls, not operating-system isolation.

## Real execution

Each case must exercise the relevant current SAEBase production interface. Task cases use the configuration, registry, runner and real task implementation. API cases call the designated production API. Artifact cases operate on files actually produced by SAEBase. Do not duplicate the tested software inside Harness and count that replacement as integration.

`configured_only`, `task.execute=false`, printed configurations and registry membership establish configuration only. Model loading, capture, encoding, intervention, training and evaluation each need the corresponding actual outputs.

Synthetic samples, small deterministic models and local test checkpoints are valid only when the case permits them and they are labeled synthetic. A case requiring a pretrained model must load the specified weights and retain the actual model/SAE identity, revision, content hash, layer and hook. Missing dependencies or weights are `ENV_BLOCKED`; a fake model cannot satisfy a real-checkpoint requirement.

Independent review checks actual import paths and production calls. Fixtures and explicit fault injection may support tests, but success paths cannot replace core computation with canned results or monkeypatches.

## Components, tasks and subtasks

Built-in components use `saebase/registry.py`, the component `plugins.py` files and `saebase/runner.py`. Third-party model/SAE/metric providers use the existing explicit `saebase.extensions` API and registry; candidates do not create another registry. Preserve `saebase.datasets` and the external Hugging Face `datasets` namespace.

Adding a dataset requires actual sample loading, stable identity, split semantics and invalid-input handling. Adding a model requires real construction and the required execution capability, shape/dtype/device semantics and identity. A YAML entry alone does not establish an integration.

Removal cases start with an independently supplied, runnable component whose setup identity is fixed. Remove the named experimental implementation, registration, aliases, configuration and active references. A fresh process must reject the removed ID while retained components still run. Keep historical experiments, checkpoints and failures. Removing a previously published component requires a separate compatibility decision.

The seven operation directories remain `capture`, `intervene`, `generate`, `circuits`, `explain`, `benchmark` and `train`. An explicitly approved experimental task ID may be implemented inside an existing category. `task.name` selects a registry entry. A subtask selector selects real behavior with validation and dispatch. `task.path` only defines artifact hierarchy and must begin with the actual task ID. Changing a directory segment does not add or remove a subtask.

## Parameters, experiments and sweeps

Define parameter type, default, allowed range, units, interactions, missing-value behavior and compatibility before implementation. Use Hydra's resolved configuration; explicit command-line values take precedence over defaults. Demonstrate that parameters reach computation and alter results as specified, including relevant invalid and boundary inputs.

The existing runtime `config_sha256` hashes resolved YAML text. Preserve that meaning. When semantic comparison requires key-order invariance, record a separately named canonical JSON configuration hash. Source, input, configuration and checkpoint identities are distinct.

Freeze sample/token/trial limits, seeds, time, memory, device and download scope. Record baselines, controlled changes, raw values, valid sample counts and metric denominators. Normal metrics must be finite. Failed or missing observations cannot silently become zero or disappear from aggregates.

Freeze the sweep grid, valid combinations, seeds and data use before execution. Selection experiments also freeze objective direction and tie-break rules. Pure enumeration/recovery cases explicitly omit optimization instead of inventing a target later. Every completed trial runs the real SAEBase task. Keep a complete ledger of planned, started, successful, failed, skipped and unexecuted trials with unique identities, actual commands and artifacts. Selection uses the declared evaluation split; confirmation data is used only after the configuration is locked. Resume only identity-matching, integrity-checked successful trials, preserving previous failures.

## Artifacts and recovery

The authoritative runtime directory is singular `experiment/`, resolved by `saebase/runtime/experiment_paths.py` and Hydra. Governance precreates each worktree's permitted output root. Keep the existing multirun layout and its actual MULTIRUN check.

Use the existing typed artifact directories and `create_analysis_group`. Its description starts with `目标：` and uses valid Context IDs. The current experiment validator checks analysis-group structure; passing it does not establish complete artifact integrity.

Default to `experiment.overwrite=false`. Use a new run identity for a repeat. Resume checks source, configuration, data, seed, model and checkpoint identity. Overwrite and resume have different meanings. Cleanup applies only to an explicit authorized temporary-file list. Archive and extraction checks cover member paths, sizes, hashes, corruption and path escape, preserving the original run on failure.

## Evidence and independent review

Use the WP21-specific `saebase.wp21.contract_receipt.v1` schema. It retains selected contract IDs and observable actions while permitting the frozen production-source scope. The legacy `trace.py` and acceptance flow constrain edits to Harness sandbox paths and must not grade WP21 product changes. Add a real execution evidence manifest with pinned source and candidate changes, actual import paths, input identities, model/SAE hashes, resolved configuration, observed environment, argv, working directory, timing, exit codes, stdout/stderr references and artifact path/size/SHA256 records. Record synthetic versus real-checkpoint evidence explicitly.

Exclude the manifest's own hash from its members; a separate checksum file may hash it. Do not include credentials or unrelated private machine data.

Reviewers independently run the frozen public behavior checks, inspect source changes, recompute relevant metrics from raw artifacts and test the same disclosed invariants on additional inputs. Hidden inputs do not authorize hidden new requirements. Preserve the distinction between design, preflight, candidate self-test and independent verification. Technical verification does not publish software or approve Atlas nodes.

Use the six existing categories in [FAILURE_ATTRIBUTION.md](FAILURE_ATTRIBUTION.md). Optional owner and stage fields distinguish baseline defects, candidate errors, governance gaps, environment blockers and evaluator scope changes. Governance maintains rules and evaluators; candidates report evidence and change only their authorized implementation scope.

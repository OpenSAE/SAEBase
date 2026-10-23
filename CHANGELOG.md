# Changelog

## Unreleased

## 0.5.0 — SAEBase Public Research Beta and Seven Workflows

- **OpenSAE website and Workspace:** seven example-first browser workflows at https://atheavenma.github.io/opensae/workspace.html, with token/feature inspection, intervention parameters, generation comparisons, circuit graphs, benchmark metrics, training records and interpretation diagnostics.
- **Real recorded results:** source-linked Gemma feature activations and intervention examples, native SAEBench measurements, Gemma-2-2B circuit graph, small original Sparsify training run, and independently distinguished auto-interpretation diagnostics.
- **Local operator GPU execution:** authenticated private queue for fresh prompt capture, teacher-forced SAE editing, and bounded 16–32-token generation. Results preserve model/SAE identity and hashed private input reference without publishing raw prompt text, weights or full logits.
- **Source-level interpretability infrastructure:** WP12/13 model and SAE adapters, reusable feature explorers, controls, report importers and native training/benchmark integrations. Source-level capability coverage does not imply independent live GPU verification of all 50 operations.
- **Deployment:** the public Playground runs on static GitHub Pages with example-first recorded/demo modes; fresh model execution requires a separately configured, access-controlled GPU Worker. The temporary test tunnel is not a stable public compute SLA.
- **Scientific boundaries:** low-quality or unreviewed feature explanations are not promoted as validated interpretations. Raw GPU archives, prompts without redistribution approval, checkpoints and operator tokens are not distributed. Software/package validation is separate from reproducing every GPU experiment.

## 0.4.0 — Research Lab Preview and 500-Prompt GPU Cohort

- **Research Lab browser and APIs:** responsive Results, Feature Explorer, Intervene, Analyze, Datasets and Reports workflows; read-only FastAPI and CPU Docker deployment with original verified Stage D historical results.
- **New natural-prompt cohort:** ten pinned public sources, 50 base prompt groups each (200 discovery and 300 evaluation), provenance/digest locks, conservative lexical duplication checks and private-text review gates. Source text is not redistributed in this software release.
- **Real checkpoint runs, as recorded by the GPU executor:** 48 × 500 source-group capture sites; six-site edited-logit comparisons; 300 featured deep-site statistical cards.
- **SAR integration:** 1,485 K16, 14 K8 and one K4 observed groups using the separately declared decoder-mediated residual-preserving coalition protocol and the original SAR estimator. The classic affine-decoder gate remains the default; genuine orthogonal-rotation control is pending.
- **RSS integration:** site-aware tokenizer positioning and resumable settings; executor reports 138,000 unique site-input occurrences across six sites and 500 parent groups. Meaning-preservation review of the transformation proposals is not yet complete.
- **Input Localization:** two six-factor controlled input designs with pinned synthetic parity checks, plus a real 64-state grid evaluated over its 55 observed-support feature columns.
- **Scientific and engineering provenance:** original-source pinned SAR/RSS regression, SHA-bound data manifests, identity-separated incomplete/mismatched historical sites, CPU job leasing, and an explicit privacy/authorization boundary for online GPU work.
- **Distribution boundary:** software and sanitized summaries only. Full raw GPU experiment artifacts are not present in the repository or automatically included in the versioned software packages; their independent off-machine verification and public data rights are separate release gates.


## 0.3.0 — Expanded Model Coverage and Extensible SAE Research

- **Developer onboarding and community:** add a persistent CPU-only first benchmark with saved metrics and reliability reports, short first-experiment/contributor/troubleshooting guides, local documentation link checks and a CI smoke test. Restore CODEOWNERS and GitHub feature/model-SAE contribution forms.

- **Third-party extension API:** explicitly discovered Python entry points for external model, SAE and benchmark providers, with strict hook/dimension/normalization/dtype and original source SHA256 checks. Integrates with the benchmark runner while preserving built-in model registration and scientific estimators.

- **General software release gate:** add version-independent candidate validation, six exact-commit CI checks, wheel/sdist/CLI integrity verification and SHA256 manifest generation. Publishing a new version requires explicit manual confirmation, an unused version tag and source-bound assets. The historical v0.2.0 acceptance gate remains separate.

- **CI maintenance lanes:** consolidate earlier real-checkpoint Pythia tests into one opt-in workflow, merge SAR/RSS reference parity and synthetic truth into a source-pinned research regression workflow, and add a separate weekly upstream-head compatibility check. Move expensive historical model/SAE and SAEBench validations to manual dispatch. Preserve existing GitHub run histories, source evidence and v0.2.0 release gates.

- **Packaging and namespace governance:** move command entrypoints under `saebase.cli` and plugin registration under `saebase.model_plugins` / `saebase.tasks.plugins`, retaining the old import paths as forwarding shims. Exclude archived verification records from new wheel/sdist builds (still available in Git and the dedicated evidence release). Verify all nine installed CLI commands, original package resources and a clean build from the sdist.

- **Benchmark evidence archive:** Checksum-verified compact evidence release, offline verifier, explicit raw-file recovery checks, and a publication workflow gated on successful main-branch CI. Original GPU runs are not reconstructed from summaries.

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
- **Verified model and SAE coverage (2026-10-16):** **48/48** real-checkpoint compatibility tests, **48/48** benchmark evaluations on 64 held-out prompt groups, and **48/48** reliability audits across **18** model-SAE configurations. See the [verified model and SAE matrix](docs/verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->

- **Earlier CPU checkpoint evidence:** 21/21 original CPU SAE weight/encode/decode checks with loaded-parameter hashes, a separately rechecked Gemma Scope 9B L20 source, and three genuine small-model benchmark and reliability runs (5 metric rows and 3 reliability audits per model). Original artifacts are linked from machine-readable ledgers. These CPU-era findings preceded the subsequently completed 48-site verification summarized above.

- **Early cross-model compatibility:** 15-model core coverage with three additional SAE architecture profiles; a validated 48-layer catalog and four depth-band selection, exact checkpoint provenance, CPU-only SAE-weight verification and 3 small-model genuine CPU acceptance workflows.
- **Native SAE loader compatibility:** official Qwen3 1.7B Base TopK, Mistral 7B community profile, native EleutherAI Sparsify reasoning SAE adapter and SAELens BatchTopK / Gemma Scope 2 4B / Llama Scope R1 converters.
- **Real CPU checkpoint acceptance:** Pythia 410M, Qwen3 1.7B Base and DeepSeek R1 Distill Qwen 1.5B each passed real activation capture, reconstruction, identity patch and nontrivial feature ablation with full model/SAE state hashes; benchmark and reliability audit runners and per-layer provenance were established before the subsequent 48-site evaluation.

- **Checkpoint loader validation:** actual original checkpoints for six SAEs passed isolated load/encode/decode; Qwen3.5-2B model+SAE passed live capture, reconstruction, and identity-patch checks. Added native Gemma Scope 2 and Llama Scope checkpoint converters and Llama-Scope norm folding. At the time, gated Gemma and Meta models and larger 8B model-pair checks retained access/compute blockers; later 48-site evidence is summarized above.

- **Initial model and SAE coverage:** register the official Qwen3.5-2B Base / 32K TopK SAE, align two Gemma Scope 2 release identifiers with upstream, and add a six-model 270M–8B real-checkpoint acceptance runner, workflow, reproducible evidence fields and offline registry tests. Verified status remains contingent on actual weight and hook tests.

- **Input Localization benchmarks:** exact 90%-variance scope budget and normalized-budget metrics, with source-checked full scope curves, feature assignments, design manifests, and JSON/HTML report links.
- **Localization reliability:** original paired-estimator seed stability on cached codes and matched descriptive cross-SAE comparisons, deliberately excluding invalid prompt-group bootstrap on full factorial input states.
- **Reference testing:** benchmark preflight, cache/source integrity, offline benchmark and reliability contracts and a native Localize execution example.

- **Controlled input designs:** complete binary factor enumeration, input-state permutations, stable underlying sample identities, explicit token-position selection and versioned provenance manifests that bind to existing activation/code caches.
- **Localize bridge:** accept a controlled design for live feature capture or a verified design manifest for cached SAE code analysis; retain the original optional estimator package.

- **Input Localization integration:** capture or reuse complete Boolean input grids of SAE codes, delegate optimal scope curves and optional paired estimates to the original independent Localize implementation, and provide an offline example and tests.
- **Documentation:** add Localize to the English/Chinese research integration overview and examples index.

## 0.2.0 — Reproducible SAE Evaluation, Interventions, and Reliability

- **Activation storage:** identity-bound sharded activation and dense/CSR code stores, with independently verified inputs and resumable execution.
- **Feature interventions:** batched ablation, scaling, clamping, addition, coalition and reconstruction conditions, downstream behavior measurements and reference steering.
- **Benchmarking:** versioned configurations, execution preflight, activation/measurement cache reuse, separate Core, reconstruction and intervention metrics, plus JSONL/CSV/HTML reports.
- **Research integrations:** unmodified original SAR/RSS estimators and provenance-checked native SAEBench Core and Sparse Probing imports.
- **Reliability audits:** grouped prompt bootstrap, paired comparisons, confidence intervals, sample-size curves, rank stability, known synthetic factors/interactions and negative controls.
- **Real checkpoint validation:** Pythia-70M and GPT-2 Small, with two different publicly available SAE checkpoint hashes per model, six fixed prompts per model and repeatability evidence.
- **Release integrity:** Python 3.11/3.12 checks, wheel/sdist verification, external checkpoint-source inventory and a GitHub Release with machine-readable evidence and SHA256SUMS.

**Scope:** Bounded engineering compatibility, not a broad comparative ranking of SAE quality. Third-party weights are not redistributed, and publication to PyPI is not implied.

## 0.1.0 — Initial Standalone Release Candidate

- Packaged model and SAE registry, configuration resources and core research interfaces.
- Added activation capture, residual-preserving feature edits and activation-health evaluation.
- Added versioned prompt artifacts and original-estimator integration examples.
- Added installation, command-line utilities, initial CI and contributor documentation.

This earlier release candidate did not establish broad external checkpoint or GPU generation reproducibility.

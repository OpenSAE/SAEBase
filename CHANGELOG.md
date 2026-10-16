# Changelog

## Unreleased

- **Developer onboarding and community:** add a persistent CPU-only first benchmark with M3/M4 reports, short first-experiment/contributor/troubleshooting guides, local documentation link checks and a CI smoke test. Restore CODEOWNERS and GitHub feature/model-SAE contribution forms.

- **Third-party extension API:** explicitly discovered Python entry points for external model, SAE and benchmark providers, with strict hook/dimension/normalization/dtype and original source SHA256 checks. Integrates with the original M3 benchmark and preserves built-in model registration and scientific estimators.

- **General software release gate:** add version-independent candidate validation, six exact-commit CI checks, wheel/sdist/CLI integrity verification and SHA256 manifest generation. Publishing a new version requires explicit manual confirmation, an unused version tag and source-bound assets. The historical v0.2.0 acceptance gate remains separate.

- **CI maintenance lanes:** consolidate M0/M1/M2 real Pythia smoke jobs into one opt-in workflow, merge SAR/RSS reference parity and synthetic truth into a source-pinned research regression workflow, and add a separate weekly upstream-head compatibility check. Move expensive historical model/SAE and SAEBench validations to manual dispatch. Preserve existing GitHub run histories, source evidence and v0.2.0 release gates.

- **Packaging and namespace governance:** move command entrypoints under `saebase.cli` and plugin registration under `saebase.model_plugins` / `saebase.tasks.plugins`, retaining the old import paths as forwarding shims. Exclude archived Stage D evidence records from new wheel/sdist builds (still available in Git and the dedicated evidence release). Verify all nine installed CLI commands, original package resources and a clean build from the sdist.

- **Stage D evidence archive:** Checksum-verified compact evidence release, offline verifier, explicit raw-file recovery checks, and a publication workflow gated on successful main-branch CI. Original GPU runs are not reconstructed from summaries.

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
- **Stage D support on current main (2026-10-16):** **48/48** real model-SAE pair acceptances, **48/48** 64-group engineering benchmarks and **48/48** reliability audits across **18** profiles. See the [verified layer matrix](docs/verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->

- **Stages B+C acceptance evidence:** 21/21 original CPU SAE weight/encode/decode checks with loaded-parameter hashes, a separately rechecked Gemma Scope 9B L20 source, and three genuine small-model M3/M4 runs (5 metric rows and 3 reliability audits per model). Original artifacts are linked from machine-readable ledgers. These CPU-era findings preceded the subsequently completed Stage D verification summarized above.

- **Model and SAE coverage (Stages B+C):** 15-model core coverage with three additional SAE architecture profiles; a validated 48-layer catalog and four depth-band selection, exact checkpoint provenance, CPU-only SAE-weight verification and 3 small-model genuine CPU acceptance workflows.
- **Native SAE loader compatibility:** official Qwen3 1.7B Base TopK, Mistral 7B community profile, native EleutherAI Sparsify reasoning SAE adapter and SAELens BatchTopK / Gemma Scope 2 4B / Llama Scope R1 converters.
- **Real CPU checkpoint acceptance:** Pythia 410M, Qwen3 1.7B Base and DeepSeek R1 Distill Qwen 1.5B each passed real activation capture, reconstruction, identity patch and nontrivial feature ablation with full model/SAE state hashes; M3/M4 audit runners and per-layer provenance were established before the subsequent Stage D evaluation.

- **Phase A genuine-weight acceptance:** actual original checkpoints for six SAEs passed isolated load/encode/decode; Qwen3.5-2B model+SAE passed live capture, reconstruction, and identity-patch checks. Added native Gemma Scope 2 and Llama Scope checkpoint converters and Llama-Scope norm folding. At the time, gated Gemma and Meta models and larger 8B model-pair checks retained access/compute blockers; later Stage D evidence is summarized above.

- **Phase A model coverage:** register the official Qwen3.5-2B Base / 32K TopK SAE, align two Gemma Scope 2 release identifiers with upstream, and add a six-model 270M–8B real-checkpoint acceptance runner, workflow, reproducible evidence fields and offline registry tests. Verified status remains contingent on actual weight and hook tests.

- **Input Localization benchmarks:** exact 90%-variance scope budget and normalized-budget metrics, with source-checked full scope curves, feature assignments, design manifests, and JSON/HTML report links.
- **Localization reliability:** original paired-estimator seed stability on cached codes and matched descriptive cross-SAE comparisons, deliberately excluding invalid prompt-group bootstrap on full factorial input states.
- **Reference testing:** benchmark preflight, cache/source integrity, offline M3/M4 contracts and a native Localize execution example.

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

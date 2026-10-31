# 03 — GPU Host Evidence Recovery and Remaining SAEBase Acceptance

> **Execution handoff for the GPU agent · OpenSAE/SAEBase**
>
> **Prepared:** 2026-10-16. **Status:** Handoff specification, not evidence that any task below has been executed.
>
> **Priority:** **P0 preserve the existing GPU data immediately**, **P0 independently verify and publish the original Stage D raw runs**, **P1 selectively validate the new external adapter path with real model/SAE weights only if useful**. Do not spend GPU time on tasks that only need CPU, GitHub admin settings or writing.
>
> **Latest reviewed main:** `06ce62c474115349a9a7558e636ba9b7528ee186`, but always fetch the live `main` before making changes. The current machine may be on an older experiment commit. **Do not reset or overwrite its working tree or experiment outputs.**

## 0. Read this first: what is already done

Earlier plans [01](01_stage_d_gpu_acceptance.md) and [02](02_stage_d_completion.md) were executed in subsequent Stage D PRs. Their historical "remaining site" counts and initial Mistral/Gemma bug reports are **not the current task list**. Current `main` has:

- [Final Stage D matrix](../../conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json): **18 model/SAE profiles, 48/48 selected pair sites, 48/48 M3 population benchmark reports and 48/48 M4 reliability reports**.
- [Verified layer matrix](../github_development_doc/verified-model-matrix.md), [Stage D compact evidence archive](../github_development_doc/stage-d-evidence-archive.md) and immutable [compact Evidence Release](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-evidence-20261016). The committed compact evidence contains 48 M3/M4 report summaries and checksummed metadata.
- [Raw run archive index](../../conf/coverage/stage_d_evidence/final/raw_run_archive_index.json) with **18 exact original .tar.gz filenames, byte lengths and SHA256 digests totaling 725,190 compressed bytes**. It names the former local directory `/workspace/stage_d_runs/archives`. These 18 original files **have not yet been published as publicly downloadable raw evidence**.
- [Raw archive and verification CLI](../../saebase/tools/stage_d_archive.py), [native Stage D raw-to-summary auditor](../../saebase/tools/phase_d_verify_evidence.py), [single-site acceptance](../../saebase/tools/phase_d_verify.py) and [population M3/M4 runner](../../saebase/tools/phase_d_benchmark.py).
- A new [versioned third-party extension API](../github_development_doc/extensions.md) and [first-experiment tutorial](../github_development_doc/first-experiment.md) added during engineering tasks 08–09. The tutorial's baseline is synthetic and CPU-only.
- Unresolved [Issue #21](https://github.com/OpenSAE/SAEBase/issues/21) specifically tracks the missing original 18 GPU archives.

**Do not rerun all 48 model/SAE combinations, retrain an SAE, change accepted scientific numbers, or construct new archives from compact report JSONs.** The first objective is to recover the *original bytes* and native measurements from the machine while they still exist.

## 1. What remains from engineering tasks 01–09

| Prior task | Current position | What this GPU agent should do |
| --- | --- | --- |
| 01. Support matrix and docs | Current 48-site matrix and docs are synchronized | No GPU rerun; verify current state and preserve links |
| **02. Evidence archival** | Compact release published; **original 18 raw archives missing from public release** | **P0: recover originals, validate SHA256, audit native M3/M4, publish a separate public immutable raw bundle** |
| 03. Community, CODEOWNERS, branch rules | Repository files largely done; main branch ruleset still needs GitHub admin action ([organization Issue #3](https://github.com/OpenSAE/.github/issues/3)) | **No GPU**; do not spend compute time or change repository settings |
| 04. Code quality/coverage | Ruff, mypy, CI and coverage established | **No GPU**; CPU unit tests only for any new recovery utilities |
| 05. Namespace and packaging | Source/wheel layout and backwards-compatible imports validated | **No GPU**; do not rebuild/swap experiment libraries without need |
| 06. CI lanes | Pinned research regression and manual real checkpoint workflows exist | Run **only one relevant manual checkpoint test if needed**, not every historical workflow |
| 07. General software Release Gate | Manual future software publishing built, historical `v0.2.0` preserved | **Do not publish a new Python software version** as part of raw evidence recovery |
| **08. External Model/SAE/Benchmark API** | CPU example and entry-point contract tested | **P1:** one genuinely loaded model+SAE plugin interoperability check using already cached weights, if access and budget allow |
| 09. Developer adoption | Offline first-experiment, docs and CI now exist | **No GPU** for [translation Issue #30](https://github.com/OpenSAE/SAEBase/issues/30) or [installable toy plugin Issue #31](https://github.com/OpenSAE/SAEBase/issues/31); optionally contribute one *separately labeled real checkpoint walkthrough* from P1 |

The existing population protocol uses 64 selection-separated evaluation groups based on recurring controlled prompt templates. These are valid recorded **engineering acceptance** units, not evidence of broad natural-language generalization. Do not relabel those records as a new scientific dataset.

## 2. Mandatory operating rules

1. **Preserve first, audit second, compute last.** Copy and hash evidence before installing new dependencies, cleaning HF caches, restarting the container, merging PRs or releasing the instance.
2. Work from the machine's original `/workspace/stage_d_runs` when available; it may contain locally modified Python code and files not yet on GitHub. Record both the original experiment Git SHA and current reviewed GitHub `main` SHA. Use a **separate current-source checkout** for new validators if the original checkout is dirty or uses a different revision.
3. Never write credentials, Hugging Face tokens, private model access artifacts, proprietary weights, personal input data or sensitive log lines to a public release. Check archive contents before publishing; sanitize only *new public copies*, while preserving byte-identical originals in restricted archival storage.
4. Never alter the original 18 .tar.gz files in place or overwrite an earlier record to force a pass. Byte mismatch means *failed provenance check* until the correct original is located.
5. Preserve existing `v0.2.0` software assets, original compact Evidence Release, Stage B/C historic evidence, Mistral native `hook_mlp_out` semantics, and Gemma Base/IT host identity. Do not silently change model weights, SAE normalization, dtype, hooks or benchmark protocol.
6. No new hardware rental. Use the already-running GPU only after P0 backup is secure. Reuse cached model/SAE checkpoints; keep optional additional experiments small.
7. Use precise status names: `original_raw_hash_verified`, `native_m3_m4_audited`, `public_redownload_verified`, `blocked_missing_original`, `blocked_access` and `new_experiment_only`. **A checksum pass is not numerical re-evaluation; a synthetic run is not a real checkpoint pass.**
8. Do not destroy, stop, delete or reconfigure the currently running GPU/volume without explicit owner authorization. The user may have other work on it. At completion, report the persistent-storage and GPU-rental status for a deliberate decision.

## 3. P0-A — Preserve the current GPU filesystem BEFORE anything else

Find the actual SAEBase checkout and Stage D data directories, then run the following from the **correct local repository root**. Do not assume that the original experiment checkout matches current GitHub `main`.

~~~bash
set -euo pipefail
pwd
git status --short
git rev-parse HEAD
git remote -v
nvidia-smi
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
df -h /workspace
du -sh /workspace/stage_d_runs /workspace/stage_d_runs/archives 2>/dev/null || true
find /workspace/stage_d_runs/archives -maxdepth 1 -type f -name '*.tar.gz' -print 2>/dev/null | sort
~~~

If the expected directory is missing, search mounted volumes, previous workspace roots, backup folders and already uploaded artifacts before marking it lost:

~~~bash
find /workspace -type f -name '*population_20261016*.tar.gz' -print 2>/dev/null | sort
~~~

Do **not** scan or print secrets. Collect a redacted environment/provenance inventory in a **new** output location, e.g.:

~~~text
/workspace/stage_d_recovery_03/
  inventory/
    gpu_environment.json
    experiment_git_revisions.json
    original_archive_files.json
    original_raw_tree_inventory.txt
  verification/
  publication/
  acceptance/
~~~

Required inventory fields: exact path, file size, original SHA256, model/SAE source references, original Git SHA(s), Python/PyTorch/CUDA/GPU identifiers, date of the original run, available storage, presence of native M3/M4 outputs and whether the raw data are backed up. The inventory is metadata only; no credentials.

**Create at least one durable, access-controlled backup of the original 18 archives immediately**, using an already authorized external location (owner-approved storage, a protected backup volume or other persistent destination). Prefer a second copy of the original complete native `runs/` and evidence/log directories if available. Re-download and recompute SHA256 on the backup copy before relying on it. **Do not depend solely on the current rented GPU disk.**

### A successful P0-A means

- [ ] All 18 indexed archive names are accounted for, or every missing name is explicitly recorded.
- [ ] The original files have not been modified, regenerated or renamed as if new.
- [ ] An independent durable backup exists and passes re-read checksum verification.
- [ ] Original experiment source and runtime identities are recorded, without exposing tokens.
- [ ] The owner has not lost any other unrelated data on the GPU machine.

## 4. P0-B — Verify the exact 18 original raw archives

Work from a clean checkout of the current repo for the verification saebase.tools. Let `REPO` be its absolute location:

~~~bash
cd /path/to/SAEBase
python -m pip install -e ".[dev]"
python saebase/tools/stage_d_archive.py --check
python saebase/tools/stage_d_archive.py --verify-raw /workspace/stage_d_runs/archives
~~~

This invokes the existing verifier against the [original raw-run index](../../conf/coverage/stage_d_evidence/final/raw_run_archive_index.json) and must return:

- `verified_archives = 18`.
- `original_bytes = 725190`.
- `status = byte_hash_verified`.
- All indexed filenames, individual byte lengths and SHA256 hashes match exactly; no unsafe TAR members or duplicate/missing archives.

For each original archive inspect the member listing before extraction:

~~~bash
tar -tzf /workspace/stage_d_runs/archives/<EXACT_INDEXED_FILENAME>.tar.gz | head -80
~~~

The angle-bracketed filename is a placeholder: substitute an **actual filename from the index**. Do not run the literal placeholder. Check for absolute paths, traversal entries, symlinks, checkpoint/model binaries, credentials, dataset redistribution restrictions and unexpectedly large uncompressed contents. Never use unfiltered extraction of untrusted data into an occupied working directory.

If the verifier fails, **preserve both original copies** and write a per-file error report; look for the exact matching archive on another volume or backup. Do **not** "repair" mismatched bytes, substitute a fresh experiment, rewrite the immutable index or mark P0-B complete.

### Create a checked standalone original-raw bundle

Only **after** the 18 original hashes pass:

~~~bash
python saebase/tools/stage_d_archive.py \
  --build-raw /workspace/stage_d_runs/archives \
  --output-dir /workspace/stage_d_recovery_03/publication

python saebase/tools/stage_d_archive.py \
  --verify-raw-bundle \
  /workspace/stage_d_recovery_03/publication/stage-d-evidence-20261016-original-raw.tar.gz

(
  cd /workspace/stage_d_recovery_03/publication
  sha256sum -c stage-d-evidence-20261016-original-raw.tar.gz.sha256
)
~~~

The resulting bundle is built from the **original 18 tar files**, the immutable raw index and an independent verification record. It is distinct from the already published *compact* archive. Copy the completed bundle and its own SHA256 checksum into approved persistent storage before proceeding.

## 5. P0-C — Recover native M3/M4 runs and audit all 48 sites independently

**This is a separate and stronger task than P0-B.** A 18/18 SHA256 pass proves that the original archives match the recorded bytes, not that all 48 scientific summary values can be reproduced from them.

1. Inspect which `runs/<m3_run_id>/` directories are already present under `/workspace/stage_d_runs/runs`. If absent, restore them from the exact, inspected original archives into an isolated directory. Preserve relative paths and permissions; **do not extract over the original source directory**.
2. Make an audit workspace containing the restored `runs/` plus a faithful *copy*, not an edited rewrite, of the repository's 48 compact `reports/` and canonical `status.json`.
3. Join each of the 48 [final matrix](../../conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json) rows with its compact report by `(model_key, sae_key, layer, hook_name)`. For every report, resolve its original `m3_run_id` and find the corresponding native directory.
4. Require the native files `resolved_config.json`, `plan.json`, `metrics.jsonl`, `metrics_manifest.json`, `reliability.json` and `reliability_manifest.json`. Re-check native ledgers with SAEBase's `saebase.benchmarks.results.read_metrics` and `saebase.reliability.report.read_audit`, including M3→M4 linkage and stored SHA256 hashes.
5. Compare each native M3 metric/condition row with the committed report's `metric_results`, `n_valid`, `m3_metrics_sha256` and `m4_audit_sha256`. Record exact numeric comparison results and tolerances; never silently reinterpret units, substitute means or drop a mismatched row.
6. Check the evaluation corpus and token ledgers against the saved `dataset_sha256`, `population_corpus_sha256`, selection/evaluation separation, `feature_ablated`, source hook, host identity and model/SAE weight digests. The accepted population protocol reports 64 independent evaluation groups; do not claim natural-language diversity or broad generalization.
7. Confirm Mistral's native `blocks.16.hook_mlp_out` normalization path and the genuine Gemma2 Base→IT host identity from **preserved accepted records**, rather than resurrecting the superseded reports described in 02.
8. Record for each site one of `native_m3_m4_audited`, `blocked_missing_original` or `failed_integrity`, with its original filenames, paths, source SHA and explanation. Count verified sites **only from actual native results found**.

Use the existing read-only auditor first:

~~~bash
python saebase/tools/phase_d_verify_evidence.py \
  --output /workspace/stage_d_runs \
  --evidence-root /path/to/SAEBase/conf/coverage/stage_d_evidence
~~~

If original `/workspace/stage_d_runs` lacks complete native paths, run it with `--output` pointing to the **fully assembled audit copy**, whose `status.json`, `reports/` and `runs/` must all match the canonical 48-site ledger. The `--evidence-root` directory is the **committed compact evidence** and checksum manifest. The flags above are taken from the actual script; don't invent extra CLI options.

The existing auditor compares compact summaries against source M3 rows but is **not an exhaustive numerical recomputation engine**. For each site, independently check the native M3 and M4 manifests and recorded metric values. If this requires a more stringent read-only verification script, implement it with a tampering/missing-native-artifact negative test; do not overwrite the original experiment code or repair the evidence to satisfy a new test.

Deliver:

~~~text
/workspace/stage_d_recovery_03/verification/
  native_m3_m4_48_site_audit.json
  native_m3_m4_48_site_audit.md
  raw_vs_compact_discrepancies.json
  original_artifact_paths_and_hashes.json
  verification_commands_and_stdout.txt
~~~

**P0-C passes only if exactly 48/48 native M3/M4 site reports have been located and independently checked, not merely because 48 compact summary JSON files exist.** Otherwise publish the *actual* pass count and keep Issue #21 open for remaining native evidence gaps. Do not download 48 models just to recreate missing historical output.

## 6. P0-D — Publish durable, independently downloadable original evidence

**Prerequisites:** 18/18 original byte hashes, safe content/provenance inspection, a completed 48-site native audit (or an explicitly incomplete but honest audit), and owner-approved GitHub/Hugging Face publication permissions.

Preferred archive destination: **a new, dedicated GitHub Release** under [OpenSAE/SAEBase](https://github.com/OpenSAE/SAEBase), for example `stage-d-original-raw-20261016`, or a versioned Hugging Face Dataset with immutable commit and public download links. **Do not move or overwrite**:

- `v0.2.0` software Release;
- `stage-d-evidence-20261016` compact Evidence Release;
- existing historical M3/M4 metrics or accepted site identity hashes.

Publish the generated `stage-d-evidence-20261016-original-raw.tar.gz` and its `.sha256` plus a short honest README/manifest explaining exact source run SHA, 18 original archives, contents, protocol, absence of model weights and **actual** native-audit coverage. Before publishing, check that the chosen tag and destination do not already exist and that no restricted/private data will be redistributed.

On a **separate clean checkout or machine**, download the public copy and independently run:

~~~bash
sha256sum -c stage-d-evidence-20261016-original-raw.tar.gz.sha256
python saebase/tools/stage_d_archive.py \
  --verify-raw-bundle stage-d-evidence-20261016-original-raw.tar.gz
~~~

Also re-run P0-C's native verification on the recovered bundle data, not on the original local working tree; record pass/fail site totals after redownload. A raw bundle's file hashes are not a substitute for restored-run metric audits.

Make a **small harness/github_development_doc/evidence follow-up PR** that links the immutable raw-archive release and numerical-audit report from:

- [harness/github_development_doc/stage-d-evidence-archive.md](../github_development_doc/stage-d-evidence-archive.md);
- [harness/github_development_doc/verified-model-matrix.md](../github_development_doc/verified-model-matrix.md);
- [Issue #21](https://github.com/OpenSAE/SAEBase/issues/21), with exact status.

Use a fresh branch from current `main`; run `python saebase/tools/check_documentation.py --check` and the normal Python 3.11/3.12 PR CI; merge only after sources and links have been verified. Do not place full raw caches or pretrained model/SAE weights directly in Git.

Close Issue #21 **only when** public redownload, exact 18 hashes, and all required 48-site numerical/manifest audits have passed. If an original is missing or privacy restrictions prevent publication, leave the issue open with precise blockers and preserve restricted backup copies.

## 7. P1 — One real external-extension interoperability experiment (only if cheap)

This addresses the only meaningful **new GPU-specific test** from engineering 08. It must **not** delay P0 or require re-running all 48 sites.

- Select **one already approved, preferably already cached** genuine model/SAE pair from the current catalog. First choice: `qwen3_1_7b_base` + `qwen3_1_7b_base_resid_w32k` at **L7** if both original weights and tokenizer are still cached and permitted. Alternative: an already cached Gemma3 4B PT/SAE site; use Mistral 7B only if its exact `hook_mlp_out` normalization is supported. Do not assume any files exist until checked.
- Test the **actual new `saebase.extensions` interface**, not merely the built-in registration path. Produce an external `ExtensionBundle`, explicit entry-point or registration, correct factory return values, pinned model/SAE source revisions, source license information, exact hook, layer, `d_in`, `d_sae`, dtype, normalization and tokenizer identity.
- Load **real weights** with `load_external_pair` and perform an actual activation capture, SAE encode/decode, identity patch, and one active-feature ablation. Verify finite, nontrivial outputs and hashes. Distinguish `loaded_unverified` / `probe_passed_not_checkpoint_certified` from full experimental validation.
- Optionally execute one compact original M3 Benchmark with a frozen, **separate** input set and explicit real loaded-weight provenance. If trying M4, supply a valid frozen group-aware dataset and raw measurement sources; M4 returning `unsupported` for an external metric without raw per-group observations is correct behavior and must **not** be called a full M4 pass.
- Create a separate end-to-end test report, configs and logs, without overwriting Stage D 48-site results. A single real plugin run proves **adapter interoperability on that exact checkpoint/site**, not that all external plugins work or that the SAE is generally reliable.
- Check the released public extension API is available from a **source checkout**, not the historical `v0.2.0` wheel. Issues [#31](https://github.com/OpenSAE/SAEBase/issues/31) (standalone installable toy plugin) and [#30](https://github.com/OpenSAE/SAEBase/issues/30) (Chinese docs) are CPU tasks; do not spend GPU time on them.

Deliver `verification/real_extension_interop.json` and, if reproducible, a small PR containing the example/config/test and precise source-revision references. If a required checkpoint needs a large new download, another GPU, access approval or protocol-changing adapter modifications, mark this P1 experiment blocked and prioritize the raw evidence backup instead.

## 8. P2 — Optional *new* diverse-prompt experiment (only after P0 is complete)

If the owner has authorized extra GPU evaluation and model/SAE weights are already local, optionally run **at most two** model/SAE sites (one small, one 4B–7B) on a new frozen evaluation with genuinely varied, independent requests. Keep the feature-selection split separate from evaluation. Prefer >=64 unique evaluation groups, with transparent sampling/provenance and permitted datasets.

Use the actual `saebase/tools/phase_d_benchmark.py` syntax; its supported controls include `--sae-key`, `--layer`, `--device`, `--output`, `--population` and `--population-path`. Confirm the new dataset conforms to the existing corpus contract **before** loading any weights. Do not use `--smoke-only` while describing results as a population test.

This is a **new exploratory experiment**, not an update or replacement of original Stage D population results. Save independent experiment/config IDs, exact tokenizer/weight/source hashes, separate inputs, M3/M4 metrics, checkable native raw records and logs; publish separately if authorized. No cross-model score or claim of general natural-language generalization without matching protocols and sufficient evidence.

Skip this P2 entirely if P0 evidence recovery or public redownload has not succeeded, or if the GPU has other owner-prioritized work.

## 9. Artifacts, PR and final acceptance contract

The GPU agent's **required** output is a factual handoff rather than another planning document:

~~~text
/workspace/stage_d_recovery_03/
  inventory/
    gpu_environment.json
    experiment_git_revisions.json
    original_archive_files.json
    original_raw_tree_inventory.txt
  verification/
    raw_archive_integrity.json
    native_m3_m4_48_site_audit.json
    native_m3_m4_48_site_audit.md
    raw_vs_compact_discrepancies.json
    original_artifact_paths_and_hashes.json
    verification_commands_and_stdout.txt
    real_extension_interop.json             # only if P1 was performed
  publication/
    stage-d-evidence-20261016-original-raw.tar.gz
    stage-d-evidence-20261016-original-raw.tar.gz.sha256
    public_release_and_download_checks.json # only after real publication
  final_handoff.md
~~~

Produce a final report that states:

1. GPU host location, original experiment commit, current source commit, environment versions, available disk and whether original data are backed up outside the current GPU.
2. **Original archives found/18**, **exact SHA256+size passed/18**, and total actual bytes; enumerate every missing or mismatched archive.
3. **Independently audited native M3/M4 site runs/48**, with links/hashes for source manifests and any mismatch. Do not report `48/48` on compact evidence alone.
4. Immutable destination and public URL of the raw archive, its final download SHA256, redownload test and whether a fresh machine can re-run the native audit.
5. If run, precise checkpoint identities and outcomes of P1/P2 with clear language about the new evidence and its limitations.
6. PR links, merged/unmerged state, CI results, documentation updates and Issue #21 outcome. If publication permission is unavailable, provide the exact verified local bundle plus a clear manual upload handoff, not a fictitious URL.
7. Remaining blockers and any unresolved 01–09 work that does **not** require GPU. Confirm the GPU has **not** been stopped or destroyed without approval.

### Mandatory completion gate

- [ ] P0-A: originals securely backed up outside the working GPU instance.
- [ ] P0-B: 18/18 exact original archive bytes, or an explicit, truthful missing/corrupt inventory.
- [ ] P0-C: 48/48 actual native reports independently verified, or an explicit count and list of missing/corrupt originals.
- [ ] P0-D: a separate immutable **public raw evidence** download, checked hashes and independently verified re-download, or a precise publication permission/privacy blocker.
- [ ] Documentation links and Issue #21 updated without rewriting original evidence.
- [ ] Optional P1/P2 outputs accurately separated from historical Stage D and left undone if not cost-effective.
- [ ] No model weights, API tokens, private data, original Release tags or non-GPU project files lost or overwritten.

**Work through P0-A to P0-D before doing optional GPU experiments. Do the actual recovery, verification and publication wherever permissions and original bytes permit. Preserve and report each remaining blocker instead of pretending the task is complete.**

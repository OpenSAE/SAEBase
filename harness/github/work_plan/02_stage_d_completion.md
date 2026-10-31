# 02 — Stage D GPU Completion, Scientific Fixes, and Final Acceptance

> **GPU-agent execution handoff — OpenSAE/SAEBase**
>
> **Purpose:** Finish *all remaining Stage D work*, repair findings from the independent review of PR #17, produce genuinely auditable model–SAE and M3/M4 results, and safely merge. This is a command to execute, not another proposal.
>
> **Current reference:** [PR #17](https://github.com/OpenSAE/SAEBase/pull/17) `stage-d/gpu-acceptance` at reviewed head `098103b804d4d0dd6e3df5043656014ec7619705`. It was OPEN at preparation time. The actual PR may change: inspect live GitHub state before making changes.
>
> **Prior task specification:** [01 — Stage D GPU Acceptance](01_stage_d_gpu_acceptance.md). This `02` document supersedes `01` wherever they conflict, especially the corrected **Mistral `blocks.16.hook_mlp_out`** site and the need for a genuine independent-prompt benchmark. Use `conf/coverage/stage_bc.json` as the catalog's source of truth.
>
> **Strict outcome rule:** Do not report "Stage D fully complete" on the strength of 48 three-prompt smoke tests alone. If any acceptance or larger-sample benchmark cannot be completed, preserve exact blockers and explicitly report incomplete status. Never invent weights, experiment results, GPU logs, or provenance.

## 0. First action: recover the actual live state and avoid repeating work

- [ ] Check current `main`, current PR #17 head, its CI and all discussion/comments. The initial reviewed PR was **83 changed files, 2 commits**, with all five conventional CI checks passing; do not infer that PR #17 has been merged.
- [ ] Review the real diffs, not only the PR description. In particular inspect:
  - `saebase/sae/stage_d.py` and `saebase/sae/loading.py`
  - `saebase/sae/registry.py`
  - `saebase/tools/phase_d_verify.py`, `saebase/tools/phase_d_matrix.py`, `saebase/tools/phase_d_benchmark.py`
  - `conf/coverage/stage_d_targets.json`, `conf/coverage/stage_d_evidence/status.json`
  - `conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json`
  - `tests/test_stage_d_acceptance.py`, `tests/test_stage_bc_coverage.py`
  - `harness/github_work_plan/01_stage_d_gpu_acceptance.md` and this file.
- [ ] Continue **on the existing PR #17 branch** unless PR #17 has already merged or been replaced; if merged, create a narrowly scoped follow-up PR on the actual newest `main`. Incorporate this `02` work plan into the working branch. Do not force-push, replace old evidence, or reset other agents' commits.
- [ ] Preserve the authentic previous acceptance results: 21/21 standalone CPU SAE checks, 4 prior model–SAE pair passes and their source digests. The Stage D PR claims 48/48 pair sites including 44 newly accepted GPU sites (plus one intentional Qwen3.5 2B GPU replay); independently check their actual JSON and origin.
- [ ] Inventory all recoverable GPU raw output in `/workspace/stage_d_runs`, remote persistent storage, GitHub artifacts, and any existing signed/hashed bundles **before loading or re-downloading models**. Record which M3/M4 raw runs and original configs still exist. Do not count compact summaries as equivalent to preserved raw runs.
- [ ] Produce a starting gap report `stage_d_gap_report.json` mapping *every* issue here to `unresolved / resolved_with_evidence / blocked`, the responsible file, reproduction, and expected validation. Keep and update it throughout execution.

### Reviewed baseline, not final certification

| Fact at reviewed PR #17 head | Observed value | Meaning |
| --- | ---: | --- |
| Registered catalog pairs/layers | 48 / 48 | `conf/coverage/stage_bc.json`, 18 profile entries |
| Submitted pair-engineering passes | 48 / 48 | Real 3-prompt capture + identity + feature edit artifacts, subject to targeted corrections |
| Newly performed GPU pair passes | 44 | Existing four pairs were separately accepted previously; Qwen3.5 2B L12 was also replayed for calibration |
| Heavy (4B–9B) layer pairs | 26 / 26 | Real three-prompt pair acceptance, not formal full-dataset M3/M4 |
| M3/M4 flags across full matrix | 20 / 48 | 17 representative Stage D `smoke_only` summaries plus 3 old CPU proofs |
| M3/M4 heavy layer coverage | 9 / 26 | 17 heavy layer sites still lack any M3/M4 acceptance |
| Representative Stage D reports committed | 17 | **Only compact JSON summaries**, not the full native M3/M4 inputs/cache/manifests |
| Original run dataset | 3 fixed prompts | **Not** a 64+ independent-group population audit |
| `status.json` current number of keys | 50 | Two stale/incorrect identity keys need reconciliation to the 48 canonical sites |
| `final/checksums.sha256` | Empty | End-to-end published evidence bundle is not validated yet |

A green ordinary CI run proves offline regression only; it does **not** replace a real GPU reproduction. The initial PR recorded an RTX 5090 (`~31.4 GiB usable`), PyTorch `2.11.0+cu128`, and CUDA 12.8. Retain original records and any accurate CPU/GPU source commit history.

## 1. Blocking scientific-correctness repairs: complete these FIRST

### P0-A — Restore original Mistral SAE normalization and verify the actual hook

**Current problem:** `saebase/sae/registry.py` correctly changed the Mistral checkpoint site to **`blocks.16.hook_mlp_out`**, matching its published `cfg.json`. However, `saebase/sae/loading.py` currently handles missing `TopKSAE.ln_std` during independent decode by setting `sae.cfg.normalize_activations = "none"`. This affects BOTH the input normalization before encoding and decoding, and therefore changes which TopK features are selected. The prior `resid_pre` site assertion in `01` is obsolete.

- [ ] Inspect the **exact pinned native SAE Lens version**, the published Mistral SAE checkpoint config, and the original `TopKSAE.encode` / `decode` / `process_sae_in` / `run_time_activation_norm_fn_in/out` implementations.
- [ ] Implement a source-faithful Mistral adapter that retains `layer_norm` semantics on actual captured **`hook_mlp_out`** activations. If SAE Lens' decode requires per-input `ln_mu` / `ln_std` state, preserve the per-input context correctly for baseline reconstruction AND feature-edited reconstruction. Avoid shared/stale normalization between prompts or layer patches, and avoid silently removing native normalization.
- [ ] Resolve whether upstream code fixes this directly or a small wrapper is needed. If wrapping, document the exact affine inverse and ensure no modification to the trained SAE weights or TopK feature selection; do not add arbitrary normalization or rewrite an upstream algorithm.
- [ ] Add regression tests showing codes/TopK feature indices, reconstructed activations, and edited activations agree with the unmodified original method on the same real input. Test multiple different prompts and repeated encode→decode→edit cycles (no `ln_mu/ln_std` state bleed). Assert source hook, dimensions, L0, nonzero ablation, finite reconstruction, identity patch.
- [ ] Run *real* Mistral 7B model + original SAE L16 on GPU again, with 3 engineering inputs and the **new independent evaluation protocol in `3`**. Recalculate M3/M4 with this corrected adapter; replace the previously misleading Mistral evidence only by retaining the old record in immutable audit history and issuing an explicitly superseding artifact with a new checksum and precise code/HF revisions.
- [ ] Confirm resulting SAE feature codes are not silently identical to the old disabled-normalization implementation. If they differ, describe the expected scientific change and cite the original SAE implementation.

**Exit gate:** Mistral native `hook_mlp_out` plus correct upstream SAE normalization; genuine corrected GPU intervention; new M3/M4; independent native-reference tests; old incorrect result identified as superseded.

### P0-B — Fix Gemma 2B Matryoshka Base ↔ IT host/transfer identity

**Current problem:** registry profile `gemma2_2b_it_matryoshka_l12_32k` has `model_key="gemma2_2b_base"`, `trained_on_model_key="gemma2_2b_base"`, `target_model_key="gemma2_2b_it"` and `transfer_kind="base_to_instruct"`. The submitted GPU L12 pair artifact actually loaded **`google/gemma-2-2b` (Base)** with its Base weights, while the M3/M4 report and canonical Stage D status row identify the host as **`gemma2_2b_it`**. The label is not evidence of an IT-host run.

- [ ] Keep the real Base-host result as **Base-native engineering evidence** under its genuine model identity, without claiming it verifies IT transfer.
- [ ] For the 48-site catalog's `gemma2_2b_it` L12 transfer row, actually load **`google/gemma-2-2b-it`** via an explicit `model_key_override="gemma2_2b_it"` (or properly tested equivalent), using the original trained SAE. Verify checkpoint/tokenizer compatibility and that returned `loaded_model_name` and full model-state SHA correspond to the IT host, not Base.
- [ ] Record `trained_on_model_key`, `evaluation_host_model_key`, `transfer_kind` and both exact source revisions in the acceptance JSON. Derive site identity, dataset ID, run ID, cache keys and report filename from the **actual host**.
- [ ] Reject an IT-labeled Benchmark when `loaded_model_name` / actual host disagrees, even if the SAE encoder and identity patch execute successfully. Add an intentional wrong-host regression test.
- [ ] Run real IT-host capture, reconstruction, identity patch, ablation and M3/M4 using the fixed evaluation dataset. Record a distinct source/dataset digest. Retain the Base baseline separately for an explicitly named transfer comparison, if useful.
- [ ] Rebuild `status.json` and `stage_d_final_matrix.json` from canonical catalog keys; do not simply rename the earlier Base hash as an IT model hash.

**Exit gate:** Actual IT-host evidence, not Base labeled as IT; two identities not conflated; 48 canonical site keys and checked transfer metadata.

### P0-C — Prevent false `--population` results

**Current problem:** `saebase/tools/phase_d_benchmark.py` currently has `prompts = PROMPTS if smoke_only else PROMPTS`; passing `--population` changes the label, **not the dataset**.

- [ ] Remove or hard-fail `--population` while it uses the three smoke prompts. It must never silently produce `protocol="population"` with 3 smoke inputs.
- [ ] Implement a real frozen-dataset interface: explicit evaluation dataset path, optional separate feature-selection dataset path, protocol ID, evaluation split/group IDs and deterministic input SHA. Use actual supported CLI flags *after implementation*; do not rely on flags not present in current code.
- [ ] Hard-validate `n_independent_evaluation_groups >= 64` for the final `population` label; evaluation prompt IDs/groups must not overlap the selection set. If not met, fail or mark `smoke_only`, never `population`.
- [ ] Keep `--smoke-only` or an equivalent clear option for rapid engineering tests, and retain historical `smoke_only` records as such.
- [ ] Add explicit test coverage that `--population` on three prompts fails, a frozen 64-group evaluation passes, changed dataset digests invalidate resume, duplicated groups or selection leakage are rejected, and M4 uses the frozen M3 results.

**Exit gate:** Every final `population` result points to a true precommitted evaluation dataset with at least 64 independent groups and matching manifest/hash.

## 2. Repair the evidence integrity and run-state system

### P0-D — Publish reproducible native M3/M4 evidence, not just a number

- [ ] Recover the original full M3/M4 directories `runs/<run_id>` on GPU/persistent storage for all earlier representative runs. A proper bundle includes `resolved_config.json`, `plan.json`, `metrics.jsonl`, `metrics_manifest.json`, `summary.csv`, `reliability.json`, `reliability_manifest.json`, `report.html` and any required source cache/manifests; preserve M1/M2 shards or provide immutable referenced objects and a verified retrieval protocol.
- [ ] Verify `read_metrics`, `validate_external_sources`, `read_audit` and `write_report` (existing native APIs) against the actual saved run. Recalculate the compact summary from underlying rows; do not hand-author or approximate numerical metrics.
- [ ] If original raw run caches were destroyed, say so explicitly. **Rerun those M3/M4 protocols** on the rented GPU as needed; never regenerate plausible JSON from compact summaries or try to reverse-engineer missing measurement rows.
- [ ] Store original complete evidence in durable external artifact storage approved for the project, with real read-back tests and SHA-256; GitHub only needs compact manifests, checksums, code/config snapshots and reliable retrieval links. Do not upload pretrained model/SAE weights, Hugging Face tokens, or credentials.
- [ ] Replace **empty** `conf/coverage/stage_d_evidence/final/checksums.sha256` with hashes for each committed compact result, manifests, selected config/dataset files and the external full-run bundle index. Hash by relative path, use a deterministic file ordering, and verify with `sha256sum -c` or Python. The checksum manifest should not refer to itself.
- [ ] Record the provenance chain `source_repo+revision → loaded state digest → layer/hook/dtype → dataset hash → M1/M2 identity → M3 run_id+manifest → M4 audit_id+manifest → report and source commit`. Add an automated independent audit command that rejects missing raw sources, mismatched status flags or unverifiable digests.
- [ ] Prefer compact relative or URI-based `acceptance_path` and `m3m4_summary` fields; avoid the currently committed `/workspace/stage_d_runs/...` absolute paths for portable public results.

**Exit gate:** Every counted M3/M4 result can be independently reconstructed or read and cryptographically checked from stored authentic files; `checksums.sha256` is populated and verification passes.

### P0-E — Make resume source-aware and tamper-resistant

- [ ] Current `should_skip_resume()` checks the acceptance JSON's self-checksum and three name fields but does **not** verify the current source revision, loaded checkpoint/dtype identity, hooked model version, dataset/protocol, exact git source commit, or existing benchmark files.
- [ ] Replace this with a source/config-aware resume fingerprint including at least `model_key, evaluation_host_model_key, sae_key, layer, hook_name, actual model & SAE revisions or loaded hashes, dtype, tokenizer revision/hash, selection/evaluation dataset hash, protocol version, git commit, relevant loader ABI` and the intended run mode. A changed configuration cannot be silently skipped under a matching filename.
- [ ] When revisions cannot be established without redownloading, do not optimistically assert that a skip is verified; mark `unverified_resume` or require explicit proof from a signed/cache-verifiable source artifact.
- [ ] Validate every `read_acceptance()` and `read_audit()` before trusting it. On corruption or missing dependent files, fail or schedule an explicit rerun while preserving the old failed artifact and the append-only attempt log.
- [ ] Test altered `model_state_sha256`, `sae_state_sha256`, hook, model host, dataset ID, precision, code revision, missing M3 manifest and tampered JSON. A bare recomputed self-checksum must not be accepted as proof that actual weights were loaded.

**Exit gate:** A stale/replaced checkpoint or changed dataset never yields a false `--resume` success.

### P0-F — Eliminate stale/misaligned keys in the final matrix

- [ ] The reviewed PR #17 `status.json` has **50** keys, but the registered catalog has **48**. The two extras are:
  1. Base-keyed `gemma2_2b_base|gemma2_2b_it_matryoshka_l12_32k|L12|blocks.12.hook_resid_post` while the canonical catalog expects an IT row. Resolve as P0-B.
  2. Failed legacy Mistral `blocks.16.hook_resid_pre` entry while current canonical source hook is `blocks.16.hook_mlp_out`. Preserve the failed attempt in `attempts.jsonl`, not as an active site.
- [ ] Rebuild one canonical `status.json` set containing **exactly 48 keys** from `conf/coverage/stage_bc.json` and exact `get_profile().hook_name(layer)`; keep superseded attempts in append-only history. Reject extras and missing rows during finalization.
- [ ] `build_final_matrix()` must derive `pair_verified` from a verified canonical acceptance object or the archived authentic prior CPU evidence, **not** from a manually edited `status` string or `skipped_prior` without provenance.
- [ ] Verify PR #17's **45 new acceptance JSON files** correspond to 44 new sites plus the one deliberate Qwen3.5 2B L12 calibration replay. The three other previously verified CPU sites must reference their actual earlier artifacts and digests, not a fake newly run acceptance.
- [ ] Replace `if args.finalize or True:` in `saebase/tools/phase_d_matrix.py` with real documented semantics and refuse to publish a "complete" summary when the stated acceptance gates have not passed.
- [ ] Confirm no science-relevant profile/hook change invalidates an existing SAE-only or pair/benchmark result; if it does, rerun ONLY the impacted profile's evidence and record the reason.

**Exit gate:** 48/48 canonical rows, no orphaned live keys, zero unaccounted aliases, every pass linked to an actual compatible authentic artifact.

## 3. Full scientifically honest evaluation protocol (mandatory for all remaining work)

### Frozen selection/evaluation data — implement before M3 reruns

- [ ] Use a versioned **public/non-sensitive** prompt corpus. Predefine at least **16 independent feature-selection prompts** and **64 independent evaluation prompt groups** per comparable protocol. Save a single canonical JSON/JSONL dataset, prompt IDs, groups, target split, tokenizer identity and complete token IDs with SHA-256. Do not generate or regenerate prompts based on the results.
- [ ] For a particular model/SAE layer, use only the selection split to choose feature ID(s), then **freeze** those IDs when testing the independent evaluation split. Never choose the strongest active feature separately on each held-out test prompt if claiming comparable intervention statistics.
- [ ] Predefine at least `identity` and fixed-ID `feature-ablation` conditions. Define and log hook, layer, selected token/readout positions, model weight precision, SAE decoder and group IDs; preserve changes in independent `run_id` values.
- [ ] Evaluate `core.l0`, `core.reconstruction_mse`, `core.explained_variance` (and `core.fvu` when stable), `intervention.logit_delta_l2` and optional registered `intervention.logit_delta_max_abs` and `intervention.kl_baseline_to_edited`. Use the existing exact MetricSpec identifiers and existing M3/M4 result schemas. **No invented metric names.**
- [ ] For the same model and prompt set, cache activations once per hook site, reuse for SAE comparisons and M4. Do not compare raw ablation magnitudes from different models/hooks as a universal SAE-quality ranking.
- [ ] M4 bootstrap only **independent prompt groups**, not repeated tokens or correlated variants; use evaluation-only seeds and predeclared resampling. Suggested `bootstrap_replicates=256`, `evaluation_seeds=(0,1,2,3,4)`, `sample_sizes=(8,16,32,64)` (all within 64 independent groups). If a metric is undefined due to sample variance, explicitly record `undefined`; do not fabricate confidence intervals.
- [ ] Original Localize metrics are **out of this matrix-wide arbitrary-prompt protocol** unless a separate complete controlled binary-factor design is constructed. Never fake a Boolean factorial grid for `localization.budget_at_90pct`.
- [ ] New population results must preserve frozen config, original token IDs, selection/evaluation hashes, model and SAE checkpoint SHA-256, native M3 files and M4 audit, evaluation group counts and metric provenance.

### GPU acceptance and budget-aware ordering

- [ ] Run the P0-A and P0-B targeted corrective experiments FIRST; do not repeat the 43 unaffected Stage D GPU pair sites.
- [ ] Confirm the 48-site real GPU/previous CPU **pair** evidence remains valid after corrections. Then prioritize the missing **17 heavy-layer M3/M4 runs** before chasing nonessential extra models, followed by all remaining small-layer Benchmark/Audit runs.
- [ ] **Minimum definitive Stage D scope:** all **26 heavy layer sites** must complete native M3 and M4 on the frozen independent dataset, and all **18 catalog SAE profiles** must have a completed representative independent-prompt M3/M4. To satisfy the user's request to finish **all** Stage D tasks, continue to all **48 canonical sites** and require `population` M3/M4 with >=64 independent evaluation groups, unless an explicitly documented external blocker prevents a site. The final report must distinguish 48 pair passes vs 48 full population M3/M4 passes and never present lesser results as the same achievement.
- [ ] Reuse exact model weights across sites of the same model, process checkpoints sequentially and unload old families to stay within a single 32GB RTX 5090 when possible. Run `microbatch=1` and short contexts initially; release logits, SAE modules and CUDA cache between families. Record actual peak VRAM, disk usage, duration and hardware rental rate/spend.
- [ ] Model-scale order: correct Mistral and Gemma transfer; complete Qwen3 8B, Llama3.1 8B, Gemma3 4B PT/IT, Gemma2 9B, Qwen3.5 9B and Qwen3.5 4B multi-layer M3/M4; then complete smaller uncovered layers and architecture profiles. Do not retrain or add 12B/27B/70B models beyond the catalog.
- [ ] The existing `phase_d_benchmark.py` `--population` flag is BROKEN as described above. The commands below are legitimate **existing smoke/test invocations**; do not claim they implement the future population interface until that interface is committed and tested.

```bash
# Work from the live PR #17 source checkout, with your access to actual GPU weights.
python -m pytest -q
python -m pip check

# No-download Stage D inventory:
python saebase/tools/phase_d_matrix.py --dry-run --output /workspace/stage_d_runs --device cuda

# Targeted current real pair acceptance, only if the existing evidence is invalid:
python saebase/tools/phase_d_verify.py --sae-key mistral7b_instruct_tc_l16_131k --layer 16 \
  --device cuda --output /workspace/stage_d_runs

# Existing M3/M4 command is a 3-prompt SMOKE until the real frozen dataset mode is implemented:
python saebase/tools/phase_d_benchmark.py --sae-key mistral7b_instruct_tc_l16_131k --layer 16 \
  --device cuda --output /workspace/stage_d_runs
```

## 4. Independent numerical audit and high-error investigations

- [ ] Inspect original checkpoint loader/source semantics for each architecture, particularly Gemma Scope 2 JumpReLU, Mistral normalized TopK, the original EleutherAI Sparsify, Qwen-Scope, and Matryoshka/BatchTopK converters. Native SAE behavior takes precedence over SAEBase convenience wrappers.
- [ ] PR #17 three-prompt data contain large raw reconstruction MSE values for some later-layer checkpoints: Gemma3 270M L12 roughly `5,300–7,350`, Gemma3 4B PT/IT L29 roughly `13,800–23,300`, Qwen3 1.7B Base L23 roughly `940–970`. Treat these as *diagnostics to investigate*, **not automatic failures**: activation scales can grow by layer. Inspect `mean(h²)`, `variance(h)`, normalized FVU, source norm conventions, selected tokens, and Native SAE-vs-wrapper reconstruction. The important question is whether the same original SAE applied to the correct activations gives consistent relative errors.
- [ ] Investigate very high `core.explained_variance` together with large raw MSE, and low/negative variance explained (for example Pythia410M L12 ~`-0.0383`). Confirm the metric denominator uses the correct populations, not accidental 3-sample estimation artifacts. Do not clamp negative valid values or edit outputs to improve appearance.
- [ ] Verify the Mistral normalized encoder output is source-consistent before and after editing a feature; the source change may legitimately change L0, reconstruction MSE and downstream effect.
- [ ] Ensure the `identity_patch_max_abs_logit_error` is measured at the exact same model hook and input position, not accidentally an unpatched output. A zero measured identity difference alone is not evidence of correct source semantics; also require a nontrivial active-feature edited logit change and source/method equality.
- [ ] Confirm model weights are genuinely BF16/FP16 when described as such: PR #17 acceptance field `precision="float32"` often reflects `sae.cfg.dtype` rather than the loaded **model** dtype. Record `model_weight_dtype` and `sae_weight_dtype` separately, along with conversion/offload status and GPU driver/CUDA versions; do not globally relabel to BF16 without checking parameter tensors.
- [ ] PR #17 artifacts contain `git_commit` values from a mix of `c7dc0d8...` and earlier `e6995f7...`, which can legitimately reflect sequential GPU code updates. Record exactly which source commit ran each observation; check loader semantics for each revision and rerun evidence affected by later code changes. Never silently rewrite old provenance to the latest PR head.
- [ ] Compare the original stage-A/B model and SAE SHA-256 for reused layers when precision and checkpoint transform are genuinely comparable; differences need a documented cause, not automatic equivalence.

## 5. Provenance, deliverables, and required files

Store durable raw experimental outputs **outside Git** and compact verified results **inside the PR**. Do not redistribute pretrained weights or credentials.

```text
/workspace/stage_d_runs/
  environment.json                 # accurate GPU/RAM/driver/CUDA/versions, rental information
  access_preflight.json            # source access and approved licenses (no token values)
  catalog_snapshot.json            # frozen 48-site catalog and current source git SHA
  stage_d_gap_report.json           # each issue in this 02 work plan
  status.json                       # EXACTLY 48 canonical keys
  attempts.jsonl                    # append-only failed, superseded, restored, and completed attempts
  acceptance/<model>/<sae>/L<layer>.json
  selection/selection_dataset.jsonl # independent, predeclared feature-selection inputs
  evaluation/eval_dataset.jsonl     # >=64 independent groups, frozen token IDs and sha
  runs/<m3_run_id>/                 # native complete SAEBase M3 cache/config/results
  reports/<model>_<sae>_L<layer>_m3m4.json
  final/stage_d_final_matrix.json
  final/stage_d_summary.md
  final/checksums.sha256            # real nonempty, hash-validated listing
  final/evidence_index.json         # permanent URIs/hashes for the full external bundle
  final/cost_report.json            # actual rental rate, runtime, SSD charges, total billed
```

GitHub PR paths should include:

- [ ] New/updated tested loaders and wrappers, targeted fixing tests, benchmark population-runner tests and resume/status-validation tests.
- [ ] `conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json` reconstructed from actual verified artifacts, with exactly 48 canonical IDs, independent pair/M3/M4 statuses and no fabricated flags.
- [ ] `conf/coverage/stage_d_evidence/final/stage_d_summary.md` with counts for pair, smoke-only M3/M4, population M3/M4, all 26 heavy sites, 18 profiles and 48 total, honest blockers, hardware and total cost.
- [ ] Nonempty `conf/coverage/stage_d_evidence/final/checksums.sha256`, which must verify every published compact artifact. Include an index for external archived M3/M4 complete directories with immutable hashes and retrieval instructions.
- [ ] Existing `conf/coverage/stage_b_cpu_verified.json` and `conf/coverage/stage_c_cpu_verified.json` unchanged except documented references. Do not throw away previous 21 standalone SAE checkpoint proofs or four accepted CPU pair proofs.
- [ ] Concise `harness/github_development_doc/model-support.md` / `harness/github_development_doc/stage-bc-model-sae-coverage.md` changes reflecting real evidence and no excessive defense paragraphs. If necessary, update `harness/github_work_plan/01_stage_d_gpu_acceptance.md` notes to explain the corrected Mistral hook without rewriting the original historical plan.
- [ ] An executable, read-only `saebase/tools/phase_d_verify_evidence.py` or equivalent that checks **all** final files and refuses success if any status is untraceable to accepted full evidence. Provide a test fixture that intentionally tampers with one digest and proves the verifier fails.

## 6. Test matrix and final release gates

### CPU tests that MUST run in GitHub CI

- [ ] Exact 48 catalog IDs, no duplicate/missing status keys; stale Mistral hook and Base-vs-IT wrong-host record rejected.
- [ ] A native-normalized Mistral TopK SAE reference on distinct prompts; correct edited reconstruction with no mutable-state leakage; no acceptance for "disable layer_norm".
- [ ] Wrong-model `loaded_model_name`, trained-on vs evaluation host mismatch, unexpected source repo, wrong hook, missing normalization buffers and SAE dimensions all fail before marking passed.
- [ ] `--population` cannot be invoked with 3 smoke inputs; 64 independent groups and disjoint selection pass; changed data or feature ID invalidate resume and data-hash-based run IDs.
- [ ] M3 rows, correct source and measurement refs, M4 audit configs, split/group integrity, no model forward during M4; corrupt manifest or unavailable raw source fails.
- [ ] Stale source/dtype/dataset/SAE SHA, tampered artifact SHA, old git revision and renamed alias must not trigger false `--resume`.
- [ ] `checksums.sha256` is complete, nonempty and verifies all compact JSON/markdown where specified; mutation to one source file is detected.
- [ ] Existing public APIs, benchmark schemas, file caches, SAR/RSS/Localize paths and `v0.2.0` behavior are not broken.

### GPU tests and quantitative closure

- [ ] Re-run corrected Mistral SAE normalization and real `hook_mlp_out` in three-prompt pair mode **then** >=64-group population M3/M4.
- [ ] Re-run Gemma2 2B Matryoshka on true IT host and its independent population M3/M4, with native Base checkpoint/host separately recorded.
- [ ] Confirm **48/48 canonical pair** acceptances retain genuine hash-linked evidence and all 26 heavy site pair records remain valid.
- [ ] Complete **26/26 heavy-site M3/M4 population** runs and **18/18 representative profile** runs; then fill every remaining 48-site M3/M4 population row for full closure. Every run has a native full dataset, complete metric/audit manifests and durable proof; no smoke row is promoted or counted as population.
- [ ] Recompute and verify **all final summary numbers programmatically**, including counts, representative coverage, per-layer metrics, peak VRAM/cost, versions and statuses. The final matrix must have exactly 48 entries and matching strict source identities.
- [ ] All five standard GitHub CI jobs and new coverage/integrity regression tests pass. A failing native GPU experiment is not waived because offline CI is green.
- [ ] Independent evidence verifier passes on downloaded raw bundles in a clean environment; all externally stored files have durable URIs and validated hashes.
- [ ] Final PR #17 (or transparent replacement, if state changed) has reviewable commits, complete evidence links, accurate before/after diff, discussion of the two science fixes and an honest "full" or "incomplete" status; **merge only after every unblocked gate has passed and remaining blockers are explicitly recorded**.
- [ ] Before destroying Vast, verify all raw logs/M3/M4 caches/JSON are safely copied to persistent storage and can be read back. Then **destroy the instance and any unneeded volume**, not just stop it, to end ongoing storage charges. Confirm in final handoff.

### Definition of "all tasks finished"

`Stage D fully complete` is allowed **only if** the two scientific semantic mismatches are fixed and retested; all 48 canonical sites have genuine pair evidence; every one of the 48 sites has a valid independent-prompt M3/M4 result with actual raw provenance; all 26 heavy sites are included; an immutable evidence archive and its checksum manifest verify; GitHub regression CI is green; PR is merged; and the GPU instance is destroyed after evidence backup.

If resources/access are insufficient, **do not falsely declare full completion**. Produce the maximum authentic completed work, preserve explicit blockers, and report exactly which rows need another execution. No blocking issue may be silently converted into a waiver, a simulated dataset, a fabricated digest or a three-prompt `population` label.

## 7. Cost and GPU operational guidance

- Start with the already proven **1× RTX 5090 32GB** (or **1× RTX A6000 48GB** if cheaper/better available) with **64–96GB RAM, >=8 vCPU and ~250–300GB fast SSD** if preserving several models' checkpoints between runs. The previous 5090 environment used a 300GB disk and peaked around 22.7GB allocated VRAM for a 9B model. These are *observed engineering numbers*, not guarantees for longer evaluation sequences; scale cautiously.
- Rent **on-demand** rather than interruptible when preserving reproducible evidence; select a host with good reliability and download throughput. Check current Vast.ai prices immediately before rental; never assume the price quoted in `01` is locked.
- Pin the exact source checkout and package versions, verify `torch.cuda.is_available()`, `torch.cuda.get_device_capability()`, one CUDA allocation, HF licenses, required download size, actual free disk, and fully loadable source IDs before starting the multi-model sweep.
- Charge/budget tracking: actual rental rate × elapsed hours + storage + network/transfer charges; track cumulative cost per family and report the final total, not just a guessed estimate.
- Reuse same-model BF16/FP16 weights across all layers; process one SAE checkpoint and one frozen dataset at a time. Avoid random retries or redownloading all 48 proven files. Avoid adding quantization to fit VRAM unless the new precision/protocol is explicitly recorded as a distinct experiment.
- Pause the GPU work only to address a concrete blocker (license, OOM, invalid hook, unavailable artifact storage); repair immediately and resume. Never stop after generating a plan or partial smoke report while requested final-evaluation tasks remain executable.
- Prefer uploading each completed model's raw M3/M4 output to durable storage before deleting its local cache. Once all proof is durably backed up and verified, delete the instance and any unnecessary persistent disk.

## 8. Handoff report the agent MUST deliver

At the end provide a concise but complete factual account in Chinese or English:

1. PR URL, merged state, merge SHA, current SAEBase `main`, and code revision(s) used for GPU runs.
2. Hardware (GPU/VRAM/RAM/SSD), CUDA/PyTorch/native SAE versions, actual price, runtime, storage/network expenses and total.
3. Exactly `pair_passes / 48`, `population_M3_M4 / 48`, `heavy_population_M3_M4 / 26`, `representative_profiles / 18`, plus counts and IDs of any `smoke_only` or blocked cases.
4. Mistral original SAE normalization reference check results and corrected GPU metrics; Gemma2 Matryoshka real Base-vs-IT host/weight comparison and matched dataset protocol.
5. Full archive location, data SHA/dataset identities, all M3/M4 manifest checksums, public matrix and final checksum audit results. Clearly state what is and is not independently reproducible.
6. Outstanding numerical anomalies and their source-grounded explanation, if any. Never infer overall SAE quality or causality from larger ablation L2.
7. CI summary, how to independently rerun integrity checks, and explicit confirmation of Vast instance/volume destruction after evidence export.

**Do not return with only a progress update. Execute the fixes, experiments, verification and release steps above wherever the GPU agent has the required model access and tools.** 

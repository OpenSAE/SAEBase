# 01 — Stage D GPU Acceptance and Final SAEBase Model Coverage

> **Execution handoff · OpenSAE/SAEBase**  
> **Status:** Not started; this document is the specification, not an execution report.  
> **Prepared:** 2026-10-15. Check current `main` before executing.  
> **Goal:** Finish authentic large-model and multi-layer Model–SAE acceptance, run M3 Benchmark and M4 Reliability with original public checkpoints, and publish a verified coverage matrix. Use one economical rented GPU; retain actual evidence and leave unsupported or unrun items explicitly unfinished.

## 0. Operating contract

This is a complete handoff for the next agent. It should be executable without relying on previous conversation history. Read the repository and linked source files first; source code and live checkpoint metadata take precedence over this plan if they have changed.

- Work in [OpenSAE/SAEBase](https://github.com/OpenSAE/SAEBase). The last B+C base commit reviewed was `44adb405715c37502bcbd3d25b34695d3c30088e`. Do not reset newer changes; start a fresh `stage-d/...` branch at the actual `main` HEAD. Open a PR, pass CI, and merge after genuine evidence checks.
- **No retraining** of model or SAE. We validate independently published pretrained checkpoint weights. **No redistribution** of third-party weights in GitHub.
- **Never infer verification** from a Hugging Face repository listing, the SAE registry, a green workflow, a fabricated result, or a CPU-only SAE check. An accepted pair requires a real LLM forward, compatible original SAE, finite reconstruction, identity patch, and nontrivial feature intervention with actual checkpoint hashes.
- Preserve public Python APIs, M1 activation/code cache identities, persisted schemas, prior experiments, and the historical `v0.2.0` release. Fix compatibility with narrowly scoped adapters and tests. Avoid unrelated README redesign, invented metrics, and defensive scope sections.
- Time and cost priority: run one model at a time, reuse downloaded weights and same-model activation caches, save results immediately, and do not re-execute previously accepted cases unless investigating an incompatibility. Keep the large-GPU workload bounded to the described matrices.
- Record failures clearly. `blocked_access`, `blocked_compute`, `blocked_loader`, `failed_numeric`, `failed_integrity`, and `passed` are different states; a skipped case is not completed. Do not relabel any row `passed` just to finish the table.

## 1. Baseline and source of truth

The real checkpoint source inventory is [conf/coverage/stage_bc.json](../../conf/coverage/stage_bc.json). Its schema is `saebase.coverage.stage_bc.v1`; the Python validator is [saebase/sae/coverage.py](../../saebase/sae/coverage.py). **Exactly 15 core profile entries plus 3 architecture entries, and 48 selected SAE layer configurations** are in the present catalog. The four depth bands, when an SAE exists near them, are 25%, 50%, 65%, and 85%; never assume an unreleased layer exists.

Completed evidence that must be **reused**, not repeated:

| Evidence | What it establishes | Source |
| --- | --- | --- |
| Stage C upstream objects | 48/48 selected source files exist; no tensor execution proven by this check | [Source HEAD run](https://github.com/OpenSAE/SAEBase/actions/runs/38829795455/artifacts/12017011100) |
| Stage C native CPU SAEs | 21/21 selected SAE weight loads, finite encode/decode and SHA-256; **not LLM pair tests** | [SHA-256 index](../../conf/coverage/stage_c_cpu_verified.json) |
| Stage C Gemma2 9B PT L20 repair | Original L0 57 source did not resolve; correct checkpoint is `layer_20/width_16k/average_l0_58`; real SAE-only test passed | [Focused artifact](https://github.com/OpenSAE/SAEBase/actions/runs/38828805176/artifacts/12015864245) |
| B: Pythia 410M + TopK MLP L4 | Real CPU LLM + SAE identity patch, feature ablation; M3 (5 metric rows), M4 (3 audits) | [B evidence](../../conf/coverage/stage_b_cpu_verified.json) |
| B: Qwen3 1.7B **Base** + Qwen-Scope L7 | Real CPU LLM + SAE identity patch, feature ablation; M3/M4 | [B evidence](../../conf/coverage/stage_b_cpu_verified.json) |
| B: DeepSeek R1 Distill Qwen 1.5B + Sparsify L20 | Real CPU LLM + SAE identity patch, feature ablation; M3/M4 | [B evidence](../../conf/coverage/stage_b_cpu_verified.json) |
| A: Qwen3.5 2B Base + official TopK L12 | Real CPU LLM+SAE capture, feature intervention and M3/M4 | [A evidence](../../conf/release/phase_a_targets.json) |
| A: Gemma 270M, Gemma 1B, Gemma2 2B, Llama 8B, Qwen3 8B | SAE-only success; the respective **model–SAE pair is not verified** (model access or compute pending) | [A status](../../conf/release/phase_a_targets.json) |

The confirmed pair checks in this 48-site catalog are therefore four sites (Pythia410M L4, Qwen3 1.7B Base L7, R1-Qwen1.5B L20, Qwen3.5 2B L12). **44 selected pair/layer sites remain without a real pair acceptance**, assuming no newer results exist. Only 21/48 have the Stage C *standalone SAE* weight check; the other 27 need actual SAE loading as part of Stage D. The count is a planning baseline, not permission to skip querying newer verified artifacts.

Inspect current implementation and contracts:

- [SAE profile registry](../../saebase/sae/registry.py): exact model key, legal layer, `hook_kind`, `sae_id`, `sae_folder`, source repo and allowed host model.
- [SAE loaders](../../saebase/sae/loading.py), [Sparsify adapter](../../saebase/sae/sparsify_adapter.py), [HF hooks](../../models/hf_hooks_backend.py): original weight loading, normalization, BF16 hooks and edited activation patch.
- [Direct model/SAE example](../../saebase/tools/phase_a_verify.py), [B CPU pair acceptance](../../saebase/tools/phase_bc_cpu_pair.py): numerical protocol examples; **B CPU pair loader explicitly rejects a CUDA device**.
- [B real Benchmark](../../saebase/tools/phase_bc_cpu_benchmark.py), [A real Benchmark](../../saebase/tools/phase_a_benchmark.py): M3/M4 templates; **the B example is CPU-specific**, and the A script is hardcoded to Qwen3.5 2B.
- [Benchmark CLI](../../saebase/tools/m3_benchmark.py), [Benchmark plan/config](../../saebase/benchmarks/schema.py), [M3 runner](../../saebase/benchmarks/runner.py), [M4 CLI](../../saebase/tools/m4_reliability.py), [AuditConfig](../../saebase/reliability/schema.py).
- [Stage C validation docs](../github_development_doc/stage-bc-model-sae-coverage.md) and [A gated-model status](../github_development_doc/phase-a-model-coverage.md).

Before proceeding: run `git status --short`, `git rev-parse HEAD`, inspect newer PRs and Actions evidence, and create a new coverage ledger from the current catalog. Do not overwrite completed CPU evidence or change previous `passed` hashes silently.

## 2. Vast.ai hardware, pricing and rental controls

**Default:** one RTX A6000 with **48 GB VRAM**, **200 GB local SSD/NVMe disk** (250–300 GB if planning to keep multiple 9B models cached), **>=64 GB host RAM** (96 GB preferred), **>=8 vCPU**, fast network. Choose **on-demand** (not interruptible), a verified reliable host (ideally >=99% reliability), suitable CUDA driver, and usable SSH/Jupyter access. A GPU rental with ~45 GiB actually shown by `nvidia-smi` is typical for a marketed 48-GB GPU; use reported free VRAM, not the product label, for admission tests.

- Search [Vast.ai](https://cloud.vast.ai/) for `RTX A6000`, `1x GPU`, >=48 GB marketed VRAM, local disk 200GB, and sufficient host RAM. Prefer approximately **$0.30–0.45 per GPU-hour when available**; those are October 2026 market references, **not guaranteed offers**. Inspect the **actual total hourly quote including storage and bandwidth** before renting. [Market reference](https://llmhosting.ai/gpus/a6000) and [Vast official pricing](https://docs.vast.ai/guides/instances/pricing).
- Faster alternative: 1x **A100 80GB**, L40S 48GB or RTX 6000 Ada 48GB if their *total* cost for the same work is competitive or the A6000 has OOM/performance problems. Do not rent H100 or multi-GPU by default. A 24GB RTX 3090 is not the recommended default for 8B/9B BF16 + large SAE + logits.
- Do **not** buy excessive storage. 200GB works only if one large model/SAE family is cached at a time and completed public weights are evicted after archiving evidence. 9B BF16 model weights alone are ~18GB decimal, and 64K–131K SAE checkpoints and HF cache revisions can consume multiple additional gigabytes. Monitor free disk. When in doubt, 250–300GB prevents a more expensive instance replacement.
- Vast bills **running compute**, **allocated storage even when stopped**, and potentially **traffic**. Container disk size is fixed at instance creation. Backup evidence before destroying the instance. Official references: [Pricing](https://docs.vast.ai/guides/instances/pricing), [Storage](https://docs.vast.ai/guides/instances/storage/types), [Instance management](https://docs.vast.ai/guides/instances/manage-instances).
- Before spending, compare at least 2–3 verified offers. Set a user-approved budget ceiling. The agent must record the rented GPU name, VRAM, host RAM, vCPU, disk GB, host reliability, actual $/GPU-hour, $/GB-hour storage, bandwidth tariff, instance ID and start/stop timestamps in the run environment record. Recompute remaining cost after the first model instead of promising a fixed total runtime.
- Rent an image with an NVIDIA driver-compatible CUDA/PyTorch runtime (Python 3.11); **verify actual compatibility first**, do not rely on an unverified image tag. Choose enough SSH ports, work inside `tmux`, and use local scratch storage for HF downloads.
- The Vast marketplace is a third-party host. Use only model-access tokens whose use is allowed; do not upload unrelated private data, bake secrets into images, print tokens, store tokens in `git`, or include tokens in logs. Prefer session-only `HF_TOKEN` injection over writing a persistent token on a rented host.

### Instance preflight commands (existing environment; no new SAEBase runner assumed)

~~~bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
free -h
df -h
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('vram GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY
~~~

Stop immediately if CUDA is not usable. Do not mark a GPU run as completed by falling back silently to CPU. A quantized or CPU-offloaded run may be diagnostic, but it cannot replace the designated BF16/FP16 full-checkpoint acceptance without being labeled as a different protocol.

### Clone, dependencies and cache (run after hardware preflight)

~~~bash
# Use a pinned SHA in real runs. Recheck HEAD before checkout.
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
git rev-parse HEAD
python -m pip install -e '.[analysis,dev]'
python -m pip install 'eai-sparsify==1.3.3'
python -m pip check
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/hf_cache /workspace/stage_d_runs
python saebase/tools/phase_bc_matrix.py --json > /workspace/stage_d_runs/catalog_snapshot.json
python -m pytest -q
~~~

The actual `HF_HOME` location must be on the allocated large disk; adjust the path if the Vast image mounts workspace elsewhere. **Do not write a Hugging Face token in any command that will appear in shared build logs**. User must accept license terms in their HF account before model downloads when required. Use `hf_hub_download`/Hugging Face metadata to test access and distinguish HTTP 401/403 from 404 or broken checkpoint paths.

## 3. Full target matrix: EXACT registered profiles and layers

The following is the complete Stage D ledger input, not a speculative list. Load all rows from `conf/coverage/stage_bc.json` programmatically; this table is a human inspection aid. `L#` means the integer layer in `profile.layers`. Hook names must come from `get_profile(sae_key).hook_name(layer)` rather than being guessed from model architecture.

| Class | model_key | sae_key | Selected layers | Pair work remaining / special gate |
| --- | --- | --- | --- | --- |
| A gated small | `gemma270m` | `gemma270m_scope_resid` | 5, 9, 12 | 3 pairs; Google Gemma access required |
| A gated small | `gemma1b` | `gemma1b_scope_resid` | 7, 13, 17 | 3 pairs; Google Gemma access required |
| A gated 2B | `gemma2_2b_base` | `gemma2_2b_pt_l25_canonical` | 13, 25 | 2 pairs; canonical SAE IDs and Google access |
| A 2B, already one verified | `qwen35_2b` | `qwen35_2b_resid_w32k` | 6, **12**, 15, 20 | L12 verified; 3 more |
| Flagship 8B | `qwen3_8b` | `qwen3_8b_resid_w64k` | 9, 18, 23, 31 | All 4 model pairs; L18 SAE-only passed |
| Flagship gated 8B | `llama31_8b` | `llama31_8b_lxr_8x` | 8, 16, 21, 27 | All 4; Meta license; Llama Scope normalization |
| B small, already one verified | `pythia410m` | `pythia410m_mlp_topk` | **4**, 12, 20 | L4 verified; 2 more, **MLP** rather than residual hook |
| B small, already one verified | `qwen3_1_7b_base` | `qwen3_1_7b_base_resid_w32k` | **7**, 14, 18, 23 | L7 verified; 3 more; **Base**, not chat model |
| B small, completed site | `r1_distill_qwen15b` | `r1_distill_qwen15b_resid_l20` | **20** | Already CPU pair+M3/M4 verified; do not repeat by default |
| Flagship 4B PT | `gemma3_4b_pt` | `gemma3_4b_pt_scope_resid` | 9, 17, 22, 29 | All 4; Google Gemma access |
| Flagship 4B IT | `gemma3_4b_it` | `gemma3_4b_it_scope_resid` | 9, 17, 22, 29 | All 4; **IT SAE matched to IT model**, license |
| Flagship 7B | `mistral7b_instruct` | `mistral7b_instruct_tc_l16_131k` | 16 | SAE-only CPU not done; ~4.3GB community SAE, **resid_pre** hook |
| Flagship 9B | `gemma2_9b_pt` | `gemma2_9b_pt_scope_resid` | 9, 20, 31 | All 3; L20 SAE mapped to `average_l0_58` |
| Flagship 9B | `qwen35_9b` | `qwen35_9b_resid_w64k` | 8, 16, 21, 27 | All 4; verify HF hybrid hook mapping |
| Reasoning 8B | `r1_distill_llama8b` | `r1_distill_llama8b_llamascope_l23` | 23 | One pair; native R1 Llama Scope SAE, not regular Llama 3.1 Base |
| Architecture | `qwen3_1_7b` | `qwen3_1_7b_ak_resid_l14_65k` | 14 | BatchTopK; Qwen3-1.7B (non-Base) |
| Architecture | `gemma2_2b_it` | `gemma2_2b_it_matryoshka_l12_32k` | 12 | Matryoshka/BatchTopK; audit checkpoint trained-on/host compatibility and Gemma access |
| Architecture | `qwen35_4b` | `qwen35_4b_matryoshka_l15_65k` | 15 | Matryoshka/BatchTopK; native SAE decoder/patch semantics |

**Count invariant:** the above 18 rows contain 48 layer sites. Four sites have already passed *pair* acceptance; remaining expected unverified pair/layers = 44 unless newer evidence exists. The nine primary 4B–9B model/variant rows (Gemma 3 4B PT/IT, Mistral7B, Qwen3 8B, Llama3.1 8B, R1 Distill Llama8B, Gemma2 9B PT, Qwen3.5 9B, Qwen3.5 4B Matryoshka) contain **26** selected layer sites. The full catalog additionally includes gated smaller Gemma models, remaining layers of verified small models, and two other architecture extras. Those must not be silently omitted from the final 48-site ledger.

There is no requirement to train SAEs or download every possible checkpoint from an entire upstream repository: only the selected 48 source checkpoint entries are in scope. Gemma 3 12B/27B, other huge models and additional arbitrary SAE widths are **out of scope** unless the owner later explicitly extends the matrix.

## 4. Required engineering to implement BEFORE the costly validation sweep

Existing B/C scripts are **not** a universal GPU acceptance CLI. The following are deliverables to add on a new Stage D branch:

- [ ] `saebase/tools/phase_d_verify.py`: CLI accepting `--sae-key`, `--layer`, `--device cuda`, `--output`, `--resume` (or equivalent documented interface), with explicit `model_key_override` only when justified. Use `load_model_for_profile()` + `load_sae_for_profile()` or `load_model_and_sae()` and `ModelAdapter`, `ActivationSite`, `capture_features`, `edit_features`; preserve the original source-specific SAE decoder and physical hook. Support **all 18 catalog profiles** and the exact selected layers. A hardcoded Stage A `TARGETS` mapping is insufficient for Stage D.
- [ ] `saebase/tools/phase_d_matrix.py` or equivalent resumable orchestrator: parse the existing catalog, check previous checkpoint evidence for exact SHA/source match, schedule selected sites and one model family at a time, skip proven passes, and write results atomically. Expose `--model`, `--sae-key`, `--layer`, `--dry-run` and `--resume` to rerun one failed checkpoint without rerunning 47 others. Print progress and disk/GPU peak usage after each model.
- [ ] `saebase/tools/phase_d_benchmark.py` or documented M3 config generator + CLI: derive source-consistent BenchmarkConfig, freeze tokenizer identities, selection/evaluation sample IDs and one SAE feature for intervention; run real M3 using the correct hooked model loader; save all M3/M4 audit artifacts and validate them on re-read. Reuse M1 across SAEs at the same model+site. Consult `saebase/tools/phase_a_benchmark.py` and `saebase/tools/phase_bc_cpu_benchmark.py` for **real schema field names**. The latter is explicitly CPU only; do not pass it a fake `--device cuda`.
- [ ] `conf/coverage/stage_d_targets.json` or equivalent input overlay for **execution order, access state and statuses**, derived from `stage_bc.json` rather than another source of truth. Preserve A/B/C historical evidence unchanged; join records on `(model_key, sae_key, layer, hook_name, checkpoint SHA)`.
- [ ] Independent CPU-testable unit tests for all newly introduced code: complete 48-site scheduling, no duplicate run IDs, skip/reuse, resume after partial writes, source/dimension/hook mismatch, patch/no-op detection, empty or invalid outputs, OOM classification, corrupt/tampered SHA evidence, wrong PT/IT weights, gated model failures, and unavailable optional pretrained libraries. No real GPU is necessary for these unit tests.
- [ ] GPU smoke script and installation instructions; run `python -m pytest -q` + `python -m pip check` before checking out many GB of weights.

Important implementation caveats:

1. **HF/TransformerLens adapters.** Qwen3.5 uses a non-classic hybrid architecture with a native HF residual hook backend. This was validated on Qwen3.5 2B. Check the exact 9B and 4B decoder site under their real weights, rather than assuming identical internal module paths. BF16 precision and GPU model placement must be consistent; use inference mode.
2. **Hook identities.** Mistral SAE is at `blocks.16.hook_resid_pre`; Pythia410M TopK is `blocks.{L}.hook_mlp_out`. Do not replace either with `hook_resid_post`. Validate pre/post norm, exact layer, selected token position and SAE input width before trusting results.
3. **Original SAE implementations.** Distinguish Standard, TopK, BatchTopK, JumpReLU, Gated and Matryoshka; reuse upstream SAELens/EleutherAI Sparsify adapters. Avoid converting BatchTopK/Matryoshka into a different sparse encoder, and inspect threshold/norm folding on Llama Scope. An SAE-only synthetic input test does not establish trained-model activation compatibility.
4. **Transfer/matching.** Never evaluate a PT SAE on an IT model and label the result native IT without explicit recorded transfer mode; the `gemma2_2b_it_matryoshka_l12_32k` profile requires a check of its trained-on and target model identity. R1-Llama SAE is not automatically interchangeable with regular Llama Scope.
5. **Integrity.** Check loaded model state and SAE state SHA-256; record source repo, exact source file path, resolved revision, layer/hook, dimensions, model access status, CUDA/PyTorch/SAELens/Transformers versions, precision and `git` SHA. The source metadata SHA/ETag alone is not a loaded-weight checksum.

## 5. Exact numerical acceptance protocol for each model–SAE layer

Execute **one full original LLM + one original SAE** at the designated hook and layer. For a new layer, perform at least **3 fixed, logged real prompts** before any 64+ sample Benchmark, with fresh, deterministic tokens and a documented selection position.

- [ ] **Source match:** `get_profile(sae_key)` supports the selected layer; host model key belongs to `allowed_model_keys()`. Correct release/folder file exists and is loaded, not an anonymous placeholder. Source file/revision and actual loaded model/SAE parameter digests are captured.
- [ ] **Finite capture:** actual `capture_features` produces activation `[d_in]`, SAE code `[d_sae]`, reconstruction `[d_in]`; dimensions equal published upstream model and SAE. All values finite, expected `L0` reported, and native SAE `encode/decode` functions actually used.
- [ ] **Identity patch:** compare the baseline real model logits and an identity patch containing the **same saved activation** at the **same hook and token position**. Verify identical shapes; initial limit maximum absolute logit error `<= 1e-2` (existing Phase A/B protocol). If hardware precision induces a reproducible larger error, do not quietly relax the threshold: isolate rounding vs wrong hook, report both errors, add a documented dtype-specific acceptance rule and regression tests if scientifically justified.
- [ ] **Feature ablation:** choose at least one active SAE feature from a *separate recorded selection input*, set its activation to zero via the original SAE decoder and SAEBase intervention API, reinsert the edited activation, compute downstream logit L2 difference; require finite, nontrivial change (existing engineering threshold `>1e-8`). Do not call replacing a full reconstructed activation with no residual-preserving handling a feature-specific effect.
- [ ] **Value checks:** record per-prompt L0, reconstruction MSE, FVU/explained variance when defined, selected feature IDs, identity patch error, feature edit change, baseline/edit logits checksum or compact summary. Do not claim higher SAE quality because ablation produces a larger logit delta.
- [ ] **Reproducibility:** deterministic seed and exact input token IDs / prompt hashes; compare after reloading at least one selected site on a second invocation. Make sure resume does not overwrite previously verified results with a failed or different source.
- [ ] **Resource telemetry:** wall time for loading/download/capture/patch; peak `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` and `nvidia-smi` usage, host RAM, disk high water and any OOM retry. Clear GPU cache and release the old model before moving to a different model family.

A successful SAE-only encode/decode job without the LLM **does not** pass this protocol. If the original model cannot be accessed, create a `blocked_access` record and move on; if a model or SAE cannot fit in VRAM, create a `blocked_compute` record and move on or upgrade the GPU, but never mark it passed. Capture stack traces in a separate sanitized log.

## 6. Fast, inexpensive order of execution

Optimize for *few model reloads and early detection of real compatibility defects*, not for the number of YAMLs visited.

**D0 — Before renting (no GPU cost).** Validate or request Hugging Face access for `google/gemma-3-270m`, `google/gemma-3-1b-pt`, `google/gemma-2-2b`, `google/gemma-3-4b-pt`, `google/gemma-3-4b-it`, `google/gemma-2-9b` and `meta-llama/Llama-3.1-8B`; verify actual current repo names from profiles, not this display list. Record any owner account restrictions before renting. All other model sources should also be HEAD checked. Pin GitHub and HF revisions and prepare the test prompts and config offline. Never bypass a gated model or assume an access token is approved.

**D1 — Cheap confidence check on GPU.** Load Qwen3.5 2B Base SAE L12 (previously verified on CPU), then run one actual GPU identity/ablation smoke and verify source hashes are consistent where the dtype/protocol is the same. This is the only intentional replay for infrastructure calibration. Confirm M3 cache and outputs persist, CUDA logit patch works, and measured throughput; then unload it.

**D2 — One representative site for each not-yet-validated large family** in this order unless access dictates otherwise:

1. Qwen3 8B Base L18; Qwen3.5 9B Base L16; Mistral 7B Instruct L16 (includes its large, not-yet CPU-verified SAE).
2. DeepSeek R1 Distill Llama 8B L23; Llama 3.1 8B Base L16 if Meta access is approved.
3. Gemma 3 4B PT L17; Gemma 3 4B IT L17; Gemma 2 9B PT L20 if Gemma access is approved.
4. Qwen3.5 4B Matryoshka L15, then the other architecture-only profiles (Qwen3 1.7B BatchTopK L14 and Gemma2 2B IT Matryoshka L12).

The nine primary heavy model/variant rows together contain **26 site-level targets**, not merely nine tests. D2 validates a representative site; D4 below covers their remaining registered layers.

**D3 — Finish A's gated small models and the remaining small layers.** Gemma 3 270M L5/L9/L12, Gemma 3 1B L7/L13/L17 and Gemma2 2B PT L13/L25, plus the other registered Pythia410M L12/L20, Qwen3 1.7B Base L14/L18/L23 and Qwen3.5 2B Base L6/L15/L20. Reuse the four already-accepted CPU sites. Include the correct trained-on host for Gemma2 2B IT Matryoshka, avoiding an accidental PT host swap. These cases are small enough that some may be possible without high GPU, but are still outstanding **site-level pair** validations.

**D4 — Extend the heavy models to every catalog-selected layer.** Process each model family in sequence: load one model, cycle through its 1–4 selected layer SAE checkpoints, verify source/hook and interventions at each, cache each layer's activations, and move to the next model only after uploading its compact evidence. Do not batch multiple huge models or multiple 65K SAE decoders into VRAM simultaneously. Use the four configured depth anchors only when actually released.

**D5 — M3/M4 benchmarks and final acceptance.** For each of the 18 SAE profile rows, complete one representative real M3/M4 run unless previous proof exists at that same site; then run cross-layer M3/M4 for all 26 heavy registered sites where resource access permits. The full 48-site table should identify `pair_verified`, `benchmark_verified`, `reliability_verified` independently. Scientific cross-model rankings are **not required** and must not be constructed from raw L0/logit deltas with unequal datasets or activation definitions.

**D6 — Publish, clean up and destroy.** Consolidate verified source/weight hashes, run artifacts, metadata, cost, files and limitations into a machine-readable coverage matrix and short documentation update; run standard CI, create a PR, and merge. **Before destroying Vast**, copy all evidence to persistent storage and verify it opens. After confirming artifacts are safely stored, destroy the Vast instance (not merely stop it) to cease storage charges.

## 7. Benchmark and Reliability procedure (new runner should wrap existing APIs)

The working M3 example and its exact schema are in `saebase/tools/phase_a_benchmark.py`; use `BenchmarkConfig.from_dict(...)`, `build_plan(...)`, and `execute(config, output, real=True, loader=...)`, with the correct original GPU model/SAE wrapper. Alternatively the existing CLI is `saebase-benchmark --config <path> --dry-run|--real --output <dir>` if its default loader resolves the model properly. **First test the default loader; implement a proper loader callback for unsupported native HF variants instead of submitting a fake successful run.**

Recommended M3 metrics (all currently registered):

- `core.l0`
- `core.reconstruction_mse`
- `core.explained_variance` and optionally `core.fvu`
- `intervention.logit_delta_l2` and `intervention.logit_delta_max_abs`
- optionally `intervention.kl_baseline_to_edited` if its native measurement is finite on the selected readout protocol.

Conditions must be explicitly registered in the M3 config: `identity` and at least one controlled `ablate` condition with an actually activated feature ID. M3 schema uses `targets`, `dataset.samples` (token IDs and sample records), `metrics`, `conditions`, and capped `max_activation_rows` / `max_measurements`. The run stores `runs/<run_id>/metrics.jsonl`, `metrics_manifest.json`, `summary.csv` and `report.html`; M1/M2 may write sibling integrity-checked caches. **Do not overwrite these schema IDs or use direct invented command-line flags.** Check `saebase/tools/m3_benchmark.py`/`saebase/tools/m4_reliability.py --help` before automation.

Evaluation protocol:

1. First do the 3-prompt smoke for correct wiring. Then **preselect** a bounded evaluation set (default suggestion: 64 independent grouped prompts for the representative checkpoint, more if cost permits) and freeze its token IDs, dataset hash, grouping, seed and split *before looking at results*. Do not treat three smoke prompts as a reliable population evaluation.
2. Feature choice: identify active features on a separate *selection* split, freeze selected IDs, and evaluate on distinct held-out samples. Avoid per-test-sample cherry-picking of strongest features if making comparative claims. A two-prompt engineering smoke can select the active feature in each prompt, but must be labeled `smoke_only`.
3. Set `sample_batch=1`, `microbatch=1` initially, no grads, small sequence lengths (e.g. 32–128 tokens) and BF16 or FP16 where native/compatible; increase slowly after capturing actual peak VRAM. Use the same prompt protocol across checkpoints intended to be compared, and record selection/readout positions explicitly. Free old logits/tensors between runs.
4. Each completed M3 row needs a finite value, nonzero valid observations, and a committed measurement/cache provenance reference; `undefined` or `blocked` rows are *not* verified. One representative site per new profile is the minimum Benchmark gate; heavy 26-site layerwise coverage is the desired full Stage D deliverable.
5. Run M4 on already stored M3 results. Existing CLI syntax:

~~~bash
saebase-benchmark --config path/to/real_m3_config.json --dry-run
saebase-benchmark --config path/to/real_m3_config.json --real --output /workspace/stage_d_runs/runs
# Use the actual printed <run_id> directory:
saebase-audit --m3-root /workspace/stage_d_runs/runs/<run_id> \
  --config path/to/real_m4_config.json --dry-run
saebase-audit --m3-root /workspace/stage_d_runs/runs/<run_id> \
  --config path/to/real_m4_config.json --run
~~~

For M4, an example configuration is `metric_ids: [core.l0, intervention.logit_delta_l2]`, `evaluation_seeds: [0,1,2,3,4]`, `bootstrap_replicates: 256`, `sample_sizes: [8,16,32]`, `confidence: 0.95`, with independent prompt group IDs. This is an example **to generate with actual `AuditConfig.to_dict()`**, not a guaranteed directly accepted YAML fragment. Never give separate prompt variants the same treatment as independent groups; similarly, Localize's full Boolean product input grid is one grid and **must not** be independently prompt-bootstrapped.

For original Localize integration, only run `localization.budget_at_90pct` or `.budget_fraction_at_90pct` with a **complete controlled Boolean input design and original `sae_localize` package**. A random prompt collection is not valid input for those scope curves. The request does not require running every paper-specific metric on every model; validate only protocols with scientifically valid input designs.

Read M3 and M4 manifests back, check SHA-256, `run_id`, model/site, SAE checkpoint, dataset, evaluation seeds and provenance. M4 should load frozen codes/measurements, **not run the language model again**. Verify summaries and reports are numerically derived from the original stored rows, not hand-authored numbers.

## 8. Evidence filesystem contract and resume behavior

Proposed local output tree (a new Stage D script must implement it; these are **not current pre-existing paths**):

~~~text
/workspace/stage_d_runs/
  environment.json                  # GPU/host/disk/prices, driver, CUDA, package versions, git SHA
  catalog_snapshot.json             # copy of the exact 48-site execution catalog
  status.json                       # 48 join keys, state, last attempt, source SHA
  attempts.jsonl                    # append-only attempt start/end, errors, resources
  access_preflight.json             # gated/unrestricted access statuses, no token text
  acceptance/<model_key>/<sae_key>/L<layer>.json
  runs/<m3_run_id>/                 # SAEBase native M3 ledger and M1/M2 caches
  reports/                          # exported M3/M4 HTML, CSV, JSON and manifests
  final/stage_d_final_matrix.json
  final/stage_d_summary.md
  final/checksums.sha256
~~~

Each `acceptance/.../L<layer>.json` should include at least:

~~~json
{
  "schema": "saebase.stage_d.pair_acceptance.v1",
  "status": "passed",
  "model_key": "qwen3_8b",
  "sae_key": "qwen3_8b_resid_w64k",
  "layer": 18,
  "hook_name": "blocks.18.hook_resid_post",
  "precision": "bf16",
  "model_repo": "<actual-model-source>",
  "model_revision": "<resolved-revision>",
  "model_state_sha256": "<actual-64-hex-hash>",
  "sae_repo": "<actual-original-SAE-source>",
  "sae_file": "<actual-path>",
  "sae_state_sha256": "<actual-64-hex-hash>",
  "d_in": 4096,
  "d_sae": 65536,
  "n_prompts": 3,
  "identity_patch_max_abs_logit_error": 0.0,
  "feature_ablation_logit_l2": [1.23, 2.34, 3.45],
  "peak_cuda_allocated_bytes": 0,
  "git_commit": "<exact-SAEBase-commit>",
  "artifact_sha256": "<checksum-of-canonical-payload>"
}
~~~

**The JSON above is a shape example with dummy metric values and a placeholder peak value; it is NOT a completed observation.** Real acceptance requires computed values; any fake/example file must be in tests or docs only, never in final run directories. Extend this schema for tokenizer, prompt hashes, CUDA driver, runtime and failure records.

Write each result atomically (`.tmp` + fsync/rename), checksum its canonical content and add it to `status.json` only after the write succeeds. For `--resume`, verify the existing SHA, model/site/source/dtype/config/dataset identity and result files before skipping work. If a result is corrupt or from a different HF revision, fail or schedule an explicit new run. Keep all failures in `attempts.jsonl` so repeated error investigations do not erase earlier status.

Suggested status vocabulary:

| Status | Meaning | Counts as acceptance? |
| --- | --- | --- |
| `registered` | Config and public source metadata only | No |
| `sae_only_verified` | Downloaded original SAE and tested encode/decode, but no original LLM pair | No |
| `pair_verified` | Original model + SAE passed site/dimension/identity/feature ablation | Yes, pair only |
| `benchmark_verified` | Native M3 rows and artifacts completed and verified | Yes, Benchmark only |
| `reliability_verified` | Native M4 audit passed source-manifest integrity | Yes, Reliability only |
| `blocked_access` | License or authentication prevents model download | No |
| `blocked_compute` | OOM / insufficient RAM, disk or CUDA capability | No |
| `blocked_loader` | Original checkpoint conversion/hook unsupported | No |
| `failed_numeric` | Nonfinite codes, wrong dimensions or incorrect patch invariance | No |
| `failed_integrity` | Missing, mismatched or tampered provenance / manifest | No |

The final matrix must show these dimensions separately, not collapse them to a single ambiguous `verified` flag. Publish aggregate counts and all unpassed rows with exact blockers, including cost and lost data incidents. Never overwrite earlier A/B/C source digests: link to them as prior evidence.

## 9. Failure triage and VRAM/disk decision tree

| Failure | Required action |
| --- | --- |
| HF 401/403 | Stop attempting gated downloads, confirm the user's accepted license and token access outside logs; retry only after authorization. No bypass or alternative weights mislabeled as the requested source. |
| HF 404/unknown SAE ID | Check `profile.sae_id(layer)` and `sae_folder(layer)` against official file listing, revision and original loader converter. Fix the mapping with regression test, not by choosing the nearest unrecorded SAE. (Gemma2 9B L20's previous `average_l0_57` issue is already solved.) |
| CUDA OOM | Release old model, collect allocator trace, batch=1/short input/disable gradients, avoid two models or multiple 64K decoders simultaneously; repeat the *same* source and protocol. If still failing on 48GB, test A100 80GB, then mark blocked if unavailable. Do not silently downgrade to 4-bit and claim identical BF16 acceptance. |
| Host RAM or disk pressure | Check `free -h`, `df -h`, HF cache revision duplication and cache use. Upload completed evidence, remove **specific already-verified public checkpoint cache copies** before downloading next family. Avoid blind `rm -rf` at the filesystem root or deleting unuploaded results. Disk cannot be resized on the running container. |
| Identity patch error > 1e-2 | Inspect hook pre/post position, readout token, normalization and dtype; compare exact shape/source and separate BF16 rounding. No acceptance until resolved or an explicitly revised tested threshold is committed. |
| Feature delta zero | Ensure truly active feature, correct decoder and residual-preserving edit, same hook/position, no accidental no-op patch. Retest a fresh selection prompt; do not invent a positive delta. |
| Matryoshka/BatchTopK mismatch | Inspect original architecture thresholds, trained host, latent width and independent upstream converter. Preserve native semantics. Block unsupported protocols rather than simulating TopK. |
| M3/M4 blocked metrics or corrupt manifest | Verify sample group IDs, `max_measurements`, `max_activation_rows`, conditions and source cache keys; recover from last valid completed cache, preserve failed row. No fabricated HTML/CSV results. |
| Vast host instability or preemption | Save `status.json` and JSONL immediately after each checkpoint, backup every completed model to external persistent location, restore from documented SHA on another verified host. |

Large checkpoint variants should be unloaded between distinct **models**, not necessarily between layers of the same model. For each new large family, check actual CUDA memory headroom before attaching 65K/131K SAE weights. CPU/GPU/offload and data type are part of the scientific run metadata.

## 10. Definition of done and required final GitHub PR

**Engineering completion gates:**

- [ ] All 48 catalog-selected `(model_key, sae_key, layer)` keys are accounted for in the final matrix; 4 previously accepted pairs remain preserved, and each of the other 44 has a **real pair** result or a documented external blocker (blockers must remain visibly incomplete). Each successful new pair has its own checksum-bound JSON and original checkpoint SHA-256.
- [ ] All nine large model/variant families have at least one real forward/intervention site accepted; the **26 large selected sites** have been attempted individually, with only authentic passes counted. Gated items are not marked complete if access is unavailable.
- [ ] Mistral 7B's large community SAE is actually downloaded and encoded/decoded, and its `resid_pre` L16 hook is tested, or it remains explicitly blocked with a reproducible error.
- [ ] Stage A gated Gemma (270M, 1B, 2B) and Meta Llama blockers are either resolved with user-approved HF access and real weights, or preserved as blockers with specific actions for the owner.
- [ ] All 18 profile rows have an accepted representative M3/M4 real run where access/compute allows; the large 26-site group has multi-layer M3/M4 acceptance with shared protocol when affordable. If some runs lack suitable independent prompts, record `smoke_only` instead of a population reliability claim.
- [ ] Exact source/HF revisions, host model/SAE SHA-256, hook, dtype, tokenization, GPU/CPU RAM, rental cost, peak VRAM, artifacts and exceptions are recorded. M3/M4 manifests are validated and complete; no silent `blocked`/`undefined` metric rows.
- [ ] Core unit/regression CI and existing P0–P3/B+C tests pass with no backward-incompatible public API, schema, CLI or old release changes. Native GPU acceptance evidence is linked to the PR; public GitHub contains **only code, configs and compact validation artifacts**, not weights.
- [ ] Documentation records the updated tiered model support matrix (`registered`, `sae_only`, `pair`, `benchmark`, `reliability`), exact outstanding blockers and how to reproduce each verified case on GPU. README updates, if needed, are concise, source-linked and do not erase existing sections.
- [ ] Final PR merges only after real evidence and tests. Save all outputs persistently and verify them, then destroy the Vast instance and remove any persistent volume that is no longer needed.

**Mandatory final agent report to the owner**: PR/merge URL + SHA; GPU/VRAM/RAM/disk/actual rental rate and spend; count of **new** pair-passes vs 44 remaining baseline; count of 26 heavy layer passes; count of 18 representative M3/M4 passes; total failed/blocked by exact reason; links to machine-readable JSON and logs; source SHA/versions; confirmation that Vast instance was destroyed. Any remaining unrun cases are expressly listed as incomplete. Do not claim "full Stage D acceptance" unless these gates are met.

## 11. Quick links for the executing agent

| Purpose | Link |
| --- | --- |
| Current source | [SAEBase main](https://github.com/OpenSAE/SAEBase) |
| Stage B/C layer catalog | [stage_bc.json](../../conf/coverage/stage_bc.json) |
| B verified pair/M3/M4 proof | [stage_b_cpu_verified.json](../../conf/coverage/stage_b_cpu_verified.json) |
| C verified standalone SAE evidence | [stage_c_cpu_verified.json](../../conf/coverage/stage_c_cpu_verified.json) |
| A gated-model/pair status | [phase_a_targets.json](../../conf/release/phase_a_targets.json) |
| Hook registry | [sae/registry.py](../../saebase/sae/registry.py) |
| Public original SAE loader | [sae/loading.py](../../saebase/sae/loading.py) |
| Stage A baseline GPU script | [saebase/tools/phase_a_verify.py](../../saebase/tools/phase_a_verify.py) |
| M3 CLI | [saebase/tools/m3_benchmark.py](../../saebase/tools/m3_benchmark.py) |
| M4 CLI | [saebase/tools/m4_reliability.py](../../saebase/tools/m4_reliability.py) |
| Controlled input Localize | [Localize Benchmark Docs](../github_development_doc/localization-benchmark.md) |
| Vast rental marketplace | [Vast.ai Console](https://cloud.vast.ai/) |
| Vast pricing/billing | [Official instance pricing](https://docs.vast.ai/guides/instances/pricing) |
| Vast disk warning | [Official storage types](https://docs.vast.ai/guides/instances/storage/types) |

---

**Next action for the assigned agent:** first verify current `main` and the user's HF access approvals and GPU budget, then implement the general `phase_d_verify` and resumable matrix driver on a code branch **before renting a GPU**. Once the no-GPU dry-run/tests pass and source access is ready, rent one 48GB GPU instance, run D1 → D6 in order, and publish actual evidence. Do not reinterpret this specification as proof that GPU acceptance has already been performed.
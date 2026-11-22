# Batch Scoring Invariance and Acceleration

## Purpose and Main Functions

Keep scores identical for the same question and feature intervention across batch sizes, then choose the fastest batch under that constraint. Describe the shared execution policy, task and Precheck responsibilities, source provenance, and measured validation entry points.

## Scope

This document covers current single-GPU Capture, Intervene, Generate, and the corresponding performance prechecks. Production batches remain stopped. This round runs only bounded, real performance and consistency checks. The document records what was verified. Limited probes do not constitute a global mathematical proof across all hardware and inputs.

## Workflow and Responsibilities

1. Runtime selects `runtime.numerics` before loading any model or performing CUDA computation. It writes the current compute version into the configuration before deciding whether to resume. Historical runs with the same policy but a different version must be recomputed. Each worker process uses one numerical policy. Start a new process to switch.
2. `models/numerics.py` sets the GPU computation policy centrally and calls reused upstream operators. HF attention uses Torch's existing math backend to fix the computation path. Model adapters apply the policy before loading. Individual tasks do not independently set the environment or create inference services.
3. Tasks continue to use the existing model, SAE, feature filtering, and reference-answer scoring interfaces. Shared caching and batching are in the model API. Algorithms and metrics belong to their respective tasks.
4. Runtime loads the model and SAE once, and Precheck and production tasks share them. Precheck follows the normal task execution path, compares one-at-a-time execution with different batch sizes for identical inputs, measures throughput and GPU memory, and passes its recommendation to the production run. The `vllm` numerical policy requires zero error. On failure, record the reason. A silent fallback to one-at-a-time scoring does not count as a pass.
5. Production tasks use the same policy and keep the per-question controls. If GPU memory is insufficient, reduce the batch size and check consistency again. Save the numerical policy and upstream provenance with run results, performance records, normalization identity, and behavior artifacts. Caches produced by an older numerical policy cannot be mixed with these results.

## Reused Implementations

Numerical settings, matrix multiplication, and log-softmax come from vLLM 0.27.1, pinned to commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, at `vllm/model_executor/layers/batch_invariant.py`. Softmax uses a newer fixed row-reduction implementation from commit `68e478651a20e3d23bf939912c38883d511ac3ee` at `vllm/model_executor/determinism/batch_invariant.py`. The integration preserves FP32 accumulation for FP16/BF16 and the output type semantics.

Copied function definitions, licenses, and provenance are in `third_party/accelerator_kernels/`. The original matrix kernel explicitly specifies IEEE FP32 and uses an upstream tile configuration that fits this GPU's shared memory. Each adaptation is recorded in the provenance manifest. Upstream provides the GPU computation code. This project maintains integration, lifecycle handling, and validation in actual tasks.

The original local environment uses Torch 2.11. The complete dependencies for the project's listed vLLM 0.27.1 require Torch 2.13. We reuse the needed capabilities while retaining verified model and SAE loading and intervention support, avoiding a full experimental environment upgrade to obtain a few computation operators. The existing vLLM entry point for ordinary text generation does not establish support for scoring the project's intermediate-layer interventions.

The standard Runtime configuration uses `runtime.numerics=vllm`. Direct low-level model API use without an explicit policy remains `native`, and historical configurations without the field also use `native`. The setting preserves weights and metric definitions. It can change historical low-precision rounding, so the identity and provenance must be recorded. CPU execution does not install CUDA operators, and a configuration check does not configure the GPU.

## Engineering Defaults

The standard configuration uses the following verified settings. Both the normal Runtime and matrix workers apply the same configuration before loading a model:

| Item | Default | Purpose |
| --- | --- | --- |
| CPU compute threads | `runtime.cpu_threads=4`, applied to both Torch and BLAS | Uses the previously measured four-model setting and prevents normal launches from reverting to 24 local Torch threads |
| Tokenizer parallelism | Disable the separate thread pool | Single-model workers already batch work, reducing extra thread contention |
| GPU computation | `runtime.numerics=vllm`, with the math attention backend | Preserve verified batch consistency |
| Compute precision | Preserve each model and SAE's original load precision, with IEEE FP32 for the stable policy | Preserve the model identity and scoring definition |
| Batch size | Measure automatically, default candidates up to 64, and reserve 10% GPU memory | Choose by throughput. Reduce the size if memory is insufficient, then check results |
| Observations and caching | Intervene and Generate reuse observations by default. Load the model and SAE once | Reuse shared computations that passed measurement |
| Compilation | Keep the existing direct execution path and use the reused Triton operators | Control startup cost and retain compatibility with dynamic interventions |
| Model processes | The existing matrix scheduler runs models sequentially | One matrix entry point runs one model worker at a time |

`base/runtime/environment.py` controls CPU threads. It sets thread environment variables when workers start. Before loading, the normal Runtime limits already-loaded thread pools using existing Torch and `threadpoolctl` APIs. `runtime.cpu_threads` can be overridden explicitly. Configuration inspection stays side-effect-free. If an older configuration lacks the field, direct Runtime execution retains its existing CPU settings. Model subprocesses use the shared launch defaults. The GPU allocator retains its verified default behavior, and batch search and memory headroom control the workload.

Performance records include actual Torch thread counts, loaded BLAS thread pools, and relevant environment settings. Record fixed engineering settings separately from automatic batch search so that benefits of existing optimizations are not counted again as gains from these CPU settings. The previous measurements on the four Capture modes and on Intervene and Generate already used four threads and observation reuse. This change makes those settings the default in normal entry points.

Default entry-point verification: one GPT2/GSM8K question ran through the normal Runtime Generate path with the default 8 tokens. All Top 10 comparisons completed. Automatic precheck passed, and the production task reused batch size 10. Scoring and discrete output control errors were both 0. One exhaustive I question in the matrix also completed, with identical inputs, answers, candidates, and all 1024 scores compared individually with the previous v4 artifacts. The actual Torch and BLAS thread counts were both 4. [Original run paths and verification results](../../../../../assets/history_experiment/maintenance_evidence/files/batch_invariance/defaults_verification.json) · [Execution record](../../../maintenance/batch_invariance/execution.md)

## Verification and Artifacts

### Qwen Accumulation and Scoring Acceleration Fix (2026-11-17)

A real run on 100 questions exposed a Qwen GDN accumulation difference missed by earlier small samples. The CUDA `cumsum` reduction layout changed with the number of rows. Operator-by-operator tracing first found a difference of about 1.9e-6 at accumulation, amplified by subsequent low-precision operations. The fix reuses the row-wise FLA scan kernel included in vLLM, with a fixed warp configuration. The current Transformers 5.23.0 GDN function keeps its algorithm, with only the accumulation replaced. See `third_party/accelerator_kernels/gdn_source.json`, `vllm_scan.py`, `hf_qwen_gdn.py`, and `FLA_LICENSE` for source version, SHA, and license.

The Qwen GPU numerical identity was upgraded separately to `saebase.vllm_batch_invariant.qwen_gdn_scan.v5`. Mathematical metrics, original precision, model weights, and questions are unchanged. The fixed reduction order changes prior Qwen rounding, so old Qwen Capture results cannot be mixed with new results and must be recomputed. Other models remain on v4 and completed results remain reusable. Bind the corresponding version into the configuration before comparing resume eligibility, and install the appropriate model implementation at load time.

Native answer scoring for HF models now reuses existing `SiteReplay` and `AnswerReplay` and caches earlier layers as it does for TL models. One-at-a-time and batched scoring reuse that cache. Keep a full native forward-pass control for each question to avoid comparing only two versions of the same optimized implementation. Reuse batch policies by a power-of-two bucket for the full scoring sequence length and by the number of control states. This avoids repeated search at each exact answer length. Precheck and Capture modes share policies for the same loaded model and SAE. After an out-of-memory error, save the reduced batch selection so later runs in the same bucket do not start from an oversized batch.

Native and batch-size 1/2/8/16 controls all passed for the 54 questions that previously failed. For one question, all 1024 states matched one-at-a-time results exactly at batch sizes 2/8/16/32/64. [Original measurement, software check, and production precheck evidence](../../../../../assets/history_experiment/maintenance_evidence/files/capture_acceleration.tar.gz)

For the final v4 probes, each group used its existing model, SAE, and layer, three GSM8K questions, the four Capture modes, Intervene, and Generate. Each ran 1 warmup and 2 measurements, and Generate produced 2 tokens. I computed all 1024 states for 10 candidates and independently recomputed all 1024 states from each model's saved artifacts. Scoring controls compared up to batch size 64. Regression evidence for different lengths and orderings, tail batches, probability accumulation precision, and resume isolation is stored separately. [Original batch invariance records and results](../../../../../assets/history_experiment/maintenance_evidence/files/batch_invariance.tar.gz)

When reading performance reports, distinguish throughput for batched versus one-at-a-time scoring under the same policy, total time per question, initial search cost, and shared stages such as model loading. Report measured gains only. The report makes no speedup estimates for unmeasured cases.

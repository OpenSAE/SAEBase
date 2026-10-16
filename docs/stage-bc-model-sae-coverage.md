# Model and SAE Coverage — Stages B+C

**Upstream source validation:** 48/48 selected SAE checkpoint file paths returned valid source metadata (size, ETag and upstream commit) without downloading weights. This is a lighter check than the 21 actual checkpoint evaluations. [Full 48-file metadata artifact](https://github.com/OpenSAE/SAEBase/actions/runs/38829795455/artifacts/12017011100). Both Matryoshka SAE folders have been mapped to their published `sae_weights.safetensors` files.\n\n**Validated CPU evidence:** [21/21 genuine SAE checkpoint ledger](../conf/coverage/stage_c_cpu_verified.json) and [three real model–SAE + M3/M4 acceptance records](../conf/coverage/stage_b_cpu_verified.json). The 21 SAE checkpoints were independently decoded using original public weights; one Gemma Scope 9B PT L20 checkpoint was fixed from nonexistent L0 57 to upstream-mapped L0 58, then [validated separately](https://github.com/OpenSAE/SAEBase/actions/runs/38828805176/artifacts/12015864245) without replaying the other 20. The other 27 registered layer positions have **not** undergone this CPU weight-acceptance protocol. CPU SAE-only acceptance is not a full LLM–SAE pair test.

**Real M3/M4 acceptance:** Three smaller models each produced 5 completed Benchmark metric rows and 3 completed Reliability audits, with manifest and JSONL checksums verified against the archived results. See [Pythia 410M](https://github.com/OpenSAE/SAEBase/actions/runs/38827730625/artifacts/12016370395), [Qwen3 1.7B Base](https://github.com/OpenSAE/SAEBase/actions/runs/38827730625/artifacts/12015962255), and [DeepSeek R1 Distill Qwen 1.5B](https://github.com/OpenSAE/SAEBase/actions/runs/38827730625/artifacts/12015698101). Two real prompts per model establish engineering compatibility, not statistical SAE quality.

SAEBase selects representative, independently published open SAE checkpoints across model scales. **Registered** models are catalog entries; a **real-weight verified** checkpoint must have a successful GitHub Actions evidence artifact.

## Core model coverage

| Model | SAE source | Layers | SAE architecture | CPU verification |
| --- | --- | --- | --- | --- |
| Pythia 70M | [EleutherAI / ctigges](https://huggingface.co/ctigges/pythia-70m-deduped__res-sm_processed) | 3 | Standard residual | Phase A/M5 verified |
| GPT-2 Small | [jbloom GPT-2](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted) | 8 | Standard residual | Phase A/M5 verified |
| Gemma 3 270M | [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-270m-pt) | 5, 9, 12 | JumpReLU residual | SAE-only CPU |
| Pythia 410M | [EleutherAI](https://huggingface.co/EleutherAI/sae-pythia-410m-65k) | 4, 12, 20 | TopK MLP | Real CPU pair |
| Gemma 3 1B | [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-1b-pt) | 7, 13, 17 | JumpReLU residual | SAE-only CPU |
| DeepSeek R1 Distill Qwen 1.5B | [analist](https://huggingface.co/analist/sae-r1-distill-qwen-1.5b-resid-l20-reasoning) | 20 | Sparsify TopK residual | Real CPU pair |
| Qwen3 1.7B Base | [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50) | 7, 14, 18, 23 | TopK residual | Real CPU pair |
| Gemma 2 2B | [Gemma Scope](https://huggingface.co/google/gemma-scope-2b-pt-res) | 13, 25 | JumpReLU residual | SAE-only CPU |
| Qwen3.5 2B Base | [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50) | 6, 12, 15, 20 | TopK residual | Real CPU pair + M3/M4 |
| Gemma 3 4B PT / IT | [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-4b-pt) | 9, 17, 22, 29 | JumpReLU residual | SAE-only CPU |
| Mistral 7B Instruct | [tylercosgrove](https://huggingface.co/tylercosgrove/mistral-7b-sparse-autoencoder-layer16) | 16 | TopK `blocks.16.hook_mlp_out` (native `layer_norm`) | Stage D pair + population; SAELens LN stats preserved per prompt via `FeatureSnapshot` |
| Qwen3 8B Base | [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50) | 9, 18, 23, 31 | TopK residual | SAE-only CPU |
| Llama 3.1 8B Base | [Llama-Scope](https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x) | 8, 16, 21, 27 | JumpReLU residual | SAE-only CPU |
| DeepSeek R1 Distill Llama 8B | [Llama-Scope R1](https://huggingface.co/fnlp/Llama-Scope-R1-Distill) | 23 | JumpReLU reasoning | SAE-only CPU |
| Gemma 2 9B PT | [Gemma Scope](https://huggingface.co/google/gemma-scope-9b-pt-res) | 9, 20, 31 | JumpReLU residual | SAE-only CPU |
| Qwen3.5 9B Base | [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50) | 8, 16, 21, 27 | TopK residual | SAE-only CPU |

The 15-core-model catalog spans 270M to 9B and adds three independently registered architecture checks: [Qwen3 1.7B BatchTopK](https://huggingface.co/adamkarvonen/qwen3-1.7b-saes), [Gemma 2 2B Matryoshka BatchTopK](https://huggingface.co/chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40), and [Qwen3.5 4B Matryoshka](https://huggingface.co/decoderesearch/qwen-3.5-saes). The latter are distinct from the 15 core representatives.

## Layer and architecture coverage

Run:

```bash
python scripts/phase_bc_matrix.py
python scripts/phase_bc_matrix.py --json > stage_bc_plan.json
```

The input file [stage_bc.json](../conf/coverage/stage_bc.json) declares **48 selected layers across 15 core model entries and three extra architectures**. The offline catalog validates exact registry membership, model association, legal layer indices, source IDs, and a non-duplicated mapping to 25%, 50%, 65%, and 85% depth anchors wherever actual released layers permit. Four depths are a target, not a fabricated guarantee for single-layer SAE releases.

SAE implementations are loaded from their published sources: SAE Lens JumpReLU/BatchTopK, official Qwen TopK and original EleutherAI Sparsify. Sparsify's native top-k outputs are translated into SAEBase's dense code vectors while retaining source checkpoint weights and actual decoder output. Sparsify decoder and feature edit tests do not reimplement training code.

## Evidence: separate checkpoints from model behavior

```bash
# Download one original SAE into a CPU environment and test actual weights.
python scripts/phase_bc_cpu_sae.py --sae-key gemma1b_scope_resid --layer 7

# Run small model + original SAE + genuine downstream feature intervention.
python scripts/phase_bc_cpu_pair.py --sae-key pythia410m_mlp_topk

# Execute the full real M3 Benchmark and M4 Reliability on that small model.
python scripts/phase_bc_cpu_benchmark.py --sae-key pythia410m_mlp_topk
```

The [Stage B CPU pairs](https://github.com/OpenSAE/SAEBase/actions/workflows/stage-bc-cpu-pairs.yml) workflow evaluates genuine Pythia 410M, Qwen3 1.7B Base, and R1Distill Qwen 1.5B checkpoints, including identity patch and nontrivial feature ablation. For each model, a subsequent M3/M4 pass saves exact benchmark and audit checksum evidence. The [Stage C CPU SAEs](https://github.com/OpenSAE/SAEBase/actions/workflows/stage-bc-cpu-sae.yml) workflow executes 21 per-layer authentic SAE-only checks on hosted CPU, separately checks upstream metadata without tensor downloads, and aggregates successful and blocked artifacts.

Each SAE-only acceptance reports full loaded state SHA-256, input/output dimensions, finite encode/decode and synthetic-test statistics. Synthetic SAE inputs **do not** establish valid end-to-end LLM activation semantics. The real model-pair acceptance additionally checks actual activation site, unchanged identity-patched logits, feature ablation effects, and both model and SAE state digests. M4 uses previously frozen observations without LLM reload.

Individual weight licenses, gated model permissions, and SAE architecture/site contracts remain the responsibility of the individual source. Do not treat an unvalidated YAML or model card as verified.

## Stage D boundary

No 4B–9B language-model forward passes, feature interventions, cross-layer large-model Benchmark or GPU throughput claims are required by B+C. Those remain Stage D work on actual suitable GPU runners, along with gated Gemma and Meta Llama access.

[Model Support](model-support.md) · [Phase A Results](phase-a-model-coverage.md) · [Research Integrations](research-integrations.md)

# Phase A — Six Model and SAE Pairs

## Actual acceptance evidence — October 15, 2026

Real checks ran on GitHub-hosted CPU with original public SAE checkpoints. These results describe what actually completed, not merely the registry.

| Model | Original SAE weights, encode/decode | Model + SAE real capture and patch | Reason where incomplete |
| --- | --- | --- | --- |
| Gemma 3 270M | Passed, 65,536 features | Blocked | Gated model returned HTTP 401, missing authorized `HF_TOKEN` |
| Gemma 3 1B | Passed, 16,384 features | Blocked | Gated model returned HTTP 401, missing authorized `HF_TOKEN` |
| Gemma 2 2B | Passed, 16,384 features | Blocked | Gated model returned HTTP 401, missing authorized `HF_TOKEN` |
| Qwen3.5 2B Base | Passed, 32,768 features | **Passed** | Real model/SAE capture, reconstruction, identity patch and active feature ablation |
| Qwen3 8B Base | Passed, 65,536 features | Not yet run | No connected high-memory GPU runner |
| Llama 3.1 8B Base | Passed, 32,768 features | Not yet run | Meta model gated, no connected high-memory GPU runner |

Proof: [Qwen3.5 real model and feature-ablation artifact](https://github.com/OpenSAE/SAEBase/actions/runs/38824111274/artifacts/12014521241), [SAE-only six-checkpoint results](https://github.com/OpenSAE/SAEBase/actions/runs/38824111301), [real Gemma access failures](https://github.com/OpenSAE/SAEBase/actions/runs/38822802463). Full provenance and checkpoints are recorded in [phase_a_targets.json](../../../conf/release/phase_a_targets.json). The SAE-only status is separate from full model validation.

Real Qwen3.5-2B M3/M4 acceptance also passed: 5 benchmark metric rows, 3 reliability metric audits, L0 50, reconstruction MSE 0.01046, explained variance 0.55957, and measured feature-ablation logit L2 91.89. See the [full M3/M4 evidence](https://github.com/OpenSAE/SAEBase/actions/runs/38824324783/artifacts/12014277351). Three short prompts establish engineering integration, not statistical SAE quality.\n\nThe Gemma models require accepting upstream access terms and then adding an appropriately authorized Hugging Face token as the `HF_TOKEN` GitHub Actions secret. For both 8B models, connect a GPU runner with sufficient VRAM, then invoke [Phase A real checkpoint workflow](../../../.github/workflows/phase-a-real.yml) with the appropriate model and CUDA device. Do not promote any pair to Verified until its original weight/activation/feature-patch evidence JSON says `status=passed`.


SAEBase's first expansion target is a **real-weight six-model coverage matrix** across 270M, 1B, 2B, and 8B. The baseline M5 verified checkpoint matrix (Pythia-70M and GPT-2 Small) remains separately preserved.

| Model | Public SAE source | SAE profile | Layer | Current evidence |
| --- | --- | --- | --- | --- |
| [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m) | [Gemma Scope 2 270M](https://huggingface.co/google/gemma-scope-2-270m-pt) | `gemma270m_scope_resid` | 5 | Registered, real acceptance pending |
| [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-pt) | [Gemma Scope 2 1B](https://huggingface.co/google/gemma-scope-2-1b-pt) | `gemma1b_scope_resid` | 13 | Registered, real acceptance pending |
| [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) | [Gemma Scope 2B](https://huggingface.co/google/gemma-scope-2b-pt-res) | `gemma2_2b_pt_l25_canonical` | 25 | Registered, real acceptance pending |
| [Qwen3.5 2B Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | [Qwen-Scope 32K](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50) | `qwen35_2b_resid_w32k` | 12 | New model+SAE profile; real acceptance pending |
| [Qwen3 8B Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | [Qwen-Scope 64K](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50) | `qwen3_8b_resid_w64k` | 18 | Registered, real acceptance pending |
| [Llama 3.1 8B Base](https://huggingface.co/meta-llama/Llama-3.1-8B) | [Llama Scope](https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x) | `llama31_8b_lxr_8x` | 16 | Registered, real acceptance pending |

These are **available upstream SAE checkpoint sources**, not a claim that all six can already run on all backends or hardware. Some model repositories require explicit license acceptance or a Hugging Face access token. Qwen3.5-2B-Base has a multimodal/HF architecture; its exact residual-site contract must pass the real model test.

## Verify with actual weights

In an environment with model access and enough memory:

```bash
python -m pip install -e ".[analysis]"
python saebase/tools/phase_a_verify.py --model gemma270m --device cpu --output phase-a-results
python saebase/tools/phase_a_verify.py --model qwen35_2b --device cuda --output phase-a-results
python saebase/tools/phase_a_verify.py --model qwen3_8b --device cuda --output phase-a-results
python saebase/tools/phase_a_verify.py --model llama31_8b --device cuda --output phase-a-results
```

Or use GitHub Actions **Phase A real model SAE acceptance** (`.github/workflows/phase-a-real.yml`), selecting a single model, CPU or CUDA device and a suitable runner. Large 8B experiments need a configured high-memory GPU runner; the standard hosted runner does not provide such a GPU. If gated weight access is necessary, use a permitted `HF_TOKEN` repository secret with the requisite license approval.

The script requires **real weights** and records checkpoint SHA-256 digests of the actual loaded model and SAE parameters, input prompt hashes, activation dimensions, feature L0, reconstruction MSE, and identity-patch maximum logit error. It exits nonzero if any acceptance check fails. Completed artifact JSON is uploaded by the Actions workflow on actual success. If no run artifact exists, a pair remains *Registered / Pending Real Validation*.

A successful probe covers a handful of prompts and one layer; it establishes loader/hook/patch compatibility, **not** a universal SAE quality claim, all-layer coverage, or correctness of a novel model architecture. A second independent SAE checkpoint at the same model hook is required for a verified multi-checkpoint matrix.

## Existing release unaffected

The original verified Pythia-70M/GPT-2 release acceptance remains at [Model Support](model-support.md). The Phase A status ledger is [phase_a_targets.json](../../../conf/release/phase_a_targets.json); update each target to verified **only after** recording a successful genuine checkpoint run and corresponding provenance evidence.

[Real Checkpoint Acceptance](reference-checkpoint-validation.md) · [Model Support](model-support.md)

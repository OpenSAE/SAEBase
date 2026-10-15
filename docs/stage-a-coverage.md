# Stage A Model and SAE Coverage

SAEBase's Stage A selects six externally published model–SAE pairs spanning 270M to 8B. All checkpoints remain hosted by their original maintainers. Every entry is independently identified by model, SAE source, layer, hook, width and upstream license.

## Selected roster

| Base model | Official/independent SAE | Layer | SAE width | Access |
| --- | --- | ---: | ---: | --- |
| [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m) | [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-270m-pt), residual stream | 9 | 16K | Gemma license acceptance |
| [Gemma 3 1B PT](https://huggingface.co/google/gemma-3-1b-pt) | [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-1b-pt), residual stream | 13 | 16K | Gemma license acceptance |
| [Gemma 2 2B PT](https://huggingface.co/google/gemma-2-2b) | [Gemma Scope](https://huggingface.co/google/gemma-scope-2b-pt-res), residual stream | 25 | 16K | Gemma license acceptance |
| [Qwen3.5 2B Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | [Qwen-Scope W32K](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50), TopK=50 | 12 | 32K | Public model; SAE Qwen license |
| [Qwen3 8B Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | [Qwen-Scope W64K](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50), TopK=50 | 18 | 64K | Public model; SAE Qwen license |
| [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | [Llama Scope](https://huggingface.co/fnlp/Llama3_1-8B-Base-LXR-8x), JumpReLU | 16 | 32K | Meta Llama access/license required |

Only actual recorded acceptance runs establish real-weight support. The previously verified Pythia 70M and GPT-2 Small matrix remains separate.

## Run without model downloads

```bash
pip install -e ".[dev]"
saebase-stage-a --dry-run
saebase-stage-a --dry-run --target qwen35_2b
pytest -q tests/test_stage_a.py
```

The preflight checks exact registered model/SAE relationships, hook names, layers and public source mapping. It does **not** download checkpoints or label them verified.

## Real weights and checkpoints

A CPU attempt on the smaller public Qwen3.5 2B model:

```bash
pip install -e ".[stage-a]"
saebase-stage-a --real --target qwen35_2b --device cpu --output stage-a-results
```

For Qwen3 8B and Llama 3.1 8B, use a CUDA GPU with sufficient VRAM (generally 32–48 GiB of usable GPU memory for BF16 model, SAE, activations and headroom, depending on batch sizes):

```bash
saebase-stage-a --real --target qwen3_8b --device cuda --output stage-a-results
```

Gemma Scope 2 and Llama Scope require SAE Lens in addition to the model stack; install `sae-lens` in a compatible environment. If using an authenticated/gated model, first accept the upstream license and authenticate with a token **outside the repository** (for example, `hf auth login`); do not commit secrets or redistribute checkpoints.

Each attempted model writes an individual `<model_key>.json` evidence record. Success requires real model and SAE weights, exact model/SAE state SHA-256, SAE dimensional and layer match, live hook capture, finite SAE reconstruction, identity logit patch, and a nontrivial latent ablation with downstream logit differences on two prompts. If download, license, hardware, compatibility or any numerical gate fails, the result is explicitly `blocked`, and the command exits unsuccessfully.

## What is checked

- Exact public SAE release/ID and selected decoder layer rather than only a model key
- SAE shape and encoder/decoder contract; Qwen-Scope TopK reads official `layerN.sae.pt` tensor maps using `torch.load(weights_only=True)`
- Frozen external checkpoint provenance and complete state digests
- HF layer-output residual capture, one-to-one activation dimensions and live identity patch
- SAE feature activation, reconstruction and effect of feature ablation on logits
- Stable JSON evidence, integrity checksum, and explicit verification status

**Compatibility note:** Qwen3.5's composite configuration stores decoder depth under `text_config.num_hidden_layers`. The HF model adapter reads that field; the legacy TransformerLens registry loader is not substituted for real Stage A capture. The original SAE Lens checkpoints for Gemma/Llama are loaded with their named upstream releases. Support for license-gated models is not implied by the presence of a working configuration.

[Supported Models](model-support.md) · [Activation Store](activation-store.md) · [Getting Started](getting-started.md)

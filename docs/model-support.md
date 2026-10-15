# Supported Models and SAE Checkpoints

SAEBase keeps model definitions separate from SAE checkpoint profiles, experimental activation sites, tokenizer identities and source artifacts. **Registered** does not necessarily mean **verified with real checkpoint weights**.

## Verified public model/SAE pairs

The real multi-checkpoint release acceptance covers exactly two independent language-model families, each with two different SAE weight hashes:

| Model | Hook site | SAE sources | Validation |
| --- | --- | --- | --- |
| [Pythia-70M deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped) | `blocks.3.hook_resid_post` | Original `pythia70m_resid_sm` residual SAE and `pythia70m_saebench_gated_l3` gated SAE | Genuine CPU checkpoint loading, activation/code reuse, intervention and audit |
| [GPT-2 Small](https://huggingface.co/openai-community/gpt2) | `blocks.8.hook_resid_pre` | `gpt2small_res_jb_l8` and `gpt2small_fs_6144_l8` | Genuine CPU checkpoint loading, activation/code reuse, intervention and audit |

The first GPT-2 SAE was released by [jbloom](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted); the second comes from the [GPT-2 Small feature-splitting experiment](https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8). The gated Pythia checkpoint was evaluated through the original [SAEBench](https://github.com/adamkarvonen/SAEBench) infrastructure.

The verified release matrix has six distinct prompts per model. It checks compatibility, exact state SHA-256 and downstream measurement integrity, not broad ranking or interpretability quality.

## Phase A 270M–8B expansion

Six model/SAE pairs are tracked in [Phase A Model Coverage](phase-a-model-coverage.md). **Qwen3.5-2B Base + the original Qwen-Scope 32K SAE has passed genuine CPU model+SAE capture and identity-patch acceptance.** All six original SAE checkpoints independently loaded and encoded/decoded with genuine weights; the other five model+SAE pairs remain blocked or untested pending gated model access or high-memory GPU compute. Details and raw evidence are linked in the Phase A page.

## Registered configurations and optional models

Additional model and SAE profiles exist for Pythia, Gemma, Llama and other families. Their availability in `saebase.sae.registry` or Hydra configurations should not be read as evidence that every checkpoint was downloaded, that every hook was tested, or that results are comparable across training/data protocols.

Some Google Gemma weights require separate access approval and different hardware resources. Optional GPT-2 feature-splitting widths are configured but are **not part of the default verified four-checkpoint release matrix**. vLLM/NNsight backends are experimental integrations; verified GPU throughput is not claimed.

Inspect configured entries without weights:

```bash
saebase --list
python -m saebase.sae.verify_registry --list
saebase-release-matrix --config conf/release/m5_matrix.json --dry-run
```

Real matrix evaluation is opt-in:

```bash
saebase-release-matrix --config conf/release/m5_matrix.json --real --device cpu
```

## Identity and checkpoint sources

The repository stores SAE profile identifiers, public source cards, model and tokenizer revisions when available, and exact loaded state SHA-256 values for completed experiments. A key or model card alone is not sufficient to establish loaded weight identity.

Model, SAE, dataset and library licensing is upstream-specific. The [asset inventory](../conf/release/m5_asset_provenance.json) records known and explicitly unverified terms; no pretrained model or SAE weights are redistributed with the package.

[Release Checkpoint Validation](checkpoint-validation.md) · [Validation](validation.md) · [Getting Started](getting-started.md)

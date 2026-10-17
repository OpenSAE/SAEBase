# Supported Models and SAE Checkpoints

SAEBase keeps model definitions separate from SAE checkpoint profiles, experimental activation sites, tokenizer identities and source artifacts. **Registered** does not necessarily mean **verified with real checkpoint weights**.

External model/SAE providers may be registered with the [Third-Party Extension API](extensions.md). Registration or a limited tensor probe does not establish the checkpoint compatibility and reliability reported in the verified matrix.

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
## Verified checkpoints and layers

As of 2026-10-16, **18** model-SAE configurations spanning 270M-9B cover **48/48** selected model/layer sites with genuine checkpoint compatibility checks. **48/48** completed benchmark evaluation on 64 held-out prompt groups, and **48/48** completed reliability auditing. See the [per-layer verification reports](verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->

## Historical `v0.2.0` release acceptance

The historical `v0.2.0` acceptance covered two independent model families, each with two SAE checkpoint weight hashes:

| Model | Hook site | SAE sources | Validation |
| --- | --- | --- | --- |
| [Pythia-70M deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped) | `blocks.3.hook_resid_post` | Original `pythia70m_resid_sm` residual SAE and `pythia70m_saebench_gated_l3` gated SAE | Genuine CPU checkpoint loading, activation/code reuse, intervention and audit |
| [GPT-2 Small](https://huggingface.co/openai-community/gpt2) | `blocks.8.hook_resid_pre` | `gpt2small_res_jb_l8` and `gpt2small_fs_6144_l8` | Genuine CPU checkpoint loading, activation/code reuse, intervention and audit |

The first GPT-2 SAE was released by [jbloom](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted); the second comes from the [GPT-2 Small feature-splitting experiment](https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8). The gated Pythia checkpoint was evaluated through the original [SAEBench](https://github.com/adamkarvonen/SAEBench) infrastructure.

The verified release matrix has six distinct prompts per model. It checks compatibility, exact state SHA-256 and downstream measurement integrity, not broad ranking or interpretability quality.

## Early checkpoint compatibility testing

Early [checkpoint tests](phase-a-model-coverage.md) examined six model and SAE pairs. At that stage, genuine model execution was verified for Qwen3.5-2B Base with the original Qwen-Scope 32K SAE, while all six SAE checkpoints independently passed weight loading and encode/decode tests. Other full-model checks were not yet available at the time. These records are retained as a historical baseline; consult the current [verified matrix](verified-model-matrix.md) for later multi-model results.

## Additional CPU checkpoint testing

The [earlier CPU verification records](stage-bc-model-sae-coverage.md) cover 15 model variants (270M–9B), three additional SAE architectures, and selected layers later included in the full validation matrix. Real feature-intervention tests passed for Pythia 410M, Qwen3 1.7B Base, and DeepSeek R1 Distill Qwen 1.5B. The original CPU evidence includes **21/21 SAE-weight checks** with [checkpoint hashes](../conf/coverage/stage_c_cpu_verified.json), alongside three small-model benchmark and reliability runs in an [earlier run ledger](../conf/coverage/stage_b_cpu_verified.json). These checks preceded the 48-site evaluation and should not be interpreted as broader validation on their own.

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

The latest evidence and original run archives are available through [Verification Evidence and Reproducibility](stage-d-evidence-archive.md).

## Identity and checkpoint sources

The repository stores SAE profile identifiers, public source cards, model and tokenizer revisions when available, and exact loaded state SHA-256 values for completed experiments. A key or model card alone is not sufficient to establish loaded weight identity.

Model, SAE, dataset and library licensing is upstream-specific. The [asset inventory](../conf/release/m5_asset_provenance.json) records known and explicitly unverified terms; no pretrained model or SAE weights are redistributed with the package.

[Release Checkpoint Validation](checkpoint-validation.md) · [Validation](validation.md) · [Getting Started](getting-started.md)

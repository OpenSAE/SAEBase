# Provenance and third-party assets

## Source inventory

This standalone SAEBase package was prepared from the user-supplied SAEBase
source and Phase A/B/C transformations. The original internal SAEReasoning
workspace is not part of this source distribution. The public release bundles
SAEBase's own model adapters, registries, Hydra configurations, evaluation
code, examples, and tests.

The `LICENSE` file states the intended Apache-2.0 distribution license.
**Repository maintainers must verify the ownership and licensing rights of every
included source file before publishing the release.** In particular, confirm
whether code adapted from separate repositories requires additional notices.

## Third-party components

| Component | Purpose | Distribution policy |
| --- | --- | --- |
| PyTorch and NumPy | Core tensor/numerical operations | Installed separately; use their licenses |
| Hydra and OmegaConf | Experiment configuration | Installed separately; use their licenses |
| TransformerLens and SAE Lens | Models, hooks and pretrained SAE loaders | Optional analysis dependencies; retain original licenses |
| Hugging Face Hub, Transformers and Datasets | Checkpoints and data access | Optional analysis dependencies; retain source licenses |
| vLLM and NNsight | Experimental GPU generation and intervention | Optional dependencies; license and compatibility review required |
| External model/SAE checkpoints | Evaluated models and dictionaries | Downloaded on demand; check model cards and access conditions |
| SAR and RSS estimators | Paper-specific metrics | Loaded from separate user-supplied repositories, not redistributed |

## Maintainer review before public release

- [ ] Confirm contributors' permission to license SAEBase source under Apache-2.0.
- [ ] Search modified/imported source for upstream code requiring attribution.
- [ ] Check every model and SAE checkpoint's license and access requirements.
- [ ] Verify dataset redistribution and usage conditions where applicable.
- [ ] Review the final wheel, sdist and repository for credentials or private files.
- [ ] Confirm references, author/citation metadata, and repository ownership.

This is a provenance checklist, not a legal certification or evidence that every
upstream asset is Apache-licensed.

## v0.2.0 external asset inventory

The machine-readable [M5 asset inventory](../conf/release/m5_asset_provenance.json)
lists external model/checkpoint cards and documented license status.
The SAEBase source archive and wheel **do not include checkpoint weights,
tokenizers, or third-party evaluation datasets**. Real model and SAE
weight state SHA-256 values are recorded in the opt-in M5 test evidence.

Unverified checkpoint license terms are not treated as permission to
redistribute weights. Software licensing rights and contributor ownership
are distinct from model checkpoint licensing, and legal review remains
the maintainers' responsibility.

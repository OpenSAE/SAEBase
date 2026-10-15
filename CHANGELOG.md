# Changelog

## 0.2.0 — P0 M0–M5 release

- M1 identity-bound, sharded activation and SAE code stores; M2 batch interventions and steering.
- M3 versioned SAE metrics, source-pinned original SAR/RSS estimators, native SAEBench parity, JSONL/CSV/HTML outputs and shared caches.
- M4 grouped bootstrap, paired SAE confidence intervals and ranking stability; original-paper synthetic known truth and negative controls.
- M5 cross-family checkpoint validation on Pythia-70M and GPT-2 Small with two distinct public SAE checkpoints per model, six reproducible prompts per model, M1/M2/M3/M4 evidence, checkpoint SHA-256, runtime stats and replay validation.
- New fail-closed source, build and provenance checks; SHA256SUMS and checksum-verifiable release evidence.
- External model/SAE weights stay upstream and are **not redistributed**. No single SAE-quality score is computed.

**Limits:** bounded engineering acceptance, not large-scale comparative model quality or legal certification. vLLM/NNsight GPU acceptance and optional extended SAE widths are outside this release.

## 0.1.0 — initial standalone release candidate

- Packaged SAEBase's model/SAE registry and configuration resources.
- Provided public activation capture and residual-preserving SAE feature editing.
- Added versioned prompt artifacts and SAE activation-health evaluation.
- Added SAR and RSS integration examples that load paper estimators externally.
- Added package installation, console commands, CI, source/build verification,
  contribution guidelines, and a release-readiness checklist.

**Scope:** The public checkpoint and GPU generation integration tests require a
compatible external runtime; successful local offline tests are not publication
result reproductions. Version 0.1.0 is a release candidate pending sign-off.

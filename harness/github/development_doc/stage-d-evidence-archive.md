# Benchmark Evidence and Reproducibility

SAEBase publishes the original verification records for **18 model–SAE configurations** evaluated at **48 model/layer sites**. Each site completed real-checkpoint compatibility tests, benchmark evaluation, and reliability auditing. The evidence is available separately from the SAEBase Python distribution so that the measurements and their sources can be inspected independently.

## Evaluation scope

- **Checkpoint compatibility:** real model and SAE weights, exact layers and hooks, identity replay, and nontrivial feature edits.
- **Benchmark evaluation:** measured reconstruction and downstream intervention effects on **64 independent held-out prompt groups per site**, separated from feature selection.
- **Reliability:** statistical checks and audit records corresponding to the measured benchmark outputs.
- **Provenance:** original configurations, model/SAE identities, per-site reports, raw measurements, token ledgers, manifests, and SHA256 checksums.

The input groups use recurring controlled prompt templates. This evidence verifies the supported software workflow under a recorded evaluation protocol; it is not a general ranking of SAE quality or interpretability.

Browse the [verified models, SAE checkpoints, and per-layer results](verified-model-matrix.md).

## Published evidence

| Archive | Contents |
| --- | --- |
| [Compact verification records](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-evidence-20261016) | Frozen configuration and result summaries, per-site provenance, token ledgers, and a standalone verifier |
| [Original run archives](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-original-raw-20261016) | 18 original family archives, native benchmark/reliability records, and checksum manifests |
| [Additional exploratory experiments](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-recovery-03-experiments-20261016) | Independent extension probe and new-prompt experiments, kept separate from the 48-site verified matrix |

The 18 original family archives total **725,190 compressed bytes**. Their combined original-run bundle has SHA256:

`62f5c5b7af719114b999b4b59afb2ddf4798ae337f1a17fd49c11c87c6704958`

A fresh download of the original-run bundle passed checksum and archive verification. The native benchmark and reliability audit matched **48/48** per-site records. The compact snapshot and original runs have separate immutable GitHub release assets.

## Download and verify

The compact snapshot can be verified without a model checkpoint or GPU:

~~~bash
gh release download stage-d-evidence-20261016 -R OpenSAE/SAEBase --dir compact-evidence
(
  cd compact-evidence
  sha256sum -c SHA256SUMS
  python verify-stage-d.py --verify-bundle stage-d-evidence-20261016-compact.tar.gz
)
~~~

From an SAEBase source checkout, independently check the original run bundle:

~~~bash
gh release download stage-d-original-raw-20261016 -R OpenSAE/SAEBase \
  -p 'stage-d-evidence-20261016-original-raw.tar.gz*' --dir original-runs

(
  cd original-runs
  sha256sum -c stage-d-evidence-20261016-original-raw.tar.gz.sha256
)

python saebase/tools/stage_d_archive.py \
  --verify-raw-bundle original-runs/stage-d-evidence-20261016-original-raw.tar.gz
~~~

The downloaded original runs can also be inspected using the native benchmark and reliability readers. File checksums establish integrity; they are not a claim that the original experiments were re-executed.

## Provenance and compatibility

Existing asset filenames, source schema IDs, and archive tags retain their original identifiers for reproducibility. The [original archive index](../../../conf/coverage/stage_d_evidence/final/raw_run_archive_index.json), [verification ledger](../../../conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json), and [file checksum manifest](../../../conf/coverage/stage_d_evidence/final/checksums.sha256) provide the underlying records.

No third-party pretrained model or SAE weights are included. Access and redistribution terms remain with their original publishers.

[v0.3.0 Software Release](https://github.com/OpenSAE/SAEBase/releases/tag/v0.3.0) · [Validation and Reproducibility](validation.md)

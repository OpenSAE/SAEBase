# M5 — Multi-family real-checkpoint release acceptance

M5 completes SAEBase's P0 roadmap. The release matrix checks genuine public
model and SAE weights, deterministic M1/M2/M3 execution and an M4 reliability
audit. It does **not** combine unrelated metrics into a single SAE quality
number or constitute a large-scale scientific comparison.

## Validated release matrix

The default, strictly versioned `conf/release/m5_matrix.json` specifies:

| Model family | Language model | Hook position | SAE checkpoint |
| --- | --- | --- | --- |
| Pythia | EleutherAI Pythia-70M deduped | `blocks.3.hook_resid_post` | `pythia-70m-deduped-res-sm`, `blocks.3.hook_resid_post` |
| Pythia | EleutherAI Pythia-70M deduped | `blocks.3.hook_resid_post` | SAEBench gated trainer 5, `blocks.3.hook_resid_post__trainer_5` |
| GPT-2 | OpenAI GPT-2 Small | `blocks.8.hook_resid_pre` | `gpt2-small-res-jb`, `blocks.8.hook_resid_pre` |
| GPT-2 | OpenAI GPT-2 Small | `blocks.8.hook_resid_pre` | `gpt2-small-res-jb-feature-splitting`, `blocks.8.hook_resid_pre_6144` |

These are four independently SHA-256-fingerprinted weights across two
language-model families, with two checkpoint choices sharing each model's
hook position. Each family evaluates six distinct prompt groups under
identity/reconstruction conditions, with separate M3 Core and intervention
metrics, verified M1/M2 reuse, and M4 grouped reliability results.

An optional `m5_extended_gpt2.json` lists two extra GPT-2 feature-splitting
widths. The extended six-checkpoint matrix is **not** a prerequisite for
`v0.2.0` and is not claimed validated by the minimum four-checkpoint gate.

## Reproduce a release evaluation

```bash
python -m pip install -e '.[analysis,dev]'
saebase-release-matrix --config conf/release/m5_matrix.json --dry-run
saebase-release-matrix --config conf/release/m5_matrix.json --real --device cpu
saebase-release-matrix --config conf/release/m5_matrix.json --report-only
```

The dry-run downloads nothing. The real run requires network access and
sufficient RAM. The same GitHub Actions test also reruns in `--resume`
mode and validates read-only reports without repeating model forwards.

M1 activations must be reused across both SAEs sharing the model/hook.
M2 measurements retain identity-bound original shards, and M3/M4 reports
keep distinct per-metric values and prompt-group uncertainty.
The release verifier compares two model families, ≥2 SAE hashes per model,
six prompt IDs per model, valid M3/M4 hashes, elapsed time and peak RSS.
Failure, unavailable checkpoints or an unsupported metric prevents release.

The archived evidence has `matrix_results.json`,
`matrix_manifest.json`, portable `matrix_summary.csv`,
`matrix_report.html` and per-model M3/M4 manifests, raw metric rows
and reliability results. The identity of the real matrix is derived
from the complete, versioned input specification. Separately injected
synthetic model tests are labeled `synthetic_test`, never `passed`.

## Reproducibility, performance and recovery

- CPU or CUDA can be selected explicitly; reported performance refers to the
  specific hardware, prompt set, batch schedule and installed dependency versions.
- The matrix records elapsed seconds per model, peak process RSS,
  M1 activation/code creation vs reuse, M2 batching and retries.
  Estimated time and RAM are null at preflight rather than fabricated.
- A synthetic fault-injection test interrupts after the first family and
  resumes from previously committed M1/M2 shards. Real mode separately checks
  complete-run replay and report integrity. Modified M3/M4 or matrix manifest
  digests fail verification.
- No GPU vLLM/NNsight throughput or unrestricted/Gemma checkpoints are
  claimed validated by this matrix.

## Source ownership and license boundaries

The inventory in `conf/release/m5_asset_provenance.json` records public
source cards and declared or unverified licensing statuses. Checkpoints,
tokenizers and datasets are **not** redistributed in source or wheel
artifacts. Rights for third-party assets are not inferred from their
availability. This is a software release, not a release of SAE/model weights.

## Publishing

A tagged release requires all of the following on the same Git commit:

1. Passing Python 3.11/3.12 CI, analysis dependencies and wheel/sdist smoke.
2. Successful real Pythia+GPT-2 matrix and uploaded SHA-256-compatible M3/M4 evidence.
3. Model/SAE source inventory and a no-checkpoint-artifacts scan.
4. `saebase-release-gate` producing `release-evidence.json` plus `SHA256SUMS`.
5. GitHub Release for `v0.2.0` containing the verified wheel, sdist, evidence
   and checksums. No PyPI push is performed.

See [v0.2.0 release notes](M5_RELEASE_NOTES.md),
[provenance](PROVENANCE.md) and [release checklist](RELEASE_CHECKLIST.md).

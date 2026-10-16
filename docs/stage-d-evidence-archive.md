# Stage D Evidence Archive

The [Stage D compact evidence release](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-evidence-20261016) provides a versioned snapshot of original **committed evidence records**, distinct from the SAEBase Python package release.

## Contents

- The frozen 18-profile, 48-layer Stage D catalog, final matrix and per-layer status ledger.
- 48 M3/M4 compact summaries, 48 selection and 48 evaluation token ledgers, and the Stage D pair-acceptance JSONs. Three earlier CPU pair acceptances retain their original Stage B ledger rather than fabricated new GPU records.
- All 200 checked-in Stage D evidence files, including the manifest covering 199 of them (excluding the checksum manifest itself).
- Original Stage B/C ledgers and the frozen Stage D evaluation corpus.
- A file-by-file SHA256 bundle manifest, compressed archive checksum, exact Git source revision and a self-contained Python verifier.

The reports record 48/48 real model-SAE pair and 48/48 population benchmark/reliability engineering acceptances. The evaluation uses 64 separate groups and recurring controlled prompt templates. These results demonstrate the reported engineering protocol, not general SAE quality or interpretability.

## Scope of the archive

**The compact release does not contain original GPU run caches or pretrained weights.** Those original bytes are the separate [original raw release](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-original-raw-20261016): 18 family archives totaling 725,190 bytes, plus `stage-d-evidence-20261016-original-raw.tar.gz` (`62f5c5b7af719114b999b4b59afb2ddf4798ae337f1a17fd49c11c87c6704958`). A clean redownload of that bundle matched the index and 48/48 native M3/M4 manifests. The [raw-run archive index](../conf/coverage/stage_d_evidence/final/raw_run_archive_index.json) still records the historical machine path; the public download is the release above.

The compact evidence is independently hash-verifiable, but it cannot on its own support recomputing all metrics from the unavailable raw M3/M4 run directories. The original `README.json` value of 17 M3/M4 summaries is an earlier in-progress snapshot; the committed final matrix and 48 per-site reports are the final source. These historical records have not been rewritten.

## Download and verify

Download all four assets from the [GitHub evidence release](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-evidence-20261016) or run:

~~~bash
gh release download stage-d-evidence-20261016 -R OpenSAE/SAEBase --dir stage-d-download
cd stage-d-download
sha256sum -c SHA256SUMS
python verify-stage-d.py --verify-bundle stage-d-evidence-20261016-compact.tar.gz
~~~

The first command checks download integrity; the second safely extracts into a temporary location, checks each file against its SHA256 manifest, then validates the original 199 Stage D hashes, catalog and status ledger and all 48 per-site reports without model weights.

A source checkout can be verified or repackaged entirely offline:

~~~bash
python scripts/stage_d_archive.py --check
python scripts/stage_d_archive.py --build --output-dir dist/stage-d-evidence
python scripts/stage_d_archive.py --verify-bundle dist/stage-d-evidence/stage-d-evidence-20261016-compact.tar.gz
(cd dist/stage-d-evidence && sha256sum -c SHA256SUMS)
~~~

The bundle's source-commit SHA records the packaging revision, distinct from the historical GPU experiment commits contained in the Stage D JSON.

## Recover original raw M3/M4 archives

Retrieve the **original bytes** from the GPU machine or an existing backup, then run the following against the original archive directory:

~~~bash
python scripts/stage_d_archive.py --verify-raw /workspace/stage_d_runs/archives
python scripts/stage_d_archive.py --build-raw /workspace/stage_d_runs/archives --output-dir dist/stage-d-raw
python scripts/stage_d_archive.py --verify-raw-bundle dist/stage-d-raw/stage-d-evidence-20261016-original-raw.tar.gz
(cd dist/stage-d-raw && sha256sum -c stage-d-evidence-20261016-original-raw.tar.gz.sha256)
~~~

Verification refuses absent files, incorrect sizes or SHA256 digests, and unsafe TAR members. The checked originals are published as [stage-d-original-raw-20261016](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-original-raw-20261016). Restoring their raw run directories and applying `scripts/phase_d_verify_evidence.py` audited 48/48 native M3/M4 manifests against the compact summaries. Hash checking alone is not re-execution or independent numerical verification.

Original model/SAE weights remain subject to upstream access and license restrictions and are not included.

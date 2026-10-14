# M1 — Activation Store

M1 is the first implementation milestone of SAEBase v0.2. It provides a
sharded, resumable local store of **raw activations** and **separately keyed,
lossless SAE feature codes**. It does not implement the batch intervention,
benchmark runner, or reliability audit planned for M2–M4.

## Quickstart

```bash
python -m pip install -e '.[dev]'
python examples/10_activation_store_offline.py
python -m pytest -q tests/test_m1_store.py
```

The example is synthetic, with no model downloads. A real, opt-in Pythia
acceptance run is defined in `.github/workflows/m1-real.yml`, which checks the **frozen real M0 reference** in
`tests/fixtures/m0_reference_public.json` against live loaded checkpoint
weights and activations. The fixture was copied byte-for-byte from the
successful [M0 run](https://github.com/OpenSAE/SAEBase/actions/runs/38634372619),
so future M1 reruns do not depend on an expiring Actions artifact.

## Python API

```python
from saebase.api import ActivationSite, ModelAdapter
from saebase.store import (
    ActivationIdentity, ActivationWriter, CodeIdentity, SampleRecord,
    ShardedReader, TokenSelection, collect_activations, encode_from_store,
)

# model: loaded model wrapped in ModelAdapter; sae: loaded SAE.
# Model/SAE SHA-256 values are computed from the actual loaded states.
site = ActivationSite(layer=3, hook_name="blocks.3.hook_resid_post")
model_identity = ActivationIdentity(
    model_key="pythia70m", model_state_sha256=model_weights_sha256,
    tokenizer_sha256=tokenizer_sha256, hook_name=site.hook_name, layer=site.layer,
    preprocessing={"bos_policy": "tokenizer-default"},
    execution={"model_mode": "eval"},
)
records = [
    SampleRecord(sample_id="prompt-001", token_ids=tuple(token_ids),
                 prompt_group_id="sample-group-1", split="evaluation",
                 selection=TokenSelection("last_valid")),
]
with ActivationWriter("./activation-cache", model_identity, width=512) as writer:
    collect_activations(model, records, writer, site=site)
    activation_path = writer.path

raw = ShardedReader(activation_path)
print(raw.get("prompt-001").positions, raw.get("prompt-001").values.shape)

sae_identity = CodeIdentity(
    sae_key="pythia70m_resid_sm", sae_state_sha256=sae_weights_sha256,
    d_in=512, d_sae=32768, inference_mode="per_token",
)
code_path, stats = encode_from_store(activation_path, sae, sae_identity,
                                     layout="csr", max_rows=128)
print(stats.collected, ShardedReader(code_path).get("prompt-001").values.shape)
```

A second SAE with a **different verified checkpoint identity** can be encoded
from the same `activation_path`. `encode_from_store` accepts no model object and
therefore performs **zero additional model forward passes**.

### Site and sample identity

The raw cache key includes loaded model weight SHA-256, tokenizer SHA-256,
chat-template hash when present, the hook and layer, requested dtype,
preprocessing configuration, and execution configuration. Live collection verifies the full actual model
state digest before capturing; SAE encoding likewise verifies the actual SAE
state digest. It deliberately does
**not** include the SAE. The code cache key is a child of the activation cache
and includes SAE full-state SHA-256, dimensions, checkpoint revision, inference
mode and normalization details.

`SampleRecord` stores the **exact input token IDs** (without storing raw
prompt text), optional attention mask, prompt group, split, variant role and
target site policy. Reusing a `sample_id` with different input tokens or split
metadata raises `CacheIntegrityError` rather than silently treating it as a
hit. A code sample also binds to the byte digest of its source activation rows:
if a repaired/raw sample changes its activations, stale SAE codes are rejected
instead of silently reused. Rebuild the affected code cache after auditing the
changed source. Caller-provided full-weight and tokenizer digests must be truthful;
hashing the entire model state can be expensive. An unpinned branch name alone
is **not** an acceptable substitute for a weight digest.

Supported token selection: `last_valid`, `indices` (absolute token IDs),
`all_valid`, and `span` ([start,end) character offsets). For `span`, the caller
must provide tokenizer **offset mapping** in `SampleRecord.token_offsets`;
otherwise it is rejected and explicit indices can be supplied instead.
Special-token `(0,0)` offsets are excluded. Empty masks, duplicate positions,
and out-of-range selections are errors.

Live collection currently supports unpadded or **right-padded causal input**
only, because the existing `ModelAdapter` does not forward attention masks.
Left padding is explicitly rejected. Readers can store arbitrary validated
masks when data has been captured externally under a specified backend.

### On-disk layout

```text
activation-cache/<activation-identity-sha256>/
  manifest.json
  samples.jsonl                 # Convenience index, regenerated on close
  shards/
    shard-00000.npy             # Dense, mmap-compatible activation rows
    shard-00000.json            # Sample metadata, positions, row ranges
    ...
  codes/<activation+sae-identity-sha256>/
    manifest.json
    samples.jsonl
    shards/
      shard-00000-values.npy    # CSR: every nonzero code
      shard-00000-indices.npy
      shard-00000-indptr.npy
      shard-00000.json
```

Dense SAE codes are supported as an alternative to CSR. CSR preserves **all
nonzero feature activations**, including very small nonzero values, without
display-oriented top-k truncation. Dense FP32 activations are the default;
FP16/BF16 are opt-in with a configurable `max_quantization_error`. BF16 uses
little-endian NumPy uint16 payload bits and a documented conversion on read.
Readers return float32 for BF16; other dtypes retain native precision.

The default shard target is **128 MiB**, and `iter_batches(max_rows=...)`
provides bounded streaming iteration. A single sample (all selected positions)
**must fit one shard**; if it does not, reduce the positions per sample or
increase the bound. M2 will implement true intra-sample batching. Writers also
bound per-sample SAE encoding through `max_rows`; do not select more than that
many positions per sample for code generation.

### Crash consistency and corruption recovery

Data arrays and their JSON sidecar are written to temporary files and
atomically renamed **before** their shard descriptor is appended to
`manifest.json`. The manifest is the commit barrier: orphan files left after
an interruption are ignored and never overwritten. A local advisory writer
lock prevents concurrent writers in one directory. Open with `resume=True`
(default) to skip exactly matching committed samples.

`ShardedReader(..., verify_shards=True)` validates SHA-256 checksums, matrix
shapes, CSR index ranges, row pointers and sample metadata. Cached content is
not trusted by default. A truncated, deleted, modified or corrupted committed
shard blocks reads. `repair_corruption=True` **explicitly** drops affected
shards from the manifest, logs the repair and leaves broken files on disk; the
caller then recaptures only missing samples. This is not silent data repair.

`max_quantization_error` is an **absolute activation error in model units**.
Low-precision caches must separately demonstrate acceptable downstream metric
error before use in a publication. M1's reference storage remains FP32.

### Reproducibility and release boundary

- `tests/test_m1_store.py` covers format, selection, scalar validation,
  parameter-identity drift, synthetic TL/HF capture, resume after an injected
  crash, corruption repair, dense/CSR exact roundtrip, and two-SAE reuse.
- `scripts/m1_real_smoke.py` loads **actual pretrained checkpoints**, compares
  their SHA-256 digests and activation probes with M0, then independently
  checks last-token and multi-position caches plus CSR SAE codes.
- `.github/workflows/m1-real.yml` is manual or opt-in by `[m1-real]` commit.
  It reads the verified M0 reference from a committed fixture, runs on CPU,
  and publishes a `m1_reference.json` artifact. Never label the job passed
  until its real-model step and the numeric checks have succeeded.
- User prompts, activations, checkpoint weights and local cache files should
  not be committed to the source repository. The M1 acceptance report contains
  only digests, sample counts, error measures and aggregate statistics.

The data format is **versioned as v1**. A future format upgrade must provide
an explicit migration; silently interpreting older shards with new semantics
is not allowed.

## M1 streaming-size acceptance

An additional offline acceptance ran 4,096 samples of 512-dimensional FP32
activations (8,388,608 raw numeric bytes), split into 128 shards using a 64 KiB
writer buffer target. The observed pending numeric buffer never exceeded
65,536 bytes. Iteration in 64-row windows returned all 4,096 samples while
the reader retained at most two shard memory maps. This is an engineering
I/O check on synthetic arrays, not a claim about GPU forward throughput.

## M1 verified release acceptance — October 14, 2026

M1 is now **accepted** as the resumable local Activation Store foundation for the v0.2 roadmap. This records evidence from the actual `main` branch, not from synthetic tests alone.

| Gate | Evidence | Result |
| --- | --- | --- |
| Offline regression | [main CI run #38642950680](https://github.com/OpenSAE/SAEBase/actions/runs/38642950680), Python 3.11 and 3.12 | Passed |
| Analysis dependencies | Same clean GitHub Actions CI, separate `analysis-install` job | Passed |
| Wheel and sdist | Same CI, `package` job | Passed |
| Real pretrained checkpoint | [real M1 run #38642950452](https://github.com/OpenSAE/SAEBase/actions/runs/38642950452), public Pythia-70M + registered pretrained SAE | Passed |
| Original reference comparison | [M0 fixture](../tests/fixtures/m0_reference_public.json), same model/SAE full-state SHA-256 and stored activation probes | Passed |
| Resume and SAE code reuse | Actual real-model artifact, zero model calls during resumed capture/SAE code encoding | Passed |

The [real-model artifact](https://github.com/OpenSAE/SAEBase/actions/runs/38642950452/artifacts/11931745526) contains `m1_reference.json` with:

- `real_checkpoint_validated=true`, `m0_reference_verified=true`, `status=passed`.
- `sample_count=4` (three original M0 prompts plus one two-position sample), `row_count=5`, `code_sample_count=4`.
- `activation_shards=3`, `code_shards=1`.
- `max_m0_activation_probe_abs_error=0.0` (for the stored reference probes; this is not a bound on all hidden-state coordinates).
- `resumed_samples=4`, `encoded_samples=4`, `repeat_code_cache_reused=4`.
- `model_forward_calls_during_resume_and_sae_encoding=0` (deliberately blocked live model cache calls during those phases).
- SAE L0 for M0 prompts was `[172, 66, 53]`, matching the saved real M0 reference.

The code also has tested crash recovery, sparse/dense round-trip, precision errors, identity invalidation, and deliberate corrupted-shard rejection. A separate local synthetic streaming test processed 4,096 activation samples with bounded pending data and two active memory maps. Neither test implies that all future model/SAE/checkpoint combinations are already supported.

The authoritative release is the merged source at [OpenSAE/SAEBase](https://github.com/OpenSAE/SAEBase) (merge commit `88248988b33a1379a8db8012e24e6ebca27d6282`). M2 batch interventions, M3 benchmark runner and M4 reliability audit remain separate milestones.

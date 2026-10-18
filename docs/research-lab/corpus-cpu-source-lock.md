# Frozen 10×50 candidate corpus, CPU source preflight

This is the **candidate source lock**, not a released/rights-reviewed
public text dataset, not a benchmark result and not 500 independent
training examples certified to be absent from any model's training set.

## Recorded evidence

The source ledger is stored at:

- `apps/research-lab/data/source_lock_v1.json`

It pins **ten exact Hugging Face upstream commits**, the deterministic
source-export SHA-256, 50 rendered candidate records per source, and
the 500-row redacted audit fingerprint. The lock contains **no raw prompt
text** and indicates `rights_review=pending`,
`human_content_review=pending`, `near_duplicate_review=pending`.

The first successful 500-row CPU audit was produced by
[GitHub Actions run 39173005702](https://github.com/OpenSAE/SAEBase/actions/runs/39173005702),
artifact `opensae-corpus-cpu-preflight`. The source scripts have since
added strict HellaSwag class quotas and an automatic lexical near-duplicate
screen; the locked candidate SHA and aggregate fingerprint must be
updated after a new complete CPU run. Refer to Work Plan 05 for the
final accepted manifest before generating any real model activations.

## Reconstruct on an Internet-connected CPU machine

Start from the actual SAEBase checkout:

```bash
python -m pip install datasets huggingface_hub
cd apps/research-lab

python -m backend.source_audit \
  --source-lock data/source_lock_v1.json \
  --max-raw-rows 2500 \
  --output /tmp/opensae-corpus-v1-audit.json

python -m backend.corpus_lock \
  --audit /tmp/opensae-corpus-v1-audit.json \
  --lock data/source_lock_v1.json
```

The audit uses a bounded deterministic stream for most sources. For
**WikiText-103** it scans the original *ordered* text lines before grouping
passages under article headings: shuffling line-level rows would destroy
source attribution. For **SQuAD** the scan is 35,000 raw questions so
50 distinct original article groups can be selected. It does not claim
whole-dataset uniform sampling. If a pinned upstream commit is unavailable,
the workflow stops rather than selecting a new revision.

The CPU audit writes no full source text into its redacted output. Within
a temporary directory it reconstructs and hashes complete candidate rows;
that directory is destroyed after the audit. To build an internal working
copy, use `python -m backend.acquire` followed by
`python -m backend.ingest dataset`, with the source commit from the lock.
By default the latter marks data **unreviewed** and blocks the public API.

The source lock is considered valid only when all 10 exact commits,
all 500 candidate hashes, the 200/300 group split, source-export digests
and the complete overall fingerprint match. A changed sample is not
silently accepted, even when the count stays exactly 50.

## Checks completed by software vs human checks

Software checks:

- Exactly 50 unique original groups/source; original source row IDs pinned.
- No use of model correctness, feature activations or post-hoc selection.
- 20 discovery / 30 evaluation groups/source.
- One source group cannot cross usage splits.
- Choice-label balancing for BoolQ and HellaSwag.
- Source-specific render identity and withheld gold labels.
- Exact duplicate SHA check across the full 500 rows.
- CPU lexical five-word Jaccard near-duplicate screening with
  redacted sample IDs; heuristic, not a semantic proof.

Still requiring human judgment or deployment privileges:

- Verify upstream file-level terms, attribution and derivative rights,
  especially inconsistent WikiText metadata.
- Inspect selected prompt text, unsupported/unexpected examples,
  incomplete tasks, offensive material and answer leakage.
- Review automatic near-duplicate flags and semantically similar
  questions that lexical Jaccard may miss.
- Public release or HF upload requires explicit authorization.
- Tokenizer-specific lengths and offsets for each *selected model*
  must be validated when the exact tokenizer assets are available.
  This does not require GPU, but some checkpoint/tokenizer assets can
  require separate access.

Do not set `--license-reviewed` or `--human-reviewed` based solely on a
successful automated audit. A 500-prompt candidate batch is **not** a
large statistically representative model benchmark. Future confirmatory
scientific claims need independent collection and design.

## Related resources

- [Work Plan 04](../../work_plan/04_research_platform_execution_plan.md)
- [Work Plan 05](../../work_plan/05_research_lab_cpu_closure_and_gpu_handoff.md)
- [Issue #41](https://github.com/OpenSAE/SAEBase/issues/41)
- [Research Lab CPU-to-GPU handoff](cpu-gpu-handoff.md)

# Frozen 10×50 candidate corpus, CPU source preflight

This is the **candidate source lock**, not a released/rights-reviewed
public text dataset, not a benchmark result and not 500 independent
training examples certified to be absent from any model's training set.

## Recorded evidence

The source ledger is stored at:

- `saebase/research_lab/data/source_lock_v1.json`

It pins **ten exact Hugging Face upstream commits**, the deterministic
source-export SHA-256, 50 rendered candidate records per source, and
the 500-row redacted audit fingerprint. The lock contains **no raw prompt
text** and indicates `rights_review=pending`,
`human_content_review=pending`, `near_duplicate_review=pending`.

The final metadata-only 500-group and 22,500-variant CPU audit is at
https://github.com/OpenSAE/SAEBase/actions/runs/39174535881
Artifact 12154265086, all upstream source SHA values pinned. Zero exact
duplicates and zero five-word Jaccard >= 0.85 candidate pairs were observed;
this does not replace a human semantic or rights review.

A complete pinned replay and source-lock verification passed at
https://github.com/OpenSAE/SAEBase/actions/runs/39175204013.
The final source lock includes all ten source commits, every source export
SHA, the full candidate manifest fingerprint, the variant population SHA
and the lexical screening summary. Do not silently replace the cohort.

## Reconstruct on an Internet-connected CPU machine

Start from the actual SAEBase checkout:

```bash
python -m pip install datasets huggingface_hub
cd saebase/research_lab

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

- [Work Plan 04](../../github_work_plan/04_research_platform_execution_plan.md)
- [Work Plan 05](../../github_work_plan/05_research_lab_cpu_closure_and_gpu_handoff.md)
- [Issue #41](https://github.com/OpenSAE/SAEBase/issues/41)
- [Research Lab CPU-to-GPU handoff](cpu-gpu-handoff.md)

## Revised CPU protocol under review

Work Plan 05 additionally builds task-aware **unapproved** RSS variant
proposals for all ten source classes, including prose continuation prompts
and Python-comment-only wrappers for HumanEval. For each base source group
the proposal stage creates 45 variants (15 selection and 30 held-out across
the three settings). The proposal manifest exports only SHA-256 identities,
never the transformed source text. Actual semantic equivalence and
model-token alignment remain separate review gates.

HellaSwag now requires a frozen 13/13/12/12 balance of answer labels, and
the CPU source audit checks cross-source near-copies using a five-word
shingle similarity screen. That screen is a conservative lexical warning,
not a certificate of no semantic duplicates.

The source lock now pins the final balanced and task-aware candidate protocol.
A later change must use a new versioned source lock and independent review.

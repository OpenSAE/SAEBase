# WP13 after LIVE-00, LIVE-01 and LIVE-02

Updated: 2026-10-22. This is a **small trusted GPU-Agent handoff**, not another strict numerical audit.

Source: [Agent-reported measurements](13_live_00_02_report_20261022.json) · [11-job task queue](13_live_execution_matrix.csv) · [GPU handoff](13_gpu_agent_real_environment_handoff.md).

## Where we stand

| Job | GPU Agent result | Product status |
| --- | --- | --- |
| LIVE-00 | PASS_WORKING, RTX 5070 12,227 MiB, 326 GiB scratch free, Python 3.12.3, existing legacy venv | Ready for small local model tasks. Docker/`OPENSAE_LAB_DATA` missing **on GPU host only** |
| LIVE-01 | PASS_WORKING, Pythia-70M L3, 5.043 seconds, 557,487,104 bytes peak VRAM, real feature-ablation logit delta | Functional real model path; not yet a published intervention gallery |
| LIVE-02 | PASS_WORKING for local GPU capture and SQLite import: Gemma-3-270M L5/L9, 20 complete traces, 16,788 activations, 5,804 unique features, 51.257 seconds and 2,896,735,744 bytes peak VRAM | **Private local data only. Zero reviewed excerpts; no public trace pages or Space deployment yet** |

Source git SHA used by GPU Agent: `903d837` on the WP13 branch. It did not rerun the historical four GPU-00 sites. A first LIVE-01 wrapper `TypeError` preceded model loading, then the normal CLI passed. SAELens emitted a direct-repo-ID compatibility warning but loaded. Do not create bugs or GPU reruns from a harmless warning alone.

For product development we accept this agent report. We do **not** require moving 20 full traces, 5,804-feature SQLite, large raw arrays or a 150 MB ZIP to chat.

## 1. Minimal small-file handoff, without GPU

The Integration Agent has added `scripts/wp13_prepare_public_bundle.py`. It runs on the existing saved raw traces, performs CPU checks of sites and complete sparse token rows, then creates a **small metadata-only** file. It contains sample/source hashes and counts, not prompt text or activation values.

```bash
cd /workspace/SAEBase-wp13
git fetch origin docs/wp13-real-world-integration
git status --porcelain
# Use a clean worktree. Never reset a dirty tree.
git merge --ff-only origin/docs/wp13-real-world-integration

source /workspace/venvs/wp12-gpu00/bin/activate
python -m pip install -r apps/research-lab/requirements.txt

python scripts/wp13_prepare_public_bundle.py \
  --private-raw /workspace/wp13/private/runs/LIVE-02-20261022T1912Z-903d837/private_raw \
  --small-report /workspace/wp13/private/LIVE-02-small-inventory.json \
  --review-template /workspace/wp13/private/LIVE-02-review-template.json
```

If the existing GPU venv is pinned and modifying it is undesirable, run this CPU-only tool in a fresh small FastAPI environment. No new torch, model loading or CUDA is required.

**Allowed small GitHub handoff:** GPU Agent commits only the sanitized `LIVE-02-small-inventory.json` (normally a few KB) and a brief status note on a separate branch such as `data/wp13-live02-small-report` targeting WP13 PR #54. It should check the file for sensitive paths/text before pushing. **Do not commit or transfer** the full trace JSONs, the unreviewed SQLite database, weights, keys, raw prompt texts or old evidence ZIP.

The new queue reports LIVE-00, LIVE-01 and LIVE-02 as `WORKING`, with separate privacy/publication status. It does **not** mark LIVE-02 `DEPLOYED`.

## 2. Building a genuine public browser example later

The ten original engineering prompts are already present in the WP13 candidate source list, but they and their measured Trace values currently carry `public_text_reviewed=false` and `rights_reviewed=false`. Before public release an actual operator must review the original text/source scope and record approval. The GPU Agent can **propose** approval and verify source digests; it should not silently change rights flags and publish everything.

The tool provides a digest-specific **review template** with all flags false. The operator selects only the actual approved samples, records their original file/prompt hashes, named reviewer, and explicit consent. The tool then constructs the real public data locally, not in this chat:

```bash
python scripts/wp13_prepare_public_bundle.py \
  --private-raw /workspace/wp13/private/runs/LIVE-02-20261022T1912Z-903d837/private_raw \
  --small-report /workspace/wp13/private/LIVE-02-release-check.json \
  --review-file /workspace/wp13/private/LIVE-02-review-approved.json \
  --publish-reviewed \
  --release-root /workspace/wp13/candidates/wp13-reviewed-demo-v1
```

After source review, this creates:
- `features/manifest.json` and `features/features.sqlite3`, built from actual approved sparse activations;
- `explore/manifest.json` and `explore/records.json`, containing the actual reviewed token traces;
- `release-manifest.json` with data checksums and **staging-only** status.

It refuses absent review, changed original file SHA, unsupported site/weights, invalid sparse records, incompatible viewer size limits and overwrites. It does **not** automatically deploy any public Space. Start with just **one approved real Gemma L5 prompt** and confirm it opens as a feature card; only then scale to the 20 reviewed traces.

Transfer approved data only through a small Git commit or a versioned, rights-reviewed dataset/volume when genuinely needed for the public Space. Size determines the transport. Do not make raw GPU artifact transfer a routine dependency.

## 3. The special-token finding becomes a product fix

From the GPU Agent's actual data: a content token `Paris` at L5 position 1 activated F506 around 225.43, while shared `<bos>` could activate F923 near 3,405.25. A whole-prompt max-activation ranking can therefore be dominated by a tokenizer control token rather than sentence content.

The updated Research Lab now:
- excludes **known special tokens** such as `<bos>` from the default *whole-prompt content-feature ranking*;
- preserves all original sparse codes and displays the real BOS activation if the researcher explicitly selects that token;
- labels that content-only ranking as a deliberate presentation choice;
- does not invent a semantic explanation for F506 or recompute old metrics.

## 4. Next real GPU job: LIVE-03

No need to run LIVE-00, LIVE-01 or LIVE-02 again.

Select the actually activated L5 `Paris` / F506 at its original recorded prompt and token; also choose one **measured non-special-token** L9 feature from existing traces. Do not select shared BOS F923 solely because it is numerically dominant.

Run baseline, identity/no-op, actual F506 code ablation and one scale operation on the same prompt/site/position. Save the full logits privately but **commit only a tiny preview** with:
`model/SAE SHA, site, prompt SHA, feature ID, token position, baseline/edit operation, observed logit delta, whether original model output changed, measured duration/VRAM`. The data should distinguish teacher-forced logits from genuinely generated text.

An activation edit with little effect is a valid real result. No effect should not be replaced with a stronger-looking synthetic example.

In parallel, the Integration Agent will work on source-reviewed publish staging and private queue/UI plumbing. LIVE-04 (local explanation), LIVE-05 native SAEBench, LIVE-06 training and LIVE-07 circuits remain next only when the first public saved prompt/feature journey works.

## 5. Working conventions

- **Trust the GPU Agent's reported execution** for development. Preserve one small example and failure summary; no large archive transfer or independent 576-tensor audit.
- **Keep public release approval separate** from correct private compute. At the moment review count = 0.
- GPU Agent owns actual weights/CUDA/private originals and only commits redacted metadata via a separate branch/PR.
- Integration Agent owns UI, approved ingestion, CPU Docker staging, source/version manifest and PR merges.
- The RTX 5070 is the available GPU, not a hypothetical RTX 5090.
- If the CPU build lacks Docker on the GPU host, run staging on a different CPU host or GitHub/HF build system instead of blocking model computation.

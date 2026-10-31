# QA500 local-agent runbook

The WP22 cohort has 10 QA datasets × 500 frozen questions × 20 verified
model/SAE layer **sites** = up to 100,000 capture observations. Models are
loaded locally. Running the commands below can download gated checkpoints,
consume substantial disk, GPU memory and compute. Explicit local authorization
is required. **Do not publish HF or alter historical records from this guide.**

## 0. Install and test (no model weight download)

```bash
git fetch origin
git checkout feat/qa-evaluation-500-v1
python -m pip install -e '.[analysis,dev]'
python -m pytest -q harness/github_work_test/test_qa_standard_500.py
python -m saebase.datasets.qa_pipeline sites
python .cursor/hooks/validate_repo.py --full
```

The physical code lives in `datasets/`. The public import is
`saebase.datasets.qa_pipeline` via the existing source-tree alias. Check
installed wheel imports separately before release.

## 1. Freeze the ten upstream datasets

Authenticate only where upstream conditions require it. Restricted GPQA
requires accepted terms and a local HF token. **No silent substitution.**

```bash
python -m saebase.datasets.qa_pipeline prepare --dataset mmlu_pro
python -m saebase.datasets.qa_pipeline verify --dataset mmlu_pro
python -m saebase.datasets.qa_pipeline prepare --dataset all
python -m saebase.datasets.qa_pipeline verify --dataset all
```

Generated `experiment/qa500/inputs/<dataset_id>/questions.jsonl` has
500 records. Each corresponding `manifest.json` records exact HF revision,
split/config, seed, checksum and source ID coverage. Keep these files outside
Git and preserve their checksums with the GPU archive. Ten datasets must be
frozen once, then reused verbatim for every model.

Data limitations: GPQA is gated; mathematical final-answer normalization,
scoring and option-level likelihood are *not implemented in the first capture
milestone*. Long contextual Hotpot prompts may be marked as failures if their
tokens exceed the configured limit. Do not choose shorter replacement samples
after seeing model failures.

## 2. One real model/SAE smoke

```bash
python -m saebase.datasets.qa_pipeline capture \
  --dataset mmlu_pro --model gemma2_2b_base --layer 13 \
  --device cuda --limit 10 --max-tokens 2048
```

This operation loads the original model and selected real SAE weights.
It captures the **last prompt token only**, asserts expected hook name and SAE
dimensions, checks finiteness, computes capture L0 and one-token reconstruction
MSE, records baseline next-token argmax and the top ten active SAE coordinates.
It does **not** compute QA accuracy or intervene on model behavior.

Review actual tokenization, prompt text, model family, SAE checkpoint, hook,
code activations and input lengths from a local controlled inspection. The
`prompt_sha256` identifies the exact rendered prompt; reconstruct it from
the frozen input manifest when auditing. Check a replay of the same 10 sample
IDs and verify the completed manifest, unchanged result rows and SAE SHA256.

## 3. Full capture once accepted

```bash
python -m saebase.datasets.qa_pipeline capture \
  --dataset mmlu_pro --model gemma2_2b_base --layer 13 \
  --device cuda --max-tokens 2048
```

Repeat for 10 dataset IDs × the 20 targets printed by `sites`. Schedule
sequentially per model to reuse checkpoint downloads and bound GPU occupancy.
If a run fails, retain the failure receipt and resume the **exact same**
command. Pilot uses `pilot.jsonl`; full uses `capture.jsonl`, never
promote partial pilot rows into full completion without recomputation.

Output example:

```text
experiment/qa500/results/qa_standard_500_v1/
  gemma2_2b_base/gemma2_2b_pt_l25_canonical/L13/mmlu_pro/
    capture_config.json
    capture.jsonl
    capture_failures.jsonl        # only when needed
    capture_summary.json
```

Capture rows contain `protocol_id`, `status=executed`, `sample_id`,
`source_revision`, `prompt_sha256`, `site_id`, `model_key`, `sae_key`,
`layer`, `hook_name`, `sae_sha256`, `sae_d_in`, `sae_d_sae`,
`token_count`, `capture_position`, `l0`, `reconstruction_mse`,
`baseline_argmax_id`, `top_features` and associated feature explanations.
The summary declares completed/expected/missing. Keep code commit SHA,
GPU identifiers, dependency versions and source provenance in an independent
agent execution receipt.

## 4. External feature annotations

If available, download official public Neuronpedia feature exports in a
separate, explicit, rate-limited step. Do **not** invent a mapping from a
Gemma Scope HF repo name to a Neuronpedia SAE ID.

Local mapping JSON:

```json
{
  "gemma2_2b_base:gemma2_2b_pt_l25_canonical:L13": {
    "model": "EXACT_VERIFIED_NEURONPEDIA_MODEL",
    "source": "EXACT_VERIFIED_SAE_SOURCE",
    "sae_sha256": "ACTUAL_64_CHARACTER_HEX_DIGEST"
  }
}
```

Local Neuronpedia line-oriented JSON:

```json
{"model":"EXACT_VERIFIED_NEURONPEDIA_MODEL","source":"EXACT_VERIFIED_SAE_SOURCE","feature_id":17,"description":"Source-reported explanation"}
```

Pass `--neuronpedia-map /path/map.json --neuronpedia-records
/path/annotations.jsonl` to capture. Matching **real checkpoint digest**
is mandatory. Without mapping, the result deliberately contains
`explanation=null, explanation_status=unmapped`. This is correct behavior.
Do not label source explanations as newly generated or validated by SAEBase.

## 5. Required checks before additional tasks and HF publication

1. Verify 500 unique ID/manifest rows per dataset; valid answer index and
   zero answer contamination. Confirm effective chat/base input formatting.
2. Perform actual model identity replay, hook capture dimension, SAE
   encode/decode reference, nontrivial active codes and finite output checks.
3. Compare repeat runs at a fixed model/site/input in the same precision
   mode. Run intervention identity and ablation controls separately.
4. Implement explicit downstream answer scorers and accepted seven-task
   protocols. Current capture data is not automatically seven-task data.
5. Validate sample-level and aggregate metric definitions; keep unknown,
   blocked, unsupported and failed results separate. Record agent GPU receipts.
6. Independently audit HF license and whether benchmark questions or
   Neuronpedia descriptions can be redistributed before publishing.

This branch intentionally leaves HF unchanged. For handoff, report the actual
command, source commit, dataset manifest hashes, completed site coverage and
stoppers. Publish only with a human-reviewed publication gate.

## Agent prompt

> In the SAEBase repository, check out
> `feat/qa-evaluation-500-v1`, read
> `harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md` and
> `harness/github_development_doc/qa500-agent-runbook.md`, then follow the
> existing Harness contracts. Execute the offline tests and run a real
> 10-question capture smoke before starting the frozen 500-question batches.
> Use exact upstream revisions and real model/SAE weights. Keep input,
> activation, feature, explanation and checkpoint identities explicit. Report
> failures, partial coverage and unavailable GPU/gated dependencies truthfully.
> Do not publish to HF or mark seven tasks complete based on capture alone.
> Preserve the prior evaluation dataset, report complete evidence and propose
> only source-backed changes for any further required task adapters.

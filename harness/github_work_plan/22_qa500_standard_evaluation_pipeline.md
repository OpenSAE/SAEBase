# WP22 — Frozen QA500 real-model evaluation pipeline

## Goal and boundary

Prepare a **new, separate** benchmark cohort for ten upstream QA datasets, 500
actual questions per dataset, ten registered models, and two verified SAE
layer sites per model. The QA cohort is independent of Stage D's 48 historical
sites and the later 500-prompt Research Lab cohort. Preserve all historical
checkpoints, results, metrics, and provenance.

This WP is an executable **capture-first** integration milestone. Availability
in the registry does not mean a task completed on GPU. The real GPU matrix,
intervention, generation, explanation, circuits, training and HF publishing
remain conditional upon their own acceptance receipts. Do not mark seven
tasks executed because the capture command ran.

## Repository implementation

- `datasets/qa_standard.py`: source-owned HF adapter, ten exact dataset
  sources, normalization, answer isolation, stratified sampling, deterministic
  IDs, pinned revisions, 500-record freezing, checksum verification.
- `datasets/qa_pipeline.py`: 20-site checkpoint matrix, capture runner,
  model/token/SAE shape and finiteness checks, per-question evidence, resume.
- `datasets/qa_neuronpedia.py`: opt-in checkpoint-verified explanations.
- `harness/github_work_test/test_qa_standard_500.py`: offline contract tests.
- `harness/github_development_doc/qa500-agent-runbook.md`: local operations.
- This plan is the single WP22 scope and audit record; all seven existing
  task contracts remain authoritative, as do old task entrypoints.

## Major correctness invariants

1. The input is rendered from an actual pinned upstream question with no gold
   answer in the model prompt. Record prompt SHA256 before tokenization.
2. All models use the **same 500 sample IDs** per dataset. Model-specific
   token IDs and lengths differ and must be independently checked.
3. No prompt exceeds the configured token limit. Oversized, ill-formed, gated
   or unavailable rows are reported as failures, never silently truncated.
4. Each SAE uses its registered, loaded hook; source model, SAE ID, layer,
   hook, dimensions and weight hash accompany results.
5. Feature IDs are dictionary coordinates for exactly that SAE checkpoint.
   Top features are ranked by absolute nonzero activation at the specified
   last-token site, not by magnitude of the raw residual stream.
6. Numerical values are real finite tensors and match their documented
   definition. The present `reconstruction_mse` uses mean squared activation
   component error for **one token**, not a corpus FVU or native SAEBench score.
7. Only external annotations with an explicit exact SAE weight SHA256 mapping
   and Neuronpedia model/source ID can be attached. Missing explanations are
   `unmapped`, `not_available` or `checkpoint_mismatch`, never invented.
8. Complete runs must have exactly 500 unique executed sample IDs; partial
   runs remain partial and must not be published as complete.
9. Provenance includes source revision, protocol, model and SAE site, code
   commit and device/runtime receipt. Publish source terms and redistributable
   artifacts only after checking upstream licenses.
10. Calibration, feature-selection, held-out test, intervention and
    explanation records remain separate; capture scores alone are not
    evidence that the seven scientific tasks completed.

## Local agent execution contract

Start with `qa500-agent-runbook.md`. Apply existing Harness contracts
`task execution`, `dataset`, `model`, `experiment`, `evidence`,
`verification`, `publication` by their actual registry identifiers; the
agent must inspect the current contract index rather than invent names.

Use a fresh worktree; run preflight and offline unit tests before downloading
weights. Freeze one dataset, verify it, then perform a ten-question real-pair
pilot. Advance to 500 questions only after independent review of actual input,
hook capture, SAE dimensions, feature IDs, and identity controls.

For each (dataset, model, SAE, layer) site record:
- exact CLI invocation, Python/dependency versions, GPU name/memory
- source and model revisions, true SAE checkpoint SHA256, hook, dtype
- frozen input manifest hash and input record SHA256
- attempted, executed, skipped/failed and error causes
- per-sample raw result shard hash and aggregate manifest
- evidence of actual model weights, rather than a dry-run result
- verifier conclusion with distinct engineering and scientific statuses

Stop on invalid model hook, changed identities, corrupted cached results,
nonfinite values or out-of-range feature indices. Do not change benchmark
samples or scoring rules to make a blocked pair pass.

## Remaining required agent implementation

The capture-first code is **not a complete seven-task QA harness**. Before
publishing as a seven-task matrix, implement batch adapters for baseline QA
scoring, benchmark Core, intervene, generate, feature explain, circuits and
train, using existing official SAEBase primitives and upstream native
estimators, with frozen task-specific protocols and independent controls.

- QA correctness needs a separate answer scorer for each answer type, with
  model-appropriate chat template, no scoring from supplied gold as prompt.
- Benchmark: reconstruction/variance, original definitions, all 500 prompts.
- Intervene: frozen non-leaky feature-selection calibration, identity control,
  proper patched and unpatched outputs, selected held-out subset.
- Generate: independent baseline and edited continuations, greedy seed 0,
  bounded response lengths and task-specific answer extraction.
- Explain: feature discovery and held-out evaluation; source explanations
  are labels/proposals, not validated causal interpretations.
- Circuits: only when an independently compatible native tracer, transcoder
  and site exist; explicit unsupported status otherwise.
- Train: separate language-model training corpus and bounded native SAE
  training, not a new SAE trained on 500 test questions.

## HF publication gate

Target `OpenSAE/SAEBase-Evaluation`, add **new cohort names**:
`qa_v1_inputs`, `qa_v1_sites`, `qa_v1_runs`, `qa_v1_metrics`,
`qa_v1_per_question`, `qa_v1_task_records`.

The publishing agent must first validate provenance, redistributability,
upstream source licenses, unique sample IDs and raw result checksums.
GPU runs and HF writes require explicit execution approval and credentials
on the agent machine. This WP never authorizes automatic dataset overwrite.
Historical 25 configs and any Research Lab results remain unchanged.

## Final acceptance

- 10/10 frozen 500-record corpora with exact upstream revisions and SHA256
- 20/20 registered site preflights and actual GPU capture status recorded
- 100,000 expected site x question capture measurements, with exact coverage
- all seven task-specific status tables, with unsupported/blocked distinguished
- independently audited per-question records and reproducible report
- new HF cohort published only after all publication gates pass

Status on creation: **software implementation started, GPU run pending,
seven-task adapters pending, HF publication pending**. Do not upgrade this
status without evidence.

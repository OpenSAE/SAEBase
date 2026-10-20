# Research Lab v0.4.0 — release acceptance and remaining gates

The **software preview** and the **scientific raw-data release** are separate deliverables. The software preview can be distributed after source-bound CI and package verification. The raw-data release requires additional approvals and independent provenance checks.

## Software release — required

- [x] Source code for CPU Research Lab, GPU executor, and original-estimator interfaces available in the versioned repository.
- [x] English and Chinese README updated with real scope and runnable CPU browsing instructions.
- [x] New experiment summary separates historical Stage D 64-group results from the new 500-prompt cohort.
- [x] SAR explicitly reports the decoder-mediated intervention protocol and separate K16/K8/K4 groups.
- [x] RSS reports unique logical input occurrences and unresolved semantic-review status.
- [x] Existing automated CI and pinned estimator-regression workflow passed on the GPU-executor source branch prior to this release-preparation change.
- [ ] Six exact-source push-to-main CI contexts verified for the final v0.4.0 commit (required for tagged release).
- [ ] Wheel, sdist, SHA256SUMS and release-evidence tied to the **final** commit and independently checked.
- [ ] GitHub tag and release published and remote asset digests checked (only after the previous gates succeed).

## Science/raw-data publication — separate

- [ ] The *full* GPU experiment archive is exported from `experiments/research-lab/7fb2be8f29a1-20261019T035727Z` with raw masks, response vectors, source identity, logits/codes, exact sampled groups and execution software.
- [ ] A second host downloads that archive and verifies each original file digest. A copy under the same physical storage mount is not independent off-machine verification.
- [ ] 10×50 third-party source texts, labels, derived variations and feature excerpts pass human inspection and source-specific redistribution licensing review.
- [ ] Per-site model, tokenizer, SAE and original estimator source revisions are recorded, including the four historically unmatched/incomplete identity cases.
- [ ] SAR conditional metrics and per-K aggregate claims are independently computed from exported `g`, masks and fit/held indices.
- [ ] RSS alignment and selection/held-out logic are independently replayed on stored actual token IDs and raw feature responses; proposed input semantics independently reviewed.
- [ ] Final external research release includes controlled study limits, failure denominators, independent download verification and persistent storage, without publishing private input text.

## Product deployment — separate

- [ ] Hugging Face Space deployed with appropriate organization authorization and backup/rollback.
- [ ] Real generated text steering demonstrated beyond one-token logits.
- [ ] Online GPU service deployed only with owner authentication, compute budgets, private object storage, data deletion controls and independent negative tests.
- [ ] Three external users test the actual Research Lab workflow.

**Release naming:** `v0.4.0` identifies the **software preview**. It does not imply raw-data availability, full text-steered generation, or production GPU readiness.

References: [Research Lab results](results-20261020.md) · [Work Plan 05](../../work_plan/05_research_lab_cpu_closure_and_gpu_handoff.md) · [Source lock](corpus-cpu-source-lock.md).

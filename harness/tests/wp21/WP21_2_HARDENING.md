# WP21.2 Harness maintenance after round01

**Local examiner follow-up:** [WP21.2 local GPU and evidence work order](../../github_work_plan/21_wp21_2_local_gpu_followup.md). This specifies the exact inputs, commands, deliverables and stop conditions for R08, R17, R20, Benchmark and 20 reviewer archives.

This is a **new maintenance iteration**, not a modification of WP21.1's
frozen R01–R20 questions, machine cases, contracts, input fixtures, asset
revisions, tolerance values, or historical independent review finals.

## Corrected and tested offline

| Finding from round01 | WP21.2 change | Remaining boundary |
| --- | --- | --- |
| R17 `native_training_logs/` and R20 `benchmark_output/` rejected despite directory output | `audit_evidence.required_artifact_present` recognizes a trailing `/` as an actual nonempty directory, matches full relative path components, rejects symlinks/traversal, and accepts hashed experiment descendants | A directory's existence is not proof of model execution; independent positive and negative replay remains mandatory |
| R08 Pythia model missing during dispatch | `preflight.py --all` checks all 20 cases and separately reports Pythia-14m, model-only R09, and training R17 requirements | Local coordinator must fetch and pin the exact authorized checkpoint revision, lock bytes, and execute actual probes |
| `probe.py` fails outside a repo-set `PYTHONPATH` | Add trusted source ROOT to the import path and require local source for the paired Gemma loader | This does not prove that an internal loader consumed the expected immutable weight blobs; local reviewer must trace provenance |
| Misleading configured CUDA versus actual CPU | Actual model parameter and activation device is written to probe observations; `behavior.observed_device` validates real device and enforces CUDA when required | Default benchmark loader and padding behavior are product questions. They must be inspected in a real local GPU environment |
| A negative test's nonzero exit aborts a `set -e` coordinator loop | `record.py` CLI returns 0 after successfully **recording** a command; real child exit remains in immutable `commands/NNNN.json`. `--propagate-exit` retains previous behavior | Tool errors still exit 2; reviewers must examine child `exit_code`/`timed_out`, not recorder status |
| Reviewer scripts/observations absent from public PR archive | `archive_review.py` packages independent reviewer scripts, all check records, assertions and stdout/stderr as a SHA256-audited local ZIP | Archive excludes checkpoint, candidate source and large tensors; a human must inspect logs for private paths before publishing. Integrity alone is not independent replay |
| R17 mixed product reproducibility and evidence-auditor failure | `review_final.json` for **new rounds** retains the raw evidence rejection reason and failed check IDs. `review_diagnostics.py` supports independently supported multiple findings without changing status/score | Original R17 failure stands. A local examiner must reproduce training and identify deterministic CUDA controls |
| `100/100` can be mistaken for general engineering quality | `review_diagnostics.py` separates frozen dimension scores from optional human-reviewed maintainability, robustness and documentation; `summary.py` exposes explicit attribution and failed-check IDs | Automatic PASS still only means frozen checks pass; no inferred quality score is awarded |
| R17 same-seed `W_dec` max abs ~0.083 | `training_repro.py` computes elementwise frozen tolerance on real saved `.safetensors` or `.pt` weights and records all failing elements, identity and runtime hints | Requires **actual** three training logs, corpus/plan identity and checkpoints. Frozen rtol=1e-4 and atol=1e-5 remain unchanged |

## Local execution guide

Run these commands from a **trusted** WP21.2 examiner checkout, outside candidate
worktrees. The execution environment needs Python 3.11/3.12 and genuine model
resources for real model probes. Commands below do not download anything.

```bash
python3.12 harness/tests/wp21/tools/manage.py check-spec

python3.12 harness/tests/wp21/tools/preflight.py --all \
  --model-snapshot /readonly/gemma \
  --sae-snapshot /readonly/gemma-sae \
  --pythia-snapshot /readonly/pythia-14m \
  --output /absolute/control/preflight_all.json
```

If R08 reports a missing Pythia snapshot, prepare the exact immutable
`EleutherAI/pythia-14m` revision
`d31f659f077f9ea500c4d7faa999893d7f28cbdf` **before** the next round.
Never substitute a toy, random checkpoint, or unpinned revision. Preserve all
round01 `ENV_BLOCKED` records regardless of what the next run finds.

After new independent review finishes, archive the reviewer provenance
locally. Confirm reviewer scripts are authored and controlled independently;
archive creation is not an authorization step for candidates.

```bash
python3.12 harness/tests/wp21/tools/archive_review.py create \
  --review /absolute/wp21_reviews/R20 \
  --output /absolute/wp21_archives/R20_review.zip
python3.12 harness/tests/wp21/tools/archive_review.py verify \
  --archive /absolute/wp21_archives/R20_review.zip
sha256sum /absolute/wp21_archives/R20_review.zip
```

The ZIP includes original, potentially identifying paths and stdout/stderr.
Keep it local until explicitly reviewed for safe publication. An external
SHA256 must accompany any shared archive; a self-contained manifest alone
cannot authenticate provenance.

To diagnose R17, provide the two actual `sae_weights.safetensors` files from
the seed-7 native runs, or trusted `weights_only`-readable `.pt` state dicts.
The original three-run evidence remains immutable.

```bash
python3.12 harness/tests/wp21/tools/training_repro.py \
  --weights-a /absolute/seed7_attempt_a/sae_weights.safetensors \
  --weights-b /absolute/seed7_attempt_b/sae_weights.safetensors \
  --plan-a /absolute/seed7_attempt_a/plan.json \
  --plan-b /absolute/seed7_attempt_b/plan.json \
  --rtol 0.0001 --atol 0.00001 \
  --output /absolute/wp21_diagnostics/R17_weight_comparison.json
```

Inspect source/training logs for seeding of Python/NumPy/PyTorch CPU/CUDA,
`torch.use_deterministic_algorithms`, CuBLAS workspace configuration before
CUDA initialization, cuDNN/TF32 settings, data order, precision and TopK
kernel selection. The report captures the **diagnostic process** settings,
not proof that historical training used those settings. Correct an observed
source or environment issue in a new attempt; never relax the frozen
comparison just to declare the old attempt PASS.

For additive root-cause findings, an independent reviewer may write
`findings.json` with schema `saebase.wp21.review_findings.v1`, `case_id`, and
one or more `{cause, owner, description, evidence_refs}` objects. Supported
cause/owner codes are listed in `review_diagnostics.py`. A separate optional
quality file `saebase.wp21.engineering_quality.v1` contains three dimensions
`code_maintainability`, `robustness`, and `documentation_accuracy`; each is
`UNREVIEWED`, `REVIEWED_SATISFACTORY`, or `REVIEWED_WITH_FINDINGS`, with
concrete evidence references for reviewed dimensions. Neither can rewrite a
frozen review.

```bash
python3.12 harness/tests/wp21/tools/review_diagnostics.py \
  --review-final /absolute/wp21_reviews/R17/review_final.json \
  --findings /absolute/wp21_diagnostics/R17_findings.json \
  --output /absolute/wp21_diagnostics/R17_diagnostics.json
```

## Items requiring the local GPU examiner

1. Supply and pin the missing Pythia-14m snapshot, rerun real R08 probes, then
   attempt R08 in a **new** isolated run.
2. Run the three R17 native SAE training attempts again with actual plan,
   inputs, weights and environment receipts. Compare same-seed weights and
   separate any reproducibility failure from directory artifact audits.
3. For benchmark integration, check variable-length prompt batches through
   `saebase.benchmarks.runner` and record the observed pad token forwarding
   to `BatchEngine`. A code fix belongs in a separately reviewed **product**
   change; Harness must not modify the original product baseline silently.
4. For R12/R13/R20, record the actual device used by the benchmark loader
   and compare it with the task-requested device. If CUDA execution is needed,
   repair/verify the **product** loader on a new attempt, not by changing case
   success criteria.
5. Keep each original Reviewer script, JSON assertion, command log, input
   identity and raw numeric array in a trusted local evidence directory.
   Create review archives and separate engineering-quality observations;
   decide which redacted artifacts may be published.

## Regression and compatibility

```bash
python -m pytest -q \
  harness/github_work_test/test_wp21_local_suite.py \
  harness/github_work_test/test_wp21_exam_ready.py \
  harness/github_work_test/test_wp21_hardening.py
```

The 20 frozen cases, 22 Contract catalog and `suite_lock.json` must still pass
`manage.py check-spec`. The WP21.1 round01 results (17 PASS, R08 ENV_BLOCKED,
R17 FAIL, R20 INCOMPLETE) remain historical observations. New tool results
are separate evidence and cannot retroactively mark them PASS.

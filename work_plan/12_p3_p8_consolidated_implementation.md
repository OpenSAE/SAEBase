# WP12 P3–P8 · Consolidated Source Implementation

Updated 2026-10-22. Development branch `feat/wp12-p3-p8-platform`.
Base branch: `feat/wp12-p2-feature-explorer`, which depends on P1 and P0.

**Scope:** implement all practical CPU/code paths now, reuse original open-source
algorithms and existing SAEBase results, and defer wide numerical, hosted, GPU
and user acceptance to P8. Source code and a passing CPU CI are not sufficient
to label every feature production-ready.

## Stage-specific outputs

| Stage | Function IDs | Actual source code | What still requires native execution |
| --- | --- | --- | --- |
| P3 interventions | F04 F19 F21–F25 | `saebase/wp12/interventions.py`, existing `saebase/interventions/`, `saebase/generate/nnsight_worker.py`, Explore → Intervene | Native pyvene on a compatible architecture, generation-step edits and paired GPU baseline/edited output |
| P4 evaluation | F36–F40 | `saebase/wp12/benchmarks.py` small upstream Core/Sparse runner, `saebase/wp12/custom_saebench.py` bounded custom-SAE original Core/Sparse runner, original `scripts/m3_native_saebench_smoke.py`, fairness-matched chart | Native published source/model/SAE with original samples and full metric/result comparison |
| P5 native training | F12 F13 F46–F50 | `saebase/wp12/training.py` real SAELens and Sparsify worker invocations, original checkpoint manifest, native Sparsify true resume and a separate 2+-GPU guard | Training a real checkpoint, real resume and native encode/decode verification. F50 needs real 2+ GPUs |
| P6 circuits | F05 F26–F30 F43 F44 | `saebase/wp12/circuits.py` genuine circuit-tracer CLI, distinct representation types, typed attribution graph and raw-source hashes, native v1 graph-file conversion, filter, SVG graph explorer, local annotations, typed feature intervention request | Actual native graph + numerical parity, real transcoder intervention, official cross-layer loader |
| P7 private/NDIF | F20 and private execution | `saebase/wp12/ndif.py`, `apps/research-lab/backend/private_api.py`, `private_worker.py`, `scripts/wp12_private_worker.py` | Explicit credentials/deployed-model allowlist for real NDIF; authenticated self-hosted end-to-end worker; public GPU remains off |
| P8 final acceptance | All F01–F50 | `saebase/wp12/acceptance.py`, `scripts/wp12_p8_acceptance.py`, `work_plan/12_capability_acceptance_matrix.csv`, offline tests | Original raw artifacts, independent native numerical comparison, external 3-user test, accessibility and real two-GPU evidence |

No native GPU/NDIF/Sparsify/SAELens/circuit-tracer run was performed by
this source-only development branch. The prior GPU-00 legacy parity report
remains a separate result with independent review pending. No historical paper
metric is changed. The last release-quality check is separate from code review.

## User-facing Research Lab

Existing Features/Results/Intervene/Analyze/Datasets/Reports remain. New
navigation adds **Benchmarks**, **Training**, **Circuits** and **Jobs** to
`apps/research-lab/frontend/index.html` with real read-only APIs:

- `GET /api/wp12/benchmarks`: rights-reviewed, immutable source-matched SAE
  comparison records; plots L0 against original metric, never mixes differing
  datasets/hooks/protocols.
- `GET /api/wp12/training`: source/permission-reviewed real checkpoint run
  summaries; local config export does **not** train a model.
- `GET /api/wp12/circuits`: approved signed normalized graphs with input
  tokens, feature nodes, attribution weights, **error nodes** and target logits;
  uploaded JSON is a local browser-only view until approved.
- `GET /api/wp12-compute`: truthfully reports CPU-only public availability.
  Public `POST /api/jobs` remains deliberately disabled.
- `/api/private/*`: separate opt-in authenticated API on a private host
  (not publicly provisioned), for input upload, leased queue, owner cancellation,
  worker claim/finish, status, and now **source-hash checked private results**.
  Authentication uses distinct strong operator/worker bearer secrets.

Research Lab starts with accurate empty states if approved records have not
been created; there are no fake feature or circuit results.

## Minimal native execution

**Bounded P3 teacher-forced SAE edit** via an already loaded
`LoadedResearchPair`:

```python
from saebase.api import load_registered
from saebase.wp12.interventions import Edit, EditPlan, compare_sae_edits

pair = load_registered(sae_key="pythia70m_resid_sm", layer=3, device="cuda")
ids = pair.model.tokens(prompt="The capital of France is")
plan = EditPlan(pair.identity.sae_key, pair.identity.layer, -1, (
    Edit("ablate", (13,)),
))
result = compare_sae_edits(pair, ids, plan)
print(result["logit_delta_l2"])
```

This is actual model execution. Native PyTorch module editing is a separate
operation and **not** an implementation of pyvene's entire configuration
semantics.

**P4 native SAEBench pilot:** use the existing pinned original upstream
checkout at commit `8042bb3828c6340da8d12062324e92b2077c571c`,
the genuine small Core/SparseProbing runner in `saebase/wp12/benchmarks.py`
and custom-SAE runner in `saebase/wp12/custom_saebench.py`. The custom SAE
adapter runs the actual original small evaluators with **one reconstruction
and one sparsity-variance batch for Core**, not the upstream
`run_evals` entrypoint's unbounded defaults. It does not require an API key.
A full native benchmark suite needs separate C2/C3 approval.

**P5 Sparsify:** use a frozen local model and dataset snapshot with an
original model/dataset SHA, explicit `--hookpoints`, `--k`,
`--num_latents`, `--max_examples` and `--batch_size`. Native Sparsify
actually resolves module names, creates activations on the fly and trains.
SAELens uses its original `LanguageModelSAETrainingRunner` with dedicated
checkpoint/output folders. Dataset type and model hook compatibility must be
checked against the native worker before computation.

A native Sparsify full restart is **`--resume`** and resumes
`checkpoints/{run_name}` in the *same run folder* with its optimizer state.
`--finetune` loads weights for a new run and does **not** count as a resumed
trajectory. A full restart calls `resume_sparsify_training(...)` with the
matching immutable original plan. This has not been verified with real weights.
Sparsify's training step budget is an upper bound on input examples,
not a claim that its native trainer directly supports a `--steps` flag.

**P6 Circuit Tracer:** only the official
`circuit-tracer attribute --prompt ... --transcoder_set ... --slug ...`
command executes attribution. The local `import_native_graph_file()` now converts the original upstream
circuit-tracer frontend v1 `{metadata,nodes,links}` JSON into a typed OpenSAE
GraphRecord with exact native node IDs, signed weights and retained error nodes.
The converter reads no `.pt` pickle and reconstructs true feature indices
from `node_id`, not the upstream Cantor-encoded auxiliary `feature` column.
Actual native output parity on a real graph is still a P8 task.

**P7 private self-hosted worker** is disabled by default:

```bash
# On an isolated authorized host, NOT the public Hugging Face Space:
export OPENSAE_ENABLE_PRIVATE_JOBS=1
export OPENSAE_PRIVATE_DATA=/private/opensaebase
# Supply separate random 32+ character operator and worker secrets
# using your private secret manager; do not commit their values.
# Run the backend application after correctly configuring auth.
python scripts/wp12_private_worker.py --device cuda
```

This executes one queued job and exits. The queue/status APIs do not expose
private prompt text. The owner retrieves the full edited result only through
`GET /api/private/jobs/{job_id}/result` using its original SHA256 and
operator authorization. Any deployment requires TLS, access controls and
bounded quotas. NDIF is a different remote backend and stays blocked without
its valid deployment/credentials and explicit permission.

## P8 final, after P2–P7 development

```bash
python scripts/wp12_p8_acceptance.py \
  --ledger work_plan/12_capability_acceptance_matrix.csv \
  --output /private/wp12-release-audit

# Optionally, independently inspect exact existing evidence bytes:
python scripts/wp12_p8_acceptance.py \
  --ledger work_plan/12_capability_acceptance_matrix.csv \
  --archive /private/wp12-gpu00-evidence.zip \
  --archive-sha256 470d5136d51f58e8ca4278670cd172e4eff9484600e138edbe048c95a500a41b \
  --output /private/wp12-release-audit
```

The evidence archive must actually be present; the CLI does not fetch
external `/workspace` data or deserialize untrusted PyTorch pickle data.
The separate native numerical array comparison remains to be completed.
No 50/50 RELEASED claim is valid from source checks alone.

## What is still intentionally not solved

- **F20 NDIF:** actual remote deployment and user credentials absent.
- **F50 distributed training:** no authorized real two-GPU execution.
- **Circuit source conversion:** original frontend v1 JSON is supported
  via `import_native_graph_file()`. A separate
  `saebase/wp12/circuit_intervene.py` performs an actual single-layer
  transcoder feature edit with state/SHA checks, original-vs-edited logits,
  and preserved non-selected decoder content. Other source graph schemas
  and full native CLT interventions remain unsupported.
- **Public live prompt/steering:** no permission to deploy a continuously
  running GPU or publish unreviewed prompt text.
- **SAEBench full suite, native pyvene, SAELens/Sparsify checkpoints:**
  actual input availability, compute and original-source matching still needed.
- **Externally validated release:** ten authentic features and three external
  researcher sessions remain final acceptance work, not fake demo records.

Core implementation is now classified separately from P8 release evidence. The
[core function assessment](12_core_function_assessment.md) records 48 PASS,
0 PARTIAL, and 2 BLOCKED (F20/F50), with explicit core
tests, execution scope, remaining gaps and all P8 signoffs PENDING. PASS
means a concrete executable core route exists. It is not scientific or
public hosted-service certification.

## Subsequent F03/F10/F19/F44 executable core additions

- F03: A real local-only language model produces an SAE feature hypothesis and predicts activation on disjoint held-out positive/negative contexts. `apps/research-lab/backend/auto_interp_local.py` and `scripts/wp12_autointerpret.py`.
- F10: Projection vectors for individual TransformerLens attention heads and MLP neurons reconstitute measured original component outputs. `saebase/api/fine_components.py`.
- F19: Native NNsight v0.8 `tracer.iter` selects generation forward occurrences and compares actual baseline and intervened output token IDs. `saebase/api/nnsight_local.py`.
- F44: Gemma Scope 2 original typed CLT shards implement triangular source-to-target feature decoding, with `execute_clt_feature` editing actual downstream layer output hooks. `saebase/sae/scope2_clt.py`, `saebase/wp12/clt_intervene.py`.

All four are core implementation PASS within their explicit source scopes, not independently accepted real-checkpoint P8 release claims.

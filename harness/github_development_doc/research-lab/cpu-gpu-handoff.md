# OpenSAE Research Lab — CPU to GPU handoff

Current state: CPU-first scaffold submitted in SAEBase Work Plan 04.
Do not label a planned task as completed merely because it has an API or UI.

## What already works

1. Responsive browser with Results, Features, Intervene, Analyze, Datasets and Reports.
2. Read-only 48-site, 64-group historical verification evidence, joined to
   actual per-site metrics and original source links, with no computed values invented.
3. Reproducible ten-dataset catalog, optional bounded HF CPU source downloader,
   independent 50-group selection and 20/30 source-group usage split.
4. Checksum-verified feature index accepting only real, identity-matched
   observations and rights-approved public text excerpts.
5. Trusted-source CPU method wrappers for unchanged SAR, RSS and Localize.
6. Immutable report publisher and an API that refuses unconfigured online inference.
7. Python 3.11/3.12 and browser-JavaScript syntax tests.

Not completed: approved and released 500 base prompts, genuine new
activations, 300 real feature cards, full new cohort analyses, authenticated
GPU worker, public Space deployment, and new-model task-level comparison.

## CPU prerequisites before large GPU compute

Download/prepare the source datasets on an Internet-enabled CPU host.
Resolve exact upstream source commit and inspect the 50 selected rows
per dataset, source licenses, distribution rights and derived prompt rendering.
Do not set review flags without a real review.

All source prompt groups should be frozen before running models. Each
variant belongs to its parent and cannot enter another usage split.
Actual model-specific tokenization and truncation must be recorded.

Prepare one vertical pilot: two source prompt groups per dataset.
Confirm exact model and SAE hashes, hook, token index, task readout,
activation capture, intervention semantics and estimated resource consumption.

## Worker output format

A real activation observation JSONL row must include:

- site_id, exactly model:sae:Llayer from the 48-site acceptance matrix
- model_sha256 and sae_sha256, matching actual loaded checkpoint hashes
- source_type equal to real_checkpoint
- feature_id, token_position, measured finite activation
- prompt_group_id, sample_id, frozen corpus release identity
- optional token_text, text_excerpt and public_text_reviewed boolean

Never copy the README example hashes or values into an observed dataset.
Run the real checkpoint workflow first, preserve raw activation/code shards,
tokenizer identity, dtype, normalization and position policy. After
independent review, use the offline feature index builder.

## SAR

Standard protocol ID: sar.original.k16.m384.fit256.v1.

Each input file supplies site/checkpoint hashes, prompt_group_id,
dataset_release, protocol_id, source_type=real_checkpoint, masks with
shape 384×16, 384 measured behavior values g, fit indices of length 256,
disjoint held indices of length 128, and a separately measured g_full.

The offline methods adapter calls original sar.py from a trusted local
checkout after checking source-file SHA. No approximation or made-up
interaction matrix is substituted. One set of 384 coalitions is not
384 independent prompt groups.

## RSS

Protocol ID: rss.transform.v1.

Input includes real clean feature codes, aligned selection/held-out
differences and 5 selection variant IDs distinct from 10 held-out variant
IDs. The arrays have shapes 5×F and 10×F and preserve consistent feature
ordering; transformed prompts and their token anchors are separately
archived. The original selector and held-out persistence estimator are
called with a pinned SHA, not rewritten in the browser.

## Input Localization

Use six named binary factors and all 64 combinations in correct bitmask
order for the new controlled-grid task. Existing paper reference grids
may have different sizes; do not truncate them and claim they are complete.
Run the original installed sae_localize package to compute scope curves.
Do not apply ordinary independent-prompt bootstrap to complete grid states.

## Required GPU development order

1. Pilot 20 curated prompts on one small and one planned deep-dive site;
   measure errors, VRAM, output size and actual throughput.
2. Run base 500 groups over the 48 registered sites. Retain the original
   immutable Stage D 64-group evidence unchanged.
3. Materialize real feature observations and build index, using only
   text permitted for redisplay. Surface missing coverage instead of
   fictional descriptions.
4. On the three chosen main sites, run full K16 SAR and preserve raw g
   and masks. Keep rotation controls a separately identified protocol.
5. Generate dataset-aware RSS variants with frozen texts, independent
   held-out identities and valid aligned positions, then run their
   genuine measurement and CPU selection/evaluation.
6. Run original controlled-grid Localize with a complete design.
7. Produce tested report payloads and checksum manifests; verify archives
   after independent download before changing the public site.
8. Implement an authenticated, durable job queue with cancellation,
   resumption, bounded budgets, private text retention, idle/overload
   controls, real GPU numeric parity tests, and independent QA.
9. Only then turn on online inference and deploy the live public Space.
   Keep all existing historical read-only views available without GPU.

## Independent acceptance requirements

- Every displayed metric has source revision, checkpoint identity and
  protocol. Results with incompatible readouts are not falsely ranked.
- No fit/held leakage or duplicate independent group counting.
- Reports distinguish source provenance from scientific validity.
- Synthetic toy tests stay explicitly tagged synthetic.
- No third-party model weights or user secrets pushed to this repo.
- Simulate worker cancel/OOM/restart, database recovery and private
  artifact access control before enabling public POST /api/jobs.


## New CPU-stage artifacts prepared in the implementation PR

- saebase/research_lab/data/coverage.json fixes six deep sites, three
  standard SAR sites and the single online candidate to exact historical
  names. It does not claim all those sites have new prompt measurements.
- backend/plan.py generates 632 deterministic task specifications, containing
  only source identity, planned protocol and explicitly blocked prerequisites.
  Three historical small-model site records lack complete weight hashes; these
  remain blocked for *new* captures until checked against newly loaded weights.
- backend/feature_store.py writes an immutable paginated SQLite feature
  browse index suitable for growth beyond a single 64 MiB JSON file.
- backend/readouts.py prevents a single next-token score from masquerading
  as a complete answer likelihood, and validates RSS input anchors.
- backend/transforms.py proposes exactly 45 transformation instances per
  supported frozen prompt (selection/held-out roles included), without
  marking semantic validity as approved. Generic variants for TinyStories,
  WikiText and HumanEval are deliberately blocked.
- saebase/research_lab/Dockerfile packages the historical read-only site,
  independent of a GPU worker. CI runs its own Docker smoke for 48 historical
  site records and checks no new curated corpus has been silently fabricated.

A later operator should start with backend.plan and resolve the listed
blocking prerequisites, not replace the manifest with optimistic flags.


## Durable queue pre-integration checks (no GPU)

An internal CPU JobQueue now exists. It accepts a strictly validated
intervention configuration and an opaque SHA-256 private input reference,
with no raw prompt text in its SQLite metadata. It has fixed owner quotas,
idempotent client keys, atomic worker leases, lease expiration recovery and
owner-only status. All of these are unit-testable without a device.

The public HTTP POST /api/jobs route remains deliberately closed. GPU
deployment must explicitly connect:

1. Authenticated owner verification and per-user compute budgets.
2. Encrypted private input/result object storage with deletion/TTL policy.
3. A trusted GPU worker that claims jobs and honors cancellation/lease
   heartbeats before running any further token predictions.
4. Actual model/SAE checkpoint digest checks and full result manifest
   verification before calling JobQueue.finish with a result SHA-256.
5. Failure injection tests for two workers, stale leases, OOM and process
   restart. Do not return a success state merely because a worker ran.

No request to the public UI currently creates a persistent queued job.

## Complete six-factor inputs prepared on CPU

The code saebase/research_lab/backend/controlled_designs.py generates two 64-state project-authored grids. It provides conversion to the original SAEBase ControlledInputDesign. See harness/github_development_doc/research-lab/controlled-six-factor.md.

Run python -m backend.controlled_designs --output-dir /tmp/opensae-six-factor-v1 from saebase/research_lab after installing SAEBase.

The real-model design still requires actual SAE activations; the mathematical known-factor control remains explicitly synthetic. Neither belongs in the natural 500-prompt cohort.
 
## Work Plan 05 source lock and worker claim fencing

Read harness/github_work_plan/05_research_lab_cpu_closure_and_gpu_handoff.md first.
The ten sources have 500 metadata-locked candidates (200 discovery, 300
evaluation); raw third-party text remains private until rights and content
approval. Exact SHA values are in saebase/research_lab/data/source_lock_v1.json.

The CPU stage generated 22,500 proposed RSS variants and a lexical
near-duplicate screen. All variants retain explicit semantic-review and
tokenizer-alignment requirements before a real GPU run.

JobQueue heartbeat and finish now require the attempt returned by claim.
The GPU worker must send the current attempt on every heartbeat/commit.
A stale worker with a reused worker ID must be rejected, including after
a job timeout and re-lease. Do not open POST /api/jobs to public users
without authentication, quotas, private storage and worker parity tests.

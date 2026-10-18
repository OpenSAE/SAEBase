# OpenSAE Research Lab — CPU-first implementation

This is a **runnable** offline/browser research explorer for Work Plan 04.
It uses the 48 real SAEBase Stage D verification records that already exist.
The website does not fabricate feature activity, model completions, SAR/RSS
numbers, or claims about the 500-prompt corpus.

## Status

| Area | What works without GPU | What is not yet complete |
|---|---|---|
| Results | Browse and filter actual historical verification metrics, link original reports | New 10 × 50 population results |
| Features | Real-only index builder, API, search, token excerpt cards after ingestion | Fresh model activations and 300 reviewed cards |
| Interventions | Site/operation/feature preflight and UI | GPU steering and authenticated online worker |
| SAR / RSS / Localize | CPU adapters calling the original research estimator code on saved arrays, analysis index | New actual coalition/variant/controlled grid data |
| Datasets | 10-source catalog, optional HF source exporter, deterministic 50-record curated freezer | Source acquisition and human/rights review |
| Reports | Immutable JSON publisher with checksummed public records | Any unreviewed scientific outcome |

The prior 48-site experiments evaluated 64 controlled prompt groups per site;
they must NEVER be counted as the requested 500 new curated prompts.

## Launch without weights or GPU

From repository root:

~~~bash
cd apps/research-lab
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.api:app --host 127.0.0.1 --port 8000
~~~

Open http://127.0.0.1:8000. The read-only API docs are at
http://127.0.0.1:8000/api/docs.

Local tests (run from repository root):

~~~bash
python -m pip install -r apps/research-lab/requirements.txt pytest httpx
python -m pytest -q apps/research-lab/tests/test_platform.py
node --check apps/research-lab/frontend/app.js
~~~

The live website requires no JavaScript build tool or npm installation.
The web font is optional; offline browser fallback works.

## Acquire ten data sources on a CPU host

The source registry lives in data/catalog.json. Ten candidates are planned,
with 50 base prompt groups each and a 20/30 discovery/evaluation split.
No third-party text is silently made public.

For optional Hugging Face export, install datasets and huggingface_hub,
then from the research-lab directory run:

~~~bash
python -m pip install datasets huggingface_hub
python -m backend.acquire --source boolq --output /tmp/boolq-source.jsonl
~~~

This resolves the upstream SHA and creates a bounded deterministic shuffled
stream export. The source export manifest says it is **not** uniform random
sampling over the entire upstream collection. WikiText lines are grouped into
original articles by headings before selection. The exporter does not upload
data, mark license approval, or inspect questions for scientific quality.

After manually reviewing the export and sourcing/licensing terms:

~~~bash
python -m backend.ingest dataset --source boolq \
  --input /tmp/boolq-source.jsonl \
  --source-revision YOUR_EXACT_SOURCE_SHA \
  --output /tmp/research-lab/datasets/boolq
~~~

This writes 50 source groups and a checksum manifest, but marks the data
unreviewed by default. The independent --human-reviewed and
--license-reviewed flags must **only** be used when those reviews have
actually been performed. Unreviewed data is not returned by the public API.

Use environment variable OPENSAE_LAB_DATA to point at a persistent output
directory. Otherwise it defaults to repo/runs/research-lab.

## Import genuine feature observations

Once a real checkpoint job has emitted JSONL with verified identities, run:

~~~bash
python -m backend.ingest features --input /path/to/real-observations.jsonl
~~~

Required observation fields:

- site_id: model:sae:Llayer identity from the existing 48-site matrix
- model_sha256 and sae_sha256: loaded full state SHA-256 values
- source_type: real_checkpoint
- feature_id and token_position: nonnegative integer indices
- activation: finite measured scalar
- prompt_group_id and sample_id: fixed source sample identifiers
- token_text and text_excerpt: optional, never automatically made public
- public_text_reviewed: explicit approval gate on displaying raw source text

Unverified checkpoint identities and synthetic records are rejected, not
quietly inserted. Unobserved features do not receive fake example text.

## Analyze recorded arrays on CPU

Install NumPy separately; use the exact original research source from a
trusted local checkout. The code will not fetch or execute arbitrary user
URLs. Original source SHA-256 is mandatory for SAR/RSS CLI actions.

~~~bash
python -m pip install -r requirements-analysis.txt
python -m backend.methods sar --input /path/to/recorded-coalitions.json \
  --source /trusted/SAR/sar.py --expected-sha SOURCE_FILE_SHA256 \
  --output /tmp/sar-result.json
python -m backend.methods rss --input /path/to/recorded-variants.json \
  --source /trusted/RSS/rss.py --expected-sha SOURCE_FILE_SHA256 \
  --output /tmp/rss-result.json
~~~

Input data must bind to an existing real verified site, original loaded
weight hashes, frozen prompt group and data release. SAR requires K16,
384 real coalition values and 256/128 fit/held-out indices. RSS requires 5
selection variants, 10 distinct held-out variants and aligned feature codes.
Localize uses the original installed sae_localize implementation and a
complete, ordered six-factor Boolean input grid. No new scientific estimator
is secretly substituted by the UI.

Vetted outputs can be listed from an immutable analysis index:

~~~bash
python -m backend.methods publish-index --input /path/to/analysis-results.jsonl \
  --output /persistent/research-lab/analyses
~~~

## Publish a shareable report

Only explicitly consented and reviewed JSON summaries can be published.
Publication never makes a new model inference request.

~~~bash
python -m backend.publish --input /path/to/approved-report.json --public-consent
~~~

The publisher assigns a content-based report ID, saves a SHA-256 manifest,
and refuses to overwrite an existing report. Integrity is checked again
whenever a report is read.

## What preflight means

POST /api/jobs/preflight validates one exact site and selected intervention
operators. It never stores the input, loads a model, or commits a run.
POST /api/jobs returns 503 until an authenticated durable GPU worker,
access control and compute budgets have passed real-device tests.

Existing SAEBase batch interventions modify one selected token position
per measurement; short-text steering uses a reference full-prefix forward.
Do not claim arbitrary token editing or GPU streaming throughput.

## References and next stage

- Full scope: ../../work_plan/04_research_platform_execution_plan.md
- Exact GPU handoff: ../../docs/research-lab/cpu-gpu-handoff.md
- Source methods: OpenSAE SAR, RSS and SAE-Input-Localization repositories
- This is a local scaffold, NOT a claim that the existing OpenSAE Hugging
  Face Space has been modified or that any live GPU is attached.


## Scalable CPU feature store and task plan

For large real activation exports, prefer the paginated SQLite-based index
over the legacy compact JSON index:

    python -m backend.feature_store --input /path/to/real-observations.jsonl

The SQLite publisher checks every site/checkpoint identity, deduplicates
prompt-group counts, retains at most eight reviewed examples per feature
and commits an immutable checksum manifest. It is a **browsing index**, not
the lossless scientific SAE codes. Report/API queries are paginated.

Before any GPU environment exists, generate the full task inventory:

    python -m backend.plan
    python -m backend.plan --output /tmp/opensae-compute-plan.json

With no collected new source releases it enumerates exactly 632 plans:
480 capture tasks across 48 sites and 10 datasets, 60 intervention bundles,
30 SAR bundles, 60 RSS bundles and two controlled-grid Localize plans.
They remain marked **blocked/not run**, with explicit missing input and
checkpoint-hash reasons. When reviewed source releases arrive, eligible
capture jobs become available for operator preflight without inventing data.

The task-level scoring library in backend/readouts.py validates complete
teacher-forced candidate log-probability sequences, not single-token proxies.
It also checks unchanged input anchors and tokenizer offset mappings.

To propose transformation texts for one *already frozen* question, use:

    python -m backend.transforms --input /path/to/frozen-prompt.json \
      --anchor 'Question: original question text' \
      --output /tmp/proposed-variants.json

This generates 3 settings × 5 operators × 3 realizations = 45 proposed
input variants, with exactly 5 selection and 10 held-out per setting.
Proposals are **not automatically certified meaning-preserving**.
Code/continuation datasets are intentionally rejected by the generic
renderer until a separate, valid study protocol exists.

## CPU Docker deployment

From the SAEBase root:

    docker build -f apps/research-lab/Dockerfile -t opensae-lab:cpu .
    docker run --rm -p 7860:7860 opensae-lab:cpu

The image ships the app and already-published engineering reports, but no
SAE weights, GPU dependencies, private input data or online worker.
See docs/research-lab/cpu-deployment.md for HF Docker Space preparation,
pre-existing Space backup, rollback and acceptance checks.


## Durable job lifecycle (CPU state only)

The internal backend/jobs.py module creates a SQLite queue holding **only a
SHA-256 reference to a separately protected private prompt payload**, never
the raw input itself. A trusted service supplies an already authenticated
owner identity; the queue does not implement web login. It supports:

- Idempotency keys and per-user/global queue limits.
- Atomic one-worker job leases and heartbeats.
- Bounded attempts after an expired worker lease.
- Owner-only status inspection and cancellation.
- Sanitized failure classes and verified result-hash references.

This is **not** an active public GPU job service. The unauthenticated
POST /api/jobs endpoint still rejects submissions with HTTP 503. Before
activating, an operator must add authentication, private encrypted object
storage, per-user budgets, GPU result validation and worker cancellation
integration. A SQLite queue alone is not a secure hosted service.

The Docker build and CI smoke checks can run without any of these GPU
dependencies. The deployable CPU site is read-only and displays the original
48-site engineering protocol alongside honest empty states for missing data.

## Controlled six-factor inputs (CPU-only)

Run python -m backend.controlled_designs --output-dir /tmp/opensae-six-factor-v1 to generate two complete 64-state Boolean input grids. Factor zero is the least significant bit and all states belong to one group per design. One grid includes separate **synthetic known-factor values**, never labeled as real model activations.

Documentation: docs/research-lab/controlled-six-factor.md. The original SAEBase ControlledInputDesign remains the model-capture interface.

## Work Plan 05 candidate source lock

The platform now has 500 source-version-pinned candidates and hashes for
22,500 unreviewed RSS proposals. It does not publicly redistribute the
underlying third-party prompt text and does not claim GPU measurements.
Dataset cards intentionally remain empty until source-rights and human
content reviews are completed.

Run backend.source_audit --source-lock and backend.corpus_lock to restore
and verify the frozen input population. See docs/research-lab/corpus-cpu-source-lock.md
and Work Plan 05.

The private queue's worker heartbeat and finish require the attempt number
from claim to fence stale retries; the public inference route stays off.

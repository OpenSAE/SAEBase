# OpenSAE Research Lab

## Purpose

Description of how Research Lab views stored observations, sources and reports, and how local services operate.

## Scope

For clearly selected Lab browsing, deploying or maintaining tasks; for the development of the scientific core, use the component entry.

**A research interface for sparse autoencoder features, interventions and reproducible evidence.**

The Lab connects the SAEBase model/SAE registry to experiment browsing, feature exploration, controlled edits, SAR/RSS/Localization analysis and shareable reports. It runs locally without CUDA, and keeps new GPU measurements separate from the older verified engineering records.

## Explore locally

From the repository root:

```bash
docker build -f assets/research_lab/application/files/Dockerfile -t opensae-lab:cpu .
docker run --rm -p 7860:7860 opensae-lab:cpu
```

Open **http://127.0.0.1:7860**. The CPU image serves real published *historical* 48-site Stage D evidence. The newer 500-prompt Research Lab experiment is **not bundled** into the public Docker image, because its full input and observation archives remain private and subject to review.

For Python development:

```bash
python -m pip install -r requirements.txt
uvicorn saebase.research_lab.backend.api:app --host 127.0.0.1 --port 8000
```

The API documentation is at `/api/docs`.

## Research workflows

| Workspace | What you can do |
| --- | --- |
| **Results** | Filter real, provenance-linked model/SAE measurements by model, layer and metric |
| **Features** | Search recorded feature IDs, activity frequencies and reviewer-approved token examples |
| **Intervene** | Specify a site, feature edit, strength and prompt; validate the intervention before any GPU execution |
| **Analyze** | Browse SAR interaction, RSS redistribution and Input Localization results loaded from verified analysis indexes |
| **Datasets** | Inspect ten source definitions, frozen splits and rights-reviewed releases |
| **Reports** | Share immutable, checksummed research summaries without silently disclosing private input texts |

**New GPU results:** The 2026-10-20 recorded experiment covers 48 × 500 site/prompt captures, six-site logit interventions, a 300-card deep-site feature selection, 1,500 decoder-mediated SAR group evaluations, RSS measurements on six sites and one complete six-factor Localize grid. See the historical [Results and interpretation](../../doc/code/documentation/source/development/research-lab/results-20261020.md) for the measured cohorts and scientific qualifications.

## Work with frozen data

The ten sources are fixed to exact Hugging Face upstream revisions in `data/source_lock.json`. CPU source audit checks 50 groups per source, deterministic 20/30 discovery/evaluation splits, source checksums, and the identities of 22,500 *proposed* RSS variations.

```bash
python -m pip install datasets huggingface_hub
python -m saebase.research_lab.backend.source_audit \
  --source-lock data/source_lock.json \
  --max-raw-rows 2500 \
  --output /secure-workspace/corpus-audit.json
python -m saebase.research_lab.backend.corpus_lock \
  --audit /secure-workspace/corpus-audit.json
```

Third-party full prompt texts are not redistributed through this repository. Human content/rights review is required before making source texts public or enabling the public prompt API.

## Import verified scientific outputs

The full research record and its browse index serve their respective readers. When maintaining ingestion, queries or reports, choose the user's actual reading action first: locate the source run, understand the metric and cohort, and open the corresponding artifact. Preserve the originating task's completion information and observed reasons. Explain missing or incomplete content with the affected object and a concrete next action, and confirm the behavior through the actual API and page used by that reader.

- **Features:** `python -m saebase.research_lab.backend.feature_store --input /path/to/real-observations.jsonl` creates an immutable paginated SQLite feature index. Only recognized model/SAE weight identities can be inserted into the historical index; newly verified checkpoints require separate source identities.
- **SAR/RSS:** `python -m saebase.research_lab.backend.methods sar|rss` calls pinned local copies of the **original** estimators on actual measurement arrays. The original affine-decoder analysis and the explicitly named decoder-mediated intervention protocol are scientifically distinct.
- **Localization:** `python -m saebase.research_lab.backend.controlled_designs --output-dir /secure-workspace/six-factor` generates two full 64-state designs, including a **synthetic** truth control. The real model run uses a separately measured grid.
- **Reports:** `python -m saebase.research_lab.backend.publish --input /path/to/reviewed-public-report.json --public-consent` publishes an operator-approved, checksum-verified summary with an immutable ID.

These tools do not create model measurements by themselves.

## GPU work and reproducibility

GPU execution is an **explicit**, separately authenticated workflow. The public `POST /api/jobs` endpoint stays disabled, returning HTTP 503. The private queue supports idempotency, quotas, lease recovery and attempt fencing, but a production GPU service additionally requires credential management, private durable storage and operator acceptance tests.

Recorded GPU work used one release under `experiments/research-lab/` (ignored by Git), with task runner source under `tasks/05_research_lab_cpu_closure_and_gpu_handoff/`. The raw experiment archive is not a GitHub Release asset. Downloading this software package alone does not reproduce the reported GPU numbers; the separate original measurement artifact must first be made available to an authorized auditor.

**Documentation:** [Research Lab results](../../doc/code/documentation/source/development/research-lab/results-20261020.md) · [Source corpus protocol](../../doc/code/documentation/source/development/research-lab/corpus-cpu-source-lock.md) · [Deployment](../../doc/code/documentation/source/development/research-lab/cpu-deployment.md) · [Release acceptance](../../doc/code/documentation/source/development/research-lab/release-acceptance.md).


## Local data operations

From the root directory call to the backend to see input and output parameters using `--help`:

- `python -m saebase.research_lab.backend.auto_interp_local`: An explanation candidate is generated using local models and documented feature samples.
- `python -m saebase.research_lab.backend.private_worker`: Consumption of a clearly activated private measurement request.
- `python -m saebase.research_lab.backend.publish bundle`: Prepare source records, generate review lists and prepare public packages as mandated.
- `python -m saebase.research_lab.backend.published_records` : Organize well-sourced circuit maps into browsing cards, circuital resolution and tailoring of re-use scientific cores.

The feature index and source summary is saved and read at `backend/feature_store.py`; package preparation and publication is organized by `backend/publish.py`.

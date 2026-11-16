# OpenSAE Research Lab

## 目的和主要功能

说明 Research Lab 如何浏览已保存观测、来源与报告，以及本地服务的运行方式。

## 适用范围

用于明确选择的 Lab 浏览、部署或维护任务；科研核心的开发使用所属组件入口。

**A research interface for sparse autoencoder features, interventions and reproducible evidence.**

The Lab connects the SAEBase model/SAE registry to experiment browsing, feature exploration, controlled edits, SAR/RSS/Localization analysis and shareable reports. It runs locally without CUDA, and keeps new GPU measurements separate from the older verified engineering records.

## Explore locally

From the repository root:

```bash
docker build -f saebase/research_lab/Dockerfile -t opensae-lab:cpu .
docker run --rm -p 7860:7860 opensae-lab:cpu
```

Open **http://127.0.0.1:7860**. The CPU image serves real published *historical* 48-site Stage D evidence. The newer 500-prompt Research Lab experiment is **not bundled** into the public Docker image, because its full input and observation archives remain private and subject to review.

For Python development:

```bash
cd saebase/research_lab
python -m pip install -r requirements.txt
uvicorn backend.api:app --host 127.0.0.1 --port 8000
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

**New GPU results:** The 2026-10-20 recorded experiment covers 48 × 500 site/prompt captures, six-site logit interventions, a 300-card deep-site feature selection, 1,500 decoder-mediated SAR group evaluations, RSS measurements on six sites and one complete six-factor Localize grid. See the historical [Results and interpretation](../../doc/code/development/research-lab/results-20261020.md) for the measured cohorts and scientific qualifications.

## Work with frozen data

The ten sources are fixed to exact Hugging Face upstream revisions in `data/source_lock.json`. CPU source audit checks 50 groups per source, deterministic 20/30 discovery/evaluation splits, source checksums, and the identities of 22,500 *proposed* RSS variations.

```bash
cd saebase/research_lab
python -m pip install datasets huggingface_hub
python -m backend.source_audit \
  --source-lock data/source_lock.json \
  --max-raw-rows 2500 \
  --output /secure-workspace/corpus-audit.json
python -m backend.corpus_lock \
  --audit /secure-workspace/corpus-audit.json
```

Third-party full prompt texts are not redistributed through this repository. Human content/rights review is required before making source texts public or enabling the public prompt API.

## Import verified scientific outputs

The full research record and its browse index serve their respective readers. When maintaining ingestion, queries or reports, choose the user's actual reading action first: locate the source run, understand the metric and cohort, and open the corresponding artifact. Preserve the originating task's completion information and observed reasons. Explain missing or incomplete content with the affected object and a concrete next action, and confirm the behavior through the actual API and page used by that reader.

- **Features:** `python -m backend.feature_store --input /path/to/real-observations.jsonl` creates an immutable paginated SQLite feature index. Only recognized model/SAE weight identities can be inserted into the historical index; newly verified checkpoints require separate source identities.
- **SAR/RSS:** `python -m backend.methods sar|rss` calls pinned local copies of the **original** estimators on actual measurement arrays. The original affine-decoder analysis and the explicitly named decoder-mediated intervention protocol are scientifically distinct.
- **Localization:** `python -m backend.controlled_designs --output-dir /secure-workspace/six-factor` generates two full 64-state designs, including a **synthetic** truth control. The real model run uses a separately measured grid.
- **Reports:** `python -m backend.publish --input /path/to/reviewed-public-report.json --public-consent` publishes an operator-approved, checksum-verified summary with an immutable ID.

These tools do not create model measurements by themselves.

## GPU work and reproducibility

GPU execution is an **explicit**, separately authenticated workflow. The public `POST /api/jobs` endpoint stays disabled, returning HTTP 503. The private queue supports idempotency, quotas, lease recovery and attempt fencing, but a production GPU service additionally requires credential management, private durable storage and operator acceptance tests.

Recorded GPU work used one release under `experiments/research-lab/` (ignored by Git), with task runner source under `tasks/05_research_lab_cpu_closure_and_gpu_handoff/`. The raw experiment archive is not a GitHub Release asset. Downloading this software package alone does not reproduce the reported GPU numbers; the separate original measurement artifact must first be made available to an authorized auditor.

**Documentation:** [Research Lab results](../../doc/code/development/research-lab/results-20261020.md) · [Source corpus protocol](../../doc/code/development/research-lab/corpus-cpu-source-lock.md) · [Deployment](../../doc/code/development/research-lab/cpu-deployment.md) · [Release acceptance](../../doc/code/development/research-lab/release-acceptance.md).


## 本地数据操作

从项目根目录调用所属后端，使用 `--help` 查看输入和输出参数：

- `python -m saebase.research_lab.backend.auto_interp_local`：使用本地模型与已记录的特征样本生成解释候选。
- `python -m saebase.research_lab.backend.private_worker`：消费一次明确启用的私有测量请求。
- `python -m saebase.research_lab.backend.publish bundle`：整理来源记录、生成审阅清单，并按已有授权准备公开包。
- `python -m saebase.research_lab.backend.published_records`：将来源明确的电路图整理为浏览卡片，电路解析和裁剪复用科研核心。

特征索引与来源摘要由 `backend/feature_store.py` 保存和读取；包准备与发布由 `backend/publish.py` 组织。

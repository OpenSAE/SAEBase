"""Read-only research explorer, explicit GPU handoff, no fabricated results.

Start from apps/research-lab:
  uvicorn backend.api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import core

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
app = FastAPI(title="OpenSAE Research Lab", version="0.1.0", docs_url="/api/docs", redoc_url=None)
app.mount("/assets", StaticFiles(directory=str(FRONTEND)), name="assets")


@lru_cache(maxsize=1)
def site_cache() -> list[dict[str, Any]]:
    return core.load_sites()


def _page(rows: list[Any], limit: int, offset: int) -> dict[str, Any]:
    return {"total":len(rows), "limit":limit, "offset":offset, "items":rows[offset:offset+limit]}


def _bad_evidence(e: core.InvalidEvidence):
    raise HTTPException(status_code=503, detail="Source evidence failed integrity verification: " + str(e)) from e


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"service":"OpenSAE Research Lab","status":"online","compute":"disabled_cpu_first"}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    try:
        sites = site_cache()
        features = core.feature_index()
        analyses = core.analyses()
        dataset_counts=core.imported_counts()
    except core.InvalidEvidence as e:
        _bad_evidence(e)
    return {
        "sites_verified":len(sites),
        "historical_metrics":sum(len(s["metrics"]) for s in sites),
        "historical_protocol":"stage_d.population.v1",
        "historical_groups_per_site":64,
        "historical_scope":"engineering acceptance on controlled templates, NOT the new curated 500 prompts",
        "candidate_datasets":len(core.dataset_catalog()["sources"]),
        "curated_prompts":sum(dataset_counts.values()),
        "curated_datasets_ready":len(dataset_counts),
        "feature_cards":len(features),
        "analysis_records":len(analyses),
        "reports_published":len(core.published_reports()),
        "cpu_online":True,
        "gpu_online":False,
        "online_new_inference":False,
    }


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    info=core.dataset_catalog()
    counts=core.imported_counts()
    return {"schema":info["schema"],"sources":[{
        **item, "imported_count":counts.get(item["id"],0),
        "status":"released" if item["id"] in counts else "awaiting_source_and_review",
    } for item in info["sources"]]}


@app.get("/api/prompts")
def prompts(dataset_id: str, limit: int = Query(20,ge=1,le=50), offset: int = Query(0,ge=0)) -> dict[str, Any]:
    if dataset_id not in {d["id"] for d in core.dataset_catalog()["sources"]}:
        raise HTTPException(404,"Unknown source dataset")
    if dataset_id not in core.imported_counts():
        return _page([],limit,offset)
    path=core.data_root() / "datasets" / dataset_id / "records.jsonl"
    # Only license and human-reviewed releases reach this branch.
    with path.open("r",encoding="utf-8") as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    return _page(rows,limit,offset)


@app.get("/api/sites")
def sites(
    model: str = "", sae: str = "", limit: int = Query(80,ge=1,le=100),
    offset: int = Query(0,ge=0)
) -> dict[str, Any]:
    try:
        filtered=[{
            k:v for k,v in site.items() if k!="metrics"
        } for site in site_cache() if
            (not model or model.lower() in site["model"].lower()) and
            (not sae or sae.lower() in site["sae"].lower())]
    except core.InvalidEvidence as e:
        _bad_evidence(e)
    return _page(filtered,limit,offset)


@app.get("/api/results")
def results(
    model: str = "", metric: str = "", site_id: str = "",
    limit: int = Query(80,ge=1,le=200), offset: int = Query(0,ge=0)
) -> dict[str, Any]:
    try:
        rows=core.historical_metrics(site_cache())
    except core.InvalidEvidence as e:
        _bad_evidence(e)
    rows=[r for r in rows if
          (not model or model.lower() in r["model"].lower()) and
          (not metric or metric.lower() in r["metric"].lower()) and
          (not site_id or site_id==r["site_id"])]
    return _page(rows,limit,offset)


@app.get("/api/features")
def features(
    q: str = "", model: str = "", site_id: str = "",
    limit: int = Query(36,ge=1,le=100),offset: int = Query(0,ge=0),
) -> dict[str, Any]:
    try:
        rows=core.feature_index()
    except core.InvalidEvidence as e:
        _bad_evidence(e)
    lookup=q.casefold()
    rows=[r for r in rows if
          (not model or model.casefold() in r.get("model","").casefold()) and
          (not site_id or r.get("site_id")==site_id) and
          (not lookup or lookup in (str(r.get("feature_id",""))+" "+str(r.get("label",""))+" "+r.get("site_id","")).casefold())]
    return _page(rows,limit,offset)


@app.get("/api/analyses")
def analyses(
    kind: str = "", limit: int = Query(50,ge=1,le=100),offset: int = Query(0,ge=0)
) -> dict[str, Any]:
    try:
        rows=core.analyses()
    except core.InvalidEvidence as e:
        _bad_evidence(e)
    return _page([r for r in rows if not kind or r.get("kind")==kind],limit,offset)


@app.get("/api/reports")
def reports(limit: int = Query(40,ge=1,le=100),offset: int = Query(0,ge=0)) -> dict[str,Any]:
    return _page(core.published_reports(),limit,offset)


@app.get("/api/reports/{report_id}")
def get_report(report_id: str) -> dict[str, Any]:
    try:
        result=core.public_report(report_id)
    except (core.InvalidEvidence,OSError,ValueError) as e:
        _bad_evidence(core.InvalidEvidence(str(e)))
    if result is None:
        raise HTTPException(404,"No approved public report with this identity")
    return result


class InterventionSpec(BaseModel):
    site_id: str = Field(min_length=5,max_length=280)
    text: str = Field(min_length=1,max_length=16000)
    operation: str = "ablate"
    feature_ids: list[int] = Field(default_factory=list,max_length=8)
    strength: float = Field(default=1.0,ge=-10,le=10)
    max_new_tokens: int = Field(default=32,ge=0,le=64)


@app.post("/api/jobs/preflight")
def preflight(request: InterventionSpec) -> dict[str, Any]:
    """Validate only. Never enqueue, store input, or claim GPU execution."""
    permitted={"identity","ablate","scale","clamp","add","coalition","reconstruction"}
    if request.operation not in permitted:
        raise HTTPException(422,"Unknown intervention operation")
    if request.operation not in ("identity","reconstruction") and not request.feature_ids:
        raise HTTPException(422,"At least one feature ID is required")
    if request.operation in ("identity","reconstruction") and request.feature_ids:
        raise HTTPException(422,"This condition does not select individual features")
    if any(v<0 for v in request.feature_ids) or len(set(request.feature_ids))!=len(request.feature_ids):
        raise HTTPException(422,"Feature IDs must be distinct nonnegative integers")
    if request.max_new_tokens==0 and request.operation=="identity":
        raise HTTPException(422,"No intervention or generation requested")
    try:
        site=next((s for s in site_cache() if s["site_id"]==request.site_id),None)
    except core.InvalidEvidence as e:
        _bad_evidence(e)
    if site is None:
        raise HTTPException(422,"Site not in the verified matrix")
    return {
        "valid":True,"submitted":False,"ran_model":False,
        "site_id":site["site_id"],"hook":site["hook"],
        "model_sha256":site["model_sha256"],"sae_sha256":site["sae_sha256"],
        "operation":request.operation,"selected_features":len(request.feature_ids),
        "max_new_tokens":request.max_new_tokens,
        "status":"blocked_until_gpu_worker_and_auth_are_configured",
        "note":"Preflight checks syntax/registered site only; it does not validate model behavior or run inference.",
    }


@app.post("/api/jobs")
def submit_jobs() -> None:
    """Deliberately disabled until authenticated durable GPU service passes QA."""
    raise HTTPException(503,"New inference disabled: no authenticated GPU worker is installed.")

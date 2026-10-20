"""Write Work Plan 05 section 6 from artifacts already on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import REPO, git_commit, write_json


def _load(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(value: Any) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    )
    return text.replace("|", "/").replace("\n", " ")


def collect(root: Path) -> list[dict[str, str]]:
    commit = git_commit()
    cpu = _load(root / "GPU-00" / "cpu_lock_summary.json") or {}
    pilot = _load(root / "GPU-01" / "pilot_20" / "manifest.json") or {}
    cost = _load(root / "GPU-01" / "cost_and_capacity.json") or {}
    gpu02 = _load(root / "GPU-02" / "decision.json") or {}
    gpu03 = _load(root / "GPU-03" / "feature_cards.json") or {}
    deep_cards = _load(root / "GPU-03" / "deep_site_cards.json") or {}
    gpu04 = _load(root / "GPU-04" / "decision.json") or {}
    generation = _load(root / "GPU-04" / "generation_v2.json") or {}
    gpu05 = (
        _load(root / "GPU-05" / "decision_v2.json")
        or _load(root / "GPU-05" / "decision.json")
        or {}
    )
    gpu06 = (
        _load(root / "GPU-06" / "decision_v2.json")
        or _load(root / "GPU-06" / "decision.json")
        or {}
    )
    coverage = _load(root / "GPU-06" / "coverage_v2.json") or {}
    gpu07 = _load(root / "GPU-07" / "localize_status.json") or {}
    gpu08 = _load(root / "GPU-08" / "metric_ledger.json") or {}
    gpu09 = _load(root / "GPU-09" / "live_service_acceptance.json") or {}
    lock = cpu.get("lock_sha256", "unavailable")
    rows = [
        {
            "task_id": "GPU-00",
            "status": "passed" if cpu.get("passed") else "failed",
            "commit": commit,
            "model_sae_site": "48-site manifest audited; weights hashed only after load",
            "input_release": lock,
            "population": f"sources={cpu.get('sources')} groups={cpu.get('groups')} discovery={cpu.get('discovery')} evaluation={cpu.get('evaluation')} variants={cpu.get('variant_proposals')} gpu_measurements={cpu.get('gpu_measurements')}",
            "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run cpu",
            "resources": "CPU audit; GPU not required",
            "outputs": str(root / "GPU-00"),
            "scientific_checks": "source lock fingerprint, 10x50 split, 22500 RSS proposal hashes, review flags false",
            "decisions": "continue only if the frozen lock matches",
        },
        {
            "task_id": "GPU-01",
            "status": "passed" if pilot.get("go") else ("partial" if pilot else "not_started"),
            "commit": commit,
            "model_sae_site": "pythia410m:pythia410m_mlp_topk:L12 then qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14",
            "input_release": lock,
            "population": "20 base groups = 2 per source; variants are not extra samples",
            "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run gpu01",
            "resources": _cell(cost),
            "outputs": str(root / "GPU-01"),
            "scientific_checks": "identity patch, identity operator, ablate, scale 0.5, full reconstruction, one token position",
            "decisions": "go" if pilot.get("go") else "no-go for full SAR and full capture",
        },
        {
            "task_id": "GPU-02",
            "status": gpu02.get("status", "not_started"),
            "commit": commit,
            "model_sae_site": _cell(gpu02.get("reachable_open_sites", "not selected")),
            "input_release": lock,
            "population": _cell(
                {
                    "sites_passed": gpu02.get("sites_passed"),
                    "denominator_sites": gpu02.get("denominator_sites", 48),
                }
            ),
            "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run gpu02",
            "resources": _cell(gpu02.get("timed_model_seconds")),
            "outputs": str(root / "GPU-02"),
            "scientific_checks": "new captures are not Stage D 64-group results; failures stay in the denominator",
            "decisions": _cell(gpu02.get("reason", gpu02.get("status"))),
        },
        {
            "task_id": "GPU-03",
            "status": gpu03.get("status", "not_started"),
            "commit": commit,
            "model_sae_site": _cell(gpu03.get("sites_with_real_codes")),
            "input_release": lock,
            "population": f"historical_cards={gpu03.get('card_count')} deep_site_cards={deep_cards.get('card_count')} deep_target=300",
            "commands": "included in run all",
            "resources": "CPU index over saved codes",
            "outputs": str(root / "GPU-03" / "feature_cards.json"),
            "scientific_checks": "no invented explanations or dead-feature labels; text hidden until review",
            "decisions": _cell(
                f"historical cards={gpu03.get('card_count')} across captured sites; deep-site cards={deep_cards.get('card_count')} with public text unavailable for public release; unobserved features are not called dead"
            ),
        },
        {
            "task_id": "GPU-04",
            "status": gpu04.get("status", "not_started"),
            "commit": commit,
            "model_sae_site": _cell(gpu04.get("model_sae_site", "six deep sites")),
            "input_release": lock,
            "population": _cell(gpu04.get("population", "not started")),
            "commands": gpu04.get("commands", "not started as a full package"),
            "resources": _cell(gpu04.get("resources", "n/a")),
            "outputs": str(root / "GPU-04"),
            "scientific_checks": _cell(
                {
                    "pairs": gpu04.get("scientific_checks"),
                    "generation": generation.get("status", "not_started"),
                    "kv_cache": generation.get("kv_cache"),
                }
            ),
            "decisions": _cell(gpu04.get("reason")),
        },
        {
            "task_id": "GPU-05",
            "status": gpu05.get("status", "not_started"),
            "commit": commit,
            "model_sae_site": _cell(
                gpu05.get("model_sae_site", "three SAR sites from coverage.json")
            ),
            "input_release": lock,
            "population": _cell(gpu05.get("population", "planned 633600 coalition measurements")),
            "commands": gpu05.get("commands", "not started"),
            "resources": _cell(gpu05.get("resources", "n/a")),
            "outputs": str(root / "GPU-05"),
            "scientific_checks": gpu05.get(
                "scientific_checks", "K=16 M=384 fit=256 held-out=128 unchanged"
            ),
            "decisions": _cell(gpu05.get("reason")),
        },
        {
            "task_id": "GPU-06",
            "status": gpu06.get("status", "not_started"),
            "commit": commit,
            "model_sae_site": _cell(gpu06.get("model_sae_site", "six deep sites")),
            "input_release": lock,
            "population": _cell(
                coverage or gpu06.get("population") or gpu06.get("coverage") or "22500 proposals"
            ),
            "commands": gpu06.get("commands", "not started"),
            "resources": _cell(gpu06.get("resources", "n/a")),
            "outputs": str(root / "GPU-06"),
            "scientific_checks": gpu06.get(
                "scientific_checks", "selection and held-out roles were not swapped"
            ),
            "decisions": _cell(gpu06.get("reason")),
        },
        {
            "task_id": "GPU-07",
            "status": "partial"
            if gpu07.get("real_model_scope_on_observed_support", {}).get("status") != "blocked"
            else "partial",
            "commit": commit,
            "model_sae_site": "synthetic control plus pythia410m L12 when the pilot passed",
            "input_release": "controlled six-factor grids, not the 500 natural prompts",
            "population": "64 states, one population",
            "commands": "included in run all",
            "resources": "64 short authored prompts on the small site",
            "outputs": str(root / "GPU-07"),
            "scientific_checks": _cell(
                gpu07.get("real_model_scope_on_observed_support")
                or gpu07.get("real_model_curve_support")
            ),
            "decisions": "synthetic curve remains synthetic_logic_not_real_sae; the 55-column curve is real_model_scope_on_observed_support",
        },
        {
            "task_id": "GPU-08",
            "status": gpu08.get("status", "not_started"),
            "commit": commit,
            "model_sae_site": "only rows whose loaded hash matches the historical hash",
            "input_release": lock,
            "population": _cell(gpu08),
            "commands": "included in run all",
            "resources": "CPU",
            "outputs": str(root / "GPU-08"),
            "scientific_checks": "non-finite activations rejected; unmatched hashes not inserted",
            "decisions": "not a complete research-lab-release-v1",
        },
        {
            "task_id": "GPU-09",
            "status": gpu09.get("status", "blocked"),
            "commit": commit,
            "model_sae_site": "no public GPU worker",
            "input_release": lock,
            "population": "queue negative checks only",
            "commands": "included in run all",
            "resources": "CPU",
            "outputs": str(root / "GPU-09" / "live_service_acceptance.json"),
            "scientific_checks": _cell(
                {
                    "cpu_queue_checks": gpu09.get("cpu_queue_checks"),
                    "negative_checks": gpu09.get("negative_checks"),
                }
            ),
            "decisions": "HF Space, three external users, and public online GPU remain blocked",
        },
    ]
    write_json(root / "section6_report.json", rows)
    return rows


def write_work_plan(rows: list[dict[str, str]]) -> None:
    path = REPO / "work_plan" / "05_research_lab_cpu_closure_and_gpu_handoff.md"
    text = path.read_text(encoding="utf-8")
    marker = "## 7. CPU 阶段剩余事项"
    header = (
        "### 本次执行填报\n\n"
        "下表只记录这次运行已经写出的产物。没有证据的工作包保持 not_started、partial、failed 或 blocked。\n\n"
        "| task_id | status | commit | model_sae_site | input_release | population | commands | resources | outputs | scientific_checks | decisions |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "\n".join(
        "| "
        + " | ".join(
            _cell(row[key])
            for key in (
                "task_id",
                "status",
                "commit",
                "model_sae_site",
                "input_release",
                "population",
                "commands",
                "resources",
                "outputs",
                "scientific_checks",
                "decisions",
            )
        )
        + " |"
        for row in rows
    )
    block = header + body + "\n\n"
    if "### 本次执行填报" in text:
        start = text.index("### 本次执行填报")
        end = text.index(marker)
        text = text[:start] + block + text[end:]
    else:
        text = text.replace(marker, block + marker, 1)
    path.write_text(text, encoding="utf-8")

"""Provenance checks for the recorded Effects, Context, and RSS bundles."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "apps" / "research-lab" / "data" / "interactive_v1"
LAB = ROOT / "apps" / "research-lab" / "lab-experiments"
RUN = ROOT / "experiments" / "research-lab" / "7fb2be8f29a1-20261019T035727Z"
EXPORTER = ROOT / "scripts" / "research_lab" / "export_real_cases.py"
DEEP = [
    "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14",
    "gemma1b:gemma1b_scope_resid:L13",
    "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22",
    "gemma3_4b_it:gemma3_4b_it_scope_resid:L22",
    "llama31_8b:llama31_8b_lxr_8x:L21",
    "mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16",
]


def load_export():
    spec = importlib.util.spec_from_file_location("export_real_cases", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_mean(activations: list[float], bit: int) -> float:
    diffs = [activations[mask] - activations[mask ^ (1 << bit)] for mask in range(64) if mask & (1 << bit)]
    assert len(diffs) == 32
    return sum(diffs) / 32


def walk_strings(value, found: list[str]) -> None:
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            walk_strings(item, found)
    elif isinstance(value, list):
        for item in value:
            walk_strings(item, found)


def test_bundle_contracts():
    effects = json.loads((DATA / "effects.json").read_text(encoding="utf-8"))
    context = json.loads((DATA / "context64.json").read_text(encoding="utf-8"))
    rss = json.loads((DATA / "rss.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert [case["site_id"] for case in effects["cases"]] == DEEP
    for case in effects["cases"]:
        names = [item["operation"] for item in case["operators"]]
        assert names == ["ablate", "scale", "add", "clamp", "coalition"]
        assert "original_top_tokens" not in case
        assert "edited_top_tokens" not in case
        assert case["measured_fields"] == ["logit_l2"]
        assert case["text_stored"] is False
        assert all(math.isfinite(item["logit_l2"]) and item["logit_l2"] >= 0 for item in case["operators"])
    assert len(context["prompts"]) == 64
    assert [item["mask"] for item in context["prompts"]] == list(range(64))
    assert len(context["features"]) >= 12
    assert len(context["features"]) == len(set(context["feature_ids"]))
    assert context["feature_ids"] != list(range(len(context["feature_ids"])))
    allowed_prompts = {item["prompt"] for item in context["prompts"]}
    for feature in context["features"]:
        assert len(feature["activations"]) == 64
        assert all(math.isfinite(value) for value in feature["activations"])
        for bit, marginal in enumerate(feature["marginals"]):
            assert math.isclose(marginal["paired_mean_difference"], paired_mean(feature["activations"], bit), rel_tol=0, abs_tol=1e-9)
    assert manifest["effects_cases"] == 6
    assert manifest["context_states"] == 64
    assert manifest["rss_aggregates"] == len(rss["cases"])
    for case in rss["cases"]:
        defined = case["variants"]
        assert case["n_defined"] == len(defined) > 0
        assert case["n_undefined"] >= 0
        assert math.isclose(case["mean_rss"], sum(item["mean_rss"] for item in defined) / len(defined), rel_tol=0, abs_tol=1e-9)
        assert math.isclose(case["held_out_rate_mean"], sum(item["held_out_rate_mean"] for item in defined) / len(defined), rel_tol=0, abs_tol=1e-9)
        assert case["protocol_id"] == "rss.position_recomputed.v2"
        assert case["semantic_review"] is False
        assert "token_ids" not in case
    leaked = []
    for path in (DATA / "effects.json", DATA / "rss.json"):
        walk_strings(json.loads(path.read_text(encoding="utf-8")), leaked)
    for text in leaked:
        if text in allowed_prompts:
            raise AssertionError("authored context prompt leaked into effects or RSS")
        if len(text) > 180 and not text.startswith("https://"):
            raise AssertionError("unexpected long public string")
    lab_js = (LAB / "assets" / "lab.js").read_text(encoding="utf-8")
    page = (LAB / "index.html").read_text(encoding="utf-8")
    assert "unavailable" not in lab_js
    assert "What changes when we remove a model feature?" in page
    assert "Remove this feature" in page
    for banned in ("Selected logit L2", "Feature IDs", "Recorded prompts 1", "gpu04.paired_operator_logit_l2.v1", "The prompt text is not in the public record"):
        assert banned not in page
        assert banned not in lab_js
    pack = json.loads((LAB / "data" / "evidence.json").read_text(encoding="utf-8"))
    assert pack["effects"][0]["operators"]
    assert len(pack["contexts"]) == len(context["features"])
    assert len(pack["contexts"][0]["states"]) == 64


def test_authored_probabilities_match_full_softmax():
    import math

    payload = json.loads((DATA / "authored_effects.json").read_text(encoding="utf-8"))
    assert 3 <= len(payload["cases"]) <= 12
    assert payload["run_id"] != "7fb2be8f29a1-20261019T035727Z"
    for case in payload["cases"]:
        assert case["prompt"]
        assert case["public_text_reviewed"] is True
        groups = [(case["original_top_tokens"], case["original_logsumexp"])]
        groups.extend((item["edited_top_tokens"], item["edited_logsumexp"]) for item in case["operators"])
        for rows, logsumexp in groups:
            for row in rows:
                rebuilt = math.exp(row["logit"] - logsumexp)
                assert math.isclose(row["probability"], rebuilt, rel_tol=0, abs_tol=1e-5)
                assert 0 <= row["probability"] <= 1


def test_export_matches_sources_when_archive_present():
    if not RUN.is_dir():
        return
    subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--verify",
            "--run-root",
            str(RUN),
            "--site-registry",
            str(ROOT / "apps" / "research-lab" / "data" / "coverage.json"),
            "--out",
            str(DATA),
        ],
        check=True,
    )
    module = load_export()
    rejected = []
    context, _summary = module.build_context(RUN, rejected)
    assert context["site_id"] == "pythia410m:pythia410m_mlp_topk:L12"
    assert len(context["features"]) == len(json.loads((DATA / "context64.json").read_text(encoding="utf-8"))["features"])

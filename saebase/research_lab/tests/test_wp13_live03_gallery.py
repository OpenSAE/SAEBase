"""Small real LIVE-03 source metrics drive read-only staging cards.

No model or GPU execution here. Historical full-vocab arrays are never needed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "saebase" / "research_lab"))

from backend.api import app  # noqa: E402
from saebase.tools.wp13_stage_live03 import main, translate_summary  # noqa: E402

SOURCE = ROOT / "harness" / "github_work_plan" / "LIVE-03-small-summary.json"
client = TestClient(app)


def test_live03_real_summary_maps_to_two_source_bound_records():
    source = translate_summary(SOURCE)
    assert len(source) == 2
    assert {r["site"] for r in source} == {
        "gemma270m:gemma270m_scope_resid:L5",
        "gemma270m:gemma270m_scope_resid:L9",
    }
    assert {r["feature_id"] for r in source} == {506, 95}
    assert all(r["measurement"] == "teacher_forced_full_vocab_logits_only" for r in source)
    assert all(r["generated_text"] is False for r in source)
    assert all(r["measured"] is True for r in source)
    assert all(r["readouts"]["feature_position"]["noop"]["logit_delta_l2"] == 0 for r in source)
    assert all(
        r["readouts"]["feature_position"]["ablation"]["argmax_changed"] is True for r in source
    )
    assert all(
        r["readouts"]["final_position"]["ablation"]["argmax_changed"] is False for r in source
    )
    assert all(r["readouts"]["final_position"]["ablation"]["logit_delta_l2"] > 0 for r in source)
    assert all("prompt_text" not in r for r in source)


def test_live03_staging_requires_explicit_approval_and_hashes(tmp_path, monkeypatch):
    assert main(["--source", str(SOURCE)]) == 0
    with pytest.raises(SystemExit):
        main(["--source", str(SOURCE), "--publish-reviewed"])
    release = tmp_path / "approved-records"
    assert (
        main(
            [
                "--source",
                str(SOURCE),
                "--publish-reviewed",
                "--release-root",
                str(release),
                "--reviewer",
                "Real test reviewer",
            ]
        )
        == 0
    )
    monkeypatch.setenv("OPENSAE_LAB_DATA", str(release))
    catalog = client.get("/api/wp12/interventions")
    assert catalog.status_code == 200
    assert catalog.json()["total"] == 2
    for item in catalog.json()["items"]:
        detail = client.get("/api/wp12/interventions/" + item["id"])
        assert detail.status_code == 200
        assert detail.json()["measured"] is True
        assert detail.json()["generated_text"] is False
    with pytest.raises(FileExistsError):
        main(
            [
                "--source",
                str(SOURCE),
                "--publish-reviewed",
                "--release-root",
                str(release),
                "--reviewer",
                "Real test reviewer",
            ]
        )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("generated_text", True, "unexecuted"),
        ("logit_kind", "generated_text", "unexecuted"),
        ("status", "not_measured", "unexecuted"),
    ],
)
def test_live03_refuses_mislabeled_computation(tmp_path, field, value, error):
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source[field] = value
    file = tmp_path / "modified.json"
    file.write_text(json.dumps(source))
    with pytest.raises(ValueError, match=error):
        translate_summary(file)


def test_live03_rejects_bad_identity_and_false_noop(tmp_path):
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source["sites"][0]["sae_sha256"] = "f" * 64
    file = tmp_path / "mismatch.json"
    file.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="verified registry"):
        translate_summary(file)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source["sites"][0]["readout_at_feature_position"]["noop"]["logit_delta_l2"] = 0.2
    file.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Identity/no-op"):
        translate_summary(file)

"""Offline annotation contracts with synthetic exports; no service or model calls."""

import gzip
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from urllib.parse import parse_qs, urlsplit

from omegaconf import OmegaConf
import pytest
import torch

from saebase.api import neuronpedia as np
from saebase.results import EvaluationResult, completion_summary
from tasks.capture import operation


@pytest.fixture
def expected():
    return {
        "id": "synthetic/0-res",
        "repo": "synthetic/weights",
        "folder": "layer0",
        "d_in": 2,
        "d_sae": 4,
        "hook": "blocks.0.hook_resid_post",
    }


def source(expected):
    return {
        "modelId": "synthetic",
        "id": "0-res",
        "hfRepoId": expected["repo"],
        "hfFolderId": expected["folder"],
        "saelensConfig": {"d_in": 2, "d_sae": 4, "hook_name": expected["hook"]},
    }


def payload(expected, indices=(1, 2, 3)):
    return {
        "schema": "saebase.neuronpedia_export.v2",
        "neuronpedia_id": expected["id"],
        "source_metadata": source(expected),
        "retrieved_at_utc": "synthetic timestamp",
        "files": [],
        "export_folders": [],
        "explanations": {
            str(i): [
                {
                    "description": f"synthetic explanation {i}",
                    "method": "synthetic",
                    "explainer_model": "synthetic",
                    "id": str(i),
                }
            ]
            for i in indices
        },
    }


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "seed": 42,
            "task": {
                "top_feature_num": 1,
                "prompt": "synthetic prompt",
                "neuronpedia": {"cache_dir": str(tmp_path / "cache"), "allow_network": True},
            },
        }
    )


@pytest.fixture
def capture(monkeypatch):
    # This fixture isolates feature/annotation behavior; generation is tested separately.
    monkeypatch.setattr(
        "tasks.capture.model_answer.answers",
        lambda cfg, pair, prompts: [
            {"status": "completed", "text": "synthetic continuation"} for _ in prompts
        ],
    )
    import saebase.api

    pair = SimpleNamespace(
        model=object(),
        sae=object(),
        site=SimpleNamespace(hook_name="blocks.0.hook_resid_post"),
        identity=SimpleNamespace(model_key="synthetic", sae_key="synthetic", layer=0),
    )
    snapshot = SimpleNamespace(
        codes=torch.tensor([0.0, -8.0, 2.0, 4.0]), reconstruction=torch.tensor([1.0, 2.0])
    )
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    monkeypatch.setattr(operation, "_prompt", lambda cfg: "synthetic prompt")
    monkeypatch.setattr(saebase.api, "capture_features", lambda *args, **kwargs: snapshot)
    return pair


def online_options(cfg):
    options = operation.neuronpedia_settings(cfg)
    options["allow_network"] = True  # Transport is mocked, even under the offline CI environment.
    return options


def test_all_active_ids_and_top_ids_are_annotated(monkeypatch, expected, cfg, capture):
    monkeypatch.setattr(np, "_source", lambda pair: expected)
    monkeypatch.setattr(np, "_download", lambda *args: payload(expected))
    monkeypatch.setattr(operation, "input_identity", lambda *args: None)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    result = operation.execute(cfg)
    assert [r["feature_id"] for r in result["active_features"]] == [1, 3, 2]
    assert len(result["top_features"]) == 1
    assert result["neuronpedia"]["explained"] == 3
    assert all(row["neuronpedia"]["status"] == "available" for row in result["active_features"])
    assert result["completion"]["requested"] == 3
    assert EvaluationResult.from_payload("capture", cfg, result).exit_code == 0


def test_verified_cache_is_reused_offline_and_corruption_is_not(
    monkeypatch, expected, cfg, capture
):
    monkeypatch.setattr(np, "_source", lambda pair: expected)
    monkeypatch.setattr(np, "_download", lambda *args: payload(expected))
    options = online_options(cfg)
    info, rows = np.annotate_features(capture, [1, 2], options)
    assert info["retrieval"] == "download" and rows[1]["explanations"]
    monkeypatch.setattr(
        np, "_download", lambda *args: pytest.fail("Offline cache must not download")
    )
    options["allow_network"] = False
    info, _ = np.annotate_features(capture, [1], options)
    assert info["retrieval"] == "cache"
    path = Path(info["cache_path"])
    doc = json.loads(path.read_text())
    doc["payload"]["explanations"]["1"][0]["description"] = "corrupted"
    path.write_text(json.dumps(doc))
    info, rows = np.annotate_features(capture, [1], options)
    assert info["status"] == "blocked" and rows[1]["status"] == "blocked"
    assert "checksum" in info["cache_rejected"]


@pytest.mark.parametrize("missing", [False, True])
def test_missing_explanations_or_network_failure_preserve_capture_and_exit_two(
    monkeypatch, tmp_path, expected, cfg, capture, missing
):
    monkeypatch.setattr(np, "_source", lambda pair: expected)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)

    def download(*args):
        if missing:
            return payload(expected, indices=(2,))
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr(np, "_download", download)
    result = operation.execute(cfg)
    assert result["active_count"] == 3 and result["feature_count"] == 4
    assert result["completion"]["status"] == "incomplete"
    assert EvaluationResult.from_payload("capture", cfg, result).exit_code == 2
    monkeypatch.chdir(tmp_path)
    from saebase.runtime.sae_operations import _maybe_save

    _maybe_save("capture", result, cfg)
    assert json.loads((tmp_path / "metadata/capture_result.json").read_text())["active_count"] == 3


@pytest.mark.parametrize("enabled,expected_status", [(False, "disabled"), (True, "unsupported")])
def test_disabled_or_unmapped_sources_do_not_contact_service(
    monkeypatch, cfg, capture, enabled, expected_status
):
    cfg.task.neuronpedia.enabled = enabled
    monkeypatch.setattr(np, "_source", lambda pair: None)
    monkeypatch.setattr(np, "_download", lambda *args: pytest.fail("No exact source selected"))
    result = operation.execute(cfg)
    assert result["neuronpedia"]["status"] == expected_status
    assert result["completion"]["status"] == ("complete" if not enabled else "incomplete")


@pytest.mark.parametrize(
    "key,value", [("modelId", "other"), ("hfRepoId", "other/weights"), ("hfFolderId", "layer1")]
)
def test_source_identity_cannot_be_approximated(expected, key, value):
    data = source(expected)
    data[key] = value
    with pytest.raises(ValueError):
        np._verify_source(data, expected)


@pytest.mark.parametrize(
    "field,value", [("d_in", 99), ("d_sae", 99), ("hook_name", "blocks.1.hook_resid_post")]
)
def test_source_dimensions_and_hook_are_verified(expected, field, value):
    data = source(expected)
    data["saelensConfig"][field] = value
    with pytest.raises(ValueError, match="dimensions/hook"):
        np._verify_source(data, expected)


def test_declared_sae_metadata_is_used_without_model_name_guessing(monkeypatch, expected):
    from saebase.api import sae_profiles

    profile = SimpleNamespace(
        loader="custom", repo_id=expected["repo"], sae_folder=lambda layer: expected["folder"]
    )
    monkeypatch.setattr(sae_profiles, "get_profile", lambda key: profile)
    identity = SimpleNamespace(
        sae_key="synthetic", layer=0, d_in=2, d_sae=4, hook_name=expected["hook"]
    )
    metadata = SimpleNamespace(neuronpedia_id=expected["id"])
    pair = SimpleNamespace(
        identity=identity, sae=SimpleNamespace(cfg=SimpleNamespace(metadata=metadata))
    )
    assert np._source(pair) == expected
    metadata.neuronpedia_id = "../.."
    with pytest.raises(ValueError, match="identifier"):
        np._source(pair)


@pytest.mark.parametrize(
    "sidecar", [None, "config.json", "failed_explanation_indexes.txt", "unexpected.json"]
)
def test_multiple_explanation_exports_preserve_authors_and_models(
    monkeypatch, expected, cfg, sidecar
):
    module = ModuleType("neuronpedia.np_explanation")

    class Explanation:
        @classmethod
        def from_np_db_json(cls, row):
            return SimpleNamespace(
                index=int(row["index"]),
                text=row["description"],
                method=row.get("typeName"),
                explainer_model=row.get("explanationModelName"),
            )

    module.Explanation = Explanation
    monkeypatch.setitem(sys.modules, "neuronpedia.np_explanation", module)
    prefix = "v1/synthetic/0-res/"
    folders = [prefix + "explanations/", prefix + "explanations-another/"]
    packed = {
        folder: gzip.compress(
            json.dumps(
                {
                    "modelId": "synthetic",
                    "layer": "0-res",
                    "index": "1",
                    "id": folder,
                    "description": "synthetic label",
                    "typeName": "synthetic",
                    "explanationModelName": folder,
                    "authorId": folder,
                    "embedding": "must not be retained",
                }
            ).encode()
        )
        for folder in folders
    }

    def fetch(url, **kwargs):
        query = parse_qs(urlsplit(url).query)
        if url.endswith("source.jsonl"):
            return json.dumps(source(expected)).encode()
        if query:
            if "delimiter" in query:
                body = "".join(
                    f"<CommonPrefixes><Prefix>{folder}</Prefix></CommonPrefixes>"
                    for folder in folders
                )
            else:
                folder = query["prefix"][0]
                body = f"<Contents><Key>{folder}batch-0.jsonl.gz</Key><Size>{len(packed[folder])}</Size></Contents>"
                if sidecar:
                    body += f"<Contents><Key>{folder}{sidecar}</Key><Size>10</Size></Contents>"
            return (
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
                + body
                + "</ListBucketResult>"
            ).encode()
        folder = next(folder for folder in folders if url.endswith(folder + "batch-0.jsonl.gz"))
        return packed[folder]

    monkeypatch.setattr(np, "_fetch", fetch)
    if sidecar == "unexpected.json":
        with pytest.raises(ValueError, match="Unexpected Neuronpedia export object path"):
            np._download(expected, online_options(cfg))
        return
    data = np._download(expected, online_options(cfg))
    assert len(data["explanations"]["1"]) == 2
    assert {row["explainer_model"] for row in data["explanations"]["1"]} == set(folders)
    assert all("embedding" not in row for row in data["explanations"]["1"])
    assert data["files"] and all(len(row["sha256"]) == 64 for row in data["files"])
    with pytest.raises(ValueError, match="budget"):
        np._download(expected, {**online_options(cfg), "max_download_bytes": 1})


def test_precheck_does_not_hide_incomplete_annotations(monkeypatch):
    from tasks.precheck.worker import check
    import saebase.runtime.sae_operations

    native = {
        "status": "executed",
        "operation": "capture",
        "feature_count": 4,
        "top_features": [],
        "completion": completion_summary([{"status": "blocked"}]),
    }
    monkeypatch.setattr(saebase.runtime.sae_operations, "execute_task", lambda *args: native)
    result = check({"kind": "task", "operation": "capture", "config": {}})
    assert result["status"] == "BLOCKED" and result["native_result"] == native


@pytest.mark.parametrize(
    "field,value", [("enabled", "yes"), ("timeout_seconds", 0), ("cache_dir", "")]
)
def test_invalid_annotation_settings_fail_before_loading(monkeypatch, cfg, field, value):
    cfg.task.neuronpedia[field] = value
    monkeypatch.setattr(operation, "_pair", lambda cfg: pytest.fail("Must validate before loading"))
    with pytest.raises(ValueError, match="neuronpedia"):
        operation.execute(cfg)


def test_batch_export_snapshot_is_verified_once_and_results_cannot_poison_it(
    monkeypatch, expected, cfg, capture
):
    monkeypatch.setattr(np, "_source", lambda pair: expected)
    monkeypatch.setattr(np, "_download", lambda *args: payload(expected))
    with np.annotation_session():
        first, rows = np.annotate_features(capture, [1], online_options(cfg))
        rows[1]["explanations"][0]["description"] = "modified by caller"
        monkeypatch.setattr(np, "_download", lambda *a: pytest.fail("Snapshot must not redownload"))
        monkeypatch.setattr(
            Path, "read_text", lambda *a, **k: pytest.fail("Snapshot must not reparse disk export")
        )
        again, second = np.annotate_features(capture, [1], online_options(cfg))
        assert again["retrieval"] == "verified_session_snapshot"
        assert second[1]["explanations"][0]["description"] == "synthetic explanation 1"
        assert first["export_sha256"] == again["export_sha256"]


def test_loaded_resource_binding_is_consumed_without_resolving_again(monkeypatch, expected):
    from saebase.api import sae_loading

    monkeypatch.setattr(
        sae_loading, "resolve_sae_source", lambda *a: pytest.fail("Use loaded source")
    )
    binding = {
        "neuronpedia_id": expected["id"],
        "repo": expected["repo"],
        "folder": expected["folder"],
    }
    pair = SimpleNamespace(
        identity=SimpleNamespace(source=binding, d_in=2, d_sae=4, hook_name=expected["hook"]),
        sae=SimpleNamespace(cfg=None),
    )
    assert np._source(pair) == {**expected, "catalog_verified": True}


def test_loaded_source_conflict_is_reported(expected):
    binding = {
        "neuronpedia_id": expected["id"],
        "repo": expected["repo"],
        "folder": expected["folder"],
    }
    pair = SimpleNamespace(
        identity=SimpleNamespace(source=binding),
        sae=SimpleNamespace(cfg=SimpleNamespace(neuronpedia_id="other/0-res")),
    )
    with pytest.raises(ValueError, match="disagree"):
        np._source(pair)

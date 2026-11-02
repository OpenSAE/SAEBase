"""Offline contract fixtures do not claim real HF Dataset acceptance."""

import hashlib
import json

from tasks.precheck.dataset_checks import check_frozen_dataset


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _render(record):
    return f"Question:\n{record['question']}\nAnswer:"


def _make(tmp_path, count=3):
    root = tmp_path / "boolq"
    root.mkdir()
    records = []
    for i in range(count):
        item = {
            "dataset_id": "boolq",
            "source_revision": "a" * 40,
            "source_repo": "google/boolq",
            "source_config": "default",
            "source_split": "validation",
            "sample_id": f"sample-{i}",
            "question": f"Question {i}?",
            "answer": "Yes",
            "answer_type": "boolean",
        }
        item["prompt_sha256"] = hashlib.sha256(_render(item).encode()).hexdigest()
        records.append(item)
    data = "".join(json.dumps(item, sort_keys=True) + "\n" for item in records).encode()
    manifest = {
        "dataset_id": "boolq",
        "n": count,
        "source_revision": "a" * 40,
        "source_repo": "google/boolq",
        "source_config": "default",
        "source_split": "validation",
        "sha256": hashlib.sha256(data).hexdigest(),
        "sample_ids_sha256": _digest([x["sample_id"] for x in records]),
        "lock_status": "local_unpinned",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "questions.jsonl").write_bytes(data)
    return root, manifest, records


def _run(tmp_path, **kwargs):
    return check_frozen_dataset(
        dataset_id="boolq",
        inputs_root=tmp_path,
        expected_count=3,
        renderer=_render,
        **kwargs,
    )


def test_fixture_only_valid_local_cohort(tmp_path):
    _make(tmp_path)
    result = _run(tmp_path, expected_revision="a" * 40)
    assert result["status"] == "PASS"
    assert result["observed"] == 3
    assert result["scientific_acceptance"] == "NOT_COVERED"
    assert result["source_lock_status"] == "local_unpinned"


def test_tampered_content_fails_closed(tmp_path):
    root, _, _ = _make(tmp_path)
    (root / "questions.jsonl").write_text("corrupted\n")
    assert _run(tmp_path)["status"] == "FAIL"


def test_missing_local_source_is_blocked(tmp_path):
    result = _run(tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["code"] == "BLOCKED_ASSET"


def test_wrong_revision_fails_closed(tmp_path):
    _make(tmp_path)
    assert _run(tmp_path, expected_revision="b" * 40)["status"] == "FAIL"


def test_duplicate_sample_id_fails_even_if_manifest_is_rehashed(tmp_path):
    root, manifest, rows = _make(tmp_path)
    rows[1]["sample_id"] = rows[0]["sample_id"]
    payload = "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows).encode()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["sample_ids_sha256"] = _digest([x["sample_id"] for x in rows])
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "questions.jsonl").write_bytes(payload)
    assert _run(tmp_path)["status"] == "FAIL"


def test_label_does_not_resolve_to_choice(tmp_path):
    root, manifest, rows = _make(tmp_path)
    rows[0]["answer_type"] = "choice"
    rows[0]["answer"] = "Q"
    rows[0]["choices"] = [{"label": "A", "text": "Option"}]
    payload = "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows).encode()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "questions.jsonl").write_bytes(payload)
    assert _run(tmp_path)["status"] == "FAIL"


def test_path_traversal_rejected(tmp_path):
    result = check_frozen_dataset(
        dataset_id="../secret", inputs_root=tmp_path, expected_count=3, renderer=_render
    )
    assert result["status"] == "FAIL"


def test_symlink_cohort_is_not_allowed(tmp_path):
    root, _, _ = _make(tmp_path)
    (tmp_path / "another").mkdir()
    link = tmp_path / "another" / "manifest.json"
    link.symlink_to(root / "manifest.json")
    (root / "manifest.json").unlink()
    (root / "manifest.json").symlink_to(link)
    assert _run(tmp_path)["status"] == "FAIL"


def test_manifest_source_repo_mismatch_must_fail(tmp_path):
    root, manifest, rows = _make(tmp_path)
    rows[0]["source_repo"] = "unrelated/other"
    payload = "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows).encode()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "questions.jsonl").write_bytes(payload)
    assert _run(tmp_path)["status"] == "FAIL"


def test_explicit_matrix_never_marks_compatibility_pass():
    from tasks.precheck.compatibility_checks import compatibility_matrix

    sources = ["boolq", "arc_challenge"]
    sites = [
        {
            "model_key": "gemma270m",
            "sae_key": "gemma270m_scope_resid",
            "layer": 5,
            "hook_name": "blocks.5.hook_resid_post",
            "site_id": "gemma270m:gemma270m_scope_resid:L5",
        },
        {
            "model_key": "gemma270m",
            "sae_key": "gemma270m_scope_resid",
            "layer": 12,
            "hook_name": "blocks.12.hook_resid_post",
            "site_id": "gemma270m:gemma270m_scope_resid:L12",
        },
    ]
    declared = compatibility_matrix(sources, sites)
    assert declared["models"] == 1
    assert declared["model_dataset_targets"] == 2
    assert declared["sae_site_dataset_targets"] == 4
    assert all(row["status"] == "NOT_RUN" for row in declared["pairs"])
    assert not any(row["eligible"] for row in declared["pairs"])
    passed = {
        "dataset:boolq": {"status": "PASS"},
        "dataset:arc_challenge": {"status": "PASS"},
        "model:gemma270m": {"status": "PASS"},
        "sae:gemma270m:gemma270m_scope_resid:L5": {"status": "PASS"},
        "sae:gemma270m:gemma270m_scope_resid:L12": {"status": "PASS"},
    }
    selected = compatibility_matrix(sources, sites, passed)
    assert all(row["eligible"] for row in selected["pairs"])
    assert all(row["status"] == "NOT_RUN" for row in selected["pairs"])
    passed["dataset:boolq"] = {"status": "BLOCKED"}
    selected = compatibility_matrix(sources, sites, passed)
    assert all(
        row["status"] == "BLOCKED" for row in selected["pairs"] if row["dataset_id"] == "boolq"
    )
    assert all(
        row["status"] == "NOT_RUN"
        for row in selected["pairs"]
        if row["dataset_id"] == "arc_challenge"
    )


def test_invalid_site_layer_and_duplicates_rejected():
    from tasks.precheck.compatibility_checks import compatibility_matrix
    import pytest

    site = {
        "model_key": "gemma270m",
        "sae_key": "gemma270m_scope_resid",
        "layer": 5,
        "hook_name": "blocks.5.hook_resid_post",
        "site_id": "gemma270m:gemma270m_scope_resid:L5",
    }
    with pytest.raises(ValueError, match="Duplicate Dataset"):
        compatibility_matrix(["boolq", "boolq"], [site])
    with pytest.raises(ValueError, match="Duplicate SAE Site"):
        compatibility_matrix(["boolq"], [site, dict(site)])
    with pytest.raises(ValueError, match="identity"):
        compatibility_matrix(["boolq"], [{**site, "site_id": "incorrect"}])

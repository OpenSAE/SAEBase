"""WP12 P3-P8 bounded CPU tests. No model download or network access."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest
import torch
from torch import nn

from saebase.wp12.interventions import Edit, EditPlan, ModuleEdit, compare_module_edits
from saebase.wp12.benchmarks import ComparisonPoint, compare_sae_points
from saebase.wp12.training import (
    TrainPlan,
    _sparsify_command,
    register_checkpoint,
)
from saebase.wp12.circuits import (
    GraphNode,
    GraphEdge,
    GraphRecord,
    RepresentationAsset,
    parse_typed_graph,
    prune,
    save_graph,
    graph_feature_intervention_request,
)
from saebase.wp12.acceptance import audit_ledger, inspect_archive
from saebase.wp12.ndif import remote_ndif_capture

ROOT = Path(__file__).resolve().parents[1]


def test_p3_sequential_and_parallel_plan_roundtrip():
    plan = EditPlan("sae", 1, -1, (Edit("ablate", (1,)), Edit("scale", (3,), 2.0)))
    assert EditPlan.parse(plan.json()) == plan
    with pytest.raises(ValueError, match="disjoint"):
        EditPlan("sae", 1, -1, (Edit("ablate", (1,)), Edit("scale", (1,), 2.0)))
    sequential = EditPlan(
        "sae", 1, -1, (Edit("ablate", (1,)), Edit("scale", (1,), 2.0)), compose="sequential"
    )
    assert EditPlan.parse(sequential.json()) == sequential


def test_p3_true_torch_module_edit_and_negative_controls():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(5, 6), nn.ReLU(), nn.Linear(6, 2))
    ids = torch.randn(2, 5)
    baseline = model(ids).clone()
    result = compare_module_edits(model, ids, [ModuleEdit("0", "zero", (0, 2))])
    assert result["fired"] == {"0": 1}
    assert result["baseline"].shape == result["edited"].shape
    assert torch.allclose(model(ids), baseline)
    with pytest.raises(ValueError, match="Unknown"):
        compare_module_edits(model, ids, [ModuleEdit("bogus", "zero")])


def _point(key: str, model: str = "a") -> ComparisonPoint:
    return ComparisonPoint(
        model_sha256=model * 64,
        sae_sha256=key * 64,
        hook="blocks.0.hook_resid_post",
        dataset_sha256="d" * 64,
        native_commit="a" * 40,
        metric_id="saebench.core.mse",
        metric_value=0.5,
        l0=float(12 if key == "1" else 24),
        sae_name=key,
    )


def test_p4_fair_native_result_preflight():
    assert compare_sae_points([_point("1"), _point("2")])["comparable"]
    unequal = compare_sae_points([_point("1"), _point("2", model="b")])
    assert not unequal["comparable"]
    assert "model_sha256" in unequal["mismatches"]


def plan(tmp_path: Path, *, distributed: int = 1) -> TrainPlan:
    model, data = tmp_path / "model", tmp_path / "dataset"
    model.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    return TrainPlan(
        backend="sparsify",
        model_path=str(model),
        dataset_path=str(data),
        model_revision="a" * 40,
        dataset_revision="b" * 40,
        hookpoint="transformer.h.0",
        d_in=8,
        d_sae=16,
        k=4,
        architecture="topk",
        max_examples=20,
        steps=10,
        batch_size=2,
        ctx_len=128,
        run_name="test-p5",
        device="cpu",
        distributed_gpus=distributed,
    )


def test_p5_native_training_validation_checkpoint_without_gpu(tmp_path):
    p = plan(tmp_path)
    p.validate()
    assert _sparsify_command(p)[:3] == [__import__("sys").executable, "-m", "sparsify"]
    with pytest.raises(PermissionError):
        plan(tmp_path, distributed=2).validate()
    plan(tmp_path, distributed=2).validate(allow_large=True)
    folder = tmp_path / "native"
    folder.mkdir()
    (folder / "sae.safetensors").write_bytes(b"untrusted test-only fixture")
    output = tmp_path / "registry.json"
    record = register_checkpoint(
        folder,
        output,
        source={"upstream_commit": "a" * 40},
    )
    assert record["load_and_encode_verified"] is False
    assert len(record["tensor_file_sha256"]) == 1
    with pytest.raises(FileExistsError):
        register_checkpoint(folder, output, source={"upstream_commit": "a" * 40})


def graph() -> GraphRecord:
    return GraphRecord(
        nodes=(
            GraphNode("tok-0", "token", None, None, 0, "A"),
            GraphNode("feature-1", "feature", 2, 17, 0, "Number feature"),
            GraphNode("error-1", "error", 2, None, None, "Transcoder error"),
            GraphNode("logit-1", "logit", None, None, None, "Next token"),
        ),
        edges=(
            GraphEdge("tok-0", "feature-1", 0.3),
            GraphEdge("feature-1", "logit-1", -0.5),
            GraphEdge("error-1", "logit-1", 0.01),
        ),
        source_sha256="a" * 64,
        model_sha256="b" * 64,
        transcoder_sha256="c" * 64,
        original_tool_commit="d" * 40,
        source_format="source-normalized",
    )


def test_p6_graph_preserves_signed_edges_and_error_nodes(tmp_path):
    original = graph()
    original.validate()
    record = parse_typed_graph(
        {
            "nodes": [n.__dict__ for n in original.nodes],
            "edges": [e.__dict__ for e in original.edges],
        },
        provenance=original.payload()["provenance"],
    )
    assert record == original
    small = prune(original, min_abs_weight=0.2)
    assert len(small.edges) == 2
    assert any(n.kind == "error" for n in small.nodes)
    assert any(e.weight < 0 for e in small.edges)
    out = tmp_path / "saved-graph"
    meta = save_graph(original, out)
    assert meta["edge_count"] == 3
    asset = RepresentationAsset(
        "transcoder",
        "official/native",
        "a" * 40,
        "layer_2",
        "input.hook",
        "output.hook",
        8,
        8,
    )
    request = graph_feature_intervention_request(
        original,
        "feature-1",
        asset,
        value=0,
    )
    assert request["executed"] is False
    with pytest.raises(ValueError, match="feature node"):
        graph_feature_intervention_request(original, "error-1", asset, value=0)


def test_p7_ndif_does_not_run_without_explicit_permission():
    with pytest.raises(PermissionError):
        remote_ndif_capture(None, "private text", None, model_id="unknown")


def test_p8_release_stays_blocked_until_independent_evidence():
    report = audit_ledger(ROOT / "work_plan/12_capability_acceptance_matrix.csv")
    assert report["capabilities"] == 50
    assert not report["release_ready"]
    assert len(report["missing"]) > 0


def test_p8_archive_sha_and_members_are_recomputed(tmp_path):
    archive = tmp_path / "evidence.zip"
    blob = b"small test tensor bytes, not actual model evidence"
    failure = b"[]"
    manifest = (
        hashlib.sha256(blob).hexdigest()
        + "  native_reference/one.pt\n"
        + hashlib.sha256(failure).hexdigest()
        + "  failures.json\n"
    )
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("run/native_reference/one.pt", blob)
        zipped.writestr("run/failures.json", failure)
        zipped.writestr("run/checksums.sha256", manifest)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = inspect_archive(archive, expected_sha256=sha)
    assert result["checked_members"] == 2
    assert result["has_original_failure_log"]
    with pytest.raises(ValueError, match="SHA256"):
        inspect_archive(archive, expected_sha256="0" * 64)

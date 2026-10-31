"""Source-level native-pyvene bridge and private prompt inspection contract."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from saebase.interventions.pyvene_adapter import (
    PyvenePlan,
    PyveneRepresentation,
    build_native_pyvene,
    execute_native_pyvene,
    load_native_pyvene,
    save_native_pyvene,
)


def test_pyvene_roundtrip_parallel_serial_and_provenance(monkeypatch, tmp_path):
    class Native:
        __module__ = "pyvene.models.intervenable_base"

        def __init__(self, config, model):
            self.config, self.model = config, model

        def __call__(self, base, **kwargs):
            return base, {"true_native_run": True, "kwargs": kwargs}

        def save(self, path, **kwargs):
            assert kwargs == {"include_model": False, "save_to_hf_hub": False}
            from pathlib import Path

            Path(path).mkdir()
            (Path(path) / "native.json").write_text("native")

    def make_config(**kwargs):
        return SimpleNamespace(**kwargs)

    class PyveneModel(Native):
        @staticmethod
        def load(path, model, **kwargs):
            assert kwargs == {"from_huggingface_hub": False, "include_model": False}
            return Native(SimpleNamespace(), model)

    PyveneModel.__module__ = "pyvene.models.intervenable_base"
    fake = ModuleType("pyvene")
    fake.RepresentationConfig = make_config
    fake.IntervenableConfig = make_config
    fake.IntervenableModel = PyveneModel
    fake.VanillaIntervention = type("VanillaIntervention", (), {})
    fake.ZeroIntervention = type("ZeroIntervention", (), {})
    monkeypatch.setitem(sys.modules, "pyvene", fake)

    p = PyvenePlan(
        (
            PyveneRepresentation(0, "block_output"),
            PyveneRepresentation(1, "mlp_output", intervention="zero"),
        ),
        mode="serial",
    )
    assert PyvenePlan.parse(p.json()) == p
    wrapped = build_native_pyvene("source-model", p)
    assert wrapped.config.mode == "serial"
    assert len(wrapped.config.representations) == 2
    before, after = execute_native_pyvene(
        wrapped,
        base={"input_ids": [1]},
        sources=[{"input_ids": [2]}],
        unit_locations={"source_0->base": ([[0]], [[0]])},
    )
    assert before == {"input_ids": [1]}
    assert after["true_native_run"] is True
    folder = tmp_path / "native"
    save_native_pyvene(wrapped, folder)
    with pytest.raises(PermissionError):
        load_native_pyvene(model="source", directory=folder)
    loaded = load_native_pyvene(
        model="source",
        directory=folder,
        trusted_local_source=True,
    )
    assert isinstance(loaded, Native)
    with pytest.raises(FileExistsError):
        save_native_pyvene(wrapped, folder)
    with pytest.raises(ValueError, match="Duplicate"):
        PyvenePlan(
            (
                PyveneRepresentation(0, "block_output"),
                PyveneRepresentation(0, "block_output"),
            )
        )


def test_private_inspection_is_a_true_job_mode_with_no_steering():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "apps" / "research-lab"
    sys.path.insert(0, str(path))
    from backend import core
    from backend.jobs import JobQueue, JobError

    site = next(item for item in core.load_sites() if core.has_complete_identity(item))
    spec = {
        "site_id": site["site_id"],
        "operation": "inspect",
        "feature_ids": [],
        "strength": 0.0,
        "max_new_tokens": 0,
        "input_ref": "a" * 64,
    }
    assert JobQueue._validated_spec(spec) == spec
    with pytest.raises(JobError, match="select no feature IDs"):
        JobQueue._validated_spec({**spec, "feature_ids": [1]})
    with pytest.raises(JobError, match="no feature edits"):
        JobQueue._validated_spec({**spec, "strength": 1.0})

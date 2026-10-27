"""M5 strictly synthetic acceptance and recovery; no downloads or release claims."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ModelAdapter
from saebase.release import MatrixConfig, load_matrix, plan_matrix
from saebase.release.runner import execute_matrix, verify_matrix

CONFIG = Path(__file__).resolve().parents[1] / "conf/release/m5_matrix.json"


def mock_matrix():
    d = load_matrix(CONFIG).to_dict()
    d["prompts"] = ["1 2 3", "3 4 5", "1 3 6", "2 7 4"]
    d["bootstrap_replicates"] = 20
    return MatrixConfig.from_dict(d)


def loaders():
    models = {}
    saes = {}
    calls = {"model": 0, "sae": 0, "forward": 0}

    def load_model(target, device):
        calls["model"] += 1
        name = target.model_key
        model = ResearchToyModel(
            layer=target.site.layer,
            hook_kind="hook_resid_pre" if name == "gpt2small" else "hook_resid_post",
        ).eval()
        real = model.run_with_cache

        def counted(*args, **kwargs):
            calls["forward"] += 1
            return real(*args, **kwargs)

        model.run_with_cache = counted
        models[name] = ModelAdapter(model)
        return models[name], "synthetic_" + name

    def load_sae(target, key, device):
        calls["sae"] += 1
        obj = ResearchToySAE().eval()
        if key.endswith("gated_l3") or key.endswith("6144_l8"):
            with torch.no_grad():
                obj.bias.add_(0.2)
        saes[key] = obj
        return obj

    return load_model, load_sae, calls


def test_offline_plan_no_model_forward_or_download():
    c = load_matrix(CONFIG)
    plan = plan_matrix(c)
    assert len(plan["model_families"]) == 2
    assert plan["models"] == 2
    assert plan["distinct_checkpoints"] == 4
    assert plan["checkpoint_downloads"] == 0
    assert plan["planned_m2_measurements"] == 48
    assert c.matrix_id == MatrixConfig.from_dict(c.to_dict()).matrix_id
    with pytest.raises(ValueError, match="real=True"):
        execute_matrix(c, "/tmp/unused-model-output")


def test_matrix_rejects_wrong_hook_and_family():
    c = load_matrix(CONFIG).to_dict()
    c["targets"][0]["site"]["hook_name"] = "blocks.3.hook_resid_pre"
    with pytest.raises(ValueError, match="hook"):
        MatrixConfig.from_dict(c)
    c = load_matrix(CONFIG).to_dict()
    c["targets"][1]["family"] = "pythia"
    with pytest.raises(ValueError, match="independent model families"):
        MatrixConfig.from_dict(c)


def test_synthetic_matrix_same_hook_reuse_and_recovery(tmp_path):
    c = mock_matrix()
    mdl, sae, calls = loaders()
    with pytest.raises(RuntimeError, match="injected M5"):
        execute_matrix(
            c, tmp_path, real=True, model_loader=mdl, sae_loader=sae, fail_after_models=1
        )
    matrixroot = tmp_path / c.matrix_id
    assert (matrixroot / "matrix_progress.json").exists()
    assert not (matrixroot / "matrix_manifest.json").exists()
    after_first = calls["forward"]
    assert after_first == len(c.prompts), "SAE second profile must reuse M1 activation capture"
    recovered = execute_matrix(c, tmp_path, real=True, model_loader=mdl, sae_loader=sae)
    assert recovered["status"] == "synthetic_test"
    assert not recovered["real_checkpoints"]
    assert len(recovered["families"]) == 2
    assert recovered["n_distinct_saes"] == 4
    assert recovered["n_models"] == 2
    assert (
        recovered["models"][0]["checkpoints"]["pythia70m_saebench_gated_l3"]["cache_cost"][
            "activations_collected"
        ]
        == 0
    )
    assert recovered["models"][1]["checkpoints"]["gpt2small_fs_6144_l8"]["cache_cost"][
        "activations_reused"
    ] == len(c.prompts)
    assert calls["forward"] == 2 * len(c.prompts), "resume must reuse all committed M1 captures"
    assert (matrixroot / "matrix_report.html").exists() is False  # cannot call synthetic real proof
    with pytest.raises(ValueError, match="not validated"):
        verify_matrix(matrixroot, expected_id=c.matrix_id)
    # Synthetic status cannot be promoted by changing a flag without matching hashes.
    raw = json.loads((matrixroot / "matrix_results.json").read_text())
    raw["status"] = "passed"
    (matrixroot / "matrix_results.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checksum"):
        verify_matrix(matrixroot, expected_id=c.matrix_id)


def test_m5_compatibility_registry_and_declared_source_cards():
    from saebase.sae.registry import get_profile, MODEL_KEYS

    assert "gpt2small" in MODEL_KEYS
    for name, site in (
        ("gpt2small_res_jb_l8", "blocks.8.hook_resid_pre"),
        ("gpt2small_fs_6144_l8", "blocks.8.hook_resid_pre"),
        ("pythia70m_saebench_gated_l3", "blocks.3.hook_resid_post"),
    ):
        p = get_profile(name)
        assert p.hook_name(p.layers[0]) == site
    for target in load_matrix(CONFIG).targets:
        assert target.model_card.startswith("https://")
        assert all(target.sae_cards[k].startswith("https://") for k in target.sae_keys)

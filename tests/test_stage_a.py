"""Stage A six-model offline contracts and native Qwen-Scope checkpoint tests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ModelAdapter
from saebase.coverage.stage_a import (
    STAGE_A, StageAPair, get_pair, plan_stage_a, probe_pair, read_evidence,
    write_evidence,
)
from saebase.sae.qwen_topk import QwenResidualTopK, from_official_checkpoint
from saebase.sae.registry import get_profile, DEFAULT_SAE_FOR_MODEL
from scripts.stage_a_matrix import main


def test_all_six_public_profiles_come_from_exact_registry_and_have_public_sources():
    assert [p.model_key for p in STAGE_A] == [
        "gemma270m", "gemma1b", "gemma2_2b_base", "qwen35_2b",
        "qwen3_8b", "llama31_8b",
    ]
    assert len({p.family for p in STAGE_A}) == 5
    plan = plan_stage_a()
    assert len(plan["pairs"]) == 6
    assert all(p["status"] == "registered_not_weight_verified" for p in plan["pairs"])
    assert plan["requires_real_runs_for_verification"] is True
    for p in STAGE_A:
        assert p.sae_repo and p.model_id and p.d_in > 0 and p.d_sae >= p.d_in
        prof = get_profile(p.sae_key)
        assert prof.repo_id == p.sae_repo
        assert prof.model_key == p.model_key
        assert prof.hook_name(p.layer) == p.site.hook_name
    assert get_profile("gemma270m_scope_resid").sae_release == "gemma-scope-2-270m-pt-resid_post"
    assert get_profile("gemma1b_scope_resid").sae_release == "gemma-scope-2-1b-pt-resid_post"
    assert get_profile("gemma2_2b_stagea_resid_l25_16k").sae_id(25) == "layer_25/width_16k/average_l0_116"
    assert get_profile("qwen35_2b_resid_w32k").sae_id(12) == "layer12.sae.pt"
    assert DEFAULT_SAE_FOR_MODEL["qwen35_2b"] == "qwen35_2b_resid_w32k"


def test_stage_a_preflight_never_downloads_and_rejects_invalid_scopes(capsys):
    assert main(["--dry-run"]) == 0
    content = json.loads(capsys.readouterr().out)
    assert len(content["six_model_roster"]) == 6
    assert all(p["status"] == "registered_not_weight_verified" for p in content["pairs"])
    assert main(["--dry-run", "--target", "qwen35_2b"]) == 0
    with pytest.raises(ValueError, match="unique"):
        plan_stage_a(("qwen35_2b", "qwen35_2b"))
    with pytest.raises(ValueError, match="not in Stage A"):
        get_pair("nonexistent")


def test_qwen_official_topk_equations_and_strict_safetensors(tmp_path: Path):
    torch.manual_seed(42)
    width, d_in, k = 9, 4, 3
    w_enc = torch.randn(width, d_in)
    w_dec = torch.randn(d_in, width)
    b_enc = torch.randn(width)
    b_dec = torch.randn(d_in)
    path = tmp_path / "layer12.sae.pt"
    torch.save({"W_enc": w_enc, "W_dec": w_dec,
                "b_enc": b_enc, "b_dec": b_dec}, path)
    sae = from_official_checkpoint(path, hook_name="blocks.12.hook_resid_post",
                                   expected_hidden=4, expected_features=9, k=3)
    x = torch.randn(2, 4)
    pre = x @ w_enc.T + b_enc
    vals, ids = pre.topk(k, dim=-1)
    expected = torch.zeros_like(pre).scatter(-1, ids, vals)
    actual = sae.encode(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(sae.decode(actual), expected @ w_dec.T + b_dec)
    assert sae.cfg.d_in == 4 and sae.cfg.d_sae == 9 and sae.cfg.k == 3
    assert len(list(sae.parameters())) == 4
    assert all(not p.requires_grad for p in sae.parameters())
    with pytest.raises(ValueError, match="hidden dimension"):
        from_official_checkpoint(path, hook_name="blocks.12.hook_resid_post", expected_hidden=8)
    with pytest.raises(ValueError, match="feature width"):
        from_official_checkpoint(path, hook_name="blocks.12.hook_resid_post", expected_features=16)
    with pytest.raises(ValueError, match="between"):
        QwenResidualTopK(w_enc, w_dec, b_enc, b_dec, k=0, hook_name="hook")
    with pytest.raises(ValueError, match="dimensions"):
        QwenResidualTopK(w_enc, w_dec.T, b_enc, b_dec, k=3, hook_name="hook")
    with pytest.raises(ValueError, match="nonfinite"):
        QwenResidualTopK(torch.full_like(w_enc, float("nan")), w_dec, b_enc, b_dec,
                          k=3, hook_name="hook")
    with pytest.raises(ValueError, match="width"):
        sae.encode(torch.randn(1, 5))


def test_synthetic_probe_cannot_be_mistaken_for_real_evidence(tmp_path: Path):
    toy = StageAPair(
        "synthetic", "fixture", 0, "test-only", "no-upstream",
        "toy", "tiny", "test", "test", False, 4, 7,
    )
    result = probe_pair(
        toy, ModelAdapter(ResearchToyModel().eval()), ResearchToySAE().eval(),
        prompts=("1 2 3", "2 3 4"),
    )
    assert result["n_reference_prompts"] == 2
    assert result["backend"] == "transformer_lens"
    assert all(r["code_l0"] > 0 for r in result["measurements"])
    assert all(r["edited_logit_delta_l2"] > 0 for r in result["measurements"])
    record = {"schema": "saebase.stage_a.coverage.v1", "status": "synthetic_test",
              "model_key": "synthetic", "evidence": result}
    path = write_evidence(tmp_path / "synthetic.json", record)
    assert read_evidence(path)["status"] == "synthetic_test"
    corrupted = json.loads(path.read_text())
    corrupted["status"] = "real_weights_verified"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="integrity"):
        read_evidence(path)
    with pytest.raises(FileExistsError):
        write_evidence(path, record)


def test_stage_a_probe_rejects_checkpoint_site_and_size_mismatches():
    toy = StageAPair("synthetic", "fixture", 0, "toy", "fixture",
                     "toy", "tiny", "test", "test", False, 4, 7)
    adapter = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    with pytest.raises(ValueError, match="dimensions"):
        probe_pair(replace(toy, d_sae=9), adapter, sae, prompts=("1 2 3",))
    sae.cfg.hook_name = "blocks.5.hook_resid_post"
    with pytest.raises(ValueError, match="incompatible hook"):
        probe_pair(toy, adapter, sae, prompts=("1 2 3",))

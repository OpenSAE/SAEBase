"""Phase A offline end-to-end tests. Real model weights are intentionally not required."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


class FakePythia(torch.nn.Module):
    """Predict the second name in A B A with deterministic fake logits."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        from saebase.data.SimpleCopying.SimpleCopying import NAMES

        self.ids = {name: i + 1 for i, name in enumerate(NAMES)}
        self.cfg = SimpleNamespace(d_vocab=128)

    def to_tokens(self, text, prepend_bos=True):
        return torch.tensor([[self.ids[n] for n in text.strip().split()]], dtype=torch.long)

    def forward(self, prompt):
        names = prompt.strip().split()
        logits = torch.zeros((1, len(names), 128))
        logits[0, -1, self.ids[names[1]]] = 9.0
        return logits

    def run_with_cache(self, tokens, names_filter=None):
        batch, length = tokens.shape
        base = tokens.to(dtype=torch.float32).unsqueeze(-1)
        activation = torch.cat([base / 50 + 1, base / 50 + 2, base / 50 + 3, base / 50 + 4], dim=-1)
        return None, {names_filter: activation}


class FakeSAE:
    cfg = SimpleNamespace(d_in=4, d_sae=8)

    def encode(self, x):
        return torch.cat([x, torch.zeros((*x.shape[:-1], 4), device=x.device)], dim=-1)

    def decode(self, z):
        return z[..., :4] * 0.95


def fake_cfg(data_dir, *, require=True):
    return OmegaConf.create(
        {
            "seed": 8,
            "paths": {"root": str(ROOT), "data_dir": str(data_dir)},
            "prompt_source": {"name": "SimpleCopying"},
            "model": {"name": "pythia70m"},
            "sae": {"name": "pythia70m_resid_sm", "layer": 3},
            "task": {
                "name": "sae",
                "evaluation": "activation_health",
                "facets": {"post_run_summary": True},
            },
            "evaluation": {
                "name": "activation_health",
                "n_prompts": 4,
                "device": "cpu",
                "require_prepared_prompts": require,
            },
        }
    )


def prepare_fake(data_root):
    from saebase.data.SimpleCopying.prepare import prepare_for_model

    model = FakePythia()
    out = prepare_for_model(
        "pythia70m",
        model,
        loaded_model_name="fake-pythia70m",
        model_name="pythia-70m-deduped",
        n_generate=16,
        min_kept=8,
        seed=7,
        data_root=data_root,
    )
    return out, model


def test_prepared_prompts_manifest_and_token_ids(tmp_path):
    from saebase.data.prompt_store import load_manifest, load_prompts

    dest, model = prepare_fake(tmp_path / "sample_data")
    assert dest == (tmp_path / "sample_data" / "SimpleCopying" / "pythia70m" / "v1")
    manifest = load_manifest("SimpleCopying", "pythia70m", data_root=tmp_path / "sample_data")
    rows = load_prompts("SimpleCopying", "pythia70m", data_root=tmp_path / "sample_data")
    assert manifest["seed"] == 7 and manifest["generated_count"] == 16
    assert manifest["kept_count"] == 16
    assert len(rows) == 16
    assert all(
        x["tokenization"]["input_ids"] == model.to_tokens(x["prompt"])[0].tolist() for x in rows
    )
    assert all(x["tok_correct"] != x["tok_distractor"] for x in rows)


def test_preflight_uses_requested_data_dir(tmp_path):
    from saebase.sae.pipeline_checks import run_preflight
    from saebase.sae.prompts import load_behavior_prompts

    data = tmp_path / "custom_prompts"
    prepare_fake(data)
    info = run_preflight(fake_cfg(data))
    assert info["manifest"]["kept_count"] == 16
    _, rows = load_behavior_prompts("SimpleCopying", "pythia70m", n=3, data_root=str(data))
    assert len(rows) == 3
    with pytest.raises(FileNotFoundError, match="Prompt manifest"):
        run_preflight(fake_cfg(tmp_path / "unused"))


def test_complete_activation_health_and_json(tmp_path, monkeypatch):
    import saebase.sae.activation_health as module

    data = tmp_path / "outside_repo_data"
    _, model = prepare_fake(data)
    sae = FakeSAE()
    hook = "blocks.3.hook_resid_post"
    meta = {
        "hook_name": hook,
        "sae_release": "pythia-70m-deduped-res-sm",
        "sae_id": hook,
        "d_in": 4,
        "d_sae": 8,
        "loader": "sae_lens",
        "loaded_model_name": "fake-pythia70m",
    }
    monkeypatch.setattr(module, "load_model_and_sae", lambda *args, **kwargs: (model, sae, meta))
    monkeypatch.setattr(module, "print_vram_gb", lambda: None)
    out = tmp_path / "isolated_results"
    out.mkdir()
    monkeypatch.chdir(out)
    result = module.run_activation_health(fake_cfg(data))
    assert result.ok and result.summary["n_ok"] == 4 and result.summary["n_fail"] == 0
    assert all(r["l0"] == 4 and r["activation_shape"] == [4] for r in result.per_prompt)
    p = out / "sae_activation_summary.json"
    assert p.exists()
    body = json.loads(p.read_text())
    assert body["model_key"] == "pythia70m" and body["layer"] == 3
    assert body["hydra"]["paths"]["data_dir"] == str(data)
    assert body["manifest"]["filter_policy"]["name"] == "logit_diff"
    assert not (ROOT / "sae_activation_summary.json").exists()


def test_cli_path_overrides_resolve_from_caller(tmp_path):
    from main import parse_args, build_overrides

    args, unknown = parse_args(
        [
            "--device",
            "cpu",
            "--model",
            "pythia70m",
            "--data-dir",
            "prompts",
            "--run-dir",
            "runs",
            "--layer",
            "3",
        ]
    )
    assert unknown == []
    overrides = build_overrides(args, tmp_path)
    assert f"paths.data_dir={tmp_path / 'prompts'}" in overrides
    assert f"hydra.run.dir={tmp_path / 'runs'}" in overrides
    assert "sae.layer=3" in overrides and "model.device=cpu" in overrides


def test_unbundled_task_and_invalid_layer_fail_early(tmp_path):
    from main import parse_args, build_overrides

    args, _ = parse_args(["--task", "discovery"])
    with pytest.raises(ValueError, match="not packaged"):
        build_overrides(args, tmp_path)
    args, _ = parse_args(["--layer", "42"])
    with pytest.raises(ValueError, match="supports layers"):
        build_overrides(args, tmp_path)


def test_run_config_consistency(tmp_path):
    from saebase.cfg.config_validate import validate_hydra_cfg
    from saebase.registry import list_registered_task_keys
    from saebase.runtime import missing_packages

    assert set(list_registered_task_keys()) == {"sae", "capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"}
    validate_hydra_cfg(fake_cfg(tmp_path))
    bad = fake_cfg(tmp_path)
    bad.sae.name = "none"
    with pytest.raises((ValueError, RuntimeError)):
        validate_hydra_cfg(bad)
    assert "vllm" not in [x[0] for x in missing_packages("analysis")]


def test_help_and_listing_require_no_gpu(tmp_path):
    commands = [
        [sys.executable, str(ROOT / "main.py"), "--help"],
        [sys.executable, str(ROOT / "main.py"), "--list"],
        [sys.executable, str(ROOT / "saebase" / "tools" / "sae_prepare_prompts.py"), "--help"],
    ]
    for cmd in commands:
        p = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stderr
    assert (
        "sae"
        in subprocess.run(commands[1], cwd=tmp_path, capture_output=True, text=True).stdout
    )


def test_preparation_rejects_unsupported_behaviors_before_download():
    from saebase.tools.sae_prepare_prompts import parse_args, prepare

    a = parse_args(["--behavior", "MMLUPro"])
    with pytest.raises(ValueError, match="supports SimpleCopying"):
        prepare(a)


def test_generator_wrapper_has_no_research_import():
    from saebase.vllm_backend.generate import sampling_params_kwargs

    assert sampling_params_kwargs(max_new_tokens=2, temperature=0.1, top_p=0.9)["max_tokens"] == 2


def test_activation_summary_with_all_failures_is_strict_json(tmp_path):
    from saebase.sae.results import summarize_rows, write_run_summary

    summary = summarize_rows([{"ok": False, "prompt_idx": 0, "error": "bad hook"}])
    assert summary["l0_median"] is None
    out = write_run_summary({"summary": summary}, tmp_path / "failed.json")
    data = json.loads(out.read_text())
    assert data["summary"]["n_fail"] == 1
    assert "NaN" not in out.read_text()


def test_preflight_rejects_missing_manifest_mode(tmp_path):
    from saebase.sae.pipeline_checks import run_preflight

    prepare_fake(tmp_path)
    with pytest.raises(ValueError, match="requires prepared prompts"):
        run_preflight(fake_cfg(tmp_path, require=False))


def test_pythia_sae_loading_accepts_multiple_sae_lens_return_shapes(monkeypatch):
    import saebase.sae.loading as loading

    class SAEDouble:
        def eval(self):
            return self

        def parameters(self):
            return []

    obj = SAEDouble()
    for response in [obj, (obj,), (obj, {"name": "cfg"}), (obj, {"name": "cfg"}, 1.0)]:

        class Loader:
            @staticmethod
            def from_pretrained(**kwargs):
                return response

        monkeypatch.setattr(loading, "import_sae_lens", lambda: SimpleNamespace(SAE=Loader))
        sae, cfg, sparsity = loading.load_sae_lens(
            release="pythia-70m-deduped-res-sm", sae_id="blocks.3.hook_resid_post", device="cpu"
        )
        assert sae is obj
        assert cfg == ({"name": "cfg"} if isinstance(response, tuple) and len(response) > 1 else {})
        assert sparsity == (1.0 if isinstance(response, tuple) and len(response) > 2 else None)


def test_copying_prompt_generator_bounds():
    from saebase.data.SimpleCopying.SimpleCopying import generate_copying_prompts, NAMES

    with pytest.raises(ValueError, match="between 1"):
        generate_copying_prompts(0)
    with pytest.raises(ValueError, match="distinct name pairs"):
        generate_copying_prompts(len(NAMES) * (len(NAMES) - 1) + 1)


def test_pythia_example_config_files_have_required_groups():
    import yaml

    conf = ROOT / "conf"
    spec = yaml.safe_load((conf / "config.yaml").read_text())
    assert any(x.get("task") == "sae" for x in spec["defaults"] if isinstance(x, dict))
    assert spec["hydra"]["job"]["chdir"] is True
    task = yaml.safe_load((conf / "task" / "sae.yaml").read_text())["task"]
    assert task["evaluation"] == "activation_health"
    sae = yaml.safe_load((conf / "sae" / "pythia70m_resid_sm.yaml").read_text())["sae"]
    assert sae["name"] == "pythia70m_resid_sm"
    assert (ROOT / "saebase" / "tools" / "phase_a_smoke.sh").is_file()


def test_missing_external_batchtopk_loader_reports_boundary():
    from saebase.sae.loading import load_sae_for_profile

    with pytest.raises(RuntimeError, match="protocol-10 loader"):
        load_sae_for_profile("llama31_8b_instruct_andyrdt_l19_k64", 19, device="cpu")


def test_legacy_sae_loader_tuple_handled(monkeypatch):
    import saebase.sae.loading as loading

    class SAEDouble:
        def eval(self):
            return self

        def parameters(self):
            return []

    obj = SAEDouble()

    class LegacySAE:
        @staticmethod
        def from_pretrained_with_cfg_and_sparsity(**kwargs):
            return obj, {"legacy": True}, 0.5

    monkeypatch.setattr(loading, "import_sae_lens", lambda: SimpleNamespace(SAE=LegacySAE))
    sae, cfg, sparsity = loading.load_sae_lens(release="r", sae_id="id", device="cpu")
    assert sae is obj and cfg == {"legacy": True} and sparsity == 0.5

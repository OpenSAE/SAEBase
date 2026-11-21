"""Persistent task workflows with synthetic local resources; no downloads or GPU."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf


def test_batch_intervention_reuses_committed_measurements(tmp_path):
    from tasks.intervene.batch import run_batch
    from tests.product.model_adapters.test_reference_checkpoint import make_bundle

    pair = make_bundle()
    config = {
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "layer": 0,
        "samples": [{"sample_id": "synthetic", "token_ids": [1, 2, 3], "target_token_id": 0}],
        "conditions": [
            {"name": "identity", "operation": "identity"},
            {"name": "ablate", "operation": "ablate", "feature_ids": [0]},
        ],
    }
    first = run_batch(pair, config, tmp_path)
    shards = {p: p.read_bytes() for p in (tmp_path / "measurements").rglob("*.jsonl")}
    second = run_batch(pair, config, tmp_path)
    assert first["completion"] == second["completion"]
    assert second["completion"]["completed"] == 2
    assert all(path.read_bytes() == body for path, body in shards.items())


def test_sae_only_probe_never_loads_a_language_model(monkeypatch):
    from saebase.api import sae_loading
    from tasks.precheck.checkpoint import probe_checkpoint
    from tests.fixtures.synthetic import ResearchToySAE

    sae = ResearchToySAE()
    sae.cfg = SimpleNamespace(d_in=4, d_sae=6)
    # Use the synthetic encoder's actual output width.
    sae.cfg.d_sae = int(sae.encode(torch.zeros(1, 4)).shape[-1])
    calls = []

    def load(key, layer, device):
        calls.append((key, layer, device))
        return sae, {"hook_name": f"blocks.{layer}.hook_resid_post"}

    monkeypatch.setattr(sae_loading, "load_sae_for_profile", load)
    monkeypatch.setattr(
        sae_loading,
        "load_model_for_profile",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load LLM")),
    )
    result = probe_checkpoint("gpt2small_res_jb_l8", 8, device="cpu")
    assert result["status"] == "PASS" and result["model_loaded"] is False
    assert len(calls) == 1


def test_all_token_capture_uses_one_forward_and_preserves_sparse_codes():
    from saebase.api import ActivationSite
    from tasks.capture.explore import capture_prompt_result

    class SAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))
            self.cfg = SimpleNamespace(d_in=3, d_sae=3)

        def encode(self, x):
            return x @ self.weight

        def decode(self, z):
            return z @ self.weight.T

    class Model:
        backend = "synthetic"
        calls = 0

        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.model.tokenizer = SimpleNamespace(
                get_vocab=lambda: {"a": 1, "b": 2}, convert_ids_to_tokens=lambda ids: ["a", "b"]
            )

        def tokens(self, **kwargs):
            return torch.tensor([[1, 2]])

        def capture_sequence(self, **kwargs):
            self.calls += 1
            return torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    model = Model()
    pair = SimpleNamespace(
        model=model,
        sae=SAE(),
        site=ActivationSite(0, "blocks.0.hook_resid_post"),
        identity=SimpleNamespace(
            model_key="synthetic", sae_key="synthetic", layer=0, d_in=3, d_sae=3
        ),
    )
    cfg = OmegaConf.create({"task": {"top_feature_num": 2, "neuronpedia": {"enabled": False}}})
    result = capture_prompt_result(cfg, pair, "a b", {"sample_id": "source-row", "split": "test"})
    assert result["source_type"] == "synthetic_test"
    assert result["sample_id"] == "source-row" and result["split"] == "test"
    assert model.calls == 1 and result["output_scope"] == "all_tokens"
    assert [f["feature_id"] for f in result["tokens"][0]["features"]] == [2, 0]
    assert result["tokens"][1]["features"][0]["activation"] == 3.0
    assert result["top_features"][0]["feature_id"] == 1
    assert result["completion"]["status"] == "complete"


def test_generate_multiple_conditions_uses_existing_steering_backend(monkeypatch):
    from tasks.generate import operation
    from tests.product.model_adapters.test_reference_checkpoint import make_bundle

    pair = make_bundle()
    pair.model.model.tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda ids: list(map(str, ids)),
        decode=lambda ids, **kwargs: " ".join(map(str, ids)),
    )
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "task": {
                "prompt": "1 2 3",
                "max_new_tokens": 2,
                "feature_id": 0,
                "conditions": [
                    {"name": "noop", "operation": "identity"},
                    {"name": "ablate", "operation": "ablate", "feature_ids": [0]},
                ],
            },
        }
    )
    result = operation.execute(cfg)
    assert result["completion"]["completed"] == 2
    assert result["result"]["runs"][0]["same_continuation"] is True
    assert result["result"]["runs"][1]["condition"]["feature_ids"] == (0,)


def test_intervention_cli_uses_runtime_and_completed_resume_skips_precheck(
    tmp_path, monkeypatch, capsys
):
    import json
    from dataclasses import replace

    import saebase.api
    from tasks.intervene import cli
    from tasks.intervene.task import InterveneTask
    from tests.product.model_adapters.test_reference_checkpoint import make_bundle

    pair = make_bundle()
    pair.model.model.site_name = "blocks.1.hook_resid_post"
    pair = replace(
        pair,
        identity=replace(
            pair.identity,
            model_key="pythia70m",
            sae_key="pythia70m_resid_sm",
            layer=1,
            hook_name=pair.model.model.site_name,
        ),
    )
    loads = []
    probes = []

    def load(**kwargs):
        loads.append(kwargs)
        return pair

    monkeypatch.setattr(saebase.api, "load_registered", load)

    def precheck(cfg):
        probes.append(1)
        assert InterveneTask().prepare_execution(cfg)["status"] == "PASS"
        return None, "synthetic-precheck"

    monkeypatch.setattr("tasks.precheck.gate.automatic_precheck", precheck)
    config = {
        "model_key": "pythia70m",
        "sae_key": "pythia70m_resid_sm",
        "layer": 1,
        "samples": [{"sample_id": "synthetic", "token_ids": [1, 2, 3], "target_token_id": 0}],
        "conditions": [{"name": "identity", "operation": "identity"}],
    }
    source = tmp_path / "synthetic-edits.json"
    source.write_text(json.dumps(config))
    args = ["--config", str(source), "--real", "--device", "cpu", "--output", str(tmp_path)]
    assert cli.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["completion"]["completed"] == 1
    before = len(loads)
    assert cli.main(args) == 0
    capsys.readouterr()
    assert len(loads) == before and len(probes) == 1

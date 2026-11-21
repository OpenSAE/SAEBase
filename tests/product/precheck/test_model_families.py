"""All-model behavior contracts recorded before the family refactor; no weights loaded."""

import dataclasses
import importlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/model_families"
MAPPING = json.loads((FIXTURES / "mapping.json").read_text())
BEFORE = json.loads((FIXTURES / "behavior.json").read_text())


def norm(value):
    if dataclasses.is_dataclass(value):
        return norm(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [norm(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@pytest.mark.parametrize("key", sorted(MAPPING))
@pytest.mark.parametrize("local", [False, True])
def test_all_model_loading_behavior(key, local, monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: local)
    series = importlib.reload(importlib.import_module(MAPPING[key]["series"] + ".specs"))
    module = getattr(series, key)

    def capture(backend):
        return lambda *a, **kw: dict(backend=backend, args=norm(a), kwargs=norm(kw))

    monkeypatch.setattr("models.hooked_lm_common.load_from_spec", capture("load_from_spec"))
    monkeypatch.setattr(
        "models.hf_hooks_backend.load_hf_hooks_model", capture("load_hf_hooks_model")
    )
    monkeypatch.setattr(
        "models.hf_logits_backend.load_hf_logits_model", capture("load_hf_logits_model")
    )
    out = dict(spec=norm(module.spec), calls=[])
    monkeypatch.setattr(Path, "is_file", lambda self: local)
    for cuda in [False, True]:
        monkeypatch.setattr("torch.cuda.is_available", lambda: cuda)
        for dev in [None, "cpu", "cuda:1", ""]:
            out["calls"].append(norm(module.load_model(dev)))
        for data in [{}, {"device": None}, {"device": "cpu"}, {"device": ""}]:
            cfg = OmegaConf.create({"model": data})
            out["calls"].append(norm(module.construct(cfg)))
            out["calls"].append(norm(module.construct_lazy(cfg)))
    expected = json.loads(json.dumps(BEFORE[key][str(local)]))
    expected.pop("compat", None)
    # Keep weight identities, precision and processing choices.
    # Only default placement changes under the GPU-first policy; explicit choices remain.
    for offset, available in [(0, False), (12, True)]:
        automatic = "cuda" if available else "cpu"
        devices = [
            automatic,
            "cpu",
            "cuda:1",
            automatic,
            automatic,
            automatic,
            automatic,
            automatic,
            "cpu",
            "cpu",
            automatic,
            automatic,
        ]
        for index, device in enumerate(devices):
            call = expected["calls"][offset + index]
            (call["kwargs"] if "kwargs" in call else call)["device"] = device
    assert out == expected


def test_candidate_fallback_and_alias_failure_paths(monkeypatch):
    from models import hooked_lm_common as common

    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        if name == "bad":
            raise ImportError("missing optional backend")
        return object()

    monkeypatch.setattr(common, "load_hooked_model", load)
    assert common.load_hooked_model_with_candidates(("bad", "ok"), device="cpu")[1] == "ok"
    assert [c[0] for c in calls] == ["bad", "ok"]
    with pytest.raises(RuntimeError, match="missing optional backend"):
        common.load_hooked_model_with_candidates(("bad",), device="cpu")
    monkeypatch.setattr(common, "load_hf_causal_lm", lambda name, **kw: ("weights", "tokenizer"))
    common.load_hooked_model_with_candidates(
        ("mirror",), device="cpu", architecture_aliases={"mirror": "arch"}
    )
    assert calls[-1][1]["architecture_name"] == "arch"
    assert calls[-1][1]["hf_model"] == "weights"


def test_loading_policies_preserve_architecture_precision_and_processing():
    from models.loading_policy import default_load_kwargs

    expected = json.loads((FIXTURES / "loading_policy.json").read_text())
    for name, options in expected.items():
        assert {k: str(v) for k, v in default_load_kwargs(name).items()} == options

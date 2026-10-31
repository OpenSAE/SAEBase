"""Contract tests for saebase.generate (no vLLM install / GPU required)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from saebase.generate import generate, packed_row, resolve_backend, sampling_params_kwargs

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "tasks"

LEGACY_HF_OR_VLLM_IMPORT = {
    "01_asymmetry_symmetry_division/traces.py",
    "03_logic_chain_validation/driver.py",
    "03_logic_chain_validation/generate.py",
    "03_logic_chain_validation/gpu_hf.py",
    "03_logic_chain_validation/hf_runtime.py",
    "03_logic_chain_validation/intervene.py",
    "04_chain_guidance/driver.py",
    "04_chain_guidance/gpu_pass.py",
    "06_ordered_steering/generate_steered.py",
    "07_feature_selection_order/e1.py",
    "08_conditional_repair_gating/generate_repair.py",
    "vllm_backend/__init__.py",
    "vllm_backend/generate.py",
}

BYPASS_PATTERNS = (
    re.compile(r"\bgenerate_hf\b"),
    re.compile(r"\bgenerate_hf_packed\b"),
    re.compile(r"from vllm import"),
    re.compile(r"(?m)^import vllm\b"),
    re.compile(r"model\.generate\("),
)


def test_packed_row_keys():
    row = packed_row(prompt="hi", token_ids=[1, 2, 3, 4], prompt_len=2, text="ab")
    assert row["prompt"] == "hi"
    assert row["prompt_len"] == 2
    assert row["n_gen_tokens"] == 2
    assert row["token_ids"] == [1, 2, 3, 4]
    assert set(row) >= {
        "prompt",
        "token_ids",
        "prompt_len",
        "token_strings",
        "reasoning_mask",
        "text",
        "n_gen_tokens",
    }


def test_sampling_params_are_caller_supplied():
    kwargs = sampling_params_kwargs(max_new_tokens=4096, temperature=0.6, top_p=0.95)
    assert kwargs == {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096}


def test_resolve_backend_defaults_to_vllm_without_package(monkeypatch):
    monkeypatch.delenv("SAE_GENERATE_BACKEND", raising=False)
    monkeypatch.setattr("saebase.generate.api.vllm_available", lambda: False)
    assert resolve_backend() == "vllm"


def test_resolve_backend_honors_hf_env(monkeypatch):
    monkeypatch.setenv("SAE_GENERATE_BACKEND", "hf")
    assert resolve_backend() == "hf"


def test_generate_rejects_hf_env(monkeypatch):
    monkeypatch.setenv("SAE_GENERATE_BACKEND", "hf")
    with pytest.raises(RuntimeError, match="does not run HuggingFace"):
        generate(["hello"], max_new_tokens=8, temperature=0.6, top_p=0.95, model="x")


def test_generate_without_engine_and_autostart_raises(monkeypatch):
    monkeypatch.delenv("SAE_GENERATE_BACKEND", raising=False)
    monkeypatch.setenv("SAE_GENERATE_AUTOSTART", "0")
    monkeypatch.setattr("saebase.generate.engine.vllm_available", lambda: False)
    monkeypatch.setattr("saebase.generate.server.server_healthy", lambda timeout_s=2.0: False)
    with pytest.raises(RuntimeError, match="not in this interpreter"):
        generate(["hello"], max_new_tokens=8, temperature=0.6, top_p=0.95, model="x")


def test_generate_http_path_without_inprocess_vllm(monkeypatch):
    monkeypatch.delenv("SAE_GENERATE_BACKEND", raising=False)
    monkeypatch.setattr("saebase.generate.engine.vllm_available", lambda: False)
    monkeypatch.setattr("saebase.generate.server.server_healthy", lambda timeout_s=2.0: True)

    def fake_post(body):
        assert body["prompt"] == ["p0", "p1"]
        assert body["max_tokens"] == 8
        return {
            "choices": [
                {"index": 1, "text": " B"},
                {"index": 0, "text": " A"},
            ]
        }

    monkeypatch.setattr("saebase.generate.server.post_completions", fake_post)
    monkeypatch.setattr("saebase.generate.engine._http_tokenizer", lambda model: None)
    rows = generate(
        ["p0", "p1"],
        max_new_tokens=8,
        temperature=0.6,
        top_p=0.95,
        model="dummy",
    )
    assert [r["text"] for r in rows] == [" A", " B"]
    assert rows[0]["n_gen_tokens"] == 0


def test_generate_vllm_import_error_is_explicit():
    from saebase.generate import generate_vllm

    try:
        import vllm  # noqa: F401
    except ImportError:
        try:
            generate_vllm(["hello"], max_new_tokens=8, temperature=0.6, top_p=0.95, model="x")
        except RuntimeError as exc:
            assert "vLLM is not installed" in str(exc)
            assert ".[generation]" in str(exc)
        else:
            raise AssertionError("expected RuntimeError without vLLM")
    else:
        pass


def test_generate_vllm_requires_model_or_llm():
    from saebase.generate import generate_vllm

    try:
        import vllm  # noqa: F401
    except ImportError:
        return
    try:
        generate_vllm(["hello"], max_new_tokens=8, temperature=0.6, top_p=0.95)
    except ValueError as exc:
        assert "model=" in str(exc)
    else:
        raise AssertionError("expected ValueError without model/llm")


def test_new_tasks_do_not_bypass_saebase_generate():
    offenders: list[str] = []
    for path in sorted(TASKS.rglob("*.py")):
        rel = path.relative_to(TASKS).as_posix()
        if rel in LEGACY_HF_OR_VLLM_IMPORT:
            continue
        text = path.read_text(encoding="utf-8")
        for pat in BYPASS_PATTERNS:
            if pat.search(text):
                offenders.append(f"{rel}: {pat.pattern}")
    assert not offenders, "new task files must call saebase.generate.generate():\n" + "\n".join(
        offenders
    )

"""Phase C research integration & owned-resource tests. CPU-only, offline."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter, SAEIdentity, capture_features, edit_features
from saebase.integrations import (
    collect_rss_features,
    evaluate_rss,
    evaluate_sar,
    load_paper_module,
)


class CountingSAR:
    def __init__(self):
        self.count = 0
        self.sampled = None
        self.fitted = None

    def coalitions(self, K, M, p, n_fit, seed):
        self.count += 1
        rng = np.random.default_rng(seed)
        masks = rng.random((M, K)) < p
        order = rng.permutation(M)
        self.sampled = (masks, order[:n_fit], order[n_fit:])
        return self.sampled

    def sar(self, responses, masks, fit, held, g_full=None):
        self.fitted = (responses.copy(), masks.copy(), fit.copy(), held.copy(), g_full)
        return {"sar1": 0.21, "sar2": 0.04, "rho2": float("nan")}


class CountingRSS:
    def __init__(self):
        self.select_changes = None
        self.held_changes = None
        self.calls = 0

    def rss(self, delta, rho, kappa, tau, M, K):
        self.select_changes = delta.copy()
        self.calls += 1
        return np.array([0, 1]), np.array([1, 1]), 0.25

    def held_out_rate(self, delta, support, direction, tau):
        self.held_changes = delta.copy()
        self.calls += 1
        return (delta[:, support] * direction > tau).mean(axis=0)


class ToyHFHooks(ResearchToyModel):
    _saebase_backend = "hf_hooks"

    def get_resid(self, tokens, layer, offsets):
        assert layer == 0
        hidden = self._hidden(tokens)
        return torch.stack([hidden[0, -1 - o] for o in offsets])

    def run_with_patch(self, tokens, layer, patches, *, readout_offset=0):
        assert layer == 0
        hidden = self._hidden(tokens).clone()
        for off, value in patches.items():
            hidden[0, -1 - off] = value
        return self._logits(hidden)[0, -1 - readout_offset]


def pair(*, hook_kind="hook_resid_post", hf=False):
    m = ToyHFHooks() if hf else ResearchToyModel(hook_kind=hook_kind)
    model = ModelAdapter(m.eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, f"blocks.0.{hook_kind}")
    return model, sae, site


@pytest.mark.parametrize(
    "hf,hook_kind", [(False, "hook_resid_post"), (False, "hook_mlp_out"), (True, "hook_resid_post")]
)
def test_sar_patching_and_exact_paper_call(hf, hook_kind):
    model, sae, site = pair(hf=hf, hook_kind=hook_kind)
    paper = CountingSAR()
    result = evaluate_sar(
        model,
        sae,
        site=site,
        paper=paper,
        prompt="2 3 4",
        readout=lambda v: v[0] - v[1],
        K=3,
        M=64,
        n_fit=48,
        seed=3,
    )
    assert paper.count == 1
    assert result.masks.shape == (64, 3)
    assert (result.fit_indices == paper.sampled[1]).all()
    assert (result.held_indices == paper.sampled[2]).all()
    assert result.metrics["sar1"] == 0.21
    # Undefined paper metric keeps None only in a JSON-safe summary.
    assert np.isnan(result.metrics["rho2"])
    assert result.summary()["metrics"]["rho2"] is None
    assert len(result.feature_ids) == 3
    g = paper.fitted[0]
    snapshot = capture_features(model, sae, site=site, prompt="2 3 4")
    for j in (0, 1, 2, 11, 31, 63):
        removals = {
            int(idx): 0.0 for idx, flag in zip(result.feature_ids, result.masks[j]) if not flag
        }
        modified = edit_features(sae, snapshot, replacements=removals)
        logits = model.patch_logits(
            tokens=snapshot.tokens, site=site, activation=modified.activation
        )
        expected = float(logits[0] - logits[1])
        assert g[j] == pytest.approx(expected, abs=1e-6)
        assert torch.allclose(
            modified.activation - sae.decode(modified.codes[None])[0], snapshot.residual, atol=1e-6
        )
    assert result.full_response == pytest.approx(
        float(
            model.forward_logits(tokens=snapshot.tokens)[0]
            - model.forward_logits(tokens=snapshot.tokens)[1]
        )
    )
    assert result.source_tokens.tolist() == [[2, 3, 4]]
    assert json.loads(json.dumps(result.summary(), allow_nan=False))["n_coalitions"] == 64


def test_sar_rejects_nonaffine_decoder():
    model, sae, site = pair()
    orig = sae.decode
    sae.decode = lambda z: (
        orig(z) + torch.cat([(z[..., 1:2] * z[..., 6:7]), torch.zeros_like(z[..., 1:4])], dim=-1)
    )
    with pytest.raises(ValueError, match="not additive"):
        evaluate_sar(
            model,
            sae,
            site=site,
            paper=CountingSAR(),
            prompt="2 3 4",
            readout=lambda x: x[0] - x[1],
            K=3,
            M=48,
            n_fit=32,
        )


def test_sar_readout_and_provenance_errors():
    model, sae, site = pair()
    paper = CountingSAR()
    identity = SAEIdentity.for_custom(
        sae=sae, model_key="tiny", checkpoint="toy", hook_name=site.hook_name, layer=0
    )
    result = evaluate_sar(
        model,
        sae,
        site=site,
        paper=paper,
        prompt="2 3 4",
        readout=lambda x: x[0],
        K=3,
        M=48,
        n_fit=32,
        identity=identity,
    )
    assert result.summary()["identity"]["checkpoint"] == "toy"
    with pytest.raises(ValueError, match="site differs"):
        evaluate_sar(
            model,
            sae,
            site=ActivationSite(0, "blocks.0.hook_mlp_out"),
            paper=paper,
            prompt="2 3 4",
            readout=lambda x: x[0],
            K=3,
            M=48,
            n_fit=32,
            identity=identity,
        )
    with pytest.raises(ValueError, match="readout"):
        evaluate_sar(
            model,
            sae,
            site=site,
            paper=paper,
            prompt="2 3 4",
            readout=lambda x: np.array([1.0, 2.0]),
            K=3,
            M=48,
            n_fit=32,
        )
    with pytest.raises(ValueError, match="K must"):
        evaluate_sar(
            model, sae, site=site, paper=paper, prompt="2 3 4", readout=lambda x: x[0], K=1
        )


@pytest.mark.parametrize(
    "hf,hook_kind", [(False, "hook_resid_post"), (False, "hook_mlp_out"), (True, "hook_resid_post")]
)
def test_rss_selection_heldout_and_source_split(hf, hook_kind):
    model, sae, site = pair(hf=hf, hook_kind=hook_kind)
    sel = [f"1 2 {i}" for i in range(4, 9)]
    held = [f"1 2 {i}" for i in range(9, 14)]
    features = collect_rss_features(
        model,
        sae,
        site=site,
        clean_prompt="1 2 3",
        selection_variants=sel,
        held_out_variants=held,
    )
    assert features.selection_changes.shape == (5, 7)
    assert features.held_out_changes.shape == (5, 7)
    p = CountingRSS()
    result = evaluate_rss(features, paper=p, K=4, M=20)
    assert p.calls == 2
    assert np.array_equal(p.select_changes, features.selection_changes)
    assert np.array_equal(p.held_changes, features.held_out_changes)
    assert np.allclose(result.held_out_rates, 1.0)
    assert json.loads(json.dumps(result.summary(), allow_nan=False))["n_held_out"] == 5
    assert features.clean_position == 2
    assert features.selection_positions == (2,) * 5


def test_rss_heldout_edit_does_not_affect_selection_or_support():
    model, sae, site = pair()
    src = collect_rss_features(
        model,
        sae,
        site=site,
        clean_prompt="1 2 3",
        selection_variants=["1 2 4", "1 2 5", "1 2 6"],
        held_out_variants=["1 2 7", "1 2 8"],
    )
    p = CountingRSS()
    result1 = evaluate_rss(src, paper=p)
    from dataclasses import replace

    altered = replace(src, held_out_changes=src.held_out_changes * (-10000))
    result2 = evaluate_rss(altered, paper=p)
    assert np.array_equal(result1.support, result2.support)
    assert np.array_equal(result1.direction, result2.direction)
    assert result1.mean_rss == result2.mean_rss
    assert not np.array_equal(result1.held_out_rates, result2.held_out_rates)


def test_rss_rejects_leakage_and_invalid_position():
    model, sae, site = pair()
    with pytest.raises(ValueError, match="disjoint"):
        collect_rss_features(
            model,
            sae,
            site=site,
            clean_prompt="1 2 3",
            selection_variants=["1 2 4"],
            held_out_variants=["1 2 4"],
        )
    with pytest.raises(ValueError, match="disjoint"):
        collect_rss_features(
            model,
            sae,
            site=site,
            clean_prompt="1 2 3",
            selection_variants=["1 2 3"],
            held_out_variants=["1 2 4"],
        )
    with pytest.raises(ValueError, match="positions"):
        collect_rss_features(
            model,
            sae,
            site=site,
            clean_prompt="1 2 3",
            selection_variants=["1 2 4"],
            held_out_variants=["1 2 5"],
            selection_positions=[-1, -2],
        )
    with pytest.raises(IndexError, match="outside"):
        collect_rss_features(
            model,
            sae,
            site=site,
            clean_prompt="1 2 3",
            selection_variants=["1 2 4"],
            held_out_variants=["1 2 5"],
            selection_positions=[9],
        )


def test_rss_explicit_alignment_positions():
    model, sae, site = pair()
    result = collect_rss_features(
        model,
        sae,
        site=site,
        clean_prompt="1 2 3",
        selection_variants=["1 2 3 5"],
        held_out_variants=["1 2 3 6"],
        selection_positions=[-2],
        held_out_positions=[-2],
    )
    assert result.clean_position == 2
    assert result.selection_positions == result.held_out_positions == (2,)
    assert np.allclose(result.selection_changes, 0, atol=1e-8)


def test_rss_empty_support_json_safe():
    class NoSupport:
        rss = staticmethod(
            lambda *a, **kw: (np.array([], dtype=int), np.array([], dtype=int), float("nan"))
        )
        held_out_rate = staticmethod(
            lambda delta, support, direction, tau: np.array([], dtype=float)
        )

    model, sae, site = pair()
    features = collect_rss_features(
        model,
        sae,
        site=site,
        clean_prompt="1 2 3",
        selection_variants=["1 2 4"],
        held_out_variants=["1 2 5"],
    )
    result = evaluate_rss(features, paper=NoSupport())
    assert result.summary()["mean_rss"] is None
    json.dumps(result.summary(), allow_nan=False)


def test_unmodified_module_loaded_explicitly(tmp_path):
    src = tmp_path / "sar.py"
    src.write_text(
        "def coalitions(*args, **kw): return ([], [], [])\ndef sar(*args, **kw): return {}\n"
    )
    loaded = load_paper_module(tmp_path, paper="sar")
    assert callable(loaded.sar) and callable(loaded.coalitions)
    with pytest.raises(FileNotFoundError, match="sar.py"):
        load_paper_module(tmp_path / "missing", paper="sar")
    with pytest.raises(ValueError, match="paper must"):
        load_paper_module(tmp_path, paper="other")


def test_configured_second_site_does_not_claim_downloads():
    from saebase.sae.registry import get_profile

    first = get_profile("pythia70m_resid_sm")
    second = get_profile("gemma270m_scope_resid")
    assert first.model_key != second.model_key
    assert first.hook_name(3) == "blocks.3.hook_resid_post"
    assert second.hook_name(5) == "blocks.5.hook_resid_post"
    assert first.loader == second.loader == "sae_lens"
    assert second.sae_release and second.sae_id(5)


def test_generator_cleanup_touches_only_owned_resources(monkeypatch):
    import saebase.generate.server as server
    import saebase.generate.engine as engine
    import saebase.generate.gpu as gpu

    calls = []

    class Owned:
        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout):
            calls.append("wait")
            return 0

    server._PROC = Owned()
    monkeypatch.setattr(engine, "release", lambda: calls.append("release"))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("global process scan")),
    )
    gpu.stop_vllm()
    assert calls == ["release", "terminate", "wait"]
    assert server._PROC is None
    gpu.stop_vllm()
    assert calls == ["release", "terminate", "wait", "release"]


def test_generator_cleanup_kills_only_owned_timeout(monkeypatch):
    import saebase.generate.server as server

    events = []

    class Owned:
        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            if "kill" not in events:
                raise subprocess.TimeoutExpired("owned", timeout)
            return 0

        def kill(self):
            events.append("kill")

    server._PROC = Owned()
    assert server.stop_owned_server(timeout_s=0.01)
    assert events == ["terminate", "kill"]
    assert server._PROC is None


def test_worker_generate_fails_before_gpu_release(monkeypatch):
    import saebase.generate.launch as launch

    monkeypatch.setattr(
        launch,
        "_release_parent_gpu",
        lambda: (_ for _ in ()).throw(AssertionError("would release")),
    )
    with pytest.raises(RuntimeError, match="not shipped"):
        launch.run_worker(["generate", "--model-id", "something"])
    # Smoke is a separate command, source-independent.
    launch._verify_available_worker_mode(["smoke", "--model-id", "anything"])


def test_streaming_worker_silent_timeout(monkeypatch, tmp_path):
    import saebase.generate.launch as launch

    monkeypatch.setattr(launch, "_release_parent_gpu", lambda: None)
    monkeypatch.setattr(launch, "_child_env", lambda: os.environ.copy())
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        launch, "worker_cmd", lambda argv: [sys.executable, "-c", "import time;time.sleep(10)"]
    )
    with pytest.raises(TimeoutError, match="exceeded"):
        launch.run_worker_stream(["smoke"], timeout=0.2)


def test_cached_generator_uses_all_runtime_settings(monkeypatch):
    import types
    import saebase.generate.engine as engine
    import saebase.generate.settings as settings

    made = []

    class MockLLM:
        def __init__(self, **kwargs):
            made.append(kwargs)

    monkeypatch.setattr(settings, "apply_process_env", lambda: None)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=MockLLM))
    engine.release()
    first = engine._cached_llm(
        "model-a", gpu_memory_utilization=0.5, max_model_len=1024, dtype="float16"
    )
    assert first is engine._cached_llm(
        "model-a", gpu_memory_utilization=0.5, max_model_len=1024, dtype="float16"
    )
    second = engine._cached_llm(
        "model-a", gpu_memory_utilization=0.5, max_model_len=2048, dtype="float16"
    )
    assert second is not first
    assert len(made) == 2
    assert made[0]["max_model_len"] == 1024 and made[1]["max_model_len"] == 2048
    engine.release()


def test_fake_vllm_offline_contract(monkeypatch):
    import types
    from saebase.generate.vllm import generate_vllm

    class MockSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class MockTokeniser:
        def encode(self, x, add_special_tokens=False):
            return [len(x)]

        def decode(self, ids, skip_special_tokens=False):
            return f"decoded:{','.join(map(str, ids))}"

    class MockLLM:
        def generate(self, prompts, params):
            assert params.kwargs["seed"] == 0
            return [
                types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[20 + i], text="")])
                for i in range(len(prompts))
            ]

        def get_tokenizer(self):
            return MockTokeniser()

    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(LLM=MockLLM, SamplingParams=MockSamplingParams)
    )
    got = []
    rows = generate_vllm(
        ["hello", "world"],
        max_new_tokens=2,
        temperature=0,
        top_p=1,
        llm=MockLLM(),
        seed=0,
        on_complete=lambda idx, row: got.append((idx, row["text"])),
    )
    assert rows[0]["text"] == "decoded:20"
    assert rows[1]["text"] == "decoded:21"
    assert rows[0]["n_gen_tokens"] == 1
    assert got == [(0, "decoded:20"), (1, "decoded:21")]

    class BadLLM(MockLLM):
        def generate(self, prompts, params):
            return []

    with pytest.raises(RuntimeError, match="2 prompts"):
        generate_vllm(["hello", "world"], max_new_tokens=2, temperature=0, top_p=1, llm=BadLLM())

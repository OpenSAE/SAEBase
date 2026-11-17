"""Worker policy lifecycle and actual CUDA batch invariance of shared operations."""

import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch

from models import numerics


def test_full_state_check_reloads_artifact_and_detects_unchecked_response_changes(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from tasks.capture.performance import AnswerReplay, prepare_activations, verify_batch_artifact
    from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE

    raw, sae = ResearchToyModel(), ResearchToySAE()
    pair = SimpleNamespace(model=ModelAdapter(raw), sae=sae, site=ActivationSite(0, raw.site_name))
    snapshot = capture_features(pair.model, sae, site=pair.site, prompt="0 1 2")
    masks = ((np.arange(4)[:, None] >> np.arange(2)) & 1).astype(bool)
    activations, inverse, _ = prepare_activations(sae, snapshot, [0, 1], masks)
    expected = AnswerReplay(pair, snapshot, [1], snapshot.activation)(activations, 2)[inverse]
    payload = {
        "source_tokens": snapshot.tokens.numpy(),
        "feature_ids": np.array([0, 1]),
        "masks": masks,
        "answer_token_ids": np.array([1]),
        "readout_activation": snapshot.activation.numpy(),
        "responses": expected,
    }
    artifact = tmp_path / "states.npz"
    np.savez(artifact, **payload)
    result = verify_batch_artifact(pair, artifact, batch_size=3)
    assert result["status"] == "PASS" and result["states_checked"] == 4
    payload["responses"] = expected.copy()
    payload["responses"][-1] += 1e-8
    np.savez(artifact, **payload)
    with pytest.raises(ValueError, match="Complete I artifact"):
        verify_batch_artifact(pair, artifact, batch_size=3)


def test_configured_only_keeps_cuda_and_environment_untouched(monkeypatch):
    monkeypatch.setattr(numerics, "_PROFILE", None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Configuration-only requests must not configure GPU execution")

    monkeypatch.setattr(numerics, "configure", unexpected)
    cfg = OmegaConf.create({"task": {"execute": False}, "runtime": {"numerics": "vllm"}})
    assert numerics.configure_task(cfg)["profile"] == "native"


def test_cpu_does_not_install_cuda_numerical_policy(monkeypatch):
    monkeypatch.setattr(numerics, "_PROFILE", None)
    before = dict(os.environ)
    assert numerics.configure("vllm", device="cpu")["profile"] == "native"
    assert dict(os.environ) == before


def test_saved_legacy_config_retains_its_native_policy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        numerics, "configure", lambda profile, **kwargs: calls.append((profile, kwargs))
    )
    cfg = OmegaConf.create({"task": {"execute": True, "device": "cuda"}})
    numerics.configure_task(cfg)
    assert calls == [("native", {"device": "cuda"})]


def test_qwen_scan_revision_does_not_invalidate_other_models():
    for name, expected in [("qwen35_4b", numerics.QWEN_POLICY), ("llama31_8b", numerics.POLICY)]:
        cfg = OmegaConf.create({"model": {"name": name}, "runtime": {"numerics": "vllm"}})
        numerics.bind_configuration(cfg)
        assert cfg.runtime.numerics_protocol == expected


def test_runtime_resume_cannot_skip_an_older_numerical_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from saebase.registry import TaskModule
    from saebase.results import EvaluationResult
    from saebase.runtime import experiment_results

    class SyntheticTask(TaskModule):
        name = "capture"

        def run(self, cfg):
            raise AssertionError("Metadata inspection must not execute a task")

    monkeypatch.setattr(
        "saebase.registry.get_task_plugin", lambda name: SimpleNamespace(task_class=SyntheticTask)
    )
    monkeypatch.setattr(experiment_results, "_source", lambda: {})
    cfg = OmegaConf.create(
        {
            "task": {"name": "capture"},
            "model": {"name": "synthetic"},
            "dataset": {"name": "synthetic"},
            "seed": 42,
            "runtime": {"mode": "overwrite", "numerics": "vllm", "numerics_protocol": "old"},
        }
    )
    result = EvaluationResult(
        ok=True,
        task="capture",
        evaluation="capture",
        model_key="synthetic",
        prompt_source="synthetic",
        status="executed",
    )
    experiment_results.record_run(tmp_path, cfg, result=result)
    cfg.runtime.mode = "resume"
    original = experiment_results.inspect_run(tmp_path, expected_cfg=cfg)
    assert original["action"] == "skip", original
    numerics.bind_configuration(cfg)
    assert cfg.runtime.numerics_protocol == numerics.POLICY
    state = experiment_results.inspect_run(tmp_path, expected_cfg=cfg)
    assert state["action"] == "retry" and state["reason"] == "configuration_mismatch"


def test_loaded_worker_cannot_mix_numerical_policies(monkeypatch):
    monkeypatch.setattr(numerics, "_PROFILE", "native")
    assert numerics.configure("native")["profile"] == "native"
    with pytest.raises(RuntimeError, match="cannot change"):
        numerics.configure("vllm")


def test_late_initialization_explains_how_to_recover(monkeypatch):
    monkeypatch.setattr(numerics, "_PROFILE", None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="start a new worker"):
        numerics.configure("vllm")
    assert numerics.identity()["profile"] == "native"


def test_runtime_shares_one_loaded_pair_and_consumes_precheck_recommendation(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from saebase.registry import TaskModule
    from saebase.results import EvaluationResult
    from saebase.runner import run_task
    from saebase.runtime.performance import current, remember_recommendations

    monkeypatch.chdir(tmp_path)
    events = []
    active = []

    @contextmanager
    def loaded(pair):
        active.append(pair)
        yield
        active.pop()

    def load(cfg):
        events.append("load")
        return object()

    def precheck(cfg):
        assert active
        events.append("precheck")
        remember_recommendations(
            [
                {
                    "optimizations": [
                        {
                            "policy_kind": "answer",
                            "policy_key": ["same-pair"],
                            "selected": True,
                            "selected_batch_size": 16,
                        }
                    ]
                }
            ]
        )
        return None, "precheck.json"

    class Task(TaskModule):
        name = "capture"

        def run(self, cfg):
            assert active
            assert current().answer_policies[("same-pair",)] == 16
            events.append("run")
            return EvaluationResult(
                ok=True,
                task="capture",
                evaluation="capture",
                model_key="synthetic",
                prompt_source="synthetic",
                status="executed",
            )

    monkeypatch.setattr("tasks.sae_runtime._pair", load)
    monkeypatch.setattr("tasks.sae_runtime.loaded_pair_session", loaded)
    monkeypatch.setattr("tasks.precheck.gate.automatic_precheck", precheck)
    cfg = OmegaConf.create(
        {
            "model": {"name": "synthetic", "device": "cpu"},
            "dataset": {"name": "synthetic"},
            "task": {
                "name": "capture",
                "execute": True,
                "device": "cpu",
                "performance": {"enabled": True},
            },
        }
    )
    assert run_task(cfg, task=Task()).ok
    assert events == ["load", "precheck", "run"]
    assert not active


def test_strict_invariance_rejects_drift_instead_of_accepting_scalar_fallback(monkeypatch):
    import numpy as np
    from saebase.runtime.performance import settings, tune, numeric_validation

    monkeypatch.setattr(numerics, "_PROFILE", "vllm")
    options = settings(
        {"task": {"performance": {"warmup": 0, "repeats": 1, "batch_sizes": [1, 2]}}}
    )
    assert options["atol"] == options["logit_atol"] == options["rtol"] == 0
    reference = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Batch invariance failed"):
        tune(
            lambda batch: reference + (0 if batch == 1 else 1e-9),
            lambda: reference,
            2,
            options,
            validate=lambda a, b: numeric_validation(a, b, options),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA operations required")
def test_cuda_scores_are_exact_across_batches_positions_and_short_final_batches():
    # A numerical worker must start before CUDA initialization; isolate from
    # other tests that have already created GPU tensors in this pytest process.
    source = """
import torch
from saebase._vendor.vllm_scan import stable_chunk_cumsum
from models.numerics import configure
configure('vllm')
torch.manual_seed(12)
n_rows = 65
with torch.inference_mode():
    scans = torch.randn(65, 32, 3, 64, device='cuda')
    expected_scan = torch.cat([stable_chunk_cumsum(scans[i:i+1]) for i in range(65)])
    torch.testing.assert_close(expected_scan.cpu(), scans.cpu().cumsum(-1), atol=1e-5, rtol=1e-5)
    for batch in [2, 4, 8, 16, 32, 64, 65]:
        actual = torch.cat([stable_chunk_cumsum(scans[i:i+batch]) for i in range(0,65,batch)])
        assert torch.equal(actual, expected_scan), (batch, 'GDN scan')
    for dtype in [torch.float32, torch.bfloat16]:
        for length in [1, 7, 65]:
            inputs = torch.randn(n_rows, length, 128, device='cuda', dtype=dtype)
            weight = torch.randn(128, 257, device='cuda', dtype=dtype)
            def score(order, batch):
                result = []
                for start in range(0, len(order), batch):
                    logits = inputs[order[start:start+batch]] @ weight
                    values = logits.float().log_softmax(-1)[:, :, 3].double().mean(-1)
                    result.append(values)
                return torch.cat(result)
            order = torch.arange(n_rows, device='cuda')
            reference = score(order, 1)
            for batch in [2, 4, 8, 16, 32, 64, 65]:
                actual = score(order, batch)
                assert torch.equal(reference, actual), (dtype, length, batch, (reference-actual).abs().max())
                reverse = order.flip(0)
                actual = score(reverse, batch).flip(0)
                assert torch.equal(reference, actual), (dtype, length, batch, 'positions')
    for dtype in [torch.float32, torch.bfloat16]:
        q, k, v = [torch.randn(n_rows, 4, 65, 64, device='cuda', dtype=dtype) for _ in range(3)]
        reference = torch.cat([
            torch.nn.functional.scaled_dot_product_attention(q[i:i+1], k[i:i+1], v[i:i+1], is_causal=True)
            for i in range(n_rows)
        ])
        for batch in [2, 4, 8, 16, 32, 64, 65]:
            output = torch.cat([
                torch.nn.functional.scaled_dot_product_attention(q[i:i+batch], k[i:i+batch], v[i:i+batch], is_causal=True)
                for i in range(0, n_rows, batch)
            ])
            assert torch.equal(reference, output), (dtype, batch, 'attention', (reference-output).abs().max())
    for dtype in [torch.float16, torch.bfloat16]:
        values = torch.randn(65, 257, device='cuda', dtype=dtype) * 4
        expected = torch.softmax(values.float().cpu(), -1).to(dtype).cuda()
        actual = torch.softmax(values, -1)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual.float(), expected.float(), atol=2e-7, rtol=0.008)
        promoted = torch.softmax(values, -1, dtype=torch.float32)
        assert promoted.dtype == torch.float32
        torch.testing.assert_close(promoted.cpu(), torch.softmax(values.float().cpu(), -1), atol=2e-7, rtol=1e-6)
print('EXACT_BATCH_INVARIANCE_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "OMP_NUM_THREADS": "4"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXACT_BATCH_INVARIANCE_PASS" in result.stdout

"""Ensure deleted SAE package has one correctly owned implementation per capability."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_standalone_sae_package_remains():
    assert not (ROOT / "saebase/sae").exists()


def test_sae_capabilities_belong_to_shared_api_or_task():
    from saebase.api.sae_profiles import get_profile
    from saebase.api.sae_loading import load_model_and_sae
    from tasks.benchmark.activation_health import run_activation_health
    from tasks.train.external_training import _saelens_train

    assert callable(get_profile)
    assert callable(load_model_and_sae)
    assert callable(run_activation_health)
    assert callable(run_external_training)


def test_precheck_owns_all_real_sae_smoke_execution():
    assert not (ROOT / "saebase/setup/smoke.py").exists()
    assert (ROOT / "tasks/precheck/worker.py").is_file()

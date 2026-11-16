"""Public SAE capabilities have domain implementations without a second package."""


def test_sae_capabilities_belong_to_shared_api_or_task():
    from saebase.api.sae_loading import load_model_and_sae
    from saebase.api.sae_profiles import get_profile
    from tasks.benchmark.metrics.health import run_activation_health
    from tasks.train.backend.training import run_training

    assert callable(get_profile)
    assert callable(load_model_and_sae)
    assert callable(run_activation_health)
    assert callable(run_training)

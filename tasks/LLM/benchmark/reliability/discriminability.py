"""Known controllable degradation, paired ground-truth uncertainty and rankings."""

from __future__ import annotations
import numpy as np

from .ground_truth import make_factors, fit_matching, evaluate_matching, reconstruction_mse
from .bootstrap import paired_bootstrap
from .ranking import rank_stability
from .schema import Observation, AuditConfig


def _observations(truth, codes, *, sae_id: str, evaluation_seed: int):
    reconstructed = codes @ truth.dictionary.T
    exact = truth.test @ truth.dictionary.T
    errors = np.mean(np.square(reconstructed - exact), axis=1)
    return [
        Observation(
            model_id="analytic-factors",
            sae_id=sae_id,
            metric_id="core.reconstruction_mse",
            dataset_id=f"synthetic-{truth.scenario}-source{truth.seed}",
            split="test",
            prompt_group_id=group,
            sample_id=group,
            value=float(value),
            seed=evaluation_seed,
        )
        for group, value in zip(truth.test_ids, errors)
    ]


def evaluate_degradation(config: AuditConfig, *, fixture_seed: int = 17) -> dict:
    """Frozen synthetic factors, varied evaluation noise with recorded seeds.

    Model/SAE training is not performed. The same held-out sample groups
    receive matched controllable noise, and any feature alignment is
    selected only on the independent calibration split.
    """
    scenarios = {}
    for scenario in ("independent", "correlated", "hierarchical", "superposition"):
        truth = make_factors(scenario, seed=fixture_seed)
        trials = []
        for evaluation_seed in config.evaluation_seeds:
            rng = np.random.default_rng(evaluation_seed + 1000)
            calibration_noise = rng.normal(size=truth.calibration.shape)
            test_noise = rng.normal(size=truth.test.shape)
            quality = {}
            for label, scale in (("oracle", 0.0), ("mild", 0.2), ("degraded", 0.85)):
                candidate_cal = truth.calibration + scale * calibration_noise
                candidate_test = truth.test + scale * test_noise
                mapping = fit_matching(truth.calibration, candidate_cal)
                score = evaluate_matching(truth.test, candidate_test, mapping)
                quality[label] = {
                    "feature_recovery": score["score"],
                    "reconstruction_mse": reconstruction_mse(
                        truth.test, truth.dictionary, candidate_test
                    ),
                    "matching_split": "calibration",
                    "evaluation_split": "test",
                }
            # Real measured error must increase under this constructed
            # degradation; no fabricated numerical separability guarantee.
            passed = (
                quality["oracle"]["reconstruction_mse"] < 1e-15
                and quality["mild"]["reconstruction_mse"] > 1e-8
                and quality["degraded"]["reconstruction_mse"]
                > quality["mild"]["reconstruction_mse"]
                and quality["oracle"]["feature_recovery"] > 0.999
                and quality["degraded"]["feature_recovery"] < quality["oracle"]["feature_recovery"]
            )
            observations = {
                "oracle": _observations(
                    truth, truth.test, sae_id="oracle", evaluation_seed=evaluation_seed
                ),
                "mild": _observations(
                    truth,
                    truth.test + 0.2 * test_noise,
                    sae_id="mild",
                    evaluation_seed=evaluation_seed,
                ),
                "degraded": _observations(
                    truth,
                    truth.test + 0.85 * test_noise,
                    sae_id="degraded",
                    evaluation_seed=evaluation_seed,
                ),
            }
            # Resampling for a fixed synthetic evaluation seed must not mix
            # with other training/evaluation seed samples.
            one = AuditConfig(
                evaluation_seeds=(evaluation_seed,),
                bootstrap_replicates=config.bootstrap_replicates,
                confidence=config.confidence,
                sample_sizes=(),
                top_k=config.top_k,
                training_seed=config.training_seed,
            )
            paired = paired_bootstrap(observations["oracle"], observations["degraded"], one)
            ranking = rank_stability(observations, one)
            passed = passed and paired["status"] == "completed" and paired["interval"][1] < 0
            passed = passed and ranking["status"] == "completed"
            trials.append(
                {
                    "evaluation_seed": evaluation_seed,
                    "status": "passed" if passed else "failed",
                    "quality": quality,
                    "paired": paired,
                    "ranking": ranking,
                }
            )
        scenarios[scenario] = {
            "status": "passed" if all(t["status"] == "passed" for t in trials) else "failed",
            "scenario": scenario,
            "fixture_seed": fixture_seed,
            "n_calibration": len(truth.calibration),
            "n_test": len(truth.test),
            "dictionary_shape": list(truth.dictionary.shape),
            "trials": trials,
        }
    return scenarios

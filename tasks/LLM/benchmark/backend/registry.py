"""Metric metadata only. Mathematical estimators reside in their own source packages."""

from __future__ import annotations
from .schema import MetricSpec

_METRICS: dict[str, MetricSpec] = {}


def register_metric(spec: MetricSpec) -> None:
    if spec.metric_id in _METRICS:
        raise ValueError(f"metric already registered: {spec.metric_id}")
    _METRICS[spec.metric_id] = spec


def get_metric(metric_id: str) -> MetricSpec:
    if metric_id not in _METRICS:
        raise ValueError(f"unregistered metric {metric_id!r}; available: {sorted(_METRICS)}")
    return _METRICS[metric_id]


def metric_specs() -> tuple[MetricSpec, ...]:
    return tuple(_METRICS[k] for k in sorted(_METRICS))


# None of these scores has a cross-metric globally meaningful direction.
for name, inputs, unit, direction in (
    ("core.l0", ("m1_codes",), "token", "contextual"),
    ("core.reconstruction_mse", ("m1_activations", "m1_codes"), "token", "lower"),
    ("core.fvu", ("m1_activations", "m1_codes"), "token", "lower"),
    ("core.explained_variance", ("m1_activations", "m1_codes"), "token", "higher"),
    ("reconstruction.target_nll", ("model_reconstruction",), "prompt", "lower"),
    (
        "reconstruction.loss_recovered",
        ("model_reconstruction", "model_zero_ablation"),
        "prompt",
        "contextual",
    ),
    ("intervention.logit_delta_l2", ("m2_measurements",), "measurement", "contextual"),
    ("intervention.logit_delta_max_abs", ("m2_measurements",), "measurement", "contextual"),
    ("intervention.kl_baseline_to_edited", ("m2_measurements",), "measurement", "lower"),
    ("intervention.activation_delta_l2", ("m2_measurements",), "measurement", "contextual"),
    ("intervention.target_logit_delta", ("m2_measurements",), "measurement", "contextual"),
    ("intervention.edited_target_top1", ("m2_measurements",), "measurement", "higher"),
    ("sar.sar1", ("paper_sar",), "coalition", "lower"),
    ("sar.sar2", ("paper_sar",), "coalition", "lower"),
    ("sar.rank1", ("paper_sar",), "coalition", "higher"),
    ("sar.rank2", ("paper_sar",), "coalition", "higher"),
    ("sar.rho2", ("paper_sar",), "coalition", "higher"),
    ("rss.mean_rss", ("paper_rss",), "variant", "contextual"),
    ("rss.held_out_persistence", ("paper_rss",), "variant", "higher"),
    ("localization.budget_at_90pct", ("localization_codes",), "dataset", "contextual"),
    ("localization.budget_fraction_at_90pct", ("localization_codes",), "dataset", "contextual"),
    ("saebench.core.l0", ("saebench_core",), "token", "contextual"),
    ("saebench.core.mse", ("saebench_core",), "token", "lower"),
    (
        "saebench.sparse_probing.sae_top_1_test_accuracy",
        ("saebench_sparse_probing",),
        "dataset",
        "higher",
    ),
):
    register_metric(MetricSpec(name, "1", tuple(inputs), unit, direction))

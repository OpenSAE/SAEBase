"""Purpose: Execute frozen benchmark protocols using activation caches, interventions and method-specific evaluators.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from saebase.api import ModelAdapter, load_registered
from saebase.extensions.contracts import ExternalMetricContext
from saebase.extensions.metrics import execute_external_metric
from saebase.extensions.registry import external_metric, has_external_pair, is_external_metric
from saebase.extensions.runtime import load_external_pair
from saebase.store import (
    ActivationIdentity,
    ActivationWriter,
    CodeIdentity,
    ShardedReader,
    collect_activations,
    encode_from_store,
    sha256_state_dict,
)
from tasks.benchmark.execution.planner import build_plan
from tasks.benchmark.execution.schema import BenchmarkConfig
from tasks.benchmark.methods.localization import parse_localization_options, run_localization
from tasks.benchmark.methods.stable_support import run_rss
from tasks.benchmark.methods.sae_benchmark import _import_saebench
from tasks.benchmark.methods.response_surface import run_sar
from tasks.benchmark.metrics.core import calculate_core
from tasks.benchmark.metrics.intervention import calculate_interventions
from tasks.benchmark.metrics.reconstruction import loss_recovered, target_nll
from tasks.benchmark.metrics.registry import get_metric
from tasks.benchmark.results.metrics import _metric, read_metrics, store_results
from tasks.benchmark.results.report import write_report
from tasks.benchmark.results.validation import validate_external_sources
from tasks.intervene.editing import BatchEngine, ExperimentIdentity, ResultWriter, RunBudget


def _identity(config, target, model, sae):
    model_sha = sha256_state_dict(model.model)
    sae_sha = sha256_state_dict(sae)
    tokenizer_hash = config.options.get("tokenizer_sha256")
    if tokenizer_hash is None:
        tokenizer = getattr(model.model, "tokenizer", None)
        if tokenizer is None or not callable(getattr(tokenizer, "get_vocab", None)):
            raise ValueError("exact tokenizer identity required: set options.tokenizer_sha256")
        payload = {
            "vocab": tokenizer.get_vocab(),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
            "added_vocab": tokenizer.get_added_vocab()
            if callable(getattr(tokenizer, "get_added_vocab", None))
            else {},
        }
        tokenizer_hash = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    if (
        not isinstance(tokenizer_hash, str)
        or len(tokenizer_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in tokenizer_hash)
    ):
        raise ValueError("options.tokenizer_sha256 must be lowercase SHA-256 hex")
    act = ActivationIdentity(
        model_key=target.model_key,
        model_state_sha256=model_sha,
        tokenizer_sha256=tokenizer_hash,
        hook_name=target.site.hook_name,
        layer=target.site.layer,
        preprocessing={"dataset": config.dataset_id, "site_position": target.site.position},
    )
    return act, sae_sha


def execute(
    config: BenchmarkConfig,
    output: str | Path,
    *,
    real: bool = False,
    resume: bool = True,
    loader=None,
    localization_paper=None,
    device: str = "cpu",
):
    """Execute a frozen matrix. A caller may inject a pair loader for offline tests.

    For real weights, caller explicitly passes real=True; failure to resolve an
    external adapter produces a blocked MetricResult rather than silent zeros.
    """
    plan = build_plan(config)
    if config.options.get("device", device) != device:
        raise ValueError("Requested device conflicts with benchmark options.device")
    if not real:
        raise ValueError("execution must be explicitly enabled with real=True")
    root = Path(output) / config.run_id
    root.mkdir(parents=True, exist_ok=True)
    if (root / "metrics_manifest.json").exists() and resume:
        prior = read_metrics(root, config.run_id)
        validate_external_sources(config, prior, root=root, localization_paper=localization_paper)
        # A completed ledger is read-only. This never reexecutes model work.
        return prior
    if (root / "metrics_manifest.json").exists() and not resume:
        raise FileExistsError(f"existing benchmark results: {root}")
    results = []
    provenance = {}
    specs = [get_metric(m) for m in config.metrics]

    def built_in_or_external(model_key, sae_key, layer):
        if has_external_pair(model_key, sae_key):
            return load_external_pair(
                model_key=model_key, sae_key=sae_key, layer=layer, device=device
            )
        return load_registered(model_key=model_key, sae_key=sae_key, layer=layer, device=device)

    load = loader or built_in_or_external
    cache_root = Path(output) / "_cache" / "m1"
    wants_localization = any(s.metric_id.startswith("localization.") for s in specs)
    if wants_localization:
        parse_localization_options(config)
    native_specs = [s for s in specs if s.metric_id.startswith("saebench.")]
    local_specs = [s for s in specs if not s.metric_id.startswith("saebench.")]
    for target in config.targets:
        for sae_key in target.sae_keys:
            if not local_specs:
                results.extend(_import_saebench(config, target, sae_key, native_specs))
                continue
            pair = load(target.model_key, sae_key, target.site.layer)
            model = pair.model if hasattr(pair, "model") else pair[0]
            sae = pair.sae if hasattr(pair, "sae") else pair[1]
            if not isinstance(model, ModelAdapter):
                model = ModelAdapter(model)
            act_id, sae_sha = _identity(config, target, model, sae)
            identity = ExperimentIdentity(
                act_id.model_state_sha256,
                sae_sha,
                target.site,
                model_key=target.model_key,
                sae_key=sae_key,
                tokenizer_sha256=act_id.tokenizer_sha256,
                backend=model.backend,
            )
            prefix = f"{target.model_key}/{target.site.layer}/{sae_key}"
            provenance[prefix] = {
                "model_state_sha256": act_id.model_state_sha256,
                "sae_state_sha256": sae_sha,
                "tokenizer_sha256": act_id.tokenizer_sha256,
            }
            if hasattr(pair, "identity") and hasattr(pair, "verification"):
                provenance[prefix]["extension_identity"] = pair.identity.to_dict()
                provenance[prefix]["extension_verification"] = pair.verification
            records = [s.record for s in config.samples]
            wants_core = any(s.metric_id.startswith("core.") for s in specs)
            wants_external_codes = any(
                is_external_metric(s.metric_id)
                and bool({"m1_activations", "m1_codes"} & set(s.required_inputs))
                for s in specs
            )
            needs_interventions = any(s.metric_id.startswith("intervention.") for s in specs)
            wants_recon = any(s.metric_id.startswith("reconstruction.") for s in specs)
            core_data = {}
            n_core = 0
            cache_cost = {}
            intervention_cost = {}
            act_path = None
            code_path = None
            if wants_core or needs_interventions or wants_localization or wants_external_codes:
                with ActivationWriter(cache_root, act_id, width=int(sae.cfg.d_in)) as writer:
                    cache_stats = collect_activations(model, records, writer, site=target.site)
                    act_path = writer.path
                code_id = CodeIdentity(sae_key, sae_sha, int(sae.cfg.d_in), int(sae.cfg.d_sae))
                code_path, code_stats = encode_from_store(act_path, sae, code_id)
                cache_cost = {
                    "activations_collected": cache_stats.collected,
                    "activations_reused": cache_stats.reused,
                    "sae_codes_encoded": code_stats.collected,
                    "sae_codes_reused": code_stats.reused,
                }
            if wants_localization:
                subset = [spec for spec in specs if spec.metric_id.startswith("localization.")]
                try:
                    artifact, ref = run_localization(
                        config,
                        target,
                        sae_key,
                        activation_store=act_path,
                        code_store=code_path,
                        root=root,
                        paper=localization_paper,
                    )
                    if artifact["total_variance"] <= 0:
                        for spec in subset:
                            results.append(
                                _metric(
                                    config,
                                    target,
                                    sae_key,
                                    spec,
                                    "undefined",
                                    reason="zero SAE activation variance on complete input grid",
                                )
                            )
                    else:
                        for spec in subset:
                            value = (
                                artifact["budget_at_90pct"]
                                if spec.metric_id.endswith("budget_at_90pct")
                                else artifact["budget_fraction_at_90pct"]
                            )
                            results.append(
                                _metric(
                                    config,
                                    target,
                                    sae_key,
                                    spec,
                                    "completed",
                                    value=float(value),
                                    n_valid=1,
                                    refs=(ref,),
                                    source_sha=artifact["source_sha256"],
                                    cost=cache_cost,
                                    metadata={
                                        "design_key": artifact["design_manifest"]["design_key"],
                                        "n_factors": artifact["n_factors"],
                                        "n_features": artifact["n_features"],
                                        "interpretation": "one controlled input grid, not independent samples",
                                    },
                                )
                            )
                except (
                    ImportError,
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    RuntimeError,
                    IndexError,
                ) as exc:
                    for spec in subset:
                        results.append(
                            _metric(
                                config,
                                target,
                                sae_key,
                                spec,
                                "blocked",
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                        )
            if wants_core:
                core_data, n_core = calculate_core(
                    act_path,
                    code_path,
                    sae,
                    sample_ids=[s.record.sample_id for s in config.samples],
                )
            for spec in specs:
                if spec.metric_id.startswith("core."):
                    value = core_data.get(spec.metric_id)
                    results.append(
                        _metric(
                            config,
                            target,
                            sae_key,
                            spec,
                            "completed" if value is not None else "undefined",
                            value,
                            n_valid=n_core if value is not None else 0,
                            reason=None
                            if value is not None
                            else "no valid activation/code cache rows",
                            cost=cache_cost,
                        )
                    )
            if needs_interventions:
                meas = Path(output) / "_cache" / "m2" / act_id.key / sae_sha
                with ResultWriter(meas, identity, resume=True) as writer:
                    intervention_stats = BatchEngine(
                        model,
                        sae,
                        identity,
                        activation_reader=ShardedReader(act_path),
                        code_reader=ShardedReader(code_path),
                    ).run(
                        config.samples,
                        config.conditions,
                        writer,
                        sample_batch=config.sample_batch,
                        microbatch=config.microbatch,
                        budget=RunBudget(max_measurements=config.max_measurements),
                        tolerance=config.tolerance,
                    )
                intervention_cost = {
                    "interventions_measured": intervention_stats.measured,
                    "interventions_reused": intervention_stats.reused,
                    "baseline_batches": intervention_stats.baseline_batches,
                    "patch_batches": intervention_stats.patch_batches,
                    "microbatch_retries": intervention_stats.microbatch_retries,
                    "elapsed_seconds": intervention_stats.elapsed_seconds,
                    "backend": intervention_stats.backend,
                }
                measurements, raw = calculate_interventions(
                    meas, identity, conditions=config.conditions
                )
                for spec in specs:
                    if not spec.metric_id.startswith("intervention."):
                        continue
                    for c in config.conditions:
                        item = measurements.get((c.name, spec.metric_id))
                        if item:
                            value, n, refs = item
                            results.append(
                                _metric(
                                    config,
                                    target,
                                    sae_key,
                                    spec,
                                    "completed",
                                    value,
                                    n_valid=n,
                                    condition=c.name,
                                    refs=refs,
                                    cost=intervention_cost,
                                )
                            )
                        else:
                            results.append(
                                _metric(
                                    config,
                                    target,
                                    sae_key,
                                    spec,
                                    "undefined",
                                    reason="metric not recorded for this condition",
                                    condition=c.name,
                                )
                            )
            if wants_recon:
                observations = []
                for s in config.samples:
                    if s.target_token_id is not None:
                        observations.append(target_nll(model, sae, s, target.site))
                for spec in specs:
                    if not spec.metric_id.startswith("reconstruction."):
                        continue
                    if spec.metric_id == "reconstruction.target_nll":
                        val = (
                            sum(patched for _, patched, _ in observations) / len(observations)
                            if observations
                            else None
                        )
                        results.append(
                            _metric(
                                config,
                                target,
                                sae_key,
                                spec,
                                "completed" if val is not None else "undefined",
                                val,
                                n_valid=len(observations) if val is not None else 0,
                                reason=None if val is not None else "no target tokens",
                            )
                        )
                    else:
                        if not observations:
                            results.append(
                                _metric(
                                    config,
                                    target,
                                    sae_key,
                                    spec,
                                    "undefined",
                                    reason="loss recovered requires target tokens",
                                )
                            )
                        else:
                            base = sum(v[0] for v in observations) / len(observations)
                            patch = sum(v[1] for v in observations) / len(observations)
                            ablation = sum(v[2] for v in observations) / len(observations)
                            val = loss_recovered(base, patch, ablation)
                            results.append(
                                _metric(
                                    config,
                                    target,
                                    sae_key,
                                    spec,
                                    "completed" if val is not None else "undefined",
                                    val,
                                    n_valid=len(observations) if val is not None else 0,
                                    reason=None
                                    if val is not None
                                    else "ablation and baseline losses coincide",
                                )
                            )
            for paper in ("sar", "rss"):
                subset = [spec for spec in specs if spec.metric_id.startswith(paper + ".")]
                if not subset:
                    continue
                source = config.sources.get(paper)
                if not source:
                    for spec in subset:
                        results.append(
                            _metric(
                                config,
                                target,
                                sae_key,
                                spec,
                                "blocked",
                                reason=f"missing sources.{paper} original estimator",
                            )
                        )
                    continue
                try:
                    vals, n, digest = (run_sar if paper == "sar" else run_rss)(
                        [s.metric_id for s in subset],
                        model,
                        sae,
                        target.site,
                        source=source,
                        options=config.options,
                        samples=config.samples,
                        seed=config.seed,
                    )
                    for spec in subset:
                        item = vals.get(spec.metric_id.removeprefix(paper + "."))
                        results.append(
                            _metric(
                                config,
                                target,
                                sae_key,
                                spec,
                                "completed"
                                if item is not None and math.isfinite(item)
                                else "undefined",
                                float(item) if item is not None and math.isfinite(item) else None,
                                n_valid=(
                                    sum(
                                        s.record.variant_role == "selection" for s in config.samples
                                    )
                                    if paper == "rss" and spec.metric_id == "rss.mean_rss"
                                    else n
                                )
                                if item is not None and math.isfinite(item)
                                else 0,
                                reason=None
                                if item is not None and math.isfinite(item)
                                else "upstream estimator undefined",
                                source_sha=digest,
                                cost={"upstream_evaluations": n},
                            )
                        )
                except Exception as exc:
                    for spec in subset:
                        results.append(
                            _metric(
                                config,
                                target,
                                sae_key,
                                spec,
                                "blocked",
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                        )
            for spec in specs:
                if not is_external_metric(spec.metric_id):
                    continue
                try:
                    result, digest, revision = execute_external_metric(
                        spec.metric_id,
                        ExternalMetricContext(
                            config=config,
                            target=target,
                            sae_key=sae_key,
                            model=model,
                            sae=sae,
                            activation_store=act_path,
                            code_store=code_path,
                        ),
                    )
                    results.append(
                        _metric(
                            config,
                            target,
                            sae_key,
                            spec,
                            "completed",
                            result.value,
                            n_valid=result.n_valid,
                            refs=result.measurement_refs,
                            source_sha=digest,
                            source_commit=revision,
                            metadata={
                                "extension_source": external_metric(spec.metric_id).source,
                                "verification": "metric_source_verified_not_model_certified",
                            },
                        )
                    )
                except (OSError, ValueError, TypeError, KeyError, RuntimeError, IndexError) as exc:
                    results.append(
                        _metric(
                            config,
                            target,
                            sae_key,
                            spec,
                            "blocked",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
            results.extend(_import_saebench(config, target, sae_key, native_specs))
            # Registered but unsupported metric families must be visible in the
            # result set. Never silently drop a promised metric from a report.
            known = (
                "core.",
                "intervention.",
                "reconstruction.",
                "sar.",
                "rss.",
                "saebench.",
                "localization.",
            )
            for spec in specs:
                if not spec.metric_id.startswith(known) and not is_external_metric(spec.metric_id):
                    results.append(
                        _metric(
                            config,
                            target,
                            sae_key,
                            spec,
                            "unsupported",
                            reason="no executable adapter registered for this metric family",
                        )
                    )
    store_results(root, results, run_id=config.run_id)
    write_report(root, config, plan, results, provenance=provenance)
    return [r.to_dict() for r in results]

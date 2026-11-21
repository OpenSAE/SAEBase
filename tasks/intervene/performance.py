"""Numerically checked native feature batching for intervene and generate.

Baseline calls the unoptimized public task operation. Every timed candidate
processes identical features and, for generation, identical token budgets/seeds.
"""

from saebase.runtime.configuration.task_options import task_owner, task_value

import numpy as np
import torch
import time
from omegaconf import OmegaConf
from saebase.api import capture_features, edit_features
from saebase.api.replay import SiteReplay
from saebase.runtime.performance import settings, tune, current, stage


def validate_outputs(expected, actual, options):
    maximum = 0.0
    accepted = True

    def visit(a, b):
        nonlocal maximum, accepted
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                accepted = False
                return
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                accepted = False
                return
            for x, y in zip(a, b):
                visit(x, y)
        elif isinstance(a, float):
            error = abs(a - b)
            maximum = max(maximum, error)
            accepted &= bool(
                np.isfinite(b)
                and np.isclose(a, b, atol=options["logit_atol"], rtol=options["rtol"])
            )
        else:
            accepted &= a == b

    visit(expected, actual)
    return {
        "accepted": bool(accepted),
        "max_abs_error": maximum,
        "categorical_outputs_require_exact_match": True,
        "atol": options["logit_atol"],
        "rtol": options["rtol"],
    }


def children(cfg, selected, operation):
    values = []
    for feature in selected["top_features"]:
        child = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        task_owner(child.task).feature_selection = "configured"
        child.task.prompt = selected["prompt"]
        task_owner(child.task).feature_id = feature["feature_id"]
        if operation == "generate":
            task_owner(child.task).conditions = [
                {
                    "name": f"feature_{feature['feature_id']}",
                    "operation": task_value(cfg.task, "intervention", "add"),
                    "feature_ids": [feature["feature_id"]],
                    "value": task_value(cfg.task, "value", 1.0),
                }
            ]
        values.append(child)
    return values


def _prepare_features(cfg, operation, selected, pair, measure):
    configs = children(cfg, selected, operation)
    prompt = selected["prompt"]
    prepare_started = time.perf_counter()
    with stage("feature_prepare"):
        snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)

    def baseline():
        from tasks.intervene.editing.observations import prompt_observation_session

        with prompt_observation_session(pair, task_value(cfg.task, "reuse_prompt_observations", False)):
            return [measure(child, prompt=prompt) for child in configs]

    if operation == "generate":
        from tasks.generate.conditions import _sample, compare_prompt_conditions
        from tasks.generate.serving.native_batch import compare_many
        from tasks.intervene.editing.specs import Condition
        from tasks.intervene.editing.steering import GenerationSpec
        from saebase.results import completion_summary
        from saebase.runtime.sae_operations import _fingerprint_prompt

        sample_id = "prompt-" + _fingerprint_prompt(prompt)[:16]
        sample, _ = _sample(pair.model, prompt, sample_id)
        conditions = [
            Condition(
                f"feature_{task_owner(child.task).feature_id}",
                str(task_value(cfg.task, "intervention", "add")),
                (int(task_owner(child.task).feature_id),),
                float(task_value(cfg.task, "value", 1.0)),
            )
            for child in configs
        ]
        spec = GenerationSpec(
            max_new_tokens=int(task_value(cfg.task, "max_new_tokens")),
            seed=int(cfg.seed),
            temperature=float(task_value(cfg.task, "temperature")),
            stop_at_context_limit=task_value(cfg.task, "stop_at_context_limit", True),
        )

        def candidate(batch):
            raw = compare_many(pair, sample, conditions, spec, batch)
            output = []
            for index, child in enumerate(configs):
                result = compare_prompt_conditions(
                    pair,
                    prompt,
                    sample_id,
                    spec.max_new_tokens,
                    feature_id=int(task_owner(child.task).feature_id),
                    conditions=OmegaConf.to_container(task_owner(child.task).conditions),
                    seed=spec.seed,
                    temperature=spec.temperature,
                    stop_at_context_limit=spec.stop_at_context_limit,
                    measured_runs=[raw[index]],
                    original_snapshot=snapshot,
                )
                output.append(
                    {
                        "status": "executed",
                        "operation": "generate",
                        "model_key": pair.identity.model_key,
                        "sae_key": pair.identity.sae_key,
                        "site": pair.site.hook_name,
                        "result": result,
                        "input_scope": {"unit": "prompt", "requested": 1, "processed": 1},
                        "completion": completion_summary(
                            [{"status": "completed"}], unit="generation_condition"
                        ),
                    }
                )
            return output

        optimizations = [
            "shared_baseline_generation",
            "native_condition_batches",
            "combined_capture_and_clean_forward",
            "full_prefix_recompute_no_stale_kv",
        ]
    else:
        from tasks.intervene.editing.observations import prompt_observation_session

        replay = SiteReplay(pair.model, snapshot.tokens, pair.site)
        patches = [
            edit_features(
                pair.sae,
                snapshot,
                replacements={int(task_owner(child.task).feature_id): float(task_owner(child.task).value)},
            )
            for child in configs
        ]
        baseline_logits = pair.model.forward_logits(tokens=snapshot.tokens, readout_position=-1)
        if task_value(cfg.task, "verify_identity", True):
            pair.model.assert_identity_patch(tokens=snapshot.tokens, site=pair.site)

        def candidate(batch):
            logits = []
            for start in range(0, len(patches), batch):
                activations = torch.stack([p.activation for p in patches[start : start + batch]])
                logits.extend(replay.logits(activations)[:, -1].detach().clone().unbind())
            with prompt_observation_session(pair, True) as observations:
                key = observations._key(snapshot.tokens, pair.site)
                observations.clean[key] = (snapshot, baseline_logits)
                observations.verified.add(key)
                for patch, logit in zip(patches, logits):
                    cache_key = (
                        key,
                        str(patch.activation.dtype),
                        patch.activation.detach()
                        .contiguous()
                        .cpu()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes(),
                    )
                    observations.edited[cache_key] = logit
                return [measure(child, prompt=prompt) for child in configs]

        optimizations = [
            "shared_clean_observation",
            "single_identity_replay",
            "cached_suffix",
            "native_condition_batches",
        ]
    from saebase.runtime.performance import synchronize

    synchronize()
    setup_seconds = time.perf_counter() - prepare_started
    return configs, prompt, snapshot, baseline, candidate, optimizations, setup_seconds


def measure_features(cfg, operation, selected, pair, measure):
    options = settings(cfg)
    configs, prompt, snapshot, baseline, candidate, optimizations, setup_seconds = (
        _prepare_features(cfg, operation, selected, pair, measure)
    )
    recorder = current()
    policy_key = (
        operation,
        pair.identity.model_key,
        (snapshot.tokens.shape[1] + 63) // 64,
        len(configs),
        task_value(cfg.task, "max_new_tokens"),
    )
    policies = getattr(recorder, "policies", {}) if recorder is not None else {}
    preparation_warmup_seconds = 0.0
    if policy_key not in policies and options["warmup"]:
        # Baseline timings are warmed too. Charge cold/JIT preparation once
        # to tuning, and compare the candidate's recurring warm preparation.
        preparation_warmup_seconds = setup_seconds
        with stage("preparation_warmup"):
            configs, prompt, snapshot, baseline, candidate, optimizations, setup_seconds = (
                _prepare_features(cfg, operation, selected, pair, measure)
            )
    if policy_key in policies and policies[policy_key] == 0:
        with stage(operation + "_measurements"):
            output = baseline()
        evidence = {
            "selected": False,
            "selected_batch_size": 1,
            "policy_reused": True,
            "reason": "Original configured execution was fastest for this bucket",
        }
    elif policy_key in policies:
        selected_batch = policies[policy_key]
        with stage(operation + "_measurements"):
            try:
                output = candidate(selected_batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                selected_batch = 1
                output = candidate(1)
        with stage("per_prompt_numerical_guard"):
            guard_indices = sorted({0, len(configs) - 1})
            reference_controls = [measure(configs[i], prompt=prompt) for i in guard_indices]
            guard = validate_outputs(
                reference_controls, [output[i] for i in guard_indices], options
            )
        if not guard["accepted"]:
            if options.get("require_batch_invariance"):
                raise ValueError(
                    f"Batch invariance failed on this prompt: {guard}; no scalar fallback was accepted"
                )
            output = baseline()
            policies[policy_key] = 0
        else:
            policies[policy_key] = selected_batch
        evidence = {
            "optimization": optimizations,
            "selected_batch_size": selected_batch,
            "policy_reused": True,
            "validation": guard,
            "fallback_to_reference": not guard["accepted"],
            "selected": guard["accepted"],
        }
    else:
        with stage("autotune"):
            evidence = tune(
                candidate,
                baseline,
                len(configs),
                options,
                validate=lambda a, b: validate_outputs(a, b, options),
                candidate_setup_seconds=setup_seconds,
                baseline_includes_setup=True,
            )
        evidence.update(optimizations=optimizations, unit="feature_comparison", policy_reused=False)
        evidence["preparation_warmup_seconds"] = preparation_warmup_seconds
        evidence["total_tuning_seconds"] += preparation_warmup_seconds
        with stage(operation + "_measurements"):
            output = (
                candidate(evidence["selected_batch_size"]) if evidence["selected"] else baseline()
            )
        policies[policy_key] = evidence["selected_batch_size"] if evidence["selected"] else 0
        if recorder is not None:
            recorder.policies = policies
    if recorder is not None:
        evidence.update(policy_kind="feature", policy_key=list(policy_key))
        recorder.experiments.append(evidence)
    return output, evidence

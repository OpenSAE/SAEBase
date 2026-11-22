"""Native unedited answers attached to Capture inputs, shared across feature modes.

Use the already loaded checkpoint and upstream cached generation. Cache identity
includes weights, exact prompt tokens, tokenizer, numerics and generation settings.
Failures remain explicit answer records; existing feature measurements survive.
"""

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import torch

from conf import launch_directory

from tasks.capture.selection import undefined_reason


DEFAULTS = {"max_new_tokens": 256, "batch_size": 8}


def settings(cfg):
    raw = cfg.get("task", {}).get("capture", {}).get("model_answer", {})
    values = {key: raw.get(key, default) for key, default in DEFAULTS.items()}
    for key, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"task.capture.model_answer.{key} must be a positive integer")
    return values


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@contextmanager
def _restore_tail(model, device):
    """Temporarily restore blocks offloaded by Capture's prefix-only optimization."""
    moved = []
    try:
        for block in getattr(model, "blocks", []):
            original = next(block.parameters()).device
            if original.type == "cpu" and device.type == "cuda":
                block.to(device)
                moved.append((block, original))
        yield
    finally:
        for block, device in moved:
            block.to(device)


@torch.inference_mode()
def answers(cfg, pair, prompts):
    """Return one answer receipt per prompt, retaining order and duplicate inputs."""
    from saebase.runtime.performance import stage
    from saebase.runtime.sae_operations import pair_state_fingerprints
    from models.numerics import identity as numerical_identity

    options = settings(cfg)
    results = [None] * len(prompts)
    if not prompts:
        return results
    pending = defaultdict(list)
    owners = {}
    duplicates = {}
    started = time.perf_counter()
    try:
        raw = pair.model.model
        tokenizer = raw.tokenizer
        model_sha, _ = pair_state_fingerprints(pair)
        token_state = {
            "vocab": tokenizer.get_vocab(),
            "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
        }
        tokenizer_sha = _digest(token_state)
        backend = pair.model.backend
        native = getattr(raw, "_model", raw)
        generator = pair.model.generate if backend == "transformer_lens" else native.generate
        eos = getattr(getattr(native, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(tokenizer, "eos_token_id", None)
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        limit = getattr(getattr(raw, "cfg", None), "n_ctx", None)
        cache_root = (
            launch_directory() / (cfg.get("paths", {}).get("root") or "")
        ).resolve() / "assets/answer_cache/model_answers/files"
        base = {
            "protocol": "saebase.capture.model_answer.v1",
            "model_sha256": model_sha,
            "tokenizer_sha256": tokenizer_sha,
            "backend": backend,
            "numerics": numerical_identity(),
            "max_new_tokens": options["max_new_tokens"],
            "temperature": 0.0,
            "eos_token_ids": eos,
            "context_limit": limit,
            "dtype": str(next(native.parameters()).dtype),
        }
        for index, prompt in enumerate(prompts):
            try:
                tokens = pair.model.tokens(prompt=prompt)
            except Exception as exc:
                results[index] = {
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                continue
            ids = tokens[0].detach().cpu().tolist()
            identity = {
                **base,
                "prompt_token_ids": ids,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            key = _digest(identity)
            path = cache_root / f"{key}.json"
            if key in owners:
                duplicates[index] = owners[key]
                continue
            owners[key] = index
            if path.exists():
                try:
                    value = json.loads(path.read_text())
                    if value["identity"] == identity and value["status"] == "completed":
                        results[index] = {**value, "reused": True}
                        continue
                except (KeyError, ValueError):
                    pass
            allowed = (
                min(options["max_new_tokens"], limit - len(ids))
                if isinstance(limit, int)
                else options["max_new_tokens"]
            )
            if allowed < 1:
                results[index] = {
                    "status": "blocked",
                    "reason": "No context space for a model answer",
                    "finish_reason": "context_limit",
                    "identity": identity,
                }
                continue
            pending[(len(ids), allowed)].append((index, tokens, identity, path))
        for (_, allowed), group in pending.items():
            for offset in range(0, len(group), options["batch_size"]):
                batch = group[offset : offset + options["batch_size"]]
                try:
                    inputs = torch.cat([item[1] for item in batch])
                    kwargs = {"max_new_tokens": allowed, "do_sample": False}
                    if eos:
                        kwargs["eos_token_id"] = eos
                    if backend == "transformer_lens":
                        kwargs.update(
                            return_type="tokens",
                            verbose=False,
                            use_past_kv_cache=True,
                            stop_at_eos=bool(eos),
                        )
                    else:
                        kwargs.update(attention_mask=torch.ones_like(inputs), use_cache=True)
                        pad = getattr(tokenizer, "pad_token_id", None)
                        if pad is not None or eos:
                            kwargs["pad_token_id"] = pad if pad is not None else eos[0]
                    with stage("model_answer_generation"), _restore_tail(raw, inputs.device):
                        output = generator(inputs, **kwargs)
                    output = getattr(output, "sequences", output)
                    if (
                        output.ndim != 2
                        or output.shape[0] != len(batch)
                        or not torch.equal(output[:, : inputs.shape[1]], inputs)
                    ):
                        raise ValueError(
                            "Generated sequences do not preserve the captured prompt tokens"
                        )
                    for row, (index, _, identity, path) in enumerate(batch):
                        generated = output[row, inputs.shape[1] :].detach().cpu().tolist()
                        end = next(
                            (i + 1 for i, token in enumerate(generated) if token in eos),
                            len(generated),
                        )
                        generated = generated[:end]
                        finish = (
                            "eos"
                            if generated and generated[-1] in eos
                            else "context_limit"
                            if allowed < options["max_new_tokens"]
                            else "max_new_tokens"
                        )
                        value = {
                            "status": "completed",
                            "text": tokenizer.decode(generated, skip_special_tokens=True),
                            "token_ids": generated,
                            "finish_reason": finish,
                            "truncated": finish != "eos",
                            "identity": identity,
                            "reused": False,
                        }
                        from saebase.runtime.experiment_results import write_json

                        write_json(path, value)
                        results[index] = value
                except Exception as exc:
                    for index, _, identity, _ in batch:
                        results[index] = {
                            "status": "failed",
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                            "identity": identity,
                        }
    except Exception as exc:
        for index, value in enumerate(results):
            if value is None:
                results[index] = {
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
    for index, original in duplicates.items():
        results[index] = {**results[original], "reused": results[original]["status"] == "completed"}
    elapsed = time.perf_counter() - started
    for value in results:
        value["batch_elapsed_seconds"] = elapsed
    return results


def answer_rows(cfg, pair, selection):
    """Bound memory and retain processed rows if the Dataset iterator fails."""
    iterator = iter(selection)
    while True:
        rows = []
        error = None
        for _ in range(settings(cfg)["batch_size"]):
            try:
                rows.append(next(iterator))
            except StopIteration:
                break
            except Exception as exc:
                error = exc
                break
        if rows:
            values = answers(cfg, pair, [str(row.get("prompt", "")) for row in rows])
            yield from zip(rows, values, strict=True)
        if error is not None:
            raise error
        if len(rows) < settings(cfg)["batch_size"]:
            break


def attach(result, answer, *, extra_answers=()):
    """Attach a required answer stage while retaining Capture's own completion."""
    from saebase.results import completion_summary

    recorded = result.get("input_identity") or {}
    identity = answer.get("identity", {})
    if (
        answer["status"] == "completed"
        and recorded.get("data_sha256")
        and "prompt_token_ids" in identity
    ):
        from saebase.store import fingerprint

        actual = fingerprint(
            {"tokens": [identity["prompt_token_ids"]], "prompt_sha256": identity["prompt_sha256"]}
        )
        if actual != recorded["data_sha256"]:
            answer = {
                **answer,
                "status": "blocked",
                "reason": "Generated input tokens differ from the captured input",
            }
    result["model_answer"] = answer
    stages = result.setdefault("stages", {})
    stages.setdefault("features_and_annotations", result.get("completion", {}))
    units = []
    for value in (answer, *extra_answers):
        unit = {"status": "completed" if value["status"] == "completed" else "blocked"}
        if unit["status"] != "completed":
            unit["reason"] = value.get("reason") or value.get("error", {}).get(
                "message", "Model answer unavailable"
            )
        units.append(unit)
    stages["model_answer"] = completion_summary(units, unit="model_answer")
    previous = stages["features_and_annotations"]
    work = list(units)
    for status, count in previous.get("counts", {}).items():
        detail = {"status": status}
        if status == "undefined" and count:
            detail["reason"] = next(
                row["reason"] for row in previous["reasons"] if row["status"] == "undefined"
            )
        work.extend(dict(detail) for _ in range(count))
    result["completion"] = completion_summary(work, unit="capture and model_answer")
    result["completion"]["reasons"].extend(
        row for row in previous.get("reasons", []) if row["status"] != "undefined"
    )
    return result


def summary(values):
    from saebase.results import completion_summary

    return completion_summary(
        [
            {"status": "completed"}
            if v["status"] == "completed"
            else {
                "status": "blocked",
                "reason": v.get("reason")
                or v.get("error", {}).get("message", "Model answer unavailable"),
            }
            for v in values
        ],
        unit="model_answer",
    )


def refresh(folder, *, pair=None):
    """Supplement saved Capture answers without recomputing a feature or coalition."""
    from omegaconf import OmegaConf
    from tasks.capture.annotations import read_samples
    from saebase.runtime.sae_operations import (
        _pair,
        _prompt,
        loaded_pair_session,
        pair_state_fingerprints,
    )
    from saebase.datasets.selection import read_selection
    from saebase.runtime.experiment_results import write_json
    from saebase.results import completion_summary

    root = Path(folder).resolve()
    manifest = json.loads((root / "metadata/manifest.json").read_text())
    if manifest["task"] != "capture":
        raise ValueError("Model answers require a Capture result")
    cfg = OmegaConf.create(manifest["configuration"])
    saved = manifest["result"]["summary"]
    batched = (root / "reports/capture_samples.jsonl").exists()
    rows = list(read_samples(root)) if batched else [{"sample_id": "explicit", "result": saved}]
    if any(not row.get("result") for row in rows):
        raise ValueError(
            "Feature measurements are missing; complete Capture before supplementing answers"
        )
    if pair is None:
        from models.numerics import configure_task
        from saebase.runtime.environment import configure_worker

        configure_worker(cfg)
        configure_task(cfg)
        pair = _pair(cfg)
    if batched:
        if "sample_count" in cfg.dataset:
            selected = {r["sample_id"]: r for r in read_selection(cfg.dataset, str(cfg.model.name))}
        else:
            from saebase.datasets.prompts import read_prompts

            selected = {
                r["sample_id"]: r for r in read_prompts(cfg.dataset, model_key=str(cfg.model.name))
            }
        prompts = [
            str(cfg.task.get("prompt") or selected[row["sample_id"]]["prompt"]) for row in rows
        ]
    else:
        prompts = [_prompt(cfg)]
    rss = saved.get("mode") == "rss" and not batched
    if rss:
        for index, prompt in enumerate(saved["variants"], 1):
            prompts.append(prompt)
            rows.append(
                {
                    "sample_id": f"variant:{index}",
                    "result": {
                        "input_identity": saved.get("input_identity"),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "completion": saved.get("stages", {}).get(
                            "features_and_annotations", saved["completion"]
                        ),
                    },
                }
            )
    with loaded_pair_session(pair):
        model_sha, _ = pair_state_fingerprints(pair)
        for row, prompt in zip(rows, prompts, strict=True):
            identity = row["result"].get("input_identity") or {}
            if identity.get("model_sha256") != model_sha:
                raise ValueError("Answer checkpoint does not match the saved Capture weights")
            if identity.get("data_sha256"):
                from saebase.store import fingerprint

                actual_tokens = pair.model.tokens(prompt=prompt).detach().cpu().tolist()
                actual_data = fingerprint(
                    {
                        "tokens": actual_tokens,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                )
                if actual_data != identity["data_sha256"]:
                    raise ValueError("Answer tokenization differs from the saved Capture input")
            expected = row["result"].get("prompt_sha256")
            if expected and expected != hashlib.sha256(prompt.encode()).hexdigest():
                raise ValueError("Answer input does not match the saved Capture prompt")
        values = answers(cfg, pair, prompts)
    supplement = {row["sample_id"]: value for row, value in zip(rows, values, strict=True)}
    write_json(root / "reports/capture_model_answers.json", supplement)
    if rss:
        attach(rows[0]["result"], values[0], extra_answers=values[1:])
    work = []
    for row, value in zip(rows, values, strict=True):
        result = row["result"] if rss and row is rows[0] else attach(row["result"], value)
        reason = undefined_reason(result)
        status = (
            "blocked"
            if result["completion"]["status"] != "complete"
            else "undefined"
            if reason
            else "completed"
        )
        work.append(
            {
                "status": status,
                **({"reason": reason} if status == "undefined" else {}),
            }
        )
    update = {
        "updated_at_ns": time.time_ns(),
        "model_answer": summary(values),
        "completion": completion_summary(work, unit="sample")
        if batched
        else rows[0]["result"]["completion"],
        "path": "reports/capture_model_answers.json",
        "settings": settings(cfg),
    }
    write_json(root / "metadata/capture_model_answer_update.json", update)
    return update


def project(folder, manifest):
    """Read answer-only updates through the ordinary Task result interface."""
    path = Path(folder) / "metadata/capture_model_answer_update.json"
    if not path.exists():
        return manifest
    update = json.loads(path.read_text())
    payload = dict(manifest["result"])
    result = dict(payload["summary"])
    annotation_update = result.get("annotation_update", {})
    completion = (
        annotation_update["completion"]
        if annotation_update.get("updated_at_ns", 0) > update.get("updated_at_ns", 0)
        else update["completion"]
    )
    result.update(
        completion=completion,
        stages={**result.get("stages", {}), "model_answer": update["model_answer"]},
    )
    result["required_artifacts"] = [
        item
        for item in result.get("required_artifacts", [])
        if item["path"] not in {update["path"], "metadata/capture_model_answer_update.json"}
    ] + [{"path": update["path"]}, {"path": "metadata/capture_model_answer_update.json"}]
    if "samples_path" not in result:
        values = json.loads((Path(folder) / update["path"]).read_text())
        result["model_answer"] = values["explicit"]
        if result.get("mode") == "rss":
            result["variant_model_answers"] = [
                values[f"variant:{index}"] for index in range(1, len(result["variants"]) + 1)
            ]
    complete = completion["status"] == "complete"
    payload.update(
        summary=result,
        completion=completion,
        ok=complete and payload["status"] == "executed",
    )
    return {
        **manifest,
        "result": payload,
        "completion": completion,
        "exit_code": 0 if payload["ok"] else 2,
    }

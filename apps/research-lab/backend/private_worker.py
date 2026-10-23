"""Execute one real, hash-identified teacher-forced SAE job privately.

Only owner-scoped hashes are returned by the HTTP API. The native worker
runs on the user's machine, which must have original model/SAE weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from . import core
from .jobs import JobQueue
from .private_api import load_input


def _source_commit() -> str:
    repo = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _generate(pair, site: dict, prompt: str, spec: dict) -> dict:
    from saebase.api import ActivationSite
    from saebase.interventions.specs import Condition, MeasurementSample
    from saebase.interventions.steering import GenerationSpec, compare_steered_generation
    from saebase.store.identity import SampleRecord, TokenSelection

    started = time.perf_counter()
    token_ids = pair.model.tokens(prompt=prompt)
    ids = tuple(int(value) for value in token_ids[0].tolist())
    sample = MeasurementSample(
        record=SampleRecord(
            sample_id="private-generation",
            token_ids=ids,
            selection=TokenSelection(mode="last_valid"),
            prompt_group_id="private-generation",
            split="discovery",
        )
    )
    features = tuple(spec["feature_ids"])
    operation = spec["operation"]
    if operation == "identity":
        condition = Condition(name="noop", operation="identity")
    elif operation == "ablate":
        condition = Condition(name="ablate", operation="ablate", feature_ids=features)
    else:
        condition = Condition(
            name="scale", operation="scale", feature_ids=features, value=float(spec["strength"])
        )
    response = compare_steered_generation(
        pair.model,
        pair.sae,
        site=ActivationSite(site["layer"], pair.identity.hook_name, -1),
        sample=sample,
        condition=condition,
        spec=GenerationSpec(max_new_tokens=spec["max_new_tokens"], temperature=0.0, seed=0),
    )
    decode = pair.model.model.tokenizer.decode
    steps = [
        {
            "step": step["step"],
            "max_abs_logit_delta": step["max_abs_logit_delta"],
            "clean_argmax_token_id": step["clean_argmax_token_id"],
            "edited_argmax_token_id": step["edited_argmax_token_id"],
            "feature_activation": step["feature_activation"],
        }
        for step in response.get("steered_steps") or []
    ]
    return {
        "schema": "opensae.wp15.private_generation.v1",
        "status": "real_local_inference",
        "operation": operation,
        "model": pair.identity.model_key,
        "site_id": site["site_id"],
        "feature_id": features[0] if features else None,
        "input_sha256": spec["input_ref"],
        "baseline_continuation": decode(
            response["baseline_generated_token_ids"], skip_special_tokens=False
        ),
        "edited_continuation": decode(
            response["steered_generated_token_ids"], skip_special_tokens=False
        ),
        "same_continuation": response["same_continuation"],
        "first_different_generated_index": response["first_different_generated_index"],
        "steps": steps,
        "max_new_tokens": spec["max_new_tokens"],
        "logits_retained": False,
        "public_release": False,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "source_commit": _source_commit(),
    }


def _edit_activation(pair, site: dict, prompt: str, feature_ids: list[int]) -> tuple[float, bool]:
    from saebase.api import ActivationSite, capture_features

    snapshot = capture_features(
        pair.model,
        pair.sae,
        tokens=pair.model.tokens(prompt=prompt),
        site=ActivationSite(site["layer"], pair.identity.hook_name, -1),
    )
    codes = snapshot.codes
    activations = []
    for feature_id in feature_ids:
        if feature_id >= int(codes.numel()):
            raise IndexError("Feature outside actual SAE width")
        activations.append(float(codes[feature_id].detach().float().item()))
    active = any(value != 0.0 for value in activations)
    return activations[0], active


def execute_one(*, root: Path, device: str = "cuda") -> dict:
    from saebase.api import load_registered
    from saebase.store import sha256_state_dict
    from saebase.wp12.interventions import Edit, EditPlan, compare_sae_edits

    if os.environ.get("OPENSAE_ENABLE_PRIVATE_JOBS") != "1":
        raise PermissionError("Local private worker is disabled")
    queue = JobQueue(root / "queue" / "jobs.sqlite3", max_queued=16, max_per_owner=2)
    job = queue.claim(worker_id="opensaebase-worker", lease_seconds=600)
    if job is None:
        return {"status": "empty"}
    jid, attempt = job["job_id"], job["attempt"]
    try:
        spec = job["spec"]
        site = next(s for s in core.load_sites() if s["site_id"] == spec["site_id"])
        prompt = load_input(root, spec["input_ref"])
        pair = load_registered(
            sae_key=site["sae"], layer=site["layer"], model_key=site["model"], device=device
        )
        if (
            sha256_state_dict(pair.model.model) != site["model_sha256"]
            or sha256_state_dict(pair.sae) != site["sae_sha256"]
        ):
            raise ValueError("Original model or SAE checkpoint identity has changed")
        op = spec["operation"]
        if spec["max_new_tokens"]:
            result = _generate(pair, site, prompt, spec)
        elif op == "inspect":
            from saebase.explore import capture_prompt_trace

            trace = capture_prompt_trace(
                sae_key=site["sae"],
                layer=site["layer"],
                prompt=prompt,
                device=device,
                sample_id=jid,
                prompt_group_id=jid,
                max_tokens=128,
                pair=pair,
            )
            result = {
                "schema": "opensae.wp12.private_prompt_inspection.v1",
                "operation": "inspect",
                "status": "real_local_inference",
                "model": site["model"],
                "model_sha256": trace["model_sha256"],
                "sae_sha256": trace["sae_sha256"],
                "tokenizer_sha256": trace["tokenizer_sha256"],
                "prompt_sha256": trace["prompt_sha256"],
                "input_sha256": spec["input_ref"],
                "site_id": site["site_id"],
                "feature_id": None,
                "tokens": trace["tokens"],
                "public_release": False,
                "logits_retained": False,
                "source_commit": _source_commit(),
            }
        else:
            edit = Edit(
                op,
                tuple(spec["feature_ids"]),
                spec["strength"] if op in {"scale", "clamp", "add"} else 0.0,
            )
            plan = EditPlan(site["sae"], site["layer"], -1, (edit,))
            response = compare_sae_edits(pair, pair.model.tokens(prompt=prompt), plan)
            baseline = response["baseline_logits"].detach().float().cpu()
            edited = response["edited_logits"].detach().float().cpu()
            if baseline.ndim != 1 or baseline.shape != edited.shape:
                raise RuntimeError("Teacher-forced readout logits have an unexpected shape")
            result = {
                k: v for k, v in response.items() if k not in {"baseline_logits", "edited_logits"}
            }
            activation, active = _edit_activation(pair, site, prompt, spec["feature_ids"])
            result["max_abs_logit_delta"] = float((edited - baseline).abs().max())
            result["baseline_argmax_token_id"] = int(baseline.argmax())
            result["edited_argmax_token_id"] = int(edited.argmax())
            result["logits_retained"] = False
            result["model"] = site["model"]
            result["site_id"] = site["site_id"]
            result["feature_id"] = spec["feature_ids"][0]
            result["operation"] = op
            result["input_sha256"] = spec["input_ref"]
            result["feature_activation"] = activation
            result["feature_active_at_edit_position"] = active
            result["public_release"] = False
            result["source_commit"] = _source_commit()
            if not active:
                result["message"] = (
                    "This feature is inactive at the last token, so the edit does not change "
                    "the next-token logits."
                )
        result["job_id"] = jid
        output = root / "results" / (jid + ".json")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        content = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        tmp = output.with_suffix(".partial")
        tmp.write_bytes(content)
        os.chmod(tmp, 0o600)
        os.replace(tmp, output)
        digest = hashlib.sha256(content).hexdigest()
        accepted = queue.finish(
            job_id=jid, worker_id="opensaebase-worker", attempt=attempt, result_sha256=digest
        )
        return {
            "job_id": jid,
            "status": "succeeded" if accepted else "lease_lost",
            "result_sha256": digest if accepted else None,
        }
    except Exception:
        queue.finish(
            job_id=jid,
            worker_id="opensaebase-worker",
            attempt=attempt,
            error_code="WORKER_EXECUTION_FAILED",
        )
        return {"job_id": jid, "status": "failed", "error_code": "WORKER_EXECUTION_FAILED"}

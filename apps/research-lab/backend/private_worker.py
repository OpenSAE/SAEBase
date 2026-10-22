"""Execute one real, hash-identified teacher-forced SAE job privately.

Only owner-scoped hashes are returned by the HTTP API. The native worker
runs on the user's machine, which must have original model/SAE weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import core
from .jobs import JobQueue
from .private_api import load_input


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
        if spec["max_new_tokens"]:
            raise ValueError("Only actual teacher-forced outputs supported here")
        site = next(s for s in core.load_sites() if s["site_id"] == spec["site_id"])
        prompt = load_input(root, spec["input_ref"])
        pair = load_registered(sae_key=site["sae"], layer=site["layer"],
                               model_key=site["model"], device=device)
        if (sha256_state_dict(pair.model.model) != site["model_sha256"]
                or sha256_state_dict(pair.sae) != site["sae_sha256"]):
            raise ValueError("Original model or SAE checkpoint identity has changed")
        op = spec["operation"]
        edit = Edit(op, tuple(spec["feature_ids"]),
                    spec["strength"] if op in {"scale", "clamp", "add"} else 0.0)
        plan = EditPlan(site["sae"], site["layer"], -1, (edit,))
        response = compare_sae_edits(pair, pair.model.tokens(prompt=prompt), plan)
        result = {k: v for k, v in response.items()
                  if k not in {"baseline_logits", "edited_logits"}}
        result["baseline_logits"] = response["baseline_logits"].tolist()
        result["edited_logits"] = response["edited_logits"].tolist()
        result["job_id"] = jid
        output = root / "results" / (jid + ".json")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        content = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        tmp = output.with_suffix(".partial")
        tmp.write_bytes(content)
        os.chmod(tmp, 0o600)
        os.replace(tmp, output)
        digest = hashlib.sha256(content).hexdigest()
        accepted = queue.finish(job_id=jid, worker_id="opensaebase-worker",
                                attempt=attempt, result_sha256=digest)
        return {"job_id": jid, "status": "succeeded" if accepted else "lease_lost",
                "result_sha256": digest if accepted else None}
    except Exception:
        queue.finish(job_id=jid, worker_id="opensaebase-worker",
                     attempt=attempt, error_code="WORKER_EXECUTION_FAILED")
        return {"job_id": jid, "status": "failed", "error_code": "WORKER_EXECUTION_FAILED"}

"""Structured evaluation results shared across SAEBase tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def normalize_status(value: str) -> str:
    aliases = {
        "PASS": "executed",
        "passed": "executed",
        "completed": "executed",
        "FAIL": "failed",
        "BLOCKED": "blocked",
        "NOT_RUN": "configured_only",
    }
    status = aliases.get(value, value)
    if status not in {"executed", "configured_only", "failed", "blocked", "unsupported"}:
        raise ValueError(f"Unknown task execution status: {value!r}")
    return status


def completion_summary(
    rows: list[dict[str, Any]], *, expected: int | None = None, unit: str = "operation"
) -> dict:
    """Summarize required work without treating mathematical undefined values as errors."""
    counts = {name: 0 for name in ("completed", "undefined", "blocked", "unsupported", "failed")}
    reasons = []
    for raw in rows:
        row = raw.to_dict() if hasattr(raw, "to_dict") else raw
        status = row.get("status")
        if status not in counts:
            raise ValueError(f"Unknown work result status: {status!r}")
        if status == "undefined" and not row.get("reason"):
            raise ValueError("Undefined results require a mathematical reason")
        counts[status] += 1
        if row.get("reason"):
            reasons.append({"status": status, "reason": row["reason"]})
    requested = len(rows) if expected is None else expected
    if type(requested) is not int or requested < 0 or requested < len(rows):
        raise ValueError("Invalid requested work count")
    missing = requested - len(rows)
    complete = not missing and not any(counts[s] for s in ("failed", "blocked", "unsupported"))
    return {
        "status": "failed" if counts["failed"] else "complete" if complete else "incomplete",
        "requested": requested,
        "completed": counts["completed"],
        "unit": unit,
        "evaluated": counts["completed"] + counts["undefined"],
        "counts": counts,
        "missing": missing,
        "reasons": reasons,
    }


def result_diagnostics(task, status, summary, meta, completion, error):
    """Explain observed task problems using task-owned reasons and repair instructions."""
    if status not in {"blocked", "unsupported", "failed"} and completion.get("status") not in {"failed", "incomplete"}:
        return []
    rows = []
    for container in (meta, summary):
        if isinstance(container.get("rows"), list):
            rows.extend(container["rows"])
    precheck = summary.get("automatic_precheck", {})
    if precheck:
        rows.extend(precheck.get("meta", {}).get("rows", []))
    rows.extend(completion.get("reasons", []))
    if summary.get("reason") or summary.get("repair"):
        rows.append(summary)
    issues = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", status)).lower() in {"pass", "completed", "executed", "undefined"}:
            continue
        reason = row.get("reason")
        if not reason:
            continue
        repair = row.get("repair")
        key = (row.get("stage"), row.get("code"), str(reason), str(repair))
        if key in seen:
            continue
        seen.add(key)
        issues.append({
            "code": row.get("code", "TASK_INCOMPLETE"),
            "stage": row.get("stage", task),
            "reason": str(reason),
            "impact": "precheck failed, and this scientific mission has not yet been carried out." if precheck else "This request is still pending.",
            "next_steps": [str(repair)] if repair else ["Check the configuration of the task and the corresponding input for this reason and continue the task after it has been amended."],
            "evidence": row.get("evidence", {}),
        })
    incomplete = status in {"blocked", "unsupported", "failed"} or completion.get("status") in {"failed", "incomplete"}
    if not issues and incomplete:
        issues.append({
            "code": "TASK_INCOMPLETE" if error else "CAUSE_NOT_RECORDED",
            "stage": task,
            "reason": str(error) if error else "Existing results mark uncompleted state, for which specific reasons have not been documented.",
            "impact": "This request is still pending.",
            "next_steps": ["Read the error log and pre-check report for this task, determine the cause and correct the corresponding input or continue after it is achieved."],
            "evidence": {},
        })
    return issues


@dataclass
class EvaluationResult:
    """Return value of an SAE evaluation run."""

    ok: bool
    task: str
    evaluation: str
    model_key: str
    prompt_source: str
    sae_key: str | None = None
    layer: int | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    per_prompt: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    status: str = "executed"
    artifact_paths: dict[str, str] = field(default_factory=dict)
    completion: dict[str, Any] = field(default_factory=dict)
    format_version: int = 2
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = normalize_status(self.status)
        if self.completion and self.completion.get("status") not in {
            "complete",
            "incomplete",
            "failed",
        }:
            raise ValueError("Unknown completion status")
        if self.completion:
            counts = self.completion.get("counts", {})
            keys = {"completed", "undefined", "blocked", "unsupported", "failed"}
            if set(counts) != keys or any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError("Invalid completion counts")
            for name in ("requested", "completed", "evaluated", "missing"):
                if type(self.completion.get(name)) is not int or self.completion[name] < 0:
                    raise ValueError("Invalid completion totals")
            missing = self.completion["missing"]
            expected_status = (
                "failed"
                if counts["failed"]
                else (
                    "incomplete"
                    if missing or counts["blocked"] or counts["unsupported"]
                    else "complete"
                )
            )
            if (
                self.completion["requested"] != sum(counts.values()) + missing
                or self.completion["completed"] != counts["completed"]
                or self.completion["evaluated"] != counts["completed"] + counts["undefined"]
                or self.completion["status"] != expected_status
            ):
                raise ValueError("Inconsistent completion summary")
        if self.status in {"failed", "blocked", "unsupported"} or (
            self.completion and self.completion["status"] != "complete"
        ):
            self.ok = False

        if not self.diagnostics:
            self.diagnostics = result_diagnostics(
                self.task, self.status, self.summary, self.meta, self.completion, self.error
            )

    @property
    def exit_code(self) -> int:
        if self.status == "configured_only":
            return 0
        if self.status == "failed" or self.completion.get("status") == "failed":
            return 1
        if (
            self.status in {"blocked", "unsupported"}
            or self.completion.get("status") == "incomplete"
        ):
            return 2
        return 0 if self.ok else 1

    @classmethod
    def from_payload(cls, operation: str, cfg: Any, payload: dict[str, Any]) -> "EvaluationResult":
        """Wrap an operation payload at the task boundary without changing its measurements."""
        status = normalize_status(str(payload.get("status")))
        ok = status not in {"failed", "blocked", "unsupported"}
        dataset = cfg.get("dataset", cfg.get("prompt_source", {}))
        paths = {}
        if payload.get("result_path"):
            paths["result"] = str(payload["result_path"])
        return cls(
            ok=ok,
            task=operation,
            evaluation=operation,
            model_key=str(payload.get("model_key", cfg.get("model", {}).get("name", ""))),
            prompt_source=str(
                payload.get("resolved_protocol", {}).get("dataset") or dataset.get("name", "")
            ),
            sae_key=payload.get("sae_key"),
            layer=payload.get("layer"),
            summary=payload,
            meta={"seed": cfg.get("seed")},
            error=str(payload.get("error")) if payload.get("error") else None,
            status=status,
            artifact_paths=paths,
            completion=payload.get("completion", {}),
            diagnostics=payload.get("diagnostics", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

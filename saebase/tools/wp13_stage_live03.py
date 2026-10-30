#!/usr/bin/env python3
"""Convert a measured LIVE-03 summary into optional reviewed public intervention cards.

No GPU is run. By default this validates and prints a *private* count; it
does not publish. Source results and model/SAE SHA are never manufactured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "saebase" / "research_lab"))
from backend import core, published_records  # noqa: E402

SCHEMA = "opensae.wp13.live03_teacher_forced_summary.v1"
PUBLIC_SCHEMA = "opensae.wp12.p3.public_result.v1"
SHA40 = "0123456789abcdef"
SHA64 = SHA40


def _sha(value: Any, size: int) -> bool:
    return isinstance(value, str) and len(value) == size and all(ch in SHA40 for ch in value)


def _measure(measure: Any, *, original: int, no_op: bool = False) -> dict[str, Any]:
    if not isinstance(measure, dict):
        raise ValueError("Expected actual original-logit metrics")
    needed = {
        "max_abs_logit_delta",
        "logit_delta_l2",
        "baseline_argmax_token_id",
        "edited_argmax_token_id",
        "argmax_changed",
    }
    if set(measure) != needed:
        raise ValueError("Unknown or missing teacher-forced metric fields")
    max_delta, l2 = measure["max_abs_logit_delta"], measure["logit_delta_l2"]
    if (
        type(max_delta) not in (float, int)
        or type(l2) not in (float, int)
        or not math.isfinite(max_delta)
        or not math.isfinite(l2)
        or max_delta < 0
        or l2 < 0
        or max_delta > l2 + 1e-4
    ):
        raise ValueError("Impossible nonfinite/negative logit metrics")
    if (
        type(measure["baseline_argmax_token_id"]) is not int
        or type(measure["edited_argmax_token_id"]) is not int
        or measure["baseline_argmax_token_id"] != original
        or type(measure["argmax_changed"]) is not bool
        or measure["argmax_changed"] != (original != measure["edited_argmax_token_id"])
    ):
        raise ValueError("Reported argmax change disagrees with native token IDs")
    if no_op and (max_delta != 0 or l2 != 0 or measure["argmax_changed"]):
        raise ValueError("Identity/no-op must not report changed logits")
    return {key: measure[key] for key in sorted(needed)}


def _readout(value: Any, *, expected_position: int, seq_length: int) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "readout_position",
        "logit_kind",
        "generated_text",
        "baseline_argmax_token_id",
        "noop",
        "ablation",
        "scale",
    }:
        raise ValueError("Unexpected readout shape")
    if (
        type(value["readout_position"]) is not int
        or value["readout_position"] != expected_position
        or not 0 <= expected_position < seq_length
        or value["logit_kind"] != "teacher_forced_full_vocab"
        or value["generated_text"] is not False
        or type(value["baseline_argmax_token_id"]) is not int
    ):
        raise ValueError("Invalid source readout identity / generated-text label")
    source = value["baseline_argmax_token_id"]
    return {
        "position": expected_position,
        "baseline_argmax_token_id": source,
        "noop": _measure(value["noop"], original=source, no_op=True),
        "ablation": _measure(value["ablation"], original=source),
        "scale": _measure(value["scale"], original=source),
    }


def translate_summary(source: Path) -> list[dict[str, Any]]:
    """Validate source-bound measured data and produce allowlisted gallery records."""
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 40_000:
        raise ValueError("A small actual measurement JSON file is required")
    original_bytes = source.read_bytes()
    report = json.loads(original_bytes)
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("job_id") != "LIVE-03"
        or report.get("status") != "PASS_WORKING"
        or report.get("generated_text") is not False
        or report.get("logit_kind") != "teacher_forced_full_vocab"
        or not _sha(report.get("code_sha"), 40)
        or type(report.get("scale_factor")) not in (int, float)
        or not math.isfinite(report["scale_factor"])
        or report["scale_factor"] <= 0
        or not isinstance(report.get("sites"), list)
        or not 1 <= len(report["sites"]) <= 4
    ):
        raise ValueError("Incorrect, unbounded or unexecuted LIVE-03 source")
    sites = {item["site_id"]: item for item in core.load_sites()}
    imported = []
    seen: set[str] = set()
    for measured in report["sites"]:
        if not isinstance(measured, dict):
            raise ValueError("Bad source site record")
        name = measured.get("site_id")
        site = sites.get(name)
        if (
            not site
            or not core.has_complete_identity(site)
            or name in seen
            or measured.get("model_sha256") != site["model_sha256"]
            or measured.get("sae_sha256") != site["sae_sha256"]
            or measured.get("hook_name") != site["hook"]
        ):
            raise ValueError("The reported real checkpoint/site is not in the verified registry")
        seen.add(name)
        feature_id, pos, token_id = (
            measured.get("feature_id"),
            measured.get("token_position"),
            measured.get("token_id"),
        )
        total = measured.get("sequence_length")
        if (
            not all(type(x) is int and x >= 0 for x in (feature_id, pos, token_id))
            or type(total) is not int
            or total < 2
            or total > 128
            or pos >= total
            or not _sha(measured.get("prompt_sha256"), 64)
            or not isinstance(measured.get("sample_id"), str)
            or not measured["sample_id"].strip()
        ):
            raise ValueError("Invalid exact prompt/feature/token identity")
        for key in ("saved_activation", "live_activation", "scale_replacement_value"):
            v = measured.get(key)
            if type(v) not in (float, int) or not math.isfinite(v):
                raise ValueError("Invalid measured native SAE activation")
        if (
            measured.get("scale_factor") != report["scale_factor"]
            or abs(
                measured["scale_replacement_value"]
                - measured["live_activation"] * measured["scale_factor"]
            )
            > 0.001
        ):
            raise ValueError("Reported scale does not correspond to measured live code")
        readouts = {
            "feature_position": _readout(
                measured.get("readout_at_feature_position"),
                expected_position=pos,
                seq_length=total,
            ),
            "final_position": _readout(
                measured.get("readout_at_final_token"),
                expected_position=total - 1,
                seq_length=total,
            ),
        }
        imported.append(
            {
                "schema": PUBLIC_SCHEMA,
                "site": name,
                "sample_id": measured["sample_id"],
                "prompt_sha256": measured["prompt_sha256"],
                "model_sha256": measured["model_sha256"],
                "sae_sha256": measured["sae_sha256"],
                "hook_name": site["hook"],
                "feature_id": feature_id,
                "token_id": token_id,
                "token_position": pos,
                "sequence_length": total,
                "measured_activation": float(measured["live_activation"]),
                "scale_factor": float(report["scale_factor"]),
                "measurement": "teacher_forced_full_vocab_logits_only",
                "generated_text": False,
                "measured": True,
                "original_agent_run": "LIVE-03",
                "original_source_code_sha": report["code_sha"],
                "source_report_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "readouts": readouts,
                "interpretation": (
                    "Source-recorded real checkpoint intervention; at each readout, "
                    "a changed argmax is not equivalent to generated text."
                ),
            }
        )
    return imported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--publish-reviewed", action="store_true")
    parser.add_argument("--reviewer", type=str)
    args = parser.parse_args(argv)
    records = translate_summary(args.source)
    if not args.publish_reviewed:
        print(
            json.dumps(
                {
                    "status": "real_measurement_validated_not_publicly_released",
                    "site_count": len(records),
                    "source_report_sha256": records[0]["source_report_sha256"],
                }
            )
        )
        return 0
    if args.release_root is None or not args.reviewer or not args.reviewer.strip():
        parser.error("Public staging requires --release-root and a named source reviewer")
    # This local command stages operator-approved records only. It does not
    # deploy to HF and cannot substitute for public storage approval.
    published = []
    for record in records:
        receipt = published_records.publish_record(
            "interventions",
            {**record, "reviewed_by": args.reviewer},
            root=args.release_root,
            public_consent=True,
            rights_reviewed=True,
        )
        published.append(receipt["record_sha256"])
    print(
        json.dumps(
            {
                "status": "approved_records_locally_staged_not_deployed",
                "record_ids": published,
                "release_root": str(args.release_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

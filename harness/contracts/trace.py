"""Structural validation of compact, observable WP18 Contract receipts.

A receipt cannot independently prove that a tool ran; the evaluator cross-
checks it with source changes and actual execution logs.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.contracts.registry import ContractError, ROOT, catalog, resolve, select_tags


def ad_hoc_selection(tags: list[str]) -> list[str]:
    """Use only predeclared IDs for user-defined sandbox tasks."""
    return select_tags(tags)["contract_ids"]


def validate_receipt(record: dict, *, expected: list[str]) -> dict:
    if (
        not isinstance(record, dict)
        or record.get("schema") != "saebase.harness.contract_receipt.v1"
    ):
        raise ContractError("Invalid receipt schema")
    if record.get("registry_version") != catalog()["version"]:
        raise ContractError("Contract registry version mismatch")
    if record.get("contract_ids") != expected:
        raise ContractError("Contract IDs differ from the resolved immutable selection")
    if not isinstance(record.get("case_id"), str) or not record["case_id"]:
        raise ContractError("Missing case ID")
    for field in ("context_paths", "edited_paths", "unresolved"):
        if not isinstance(record.get(field), list) or not all(
            isinstance(x, str) for x in record[field]
        ):
            raise ContractError("Missing observable " + field)
    if not isinstance(record.get("commands"), list):
        raise ContractError("Missing executed command list")
    for command in record["commands"]:
        if not isinstance(command, dict) or set(command) != {"argv", "exit_code"}:
            raise ContractError("Command must record argv and exit code")
        if (
            not isinstance(command["argv"], list)
            or not all(isinstance(x, str) for x in command["argv"])
            or not isinstance(command["exit_code"], int)
        ):
            raise ContractError("Invalid command receipt")
    for relative in record["edited_paths"]:
        if not isinstance(relative, str) or relative.startswith("/") or "\\" in relative:
            raise ContractError("Invalid edited path")
        if any(x in ("..", ".") for x in relative.split("/")):
            raise ContractError("Edited path traversal")
        if not relative.startswith("harness/tests/"):
            raise ContractError("Exam receipt edits outside its test sandbox")
        if relative.startswith("harness/tests/wp18/acceptance/") or relative in (
            "harness/tests/README.md",
            "harness/tests/wp18/README.md",
        ):
            raise ContractError("Exam receipt cannot edit independent authority")
        if (
            not (ROOT / relative)
            .resolve(strict=False)
            .is_relative_to((ROOT / "harness/tests").resolve(strict=False))
        ):
            raise ContractError("Edited path escapes test sandbox")
    return {
        "case_id": record["case_id"],
        "contract_count": len(expected),
        "declared_edits": len(record["edited_paths"]),
        "declared_commands": len(record["commands"]),
        "unresolved": len(record["unresolved"]),
        "verified_execution": False,
    }


def check_receipt_file(path: Path, *, plan: str, case: str) -> dict:
    location = path.resolve()
    if not location.is_relative_to((ROOT / "harness/tests").resolve()):
        raise ContractError("Receipt must live inside Harness test sandbox")
    doc = json.loads(location.read_text(encoding="utf-8"))
    if doc.get("case_id") != case:
        raise ContractError("Receipt case mismatch")
    return validate_receipt(doc, expected=resolve(plan, case)["contract_ids"])

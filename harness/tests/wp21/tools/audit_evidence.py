"""Collect/check real-workflow evidence. This is NOT a behavioral grader.

Independent reviewers must still replay every frozen per-case positive and
negative check. This tool can reject missing, forged-shaped, stale, mismatched
or corrupted records; hashes cannot prove the claimed computation happened.
"""

from __future__ import annotations
import argparse
import difflib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    BASE_COMMIT,
    ROOT,
    SUITE,
    canonical_hash,
    contained,
    get_case,
    read_json,
    sha256,
    write_json,
)
from manage import scope_audit

HEX = re.compile(r"[a-f0-9]{64}")


def artifacts(workspace: Path) -> list[dict]:
    root = workspace / "experiment"
    if not root.is_dir() or root.is_symlink():
        raise ValueError("owned experiment/ directory required")
    result = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(workspace).as_posix()
        contained(workspace, rel, exists=True)
        if path.is_file():
            result.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return result


def required_artifact_present(
    name: str, *, evidence: Path, workspace: Path, experiment_files: list[dict]
) -> bool:
    """Match frozen file or directory deliverables by type, not string prefix.

    The trailing slash in a frozen required_artifact means a *nonempty
    directory*. An experiment directory counts only when a regular file below
    it appears in the independently collected, byte-hashed experiment manifest.
    Evidence directories are inspected in place and must contain a regular
    non-symlink file. Directory names are matched at component boundaries.
    """
    is_directory = name.endswith("/")
    logical = name[:-1] if is_directory else name
    from common import relative_name

    relative_name(logical)  # reject traversal, absolute and ambiguous paths
    if is_directory:
        # Most directory deliverables are directly under the case evidence root.
        target = contained(evidence, logical)
        if target.is_dir():
            for entry in target.rglob("*"):
                relative = entry.relative_to(evidence).as_posix()
                contained(evidence, relative, exists=True)  # no symlink components
                if entry.is_file():
                    return True
        # A logical directory can alternatively live inside a hashed run leaf.
        # We must match the entire logical path, not merely a prefix or suffix
        # of the basename (e.g., output/ cannot match output_backup/).
        root = workspace / "experiment"
        for folder in root.rglob("*"):
            rel = folder.relative_to(workspace).as_posix()
            contained(workspace, rel, exists=True)
            if folder.is_dir() and (rel.endswith("/" + logical)):
                prefix = rel + "/"
                if any(item["path"].startswith(prefix) for item in experiment_files):
                    return True
        return False
    target = contained(evidence, logical)
    if target.is_file():
        return True
    return any(item["path"].endswith("/" + logical) for item in experiment_files)


def evidence_folder(workspace: Path, control: Path) -> tuple[dict, Path]:
    assignment = read_json(control / "assignment.json")
    return assignment, contained(workspace, assignment["evidence_root"], exists=True)


def original_bytes(relative: str, case: dict) -> bytes:
    seed = case.get("seed_fixture")
    if seed:
        seedroot = SUITE / "fixtures" / seed
        for item in read_json(seedroot / "manifest.json")["files"]:
            if item["destination"] == relative:
                return contained(seedroot, item["source"], exists=True).read_bytes()
    path = contained(ROOT, relative)
    return path.read_bytes() if path.is_file() else b""


def collect(workspace: Path, control: Path) -> dict:
    assignment, folder = evidence_folder(workspace, control)
    scope = scope_audit(workspace, control)
    if scope["violations"]:
        raise ValueError("source scope violation: " + ", ".join(scope["violations"]))
    entries = artifacts(workspace)
    case = get_case(assignment["case_id"])
    if sum(x["size_bytes"] for x in entries) > case["budget"]["max_new_artifact_mb"] * 1024 * 1024:
        raise ValueError("artifact size exceeds frozen case budget")
    write_json(
        folder / "artifacts_manifest.json",
        {
            "schema": "saebase.wp21.artifacts.v1",
            "files": entries,
            "sha256": canonical_hash(entries),
        },
    )
    write_json(folder / "source_manifest.json", scope)
    patch = []
    for change in scope["changes"]:
        rel = change["path"]
        before = original_bytes(rel, case)
        target = contained(workspace, rel)
        after = target.read_bytes() if target.is_file() else b""
        try:
            left, right = before.decode("utf-8"), after.decode("utf-8")
        except UnicodeDecodeError:
            patch.append("Binary source change (see SHA256 manifest): " + rel + "\n")
            continue
        patch.extend(
            difflib.unified_diff(
                left.splitlines(keepends=True),
                right.splitlines(keepends=True),
                fromfile="a/" + rel,
                tofile="b/" + rel,
            )
        )
    (folder / "diff.patch").write_text("".join(patch), encoding="utf-8")
    return {
        "status": "COLLECTED_NOT_GRADED",
        "source_snapshot_sha256": scope["source_snapshot_sha256"],
        "changed_files": len(scope["changes"]),
        "artifact_files": len(entries),
    }


def contains_configured_only(value) -> bool:
    if isinstance(value, dict):
        return value.get("status") in {"configured_only", "dry_run", "planned"} or any(
            contains_configured_only(v) for v in value.values()
        )
    if isinstance(value, list):
        return any(contains_configured_only(v) for v in value)
    return False


def audit(workspace: Path, control: Path) -> dict:
    assignment, folder = evidence_folder(workspace, control)
    case = get_case(assignment["case_id"])
    report = read_json(folder / "evidence.json")
    if report.get("status") in {"BLOCKED", "ENV_BLOCKED"}:
        return {
            "status": report["status"],
            "case_id": case["id"],
            "unresolved": report.get("unresolved", []),
            "grade": None,
        }
    if assignment.get("state") != "assigned" or not assignment.get("assignment_ref"):
        raise ValueError("candidate has no actual assigned exam")
    if report.get("status") != "SELF_TESTED" or report.get("evidence_kind") != "real_checkpoint":
        raise ValueError("only SELF_TESTED real_checkpoint evidence can enter independent replay")
    if (
        report.get("schema") != "saebase.wp21.evidence.v1"
        or report.get("case_id") != case["id"]
        or report.get("run_id") != assignment["run_id"]
    ):
        raise ValueError("evidence identity mismatch")
    if report.get("base_commit") != BASE_COMMIT:
        raise ValueError("base source mismatch")
    scope = scope_audit(workspace, control)
    if (
        scope["violations"]
        or report.get("source_snapshot_sha256") != scope["source_snapshot_sha256"]
    ):
        raise ValueError("scope or final source identity mismatch")
    source = read_json(folder / "source_manifest.json")
    if source != scope:
        raise ValueError("source manifest is stale")
    actual = artifacts(workspace)
    manifest = read_json(folder / "artifacts_manifest.json")
    if manifest.get("files") != actual or manifest.get("sha256") != canonical_hash(actual):
        raise ValueError("artifact manifest mismatch or omitted/extra file")
    if not actual:
        raise ValueError("no experiment artifacts")
    if sum(x["size_bytes"] for x in actual) > case["budget"]["max_new_artifact_mb"] * 1024 * 1024:
        raise ValueError("artifact budget exceeded")
    for key in ["input_identity", "model_identity", "environment"]:
        if not isinstance(report.get(key), dict) or not report[key]:
            raise ValueError("missing concrete " + key)
    if case["id"] not in {"R08", "R09"} and not isinstance(report.get("sae_identity"), dict):
        raise ValueError("this case needs a real source/trained SAE identity")
    model = report["model_identity"]
    if (
        not isinstance(model.get("revision"), str)
        or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", model["revision"]) is None
    ):
        raise ValueError("model revision must be immutable")
    if not HEX.fullmatch(str(model.get("files_sha256", ""))):
        raise ValueError("model file identity must reference the trusted assets lock digest")
    if not HEX.fullmatch(str(report["input_identity"].get("sha256", ""))):
        raise ValueError("actual input fingerprint missing")
    cfg_rel = report.get("resolved_config_path")
    cfg = read_json(contained(workspace, cfg_rel, exists=True))
    if report.get("canonical_config_sha256") != canonical_hash(cfg):
        raise ValueError("canonical resolved-config fingerprint mismatch")
    runtime_hash = report.get("runtime_config_sha256")
    if runtime_hash is not None and not HEX.fullmatch(str(runtime_hash)):
        raise ValueError("bad original runtime config hash")
    results = report.get("executed_result_paths")
    if not isinstance(results, list) or not results:
        raise ValueError("actual result files required")
    manifest_paths = {x["path"] for x in actual}
    for relative in results:
        if relative not in manifest_paths:
            raise ValueError("executed result must be a hashed experiment artifact")
        value = read_json(contained(workspace, relative, exists=True))
        if contains_configured_only(value):
            raise ValueError("configured_only/dry-run is not model execution")
    # Required case artifacts can live in the case evidence folder or beneath
    # actual experiment/ outputs; their existence is necessary, never
    # sufficient, to establish correct case behavior.
    missing_artifacts = [
        name
        for name in case["required_artifacts"]
        if not required_artifact_present(
            name, evidence=folder, workspace=workspace, experiment_files=actual
        )
    ]
    if missing_artifacts:
        raise ValueError("missing frozen required_artifacts: " + ", ".join(missing_artifacts))
    refs = report.get("command_refs")
    if not isinstance(refs, list) or not refs or len(refs) != len(set(refs)):
        raise ValueError("distinct command records required")
    commands = []
    all_recorded = {
        "commands/" + p.name for p in (folder / "commands").glob("[0-9][0-9][0-9][0-9].json")
    }
    if set(refs) != all_recorded:
        raise ValueError("command references omit or invent an executed attempt")
    for relative in refs:
        command_path = contained(folder, relative, exists=True)
        command = read_json(command_path)
        if (
            command.get("schema") != "saebase.wp21.command.v1"
            or type(command.get("exit_code")) is not int
        ):
            raise ValueError("malformed command record")
        if not command.get("argv") or not all(isinstance(x, str) for x in command["argv"]):
            raise ValueError("record actual argv")
        if (
            command.get("case_id") != case["id"]
            or command.get("run_id") != assignment["run_id"]
            or command.get("assignment_sha256") != canonical_hash(assignment)
        ):
            raise ValueError("command was not recorded under trusted case assignment")
        if (
            command.get("cwd") != str(workspace.resolve())
            or command.get("time_budget_enforced_per_record") is not True
        ):
            raise ValueError("recorded process working tree or budget gate invalid")
        if (
            type(command.get("duration_seconds")) not in {int, float}
            or command["duration_seconds"] < 0
            or command.get("timed_out") not in {True, False}
        ):
            raise ValueError("bad recorded duration/timing state")
        if command.get("execution_claim") != "process_only":
            raise ValueError("command cannot claim more than a real process")
        for stream in ["stdout", "stderr"]:
            recorded = command[stream]
            path = contained(command_path.parent, recorded["path"], exists=True)
            if sha256(path) != recorded["sha256"] or path.stat().st_size != recorded["size_bytes"]:
                raise ValueError("command log integrity mismatch")
        commands.append(command)
    if sum(c["duration_seconds"] for c in commands) > case["budget"]["max_seconds"] + 0.1:
        raise ValueError("recorded processes exceeded cumulative case time budget")
    if not any(c["exit_code"] == 0 and not c.get("timed_out") for c in commands):
        raise ValueError("no successful process")
    receipt = read_json(folder / "contract_receipt.json")
    if (
        receipt.get("schema") != "saebase.wp21.contract_receipt.v1"
        or receipt.get("case_id") != case["id"]
        or receipt.get("registry_version") != "1.1.0"
    ):
        raise ValueError("use WP21 receipt, not the legacy sandbox receipt validator")
    if receipt.get("contract_ids") != case["contracts"]:
        raise ValueError("contract binding mismatch")
    if set(receipt.get("edited_paths", [])) != {c["path"] for c in scope["changes"]}:
        raise ValueError("receipt edits differ from observed diff")
    if receipt.get("commands") != [
        {"argv": c["argv"], "exit_code": c["exit_code"]} for c in commands
    ]:
        raise ValueError("receipt commands differ from recorded processes")
    for name in [
        "selfcheck.json",
        "report.md",
        "diff.patch",
        "expected_code_diff.json",
        "review_candidate.json",
    ]:
        contained(folder, name, exists=True)
    return {
        "status": "EVIDENCE_READY_FOR_REPLAY",
        "case_id": case["id"],
        "grade": None,
        "independent_real_execution": False,
        "artifact_files": len(actual),
        "command_count": len(commands),
        "remaining": [
            "Verify assets lock against actually loaded files and model/tokenizer/SAE identity.",
            "Replay every case-specific positive/negative check in a fresh reviewer workspace.",
            "Inspect actual imports, numerical traces, budgets, diff and frozen grading rubric.",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["collect", "audit"])
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    try:
        result = (
            collect(a.workspace.resolve(), a.control.resolve())
            if a.command == "collect"
            else audit(a.workspace.resolve(), a.control.resolve())
        )
        if a.output:
            if a.output.resolve().is_relative_to(a.workspace.resolve()):
                raise ValueError("review report belongs outside candidate workspace")
            write_json(a.output, result, exclusive=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 3 if result["status"] in {"BLOCKED", "ENV_BLOCKED"} else 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(
            json.dumps(
                {"status": "EVIDENCE_REJECTED", "reason": str(exc), "grade": None},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Record an actual process, its exit status and log hashes, without shell eval.

A successful process record is not a claim of real checkpoint execution.
Use the trusted package copy for independent reviewer replay.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, canonical_hash, contained, get_case, read_json, sha256, write_json


def case_authorization(cwd: Path, output: Path, timeout: int, control: Path | None) -> dict | None:
    """Require an externally stored dispatcher assignment for candidate jobs.

    This is a logical execution gate, NOT a container sandbox. Independent
    reviewer additionally audits actual command logs and source provenance.
    """
    marker = cwd / ".cursor/wp21_assignment.json"
    if not marker.exists() and control is None:
        return None  # General-purpose recorder and trusted reviewer workspace.
    if not marker.is_file() or control is None:
        raise ValueError("WP21 candidate recorder requires --control and existing assignment")
    control = control.resolve(strict=True)
    if control.is_relative_to(cwd) or control.is_relative_to(ROOT):
        raise ValueError("trusted control must be outside candidate and package")
    assignment = read_json(control / "assignment.json")
    if read_json(marker) != assignment or assignment.get("state") != "assigned" or not assignment.get("assignment_ref"):
        raise ValueError("no authentic assigned dispatcher record")
    case = get_case(assignment["case_id"])
    if assignment.get("case_sha256") != canonical_hash(case) or assignment.get("budget") != case["budget"]:
        raise ValueError("assignment case/budget differs from frozen examiner spec")
    if assignment.get("suite_lock_sha256") and assignment["suite_lock_sha256"] != sha256(ROOT / "harness/tests/wp21/suite_lock.json"):
        raise ValueError("candidate references a different frozen suite")
    expected = contained(cwd, assignment["evidence_root"], exists=True) / "commands"
    if output.parent.resolve() != expected.resolve() or output.suffix != ".json" or not output.stem.isdigit() or len(output.stem) != 4:
        raise ValueError("candidate command record must be commands/NNNN.json under assigned evidence root")
    if timeout > case["budget"]["max_seconds"]:
        raise ValueError("requested timeout exceeds frozen case budget")
    previous = []
    if expected.is_dir():
        for path in sorted(expected.glob("[0-9][0-9][0-9][0-9].json")):
            rec = read_json(path)
            if rec.get("schema") != "saebase.wp21.command.v1" or rec.get("case_id") != case["id"] or rec.get("run_id") != assignment["run_id"]:
                raise ValueError("prior command log identity mismatch")
            if not isinstance(rec.get("duration_seconds"), (float, int)) or rec["duration_seconds"] < 0:
                raise ValueError("invalid prior duration")
            previous.append(rec)
    spent = sum(r["duration_seconds"] for r in previous)
    remaining = case["budget"]["max_seconds"] - spent
    if remaining <= 0 or timeout > remaining:
        raise ValueError(f"case wall-clock process budget exhausted; remaining={remaining:.3f}s")
    return {"case_id":case["id"], "run_id":assignment["run_id"],
            "assignment_sha256":canonical_hash(assignment),
            "case_max_seconds":case["budget"]["max_seconds"],
            "prior_recorded_seconds":round(spent,6),"remaining_before_seconds":round(remaining,6)}


def record(argv: list[str], cwd: Path, output: Path, timeout: int = 900, *, control: Path | None = None) -> dict:
    if not argv or not all(isinstance(x, str) and "\x00" not in x for x in argv):
        raise ValueError("argv must be a nonempty string list")
    if any(x.lower().startswith(("hf_token=", "--token=", "authorization:", "api_key=")) for x in argv):
        raise ValueError("credentials belong in the runtime's existing environment, not recorded argv")
    if not 1 <= timeout <= 3600:
        raise ValueError("timeout must be 1..3600 seconds and obey the case budget")
    cwd = cwd.resolve(strict=True)
    output = output.absolute()
    authorization = case_authorization(cwd, output, timeout, control)
    if output.exists() or output.with_suffix(".stdout.log").exists() or output.with_suffix(".stderr.log").exists():
        raise FileExistsError("command record is immutable; choose a new command index")
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout = output.with_suffix(".stdout.log")
    stderr = output.with_suffix(".stderr.log")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    tick = time.monotonic()
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(cwd)
    timed_out = False
    with stdout.open("xb") as out, stderr.open("xb") as err:
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, shell=False, start_new_session=(os.name == "posix"))
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                code = 124
        except OSError as exc:
            err.write((type(exc).__name__ + ": process launch failed\n").encode())
            code = 127
    result = {"schema": "saebase.wp21.command.v1", "argv": argv, "cwd": str(cwd),
              "started_at_utc": started, "duration_seconds": round(time.monotonic()-tick, 6),
              "exit_code": code, "timed_out": timed_out, "execution_claim": "process_only",
              "hub_offline_requested": True,
              "network_isolation": "external_environment_responsibility",
              "stdout": {"path": stdout.name, "sha256": sha256(stdout), "size_bytes": stdout.stat().st_size},
              "stderr": {"path": stderr.name, "sha256": sha256(stderr), "size_bytes": stderr.stat().st_size}}
    if authorization is not None:
        result.update(authorization)
        result["time_budget_enforced_per_record"] = True
        result["remaining_after_seconds"] = round(authorization["remaining_before_seconds"] - result["duration_seconds"], 6)
    write_json(output, result, exclusive=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cwd", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--control", type=Path, help="Trusted external control/Rxx directory, mandatory for candidate runs")
    p.add_argument("--propagate-exit", action="store_true",
                   help="Legacy behavior: use the recorded child exit code as the recorder CLI exit code")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    a = p.parse_args()
    argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
    try:
        result = record(argv, a.cwd, a.output, a.timeout, control=a.control)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Recorder success means that the *observation* was durably written.
        # An expected negative test may have child exit_code != 0. Its actual
        # exit code remains in the JSON and may be propagated on request.
        if a.propagate_exit:
            return result["exit_code"] if 0 <= result["exit_code"] <= 255 else 1
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "RECORD_ERROR", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Safe B2 Native API/rclone copy, check and recovery with auditable receipts.

Configured rclone credentials stay in the user's environment. Transfers are
dry-run by default; no command deletes files or overwrites altered evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def remote_path(text: str) -> str:
    """Require a subdirectory inside a named bucket, never a whole bucket."""
    if not re.fullmatch(
        r"[A-Za-z0-9_-]+:[A-Za-z0-9._-]+/[A-Za-z0-9._/@+=-][A-Za-z0-9._/@+=-]*",
        text or "",
    ):
        raise ValueError("Remote must be configured-remote:bucket/project/subdir")
    if ".." in text.split("/"):
        raise ValueError("Parent directory segment is not allowed")
    return text


def local_path(text: str, needs_exist: bool) -> str:
    path = Path(text).expanduser().resolve(strict=False)
    if needs_exist and not path.is_dir():
        raise ValueError(f"Local source directory not found: {path}")
    if path == Path(path.anchor):
        raise ValueError("Refusing a filesystem root as transfer scope")
    return str(path)


def command(
    action: str,
    local: str,
    remote: str,
    transfers: int = 16,
    checkers: int = 16,
    *,
    execute: bool = False,
    streams: int = 4,
    fast_list: bool = True,
    chunk_size: str = "64M",
    cutoff: str = "256M",
    retries: int = 10,
    low_level_retries: int = 20,
    upload_concurrency: int = 4,
) -> list[str]:
    """Construct safe, explicit rclone argv without reading credentials."""
    if action not in ("push", "pull", "verify"):
        raise ValueError("Unsupported transfer action")
    if (not 1 <= transfers <= 64 or not 1 <= checkers <= 64
            or not 1 <= streams <= 16 or not 1 <= upload_concurrency <= 16
            or not 1 <= retries <= 20 or not 1 <= low_level_retries <= 40):
        raise ValueError("Transfer performance settings out of bounds")
    if not re.fullmatch(r"[1-9][0-9]*[KMG]", chunk_size):
        raise ValueError("Invalid transfer chunk size")
    if not re.fullmatch(r"[1-9][0-9]*[KMG]", cutoff):
        raise ValueError("Invalid multithread cutoff")
    r = remote_path(remote)
    local_ref = local_path(local, action in ("push", "verify"))
    source, dest = (local_ref, r) if action in ("push", "verify") else (r, local_ref)
    op = "check" if action == "verify" else "copy"
    args = [
        "rclone", op, source, dest,
        "--transfers", str(transfers), "--checkers", str(checkers),
        "--retries", str(retries), "--low-level-retries", str(low_level_retries),
        "--stats", "10s", "--stats-one-line",
    ]
    if fast_list:
        args.append("--fast-list")
    if action == "verify":
        args += ["--one-way"]
    else:
        # Unlike sync, copy never deletes extra destination files. Immutable
        # also fails on a changed existing target instead of overwriting it.
        args += ["--immutable", "--multi-thread-streams", str(streams),
                 "--multi-thread-cutoff", cutoff, "--multi-thread-chunk-size", chunk_size]
        if action == "push":
            args += ["--b2-chunk-size", chunk_size,
                     "--b2-upload-concurrency", str(upload_concurrency)]
    if not execute and action != "verify":
        args += ["--dry-run"]
    return args


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("push", "pull", "verify"))
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--transfers", type=int, default=16)
    parser.add_argument("--checkers", type=int, default=16)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--cutoff", default="256M")
    parser.add_argument("--chunk-size", default="64M")
    parser.add_argument("--upload-concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--low-level-retries", type=int, default=20)
    parser.add_argument("--no-fast-list", action="store_true")
    parser.add_argument("--receipt", type=Path, help="Unique immutable local result receipt")
    parser.add_argument("--execute", action="store_true", help="Explicitly access B2")
    args = parser.parse_args()
    try:
        cmd = command(
            args.action, args.local, args.remote, args.transfers, args.checkers,
            execute=args.execute, streams=args.streams, fast_list=not args.no_fast_list,
            chunk_size=args.chunk_size, cutoff=args.cutoff, retries=args.retries,
            low_level_retries=args.low_level_retries,
            upload_concurrency=args.upload_concurrency,
        )
    except ValueError as exc:
        parser.error(str(exc))
    record = {
        "schema": "saebase.b2_transfer.v1",
        "action": args.action,
        "mode": "execute" if args.execute else "preview",
        "rclone_args": cmd,
        "destructive_delete": False,
        "credentials_in_source": False,
        "started_utc": _utc(),
    }
    # Print argv only: no secret keys, remote configuration or env values.
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return 0
    start = time.monotonic()
    if shutil.which("rclone") is None:
        code = 2
        print("rclone is not installed", file=sys.stderr)
    else:
        try:
            code = subprocess.run(cmd, check=False).returncode
        except OSError as exc:
            code = 2
            print(f"rclone execution error: {exc}", file=sys.stderr)
    record.update({
        "exit_code": code, "status": "PASS" if code == 0 else "BLOCKED",
        "elapsed_seconds": time.monotonic() - start,
        "finished_utc": _utc(),
        "remote_integrity": "CHECKED" if args.action == "verify" and code == 0
        else "NOT_RUN",
        "independent_sha256_manifest": "REQUIRED_FOR_SCIENTIFIC_ARCHIVE",
    })
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Safe rclone Backblaze B2 copy/check wrapper (dry-run by default, never deletes)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def remote_path(text: str) -> str:
    # remote:bucket/nonempty-prefix. Do not let a typo transfer an entire bucket.
    if not re.fullmatch(r"[A-Za-z0-9_-]+:[A-Za-z0-9._-]+/[A-Za-z0-9._/@+=-][A-Za-z0-9._/@+=-]*",text or ""):
        raise ValueError("Remote must be of the form b2remote:bucket/project/subdir (not bucket root)")
    if ".." in text.split("/"):
        raise ValueError("Parent directory segment is not allowed")
    return text


def local_path(text: str, needs_exist: bool) -> str:
    path=Path(text).expanduser().resolve(strict=False)
    if needs_exist and not path.is_dir():
        raise ValueError(f"Local source directory not found: {path}")
    if path == Path(path.anchor):
        raise ValueError("Refusing a filesystem root as transfer scope")
    return str(path)


def command(action: str, local: str, remote: str, transfers: int, checkers: int, *, execute: bool) -> list[str]:
    if action not in ("push","pull","verify"):
        raise ValueError("Unsupported transfer action")
    if not 1<=transfers<=32 or not 1<=checkers<=64:
        raise ValueError("Concurrency out of bounds")
    r=remote_path(remote)
    l=local_path(local, action in ("push","verify"))
    source,dest=(l,r) if action in ("push","verify") else (r,l)
    op="check" if action=="verify" else "copy"
    args=["rclone",op,source,dest,"--transfers",str(transfers),
          "--checkers",str(checkers),"--retries","5","--low-level-retries","10",
          "--stats","30s","--stats-one-line"]
    if action=="verify":
        args+=["--one-way"]
    if not execute and action!="verify":
        args+=["--dry-run"]
    return args


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("push","pull","verify"))
    parser.add_argument("--local",required=True,help="Narrow local directory scope")
    parser.add_argument("--remote",required=True,help="Configured rclone remote:bucket/project-prefix")
    parser.add_argument("--transfers",type=int,default=4)
    parser.add_argument("--checkers",type=int,default=8)
    parser.add_argument("--execute",action="store_true",help="Actually upload/download; default is dry run")
    args=parser.parse_args()
    try:
        cmd=command(args.action,args.local,args.remote,args.transfers,args.checkers,execute=args.execute)
    except ValueError as exc:
        parser.error(str(exc))
    summary={"action":args.action,"mode":("execute" if args.execute else "preview"),
             "rclone_args":cmd,"destructive_delete":False,
             "note":"Remote credentials are resolved by rclone; never copy them into this repository"}
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not args.execute:
        return 0
    if shutil.which("rclone") is None:
        print("rclone is not installed",file=sys.stderr)
        return 2
    try:
        process=subprocess.run(cmd,check=False)
    except OSError as exc:
        print(f"rclone execution error: {exc}",file=sys.stderr)
        return 2
    return process.returncode


if __name__=="__main__":
    raise SystemExit(main())

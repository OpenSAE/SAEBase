"""Probe local machine capabilities; never guess or upload sensitive environment data."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def safe_call(args: list[str], timeout: int=5) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,check=False)
    except (OSError,subprocess.TimeoutExpired):
        return None
    if p.returncode!=0:
        return None
    return p.stdout.strip()


def cpu_info():
    result={"logical_count":os.cpu_count(),"physical_count":None,"model":None,"source":"python/os"}
    if Path("/proc/cpuinfo").exists():
        try:
            lines=Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
            models=[line.split(":",1)[1].strip() for line in lines if line.startswith("model name")]
            if models:
                result["model"]=models[0]
                result["source"]="/proc/cpuinfo"
        except OSError:
            pass
    return result


def memory_info():
    out={"total_bytes":None,"available_bytes":None,"source":"unknown"}
    mem=Path("/proc/meminfo")
    if mem.exists():
        try:
            vals={}
            for line in mem.read_text().splitlines():
                if ":" in line:
                    k,rest=line.split(":",1)
                    vals[k]=int(rest.strip().split()[0])*1024
            out.update({"total_bytes":vals.get("MemTotal"),"available_bytes":vals.get("MemAvailable"),"source":"/proc/meminfo"})
        except (OSError,ValueError,IndexError):
            pass
    return out


def gpu_info():
    value=safe_call(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader,nounits"])
    if not value:
        return {"status":"unknown","devices":[],"source":"nvidia-smi unavailable"}
    devices=[]
    for i,line in enumerate(value.splitlines()):
        fields=[x.strip() for x in line.split(",")]
        if len(fields)<3:
            continue
        try:
            mib=int(float(fields[1]))
        except ValueError:
            mib=None
        devices.append({"index":i,"name":fields[0],"memory_mib":mib,"driver_version":fields[2]})
    return {"status":"observed" if devices else "unknown","devices":devices,"source":"nvidia-smi"}


def storage_info():
    usage=shutil.disk_usage(str(ROOT))
    return {"total_bytes":usage.total,"used_bytes":usage.used,"free_bytes":usage.free,
            "source":"shutil.disk_usage","path_label":"project filesystem (absolute path intentionally omitted)"}


def run_capabilities():
    tools=["tmux","sbatch","docker","podman","python","nvidia-smi","rclone"]
    return {"commands":{c:bool(shutil.which(c)) for c in tools},
            "virtual_environment":sys.prefix!=sys.base_prefix,
            "python_version":platform.python_version(),
            "python_executable":Path(sys.executable).name,
            "cuda_visible_devices_set":"CUDA_VISIBLE_DEVICES" in os.environ,
            "sandbox":"unknown",
            "network":"not probed",
            "source":"PATH, sys and environment presence flags"}


def profile():
    node=platform.node()
    return {
        "schema_version":1,"kind":"environment_profile","synthetic":False,
        "captured_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hostname_hash":hashlib.sha256(node.encode("utf-8")).hexdigest()[:12] if node else None,
        "platform":{"system":platform.system(),"release":platform.release(),"machine":platform.machine(),"source":"platform"},
        "cpu":cpu_info(),"memory":memory_info(),"gpu":gpu_info(),
        "storage":storage_info(),"execution":run_capabilities(),
        "notes":["Unknown fields are intentionally not guessed","No cloud credentials, env values, or absolute user paths are collected"]
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",help="Optional destination JSON. Defaults to stdout; keep local snapshots out of Git.")
    args=parser.parse_args()
    data=profile()
    serialized=json.dumps(data,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        dest=Path(args.output)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(serialized,encoding="utf-8")
        print(f"WROTE {dest} (local runtime facts)")
    else:
        print(serialized,end="")
    return 0


if __name__=="__main__":
    sys.exit(main())

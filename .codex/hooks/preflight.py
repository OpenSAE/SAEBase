"""目的和主要功能：说明当前开发模式并读取客户端实际配置。

适用范围：客户端接入、配置变化或权限故障时按需诊断。
"""

from __future__ import annotations
import json
from pathlib import Path
import selectors
import subprocess
import time
import tomllib
from development_mode import current_mode

ROOT = Path(__file__).resolve().parents[2]


def client_state():
    process = subprocess.Popen(
        ["codex", "app-server", "--stdio", "--strict-config"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    def call(ident, method, params):
        process.stdin.write(json.dumps({"id": ident, "method": method, "params": params}) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if selector.select(0.2):
                line = process.stdout.readline()
                if not line:
                    raise ValueError("Client closed during preflight")
                reply = json.loads(line)
                if reply.get("id") == ident:
                    if "error" in reply:
                        raise ValueError("Client inspection rejected: " + str(reply["error"]))
                    return reply["result"]
        raise TimeoutError("Client inspection timed out")

    try:
        call(
            1,
            "initialize",
            {
                "clientInfo": {"name": "saebase_preflight", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        process.stdin.write('{"method":"initialized"}\n')
        process.stdin.flush()
        config = call(2, "config/read", {"cwd": str(ROOT), "includeLayers": True})
        hooks = call(3, "hooks/list", {"cwds": [str(ROOT)]})
        return config, hooks
    finally:
        selector.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def inspect_state(config, hooks):
    errors = []
    expected = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    actual = config["config"]
    if current_mode() == "intensive":
        if actual.get("approval_policy") != "never":
            errors.append("Reload the client to apply approval_policy=never")
        if actual.get("default_permissions") != ":danger-full-access":
            errors.append("Reload the client to apply intensive development permissions")
        return errors
    if actual.get("default_permissions") != "saebase" or actual.get("sandbox_mode") is not None:
        errors.append(
            "Effective permission profile is missing or overridden by legacy sandbox settings"
        )
    if actual.get("features", {}).get("hooks") is not True:
        errors.append("Hooks are disabled")
    profile = actual.get("permissions", {}).get("saebase", {})
    filesystem = {k: v for k, v in profile.get("filesystem", {}).items() if v is not None}
    if filesystem != expected["permissions"]["saebase"]["filesystem"]:
        errors.append("Effective filesystem permissions differ from reviewed policy")
    if profile.get("network", {}).get("enabled") is not False:
        errors.append("Network is enabled")
    entries = hooks.get("data", [])
    matched = [
        h
        for e in entries
        for h in e.get("hooks", [])
        if h.get("sourcePath") == str(ROOT / ".codex/hooks.json")
    ]
    required = {"preToolUse"}
    if {h.get("eventName") for h in matched} != required or len(matched) != 1:
        errors.append("Project hooks not discovered")
    if any(
        not h.get("enabled") or h.get("trustStatus") not in {"trusted", "managed"} for h in matched
    ):
        errors.append("HOOK_TRUST_NOT_VERIFIED: review and trust /hooks yourself")
    if any(e.get("errors") for e in entries):
        errors.append("Client reported hook loading errors")
    return errors


def main():
    mode = current_mode()
    errors = []
    version = "unknown"
    try:
        version = subprocess.run(
            ["codex", "--version"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        if tuple(map(int, version.split()[-1].split("."))) < (0, 153, 0):
            errors.append("Codex >= 0.153.0 is required")
        errors.extend(inspect_state(*client_state()))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, TimeoutError) as exc:
        errors.append("PREFLIGHT_BLOCKED: " + str(exc))
    report = {
        "mode": mode,
        "version": version,
        "status": "BLOCKED" if errors else "PASS",
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

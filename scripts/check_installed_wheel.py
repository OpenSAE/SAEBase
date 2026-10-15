#!/usr/bin/env python3
"""Install the built wheel without dependencies, then smoke-test it outside the source tree.

The Python executing this script supplies already-installed runtime dependencies.
No network or model weights are required. A missing optional model package is fine.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def smoke(dist_dir: Path) -> dict[str, str]:
    wheels = list(dist_dir.glob("opensae_saebase-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Pass a dist directory containing exactly one SAEBase wheel")
    with tempfile.TemporaryDirectory(prefix="saebase-wheel-") as temp:
        temp_dir = Path(temp)
        target = temp_dir / "site"
        run_dir = temp_dir / "external-workdir"
        run_dir.mkdir()
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
             "--no-build-isolation", "--target", str(target), str(wheels[0].resolve())],
            cwd=run_dir, capture_output=True, text=True, timeout=120,
        )
        if installed.returncode:
            raise RuntimeError("Wheel installation failed:\n" + installed.stdout + "\n" + installed.stderr)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        env["PYTHONNOUSERSITE"] = "1"
        script = target / "bin" / "saebase"
        commands = [
            [str(script), "--list"],
            [str(target / "bin" / "saebase-prepare-prompts"), "--help"],
            [str(target / "bin" / "saebase-check-env"), "--help"],
            [str(target / "bin" / "saebase-benchmark"), "--help"],
            [str(target / "bin" / "saebase-audit"), "--help"],
            [str(target / "bin" / "saebase-release-matrix"), "--help"],
            [str(target / "bin" / "saebase-release-gate"), "--help"],
            [sys.executable, "-c", (
                "from pathlib import Path; from importlib.resources import files; "
                "import saebase, main, conf; "
                "from saebase.api import ModelAdapter, ActivationSite; "
                "assert (files('conf')/'config.yaml').is_file(); "
                "assert (files('conf')/'release'/'m5_matrix.json').is_file(); "
                "assert (files('conf')/'release'/'m5_asset_provenance.json').is_file(); "
                "assert (files('conf')/'sae'/'pythia70m_resid_sm.yaml').is_file(); "
                "from main import parse_args, build_overrides; "
                "from pathlib import Path as _P; "
                "_a,_=parse_args(['--device','cpu']); "
                "assert 'paths.root='+str(_P.cwd()) in build_overrides(_a,_P.cwd()); "
                "print(Path(saebase.__file__).resolve()); "
                "print('wheel configuration and API OK')"
            )],
        ]
        outputs = []
        for cmd in commands:
            r = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=35)
            if r.returncode:
                raise RuntimeError(f"Installed wheel command failed: {cmd}\n{r.stdout}\n{r.stderr}")
            outputs.append(r.stdout)
        if "Tasks: sae" not in outputs[0] or "wheel configuration and API OK" not in outputs[-1]:
            raise AssertionError("Wheel smoke test output does not match expectations")
        if str(target) not in outputs[-1]:
            raise AssertionError("Wheel smoke imported the checkout, not the installed package")
        return {"installation": "OK", "registry_cli": "OK", "preparation_cli": "OK",
                "environment_cli": "OK", "package_resources": "OK", "external_working_directory": "OK"}


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?",default="dist")
    args=parser.parse_args()
    print(json.dumps(smoke(Path(args.dist_dir)), indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

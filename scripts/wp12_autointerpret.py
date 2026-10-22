#!/usr/bin/env python3
"""Run local SAE feature auto-interpretation on real reviewed activation examples.

No Hugging Face model download, external API request, public upload or
automatic approval. Outputs a private hypothesis with held-out diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-model", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Never overwrite a previously generated explanation candidate")
    if not args.examples.is_file() or args.examples.stat().st_size > 1_000_000:
        parser.error("A bounded local reviewed source examples JSON is required")
    data = json.loads(args.examples.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "site_id", "feature_id", "discovery", "heldout",
    }:
        parser.error("Require exact source site, feature, discovery and heldout fields")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "research-lab"))
    from backend.auto_interp_local import LocalAutoInterpreter

    worker = LocalAutoInterpreter.from_local_snapshot(args.local_model, device=args.device)
    result = worker.interpret(
        site_id=data["site_id"], feature_id=data["feature_id"],
        discovery=data["discovery"], heldout=data["heldout"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({
        "status": result["status"], "approved": False,
        "heldout_score": result.get("heldout_score"),
        "generator_state_sha256": result["generator_state_sha256"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

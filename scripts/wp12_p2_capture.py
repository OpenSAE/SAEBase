#!/usr/bin/env python3
"""Opt-in real SAE token feature capture; data remain private until reviewed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from saebase.explore import capture_prompt_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-key", default="gemma270m_scope_resid")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--split", choices=["discovery", "evaluation"], default="discovery")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing capture")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    row = capture_prompt_trace(
        sae_key=args.sae_key, layer=args.layer, prompt=prompt, device=args.device,
        sample_id=args.sample_id, prompt_group_id=args.group_id, split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fp:
        json.dump(row, fp, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        fp.write("\n")
    print(json.dumps({
        "status": "private_capture_written_not_published",
        "site_id": row["site_id"], "sample_id": row["sample_id"],
        "n_tokens": len(row["tokens"]), "path": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

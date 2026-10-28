"""Package SAEBase's source-generated Atlas for an isolated static website.

The large Code Tree is emitted as ordered, UTF-8-safe JSON-string chunks so
the site can reconstruct the graph without eval, dynamic imports, or requests
to an unrelated OpenResearch project.

Follow ATLAS_AUTHORING_STANDARD.md before generating or publishing Atlas data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ("core", "knowledge", "code", "environment", "context")
SHARD_CHARS = 280_000
STATIC = (
    "app.js",
    "atlas-structure.js",
    "atlas-public-balance.js",
    "locale-data.js",
    "locale-runtime.js",
    "i18n.js",
    "style.css",
    "index.html",
    "core.html",
    "knowledge.html",
    "code.html",
    "environment.html",
    "context.html",
    "glossary.html",
)


def source_snapshot() -> tuple[str, dict]:
    text = (HERE / "data.js").read_text(encoding="utf-8")
    prefix = "window.OPENRESEARCH_ATLAS = "
    if prefix not in text:
        raise ValueError("Source-generated SAEBase data.js is missing")
    raw = text.split(prefix, 1)[1].strip().removesuffix(";")
    graphs = json.loads(raw)
    if not str(graphs.get("context", {}).get("project", "")).startswith("SAEBase"):
        raise ValueError("OpenResearch graph must not enter SAEBase deployment")
    if not {"core", "knowledge", "code", "environment", "context"} <= set(graphs):
        raise ValueError("SAEBase deployment needs all five source-owned trees")
    knowledge_ids = {n["id"] for n in graphs["knowledge"]["nodes"]}
    if not any("sae" in n for n in knowledge_ids) or any("cub" in n for n in knowledge_ids):
        raise ValueError("SAEBase scientific data has foreign project content")
    return raw, graphs


def bundle(output: Path) -> dict:
    raw, graphs = source_snapshot()
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Source checkout needs a real commit identity")
    output.mkdir(parents=True, exist_ok=True)
    chunks = [raw[i : i + SHARD_CHARS] for i in range(0, len(raw), SHARD_CHARS)]
    names: list[str] = []
    hashes: dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        name = f"data-{i:03d}.js"
        # Strings are JSON-escaped. This creates no executable source fragments.
        payload = (
            "window.SAEBASE_ATLAS_PARTS=window.SAEBASE_ATLAS_PARTS||[];\n"
            f"window.SAEBASE_ATLAS_PARTS[{i}]=" + json.dumps(chunk, ensure_ascii=False) + ";\n"
        )
        (output / name).write_text(payload, encoding="utf-8")
        names.append(name)
        hashes[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    bootstrap = (
        "/* SAEBase Atlas: reconstruct the verified source graph without eval. */\n"
        "(function(){\n"
        f"  const expected={len(names)};\n"
        "  const p=window.SAEBASE_ATLAS_PARTS;\n"
        '  if(!Array.isArray(p)||p.length!==expected||p.some(x=>typeof x!=="string"))\n'
        '    throw new Error("Incomplete SAEBase Atlas source snapshot");\n'
        '  const g=JSON.parse(p.join(""));\n'
        '  if(!String(g.context?.project||"").startsWith("SAEBase"))\n'
        '    throw new Error("Foreign Atlas project rejected");\n'
        "  window.OPENRESEARCH_ATLAS=g;\n"
        "  delete window.SAEBASE_ATLAS_PARTS;\n"
        "})();\n"
    )
    (output / "data-bootstrap.js").write_text(bootstrap, encoding="utf-8")
    for name in STATIC:
        shutil.copy2(HERE / name, output / name)

    script_tags = "\n".join(
        f'  <script defer src="./{name}?v={source_commit[:12]}"></script>'
        for name in [*names, "data-bootstrap.js"]
    )
    for view in ("index", *VIEWS):
        html = (output / (view + ".html")).read_text(encoding="utf-8")
        original = '  <script defer src="./data.js"></script>'
        if original not in html:
            raise ValueError("Missing expected data.js script in " + view)
        html = html.replace(original, script_tags)
        if "SAEBase" not in html or "OpenResearch Atlas ·" in html:
            raise ValueError("Wrong Atlas project branding in " + view)
        # A content-addressed asset version avoids using stale locale files.
        html = re.sub(
            r'(\./(?:app|i18n|locale-data|locale-runtime|atlas-structure|atlas-public-balance)\.js)\?v=[^"]+',
            lambda match: match.group(1) + "?v=" + source_commit[:12],
            html,
        )
        (output / (view + ".html")).write_text(html, encoding="utf-8")

    manifest = {
        "schema": "saebase.atlas.deployment.v1",
        "project": "SAEBase",
        "source_sha": source_commit,
        "tree_counts": {key: len(value["nodes"]) for key, value in graphs.items()},
        "chunks": names,
        "chunk_sha256": hashes,
    }
    (output / "site-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bundle(args.output.resolve())
    print("Packaged independently deployed SAEBase Atlas:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

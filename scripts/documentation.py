"""Render support documentation and check current Markdown navigation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tasks.benchmark.results.evidence import EVIDENCE_REL, load_support

ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r"!?\[[^\]\n]+\]\((<[^>]+>|[^)\s]+)(?:\s+\"[^\"]+\")?\)")

DOCS = (
    "README.md",
    "README_cn.md",
    ".github/CONTRIBUTING.md",
)


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        [
            *(root / p for p in DOCS),
            *(root / "doc/code/development").rglob("*.md"),
            *(root / "scripts/examples").rglob("*.md"),
        ]
    )


def missing_links(path: Path, root: Path) -> list[str]:
    """Check that each relative Markdown destination exists inside root."""
    root = root.resolve()
    missing = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for match in LINK.finditer(line):
            target = match.group(1).strip("<>")
            if not target or target.startswith("#"):
                continue
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("//"):
                continue
            local = unquote(parsed.path)
            if not local:
                continue
            dest = (path.parent / local).resolve()
            if not dest.is_relative_to(root) or not dest.exists():
                missing.append(f"{path.relative_to(root)}:{index}: {target}")
    return missing


def validate(root: Path = ROOT) -> tuple[int, list[str]]:
    files = markdown_files(root)
    absent = [str(p.relative_to(root)) for p in files if not p.is_file()]
    if absent:
        return len(files), [f"missing Markdown page: {p}" for p in absent]
    broken = []
    receipt = root / ".researchharness/historical-documents.json"
    historical = {}
    if receipt.is_file():
        historical = {row["path"]: row for row in json.loads(receipt.read_text())["files"]}
    checked = 0
    for file in files:
        row = historical.get(file.relative_to(root).as_posix())
        if row and row.get("authorization") == "HISTORICAL_ONLY":
            # This command checks current navigation. Original historical bytes
            # and archives are maintained by the bound central history service.
            continue
        checked += 1
        broken.extend(missing_links(file, root))
    return checked, broken


def links_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on broken relative links")
    args = parser.parse_args()
    if not args.check:
        parser.error("pass --check")
    n, broken = validate()
    if broken:
        for item in broken:
            print("Broken documentation link:", item)
        return 1
    print(f"Verified local links in {n} Markdown documents")
    return 0


BEGIN = "<!-- BEGIN VERIFIED_MODEL_SUPPORT -->"

END = "<!-- END VERIFIED_MODEL_SUPPORT -->"


def render_table(s: dict) -> str:
    lines = [
        f"**Verified model and SAE coverage ({s['date']}):** {s['profiles']} "
        f"selected model-SAE configurations across {s['total']} layer sites. "
        f"Real-checkpoint checks {s['pair']}/{s['total']}; benchmark evaluations "
        f"{s['benchmark']}/{s['total']}; reliability audits {s['reliability']}/{s['total']}.",
        "",
        "| Model ID | SAE checkpoint ID | Parameters | Architecture | Verified layers (per-site results) | Compatibility | Benchmark | Reliability |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tick = chr(96)
    for target in s["catalog"]:
        model, sae = target["model_key"], target["sae_key"]
        selected = [s["rows"][(model, sae, layer)] for layer in target["selected_layers"]]
        layers = []
        for row in selected:
            label = f"L{row['layer']}"
            report = row.get("m3m4_summary")
            layers.append(
                f"[{label}](../../../{EVIDENCE_REL.as_posix()}/{report})" if report else label
            )

        def count(field):
            return sum(row.get(field) is True for row in selected)

        arch = {"topk": "TopK", "jumprelu": "JumpReLU"}.get(
            target["architecture"], target["architecture"]
        )
        lines.append(
            f"| {tick}{model}{tick} | {tick}{sae}{tick} | {target['size']} | {arch} | "
            f"{', '.join(layers)} | {count('pair_verified')}/{len(selected)} | "
            f"{count('benchmark_verified')}/{len(selected)} | "
            f"{count('reliability_verified')}/{len(selected)} |"
        )
    return "\n".join(lines)


def render_blocks(s: dict) -> dict[str, str]:
    pair = f"{s['pair']}/{s['total']}"
    bench = f"{s['benchmark']}/{s['total']}"
    rel = f"{s['reliability']}/{s['total']}"
    date, profiles = s["date"], s["profiles"]
    return {
        "README.md": (
            f"**Verified checkpoints ({date}).** Across **{profiles}** model-SAE "
            f"configurations (270M-9B), **{pair}** selected model/layer sites "
            f"passed real-checkpoint compatibility tests, **{bench}** completed "
            "benchmark evaluations on 64 held-out prompt groups, and "
            f"**{rel}** completed reliability audits. See the "
            "[verified model and SAE matrix](doc/code/development/verified-model-matrix.md)."
        ),
        "README_cn.md": (
            f"**已验证的模型与 SAE（{date}）。** 覆盖 **{profiles}** 组模型与 SAE 配置"
            f"（270M-9B），**{pair}** 个选定层位通过真实权重兼容性检查，"
            f"**{bench}** 个完成基于 64 组独立留出提示的基准评估，"
            f"**{rel}** 个完成统计可靠性审计。详见"
            "[模型支持与逐层验证结果](doc/code/development/verified-model-matrix.md)。"
        ),
        "doc/code/development/model-support.md": (
            "## Verified checkpoints and layers\n\n"
            f"As of {date}, **{profiles}** model-SAE configurations spanning "
            f"270M-9B cover **{pair}** selected model/layer sites with genuine "
            f"checkpoint compatibility checks. **{bench}** completed benchmark "
            f"evaluation on 64 held-out prompt groups, and **{rel}** completed "
            "reliability auditing. See the "
            "[per-layer verification reports](verified-model-matrix.md)."
        ),
        "doc/code/development/validation.md": (
            "## Verified model and SAE evaluations\n\n"
            f"**{profiles}** selected model-SAE configurations, **{pair}** "
            f"real-checkpoint compatibility tests, **{bench}** benchmark "
            f"evaluations on 64 held-out prompt groups, and **{rel}** "
            f"reliability audits ({date}). See the "
            "[per-layer reports](verified-model-matrix.md)."
        ),
        "doc/code/development/CHANGELOG.md": (
            f"- **Verified model and SAE coverage ({date}):** **{pair}** "
            f"real-checkpoint compatibility tests, **{bench}** benchmark "
            f"evaluations on 64 held-out prompt groups, and **{rel}** "
            f"reliability audits across **{profiles}** model-SAE configurations. "
            "See the [verified model and SAE matrix](verified-model-matrix.md)."
        ),
        "doc/code/development/verified-model-matrix.md": render_table(s),
    }


def replace_block(content: str, replacement: str, name: str) -> str:
    if content.count(BEGIN) != 1 or content.count(END) != 1:
        raise ValueError(f"{name}: expected one verified-support marker pair")
    left, remaining = content.split(BEGIN, 1)
    _, right = remaining.split(END, 1)
    return left + BEGIN + "\n" + replacement.rstrip() + "\n" + END + right


def generate(*, check: bool, root: Path = ROOT) -> list[str]:
    changed = []
    for name, block in render_blocks(load_support(root)).items():
        path = root / name
        original = path.read_text(encoding="utf-8")
        rendered = replace_block(original, block, name)
        if original != rendered:
            changed.append(name)
            if not check:
                path.write_text(rendered, encoding="utf-8")
    if "](verified-model-matrix.md)" not in (root / "doc/code/development/index.md").read_text(
        encoding="utf-8"
    ):
        raise ValueError("docs index does not link the current verified matrix")
    if check and changed:
        raise ValueError("stale model-support docs: " + ", ".join(changed))
    return changed


def support_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        changed = generate(check=args.check)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"model support validation failed: {error}\n")
    print(
        "Verified model support docs"
        if args.check
        else "Regenerated: " + (", ".join(changed) or "no changes")
    )
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "support":
        del sys.argv[1]
        return support_main()
    return links_main()


if __name__ == "__main__":
    raise SystemExit(main())

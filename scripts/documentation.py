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
    "doc/code/translated/README.md",
    ".github/CONTRIBUTING.md",
    "doc/conf.md",
)


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        [
            *(root / p for p in DOCS),
            *(root / "doc/code/documentation/source/development").rglob("*.md"),
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
            "[verified model and SAE matrix](doc/code/documentation/source/development/verified-model-matrix.md)."
        ),
        "doc/code/translated/README.md": (
            f"**Certified Models and SAE({date}).** Overwrite **{profiles}** Group Models and SAE Configure"
            f"(270M-9B),**{pair}** The selected layer is checked for real weight compatibility,"
            f"**{bench}** _Other Organiser 64 Group independent and prompt baseline assessment,"
            f"**{rel}** Audits of statistical reliability completed."
            "[Model support and layer-by-layer validation results](doc/code/documentation/source/development/verified-model-matrix.md)."
        ),
        "doc/code/documentation/source/development/model-support.md": (
            "## Verified checkpoints and layers\n\n"
            f"As of {date}, **{profiles}** model-SAE configurations spanning "
            f"270M-9B cover **{pair}** selected model/layer sites with genuine "
            f"checkpoint compatibility checks. **{bench}** completed benchmark "
            f"evaluation on 64 held-out prompt groups, and **{rel}** completed "
            "reliability auditing. See the "
            "[per-layer verification reports](verified-model-matrix.md)."
        ),
        "doc/code/documentation/source/development/validation.md": (
            "## Verified model and SAE evaluations\n\n"
            f"**{profiles}** selected model-SAE configurations, **{pair}** "
            f"real-checkpoint compatibility tests, **{bench}** benchmark "
            f"evaluations on 64 held-out prompt groups, and **{rel}** "
            f"reliability audits ({date}). See the "
            "[per-layer reports](verified-model-matrix.md)."
        ),
        "doc/code/documentation/source/development/CHANGELOG.md": (
            f"- **Verified model and SAE coverage ({date}):** **{pair}** "
            f"real-checkpoint compatibility tests, **{bench}** benchmark "
            f"evaluations on 64 held-out prompt groups, and **{rel}** "
            f"reliability audits across **{profiles}** model-SAE configurations. "
            "See the [verified model and SAE matrix](verified-model-matrix.md)."
        ),
        "doc/code/documentation/source/development/verified-model-matrix.md": render_table(s),
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
    if "](verified-model-matrix.md)" not in (root / "doc/code/documentation/source/development/index.md").read_text(
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


CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".jsonl", ".toml"}
CONFIG_NAV_START = "<!-- BEGIN CONFIG_NAVIGATION -->"
CONFIG_NAV_END = "<!-- END CONFIG_NAVIGATION -->"


def configuration_entries(root: Path, directory: Path):
    """Find immediate configuration files and populated configuration directories."""
    for entry in directory.iterdir():
        if entry.name.startswith((".", "__")):
            continue
        if entry.is_file() and entry.suffix in CONFIG_SUFFIXES:
            yield entry.relative_to(root).as_posix()
        if entry.is_dir() and any(
            file.is_file() and file.suffix in CONFIG_SUFFIXES for file in entry.rglob("*")
        ):
            yield entry.relative_to(root).as_posix() + "/"


def checked_configuration_files(root: Path) -> list[Path]:
    """Check that each configuration branch has one file and matches the source tree."""
    config = root / "conf"
    paths = []
    owners = {}
    for path in sorted(config.rglob("*")):
        relative = path.relative_to(config)
        if relative == Path("__init__.py") or "__pycache__" in relative.parts:
            continue
        directory = relative if path.is_dir() else relative.parent
        if len(directory.parts) > 2:
            raise ValueError(f"{path.relative_to(root)}: conf allows at most two nested directories")
        if not path.is_file():
            continue
        if path.suffix not in CONFIG_SUFFIXES:
            raise ValueError(f"{path.relative_to(root)}: file has no configuration role")
        if relative != Path("config.yaml"):
            source = root / relative.parent
            if source == root or not source.is_dir() or not any(source.rglob("*.py")):
                raise ValueError(
                    f"{path.relative_to(root)}: expected a source directory at {relative.parent}"
                )
        if relative.parent in owners:
            previous = owners[relative.parent]
            raise ValueError(
                f"{path.relative_to(root)}: consolidate into {previous}; one file per configuration branch"
            )
        owners[relative.parent] = path.relative_to(root)
        paths.append(path)
    return paths


def _conf_block(text: str, start: str, end: str, content: str) -> str:
    before, opening, rest = text.partition(start)
    _, closing, after = rest.partition(end)
    if not opening or not closing or text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"doc/conf.md: expected one ordered marker pair for {start}")
    return before + opening + "\n\n" + content + "\n\n" + closing + after


def _sync_parameter_defaults(text: str, root: Path) -> str:
    """Refresh literal defaults in authored tables from the file that owns them."""
    import yaml

    current = None
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            current = None
        heading = re.match(r"^#{2,6} .* `(conf/[^\x60]+\.ya?ml)`\s*$", line)
        if heading:
            current = yaml.safe_load((root / heading[1]).read_text(encoding="utf-8"))
        match = re.match(r"^\| `([\w.]+)` \| `([^\x60]+)` \|", line)
        if match and isinstance(current, dict):
            value = current
            for part in match[1].split("."):
                if not isinstance(value, dict) or part not in value:
                    break
                value = value[part]
            else:
                if yaml.safe_load(match[2]) != value and not isinstance(value, dict):
                    rendered = yaml.safe_dump(value, default_flow_style=True).split("\n...", 1)[0]
                    if "\n" not in rendered and "`" not in rendered and "|" not in rendered:
                        line = line[: match.start(2)] + rendered + line[match.end(2) :]
        lines.append(line)
    return "".join(lines)


def configuration_navigation(text: str, root: Path) -> str:
    """Validate source ownership and synchronize the navigation and current values."""
    from hashlib import sha256
    from html import escape

    files = checked_configuration_files(root)
    text = _sync_parameter_defaults(text, root)
    headings = list(re.finditer(r"^(#{2,6}) (.+)$", text, re.MULTILINE))
    explanations = {}
    parents = {}
    for index, heading in enumerate(headings):
        level, title = heading.groups()
        depth = len(level)
        parents = {number: name for number, name in parents.items() if number < depth}
        match = re.fullmatch(r"[\d.]+ (.+) `(conf/[^\x60]+)`", title)
        if match is None:
            continue
        label, path = match.groups()
        if path in explanations:
            raise ValueError(f"doc/conf.md: repeated section for {path}")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        content = text[heading.end() : end]
        anchor = re.sub(r"[^\w -]", "", title).lower().replace(" ", "-")
        explanations[path] = parents.get(depth - 1), label, anchor, content
        parents[depth] = path

    for file in files:
        path = file.relative_to(root).as_posix()
        parent = file.parent
        while parent != root / "conf":
            relative = parent.relative_to(root).as_posix() + "/"
            if relative not in explanations:
                raise ValueError(f"doc/conf.md: add a section for {relative}")
            parent = parent.parent
        if path not in explanations:
            raise ValueError(f"doc/conf.md: add a section for {path}")

    rows = ["| Configure Paths | When? |", "| --- | --- |"]
    for path, (owner, label, anchor, content) in explanations.items():
        if not (root / path).exists():
            raise ValueError(f"doc/conf.md: stale section for {path}")
        parent = Path(path.rstrip("/")).parent.as_posix() + "/"
        expected = None if parent == "conf/" else parent
        if owner != expected:
            raise ValueError(f"doc/conf.md: {path} belongs under {expected or 'conf/'}")
        if owner is None:
            purpose = re.search(r"^\*\*Purpose.\*\* (.+)$", content, re.MULTILINE)
            if purpose is None:
                raise ValueError(f"doc/conf.md: add a purpose to the section for {path}")
            rows.append(f"| [`{path}`](../{path}) | [{label}](#{anchor}),{purpose[1]} |")

    values = []
    for file in [root / "base/configuration/__init__.py", *files]:
        path = file.relative_to(root).as_posix()
        data = file.read_bytes()
        lines = data.decode("utf-8").splitlines()
        if file.suffix == ".jsonl":
            visible = []
        elif file.suffix == ".json" and len(lines) > 180:
            visible = lines[:80]
        else:
            visible = lines
        preview = "\n".join(visible)
        if remaining := len(lines) - len(visible):
            preview += ("\n" if preview else "") + f"The rest {remaining} See line source file."
        values.append(
            f"<details>\n<summary>{escape(path)}</summary>\n\n"
            f"[Open Full File](../{path})\n\n"
            f"<!-- SHA256: {sha256(data).hexdigest()} -->\n"
            f"<pre>{escape(preview)}</pre>\n\n</details>"
        )

    text = _conf_block(text, CONFIG_NAV_START, CONFIG_NAV_END, "\n".join(rows))
    return _conf_block(
        text, "<!-- BEGIN CONFIG_VALUES -->", "<!-- END CONFIG_VALUES -->", "\n\n".join(values)
    )


def conf_main() -> int:
    parser = argparse.ArgumentParser(description="Update the configuration path navigation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = ROOT / "doc" / "conf.md"
    try:
        original = path.read_text(encoding="utf-8")
        expected = configuration_navigation(original, ROOT)
        if args.check and expected != original:
            raise ValueError("run python -m scripts.documentation conf --write")
        if args.write and expected != original:
            path.write_text(expected, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"configuration navigation: {error}\n")
    print("Configuration navigation is current")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "support":
        del sys.argv[1]
        return support_main()
    if len(sys.argv) > 1 and sys.argv[1] == "conf":
        del sys.argv[1]
        return conf_main()
    return links_main()


if __name__ == "__main__":
    raise SystemExit(main())

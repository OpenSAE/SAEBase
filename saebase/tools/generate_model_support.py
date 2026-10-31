#!/usr/bin/env python3
"""Render and verify current-main model and SAE support from committed evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_REL = Path("conf/coverage/stage_d_evidence")
BEGIN = "<!-- BEGIN VERIFIED_MODEL_SUPPORT -->"
END = "<!-- END VERIFIED_MODEL_SUPPORT -->"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _evidence_file(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise ValueError(f"invalid evidence path: {rel!r}")
    base = (root / EVIDENCE_REL).resolve()
    candidate = (base / rel).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"missing or unsafe evidence file: {rel!r}")
    return candidate


def validate_data(catalog: dict, final: dict, status: dict, root: Path) -> dict:
    """Validate identity, evidence reports, ledgers, and reported totals."""
    if catalog.get("schema") != "saebase.coverage.stage_bc.v1":
        raise ValueError("unexpected catalog schema")
    if final.get("schema") != "saebase.stage_d.final_matrix.v1":
        raise ValueError("unexpected final matrix schema")
    if status.get("schema") != "saebase.stage_d.status.v1":
        raise ValueError("unexpected status schema")
    targets, rows = catalog.get("targets"), final.get("rows")
    if not isinstance(targets, list) or not isinstance(rows, list):
        raise ValueError("missing catalog targets or final matrix rows")
    expected = {}
    for target in targets:
        for layer in target["selected_layers"]:
            key = (target["model_key"], target["sae_key"], layer)
            if key in expected:
                raise ValueError(f"duplicate catalog site: {key}")
            expected[key] = target
    found, ledger_keys = {}, set()
    counts = dict(pair=0, benchmark=0, reliability=0, population=0, smoke=0)
    states = {}
    for row in rows:
        key = (row["model_key"], row["sae_key"], row["layer"])
        if key not in expected or key in found:
            raise ValueError(f"unexpected or duplicate final matrix row: {key}")
        found[key] = row
        hook = row["hook_name"]
        state = row["status"]
        states[state] = states.get(state, 0) + 1
        ledger_key = f"{key[0]}|{key[1]}|L{key[2]}|{hook}"
        ledger_keys.add(ledger_key)
        ledger = status.get("sites", {}).get(ledger_key)
        if not isinstance(ledger, dict) or ledger.get("status") != state:
            raise ValueError(f"status mismatch at {key}")
        p = row.get("pair_verified") is True
        b = row.get("benchmark_verified") is True
        a = row.get("reliability_verified") is True
        if (a and not b) or (b and not p):
            raise ValueError(f"verification hierarchy mismatch at {key}")
        counts["pair"] += int(p)
        counts["benchmark"] += int(b)
        counts["reliability"] += int(a)
        counts["population"] += int(a and b and row.get("protocol") == "population")
        counts["smoke"] += int(row.get("protocol") == "smoke_only")
        if p:
            # Older accepted CPU pairs have separate pre-Stage-D provenance.
            # Never manufacture a model SHA when the current matrix omits it.
            for name in ("model_state_sha256", "sae_state_sha256"):
                value = row.get(name)
                if value is not None and re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
                    raise ValueError(f"malformed weight hash at {key}: {name}")
                if value is None and not row.get("prior_pair_verified"):
                    raise ValueError(f"missing new-pair weight hash at {key}: {name}")
            if row.get("acceptance_path"):
                _evidence_file(root, row["acceptance_path"])
            elif not row.get("prior_pair_verified"):
                raise ValueError(f"missing pair acceptance at {key}")
        if b or a:
            report = _json(_evidence_file(root, row.get("m3m4_summary")))
            for field, wanted in zip(("model_key", "sae_key", "layer", "hook_name"), (*key, hook)):
                if report.get(field) != wanted:
                    raise ValueError(f"{field} mismatch in report at {key}")
            if report.get("status") != "passed" or report.get("protocol") != row.get("protocol"):
                raise ValueError(f"non-passing or wrong-protocol report at {key}")
            if row.get("protocol") == "population":
                if report.get("n_independent_evaluation_groups", 0) < 64:
                    raise ValueError(f"insufficient independent groups at {key}")
                if report.get("feature_selection_split") != "selection":
                    raise ValueError(f"selection/evaluation split mismatch at {key}")
            if a and report.get("reliability_metric_audits", 0) < 1:
                raise ValueError(f"missing reliability audit at {key}")
    if set(found) != set(expected):
        raise ValueError(f"missing catalog layer sites: {set(expected) - set(found)}")
    if set(status.get("sites", {})) != ledger_keys or status.get("n_sites") != len(rows):
        raise ValueError("status ledger coverage differs from final matrix")
    for field, computed in (
        ("n_sites", len(rows)),
        ("n_pair_verified", counts["pair"]),
        ("n_population_m3m4", counts["population"]),
        ("n_smoke_only_m3m4", counts["smoke"]),
    ):
        if final.get(field) != computed:
            raise ValueError(f"{field}: expected {computed}, found {final.get(field)}")
    if final.get("counts") != states:
        raise ValueError("final matrix state counts do not match rows")
    return dict(
        catalog=targets,
        rows=found,
        total=len(rows),
        profiles=len(targets),
        date=final["updated_at_utc"].split("T")[0],
        **counts,
    )


def load_support(root: Path = ROOT) -> dict:
    return validate_data(
        _json(root / "conf/coverage/stage_bc.json"),
        _json(root / EVIDENCE_REL / "final/stage_d_final_matrix.json"),
        _json(root / EVIDENCE_REL / "status.json"),
        root,
    )


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
            layers.append(f"[{label}](../../{EVIDENCE_REL.as_posix()}/{report})" if report else label)
        count = lambda field: sum(row.get(field) is True for row in selected)
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
            "[verified model and SAE matrix](harness/github/development_doc/verified-model-matrix.md)."
        ),
        "README_cn.md": (
            f"**已验证的模型与 SAE（{date}）。** 覆盖 **{profiles}** 组模型与 SAE 配置"
            f"（270M-9B），**{pair}** 个选定层位通过真实权重兼容性检查，"
            f"**{bench}** 个完成基于 64 组独立留出提示的基准评估，"
            f"**{rel}** 个完成统计可靠性审计。详见"
            "[模型支持与逐层验证结果](harness/github/development_doc/verified-model-matrix.md)。"
        ),
        "harness/github/development_doc/model-support.md": (
            "## Verified checkpoints and layers\n\n"
            f"As of {date}, **{profiles}** model-SAE configurations spanning "
            f"270M-9B cover **{pair}** selected model/layer sites with genuine "
            f"checkpoint compatibility checks. **{bench}** completed benchmark "
            f"evaluation on 64 held-out prompt groups, and **{rel}** completed "
            "reliability auditing. See the "
            "[per-layer verification reports](verified-model-matrix.md)."
        ),
        "harness/github/development_doc/validation.md": (
            "## Verified model and SAE evaluations\n\n"
            f"**{profiles}** selected model-SAE configurations, **{pair}** "
            f"real-checkpoint compatibility tests, **{bench}** benchmark "
            f"evaluations on 64 held-out prompt groups, and **{rel}** "
            f"reliability audits ({date}). See the "
            "[per-layer reports](verified-model-matrix.md)."
        ),
        "harness/github/development_doc/CHANGELOG.md": (
            f"- **Verified model and SAE coverage ({date}):** **{pair}** "
            f"real-checkpoint compatibility tests, **{bench}** benchmark "
            f"evaluations on 64 held-out prompt groups, and **{rel}** "
            f"reliability audits across **{profiles}** model-SAE configurations. "
            "See the [verified model and SAE matrix](verified-model-matrix.md)."
        ),
        "harness/github/development_doc/verified-model-matrix.md": render_table(s),
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
    if "](verified-model-matrix.md)" not in (
        root / "harness/github/development_doc/index.md"
    ).read_text(encoding="utf-8"):
        raise ValueError("docs index does not link the current verified matrix")
    if check and changed:
        raise ValueError("stale model-support docs: " + ", ".join(changed))
    return changed


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())

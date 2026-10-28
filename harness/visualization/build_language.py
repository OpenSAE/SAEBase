"""Build the source-ID English overlay for SAEBase's five Atlas trees.

Scientific Context, Core, Knowledge and Environment use reviewed translations.
Code labels are composed from real Python AST symbols, paths and interfaces.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CJK = re.compile(r"[\u3400-\u9fff]")
DOCS = {
    "doc/knowledge/foundations/sae_methods.md": "doc/knowledge/foundations/sae_methods.en.md",
    "doc/knowledge/README.md": "doc/knowledge/README.en.md",
    "doc/knowledge/CONTEXT_STYLE.md": "doc/knowledge/CONTEXT_STYLE.en.md",
    "harness/knowledge/README.md": "harness/knowledge/README.en.md",
    "harness/code/README.md": "harness/code/README.en.md",
    "harness/environment/README.md": "harness/environment/README.en.md",
    "harness/experiments/README.md": "harness/experiments/README.en.md",
    "harness/core/README.md": "harness/core/README.en.md",
    "harness/core/AGENT_FLOW.md": "harness/core/AGENT_FLOW.en.md",
    "harness/core/BOUNDARIES.md": "harness/core/BOUNDARIES.en.md",
    "harness/core/TREE_CONTRACTS.md": "harness/core/TREE_CONTRACTS.en.md",
    "harness/visualization/README.md": "harness/visualization/README.en.md",
    "harness/README.md": "harness/README.en.md",
    "harness/transfer/README.md": "harness/transfer/README.en.md",
}
ROLES = (
    ("datasets/", "dataset and prompt preparation"),
    ("models/", "model loading and compatibility"),
    ("tasks/capture/", "SAE activation capture"),
    ("tasks/intervene/", "controlled feature interventions"),
    ("tasks/generate/", "generation-time feature steering"),
    ("tasks/circuits/", "causal circuit tracing"),
    ("tasks/explain/", "feature explanation"),
    ("tasks/benchmark/", "SAE benchmarking"),
    ("tasks/train/", "sparse autoencoder training"),
    ("tasks/", "research task execution"),
    ("saebase/sae/", "SAE representation and checkpoints"),
    ("saebase/api/", "feature-level research interfaces"),
    ("saebase/data/", "prompt and artifact compatibility"),
    ("saebase/store/", "feature cache and storage"),
    ("saebase/benchmarks/", "benchmark planning and metrics"),
    ("saebase/reliability/", "statistical reliability"),
    ("saebase/research_lab/", "Research Lab services"),
    ("saebase/runtime/", "experiment configuration and execution"),
    ("saebase/tools/", "research verification tools"),
    ("saebase/", "SAEBase research infrastructure"),
    ("harness/code/", "source-code documentation"),
    ("harness/core/", "research architecture"),
    ("harness/knowledge/", "scientific knowledge"),
    ("harness/environment/", "execution resources"),
    ("harness/visualization/", "research Atlas visualization"),
    ("harness/", "research Harness"),
    ("conf/", "experiment configuration"),
)
VERBS = {
    "get": "Get",
    "read": "Read",
    "load": "Load",
    "capture": "Capture",
    "encode": "Encode",
    "decode": "Decode",
    "build": "Build",
    "construct": "Construct",
    "register": "Register",
    "validate": "Validate",
    "verify": "Verify",
    "check": "Check",
    "compute": "Compute",
    "calculate": "Calculate",
    "evaluate": "Evaluate",
    "compare": "Compare",
    "intervene": "Intervene",
    "steer": "Steer",
    "generate": "Generate",
    "train": "Train",
    "prepare": "Prepare",
    "filter": "Filter",
    "normalize": "Normalize",
    "save": "Save",
    "write": "Write",
    "export": "Export",
    "parse": "Parse",
    "select": "Select",
    "extract": "Extract",
    "run": "Run",
    "create": "Create",
    "update": "Update",
    "render": "Render",
    "inspect": "Inspect",
    "collect": "Collect",
    "measure": "Measure",
    "trace": "Trace",
    "sample": "Sample",
    "list": "List",
}
UI = {
    "组件职责": "Component responsibility",
    "函数接口": "Function interface",
    "源码组成": "Module composition",
    "类属性": "Class fields",
    "类方法": "Class methods",
    "完整源码": "Complete source code",
    "对应的真实代码": "Related source implementation",
    "为什么重要": "Why this matters",
    "其他名称": "Alternative names",
    "真实依据": "Source evidence",
    "基础解释": "Foundational reference",
    "源代码标识": "Source identifier",
    "设备与执行信息": "Runtime facts",
    "方法签名": "Method signature",
    "输入参数": "Input parameters",
    "返回值": "Return value",
    "专业词汇 · Context": "Technical terms · Context",
}
PARAMETERS = {
    "name": "The registered component or item name.",
    "key": "Key identifying a registered component.",
    "cfg": "Resolved experiment configuration.",
    "config": "Configuration supplied by the caller.",
    "device": "Requested model or tensor device.",
    "model": "Selected model instance or identity.",
    "model_key": "Registered model identity.",
    "sae": "Selected sparse autoencoder.",
    "prompt": "Input prompt for the language model.",
    "token_ids": "Input token identifiers.",
    "layer": "Selected language-model layer.",
    "hook_name": "Named activation hook.",
    "path": "Source or destination file path.",
    "seed": "Random seed for reproducibility.",
    "feature_ids": "Selected SAE feature indices.",
    "output": "Output artifact destination.",
    "argv": "Command-line arguments passed to the entry point.",
}


def human(identifier: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", identifier)
    return " ".join(x for x in re.split(r"[_\-\s]+", value) if x) or "component"


def role(path: str) -> str:
    for prefix, desc in ROLES:
        if path.startswith(prefix):
            return desc
    return "project source interfaces"


def make_code(node: dict, children: dict, by_id: dict) -> dict:
    kind = node.get("kind", "component")
    path = node.get("path") or node.get("source_path") or ""
    symbol = node.get("source_symbol") or ""
    file_stem = path.rsplit("/", 1)[-1].removesuffix(".py")
    name = symbol.rsplit(".", 1)[-1] if symbol else file_stem
    context = role(path)
    if kind in {"function", "method"}:
        prefix = name.split("_", 1)[0].lower()
        verb = VERBS.get(prefix)
        label = (verb + " " + human(name[len(prefix) :].lstrip("_"))) if verb else human(name)
        title = label.strip() + " · " + human(file_stem) + " (" + name + ")"
        desc = (
            "The Python "
            + kind
            + " "
            + (symbol or name)
            + " in "
            + path
            + " implements "
            + context
            + ". Its source signature, parameters, and return expressions are available below."
        )
    elif kind == "class":
        title = human(name) + " · " + context
        desc = (
            "The class "
            + name
            + " in "
            + path
            + " implements "
            + context
            + ". Its declared fields and methods are available in the source details."
        )
    elif kind == "module":
        title = human(file_stem) + " · " + context
        desc = (
            "The module "
            + path
            + " implements "
            + context
            + ". Child nodes enumerate the actual Python definitions."
        )
    else:
        descendants = [by_id[c] for c in children.get(node["id"], []) if c in by_id]
        first = next((x for x in descendants if x.get("path")), {})
        category = role(path or first.get("path", ""))
        title = category.capitalize() + " · " + human(node["id"].split(":")[-1])
        desc = (
            "This navigation group contains "
            + str(len(descendants))
            + " source-backed nodes associated with "
            + category
            + ". The child nodes retain their actual source identities."
        )
    return {
        "title": title,
        "summary": desc,
        "description": desc,
        "reader_explanation": desc,
    }


def build() -> str:
    source = (HERE / "data.js").read_text(encoding="utf-8")
    prefix = "window.OPENRESEARCH_ATLAS = "
    if prefix not in source:
        raise ValueError("Missing source-generated Atlas snapshot")
    graphs = json.loads(source.split(prefix, 1)[1].strip().removesuffix(";"))
    if not str(graphs["context"].get("project", "")).startswith("SAEBase"):
        raise ValueError("Foreign Context data must not be included in SAEBase")
    context = json.loads((ROOT / "doc/knowledge/i18n/context.en.json").read_text(encoding="utf-8"))
    trees = json.loads((HERE / "i18n/trees.en.json").read_text(encoding="utf-8"))
    for source_data in (context, trees):
        if source_data.get("project") != "SAEBase" or source_data.get("target_locale") != "en":
            raise ValueError("English translation belongs to another project")
    translated = {}
    for view, graph in graphs.items():
        by_id = {n["id"]: n for n in graph["nodes"]}
        children = defaultdict(list)
        for edge in graph["edges"]:
            if edge["type"] in {"contains", "refines", "addresses", "decomposes"}:
                children[edge["source"]].append(edge["target"])
        if view == "context":
            nodes = dict(context["nodes"])
            for node in graph["nodes"]:
                if node["id"].startswith("context:nav:"):
                    titles = [nodes[x]["title"] for x in children[node["id"]] if x in nodes]
                    title = (
                        ("Terminology: " + titles[0] + " to " + titles[-1])
                        if titles
                        else "Research terminology"
                    )
                    summary = (
                        "Browse " + str(len(titles)) + " defined scientific and engineering terms."
                    )
                    nodes[node["id"]] = {"title": title, "summary": summary, "description": summary}
        elif view == "code":
            nodes = {n["id"]: make_code(n, children, by_id) for n in graph["nodes"]}
        else:
            nodes = dict(trees["views"][view])
        if set(nodes) != set(by_id):
            raise ValueError(
                "Missing " + view + " translations: " + str(sorted(set(by_id) - set(nodes))[:10])
            )
        for node_id, obj in nodes.items():
            for key in ("title", "summary", "description"):
                value = obj.get(key)
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or CJK.search(value)
                    or "\ufffd" in value
                ):
                    raise ValueError("Invalid English field: " + view + "/" + node_id + "/" + key)
        translated[view] = nodes
    for original, english in DOCS.items():
        if not (ROOT / original).is_file() or not (ROOT / english).is_file():
            raise ValueError("Missing English document for " + original)
        if CJK.search((ROOT / english).read_text(encoding="utf-8")):
            raise ValueError("Chinese text appears in English Markdown: " + english)
    ui_record = json.loads((HERE / "i18n/ui.en.json").read_text(encoding="utf-8"))
    if ui_record.get("project") != "SAEBase":
        raise ValueError("Foreign UI terminology overlay")
    UI.update(ui_record["ui"])
    record = {
        "schema": "saebase.atlas.locale.v2",
        "project": "SAEBase",
        "source_locale": "zh",
        "target_locale": "en",
        "nodes": translated,
        "docs": DOCS,
        "ui": UI,
        "parameter_meanings": PARAMETERS,
    }
    return (
        "/* Generated SAEBase-only English fields; do not edit by hand. */\n"
        + "window.SAEBASE_ATLAS_LOCALES = "
        + json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        + ";\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    projection = build()
    dest = HERE / "locale-data.js"
    if args.check:
        if not dest.exists() or dest.read_text(encoding="utf-8") != projection:
            raise SystemExit("Stale SAEBase English locale projection")
        print("All SAEBase Atlas IDs and English documents validated")
    else:
        dest.write_text(projection, encoding="utf-8")
        print("Wrote", dest, "bytes", len(projection.encode("utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

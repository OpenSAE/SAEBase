"""Build a reproducible Python AST code tree; semantic context comes from reviewed knowledge."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOTS = ("saebase", "datasets", "models", "tasks")
ANNOTATIONS = ROOT / "doc/code/code_annotations.json"
DEFAULT_OUT = ROOT / "doc/code/code_tree.json"
GROUP_LABELS = {
    "saebase": "SAEBase · LLM 稀疏特征研究基础",
    "datasets": "Datasets · NLP 输入数据",
    "models": "Models · 网络构建",
    "tasks": "Tasks · 科研任务",
    "harness": "Harness · 可选工具",
}


def safe_doc(node: ast.AST) -> str:
    raw = ast.get_docstring(node, clean=True) or ""
    return " ".join(raw.split()).strip()[:360]


def args_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except (AttributeError, ValueError):
        args = "..."
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({args}){returns}"


def parse_annotations():
    if not ANNOTATIONS.is_file():
        return {}
    obj = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    result = {}
    for row in obj.get("mappings", []):
        result[row["path"]] = row
    return result


def make_node(
    key, kind, title, path="", line=0, signature="", doc="", annotations=None, bases=None
):
    known = annotations or {}
    summary = doc or known.get("summary") or ""
    return {
        "id": key,
        "kind": kind,
        "title": title,
        "path": path,
        "line": line,
        "status": "implemented",
        "signature": signature,
        "description": summary or "暂无经过验证的功能说明，请结合源代码审核。",
        "description_source": "docstring"
        if doc
        else ("reviewed_context" if known.get("summary") else "needs_review"),
        "context_ids": known.get("context_ids", []),
        "core_ids": known.get("core_ids", []),
        "bases": bases or [],
        "needs_review": not bool(summary),
        "evidence": ([f"{path}:{line}"] if path and line else []),
    }


def build(include_harness=False):
    annotations = parse_annotations()
    roots = list(CODE_ROOTS)
    if include_harness:
        roots.append("harness")
    nodes = [
        make_node(
            "code:root",
            "root",
            "SAEBase · 源代码结构",
            doc="由当前代码静态分析自动生成；点击展开包、模块、类与方法。",
        )
    ]
    edges = []
    files = []
    digester = hashlib.sha256()
    for root in roots:
        nodes.append(
            make_node(
                f"code:group:{root}", "group", GROUP_LABELS[root], doc=f"{root}/ 下的 Python 模块"
            )
        )
        edges.append({"source": "code:root", "target": f"code:group:{root}", "type": "contains"})
        for file in sorted((ROOT / root).rglob("*.py")):
            if any(p in ("__pycache__", ".venv", "node_modules") for p in file.parts):
                continue
            path = file.relative_to(ROOT).as_posix()
            source = file.read_text(encoding="utf-8")
            digester.update(path.encode() + b"\x00" + source.encode() + b"\x00")
            tree = ast.parse(source, filename=path)
            module = ".".join(file.relative_to(ROOT).with_suffix("").parts)
            mid = f"code:module:{module}"
            desc = annotations.get(path, {})
            nodes.append(
                make_node(mid, "module", path, path, 1, doc=safe_doc(tree), annotations=desc)
            )
            edges.append({"source": f"code:group:{root}", "target": mid, "type": "contains"})
            files.append((path, module, mid, tree, desc))
    classes = {}
    modules = {module: mid for _, module, mid, _, _ in files}
    for path, module, mid, tree, annotation in files:
        for definition in tree.body:
            if isinstance(definition, ast.ClassDef):
                cid = f"code:class:{module}:{definition.name}"
                bases = [ast.unparse(base) for base in definition.bases]
                nodes.append(
                    make_node(
                        cid,
                        "class",
                        definition.name,
                        path,
                        definition.lineno,
                        signature=f"class {definition.name}({', '.join(bases)})",
                        doc=safe_doc(definition),
                        annotations=annotation,
                        bases=bases,
                    )
                )
                edges.append({"source": mid, "target": cid, "type": "contains"})
                classes.setdefault(definition.name, []).append(cid)
                for method in definition.body:
                    if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    # Dunder methods are retained, but the browser collapses them by default.
                    fid = f"code:method:{module}:{definition.name}.{method.name}"
                    nodes.append(
                        make_node(
                            fid,
                            "method",
                            method.name,
                            path,
                            method.lineno,
                            signature=args_signature(method),
                            doc=safe_doc(method),
                            annotations=annotation,
                        )
                    )
                    edges.append({"source": cid, "target": fid, "type": "contains"})
            elif isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = f"code:function:{module}:{definition.name}"
                nodes.append(
                    make_node(
                        fid,
                        "function",
                        definition.name,
                        path,
                        definition.lineno,
                        signature=args_signature(definition),
                        doc=safe_doc(definition),
                        annotations=annotation,
                    )
                )
                edges.append({"source": mid, "target": fid, "type": "contains"})
    class_defs = {n["id"]: n for n in nodes if n["kind"] == "class"}
    for cid, node in class_defs.items():
        for base in node["bases"]:
            name = base.rsplit(".", 1)[-1]
            choices = classes.get(name, [])
            if len(choices) == 1 and choices[0] != cid:
                edges.append({"source": cid, "target": choices[0], "type": "inherits"})
    for path, module, mid, tree, _ in files:
        linked = set()
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                candidates = [node.module]
            for c in candidates:
                for target, tid in modules.items():
                    if c == target or c.startswith(target + "."):
                        if tid != mid and tid not in linked:
                            linked.add(tid)
                            edges.append({"source": mid, "target": tid, "type": "imports"})
    # Conservative local call edges: only same-module unambiguous symbols.
    # Dynamic dispatch (self methods, getattr, plugins) is intentionally omitted.
    local_functions = {}
    for item in nodes:
        if item["kind"] == "function":
            local_functions[(item["path"], item["title"])] = item["id"]
    for path, module, _, tree, _ in files:
        for definition in tree.body:
            if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            caller = f"code:function:{module}:{definition.name}"
            observed = set()
            for call in ast.walk(definition):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    callee = local_functions.get((path, call.func.id))
                    if callee and callee != caller:
                        observed.add(callee)
            for callee in sorted(observed):
                edges.append({"source": caller, "target": callee, "type": "calls"})
    nodes.sort(key=lambda n: n["id"])
    edges.sort(key=lambda e: (e["source"], e["type"], e["target"]))
    return {
        "schema_version": 1,
        "kind": "code",
        "project": "SAEBase",
        "source_sha256": digester.hexdigest(),
        "scope": roots,
        "extraction": "Python AST; unresolved dynamic calls deliberately omitted",
        "nodes": nodes,
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--include-harness", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Check snapshot equals current source; do not write"
    )
    args = parser.parse_args()
    graph = build(args.include_harness)
    destination = Path(args.output).resolve(strict=False)
    blob = json.dumps(graph, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != blob:
            print("STALE code_tree snapshot; regenerate after code edits", file=sys.stderr)
            return 1
        print(f"CURRENT code tree: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(blob, encoding="utf-8")
    print(f"WROTE {destination}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())

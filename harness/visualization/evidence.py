"""Connect existing scientific and project concepts to exact Python implementations.

Bindings refer only to AST-authenticated Code nodes. Every excerpt is generated
from tracked source and includes a complete callable interface for inspection.
"""

from __future__ import annotations

import copy
import ast
from harness.code.interfaces import parse_file, function_interface

CODE_TARGETS = {
    "q.modularity": "code:function:saebase.registry:register_dataset",
    "s.registry": "code:function:saebase.registry:register_dataset",
    "s.compatibility": "code:function:saebase.registry:get_dataset_plugin",
    "s.hydra": "code:function:saebase.runtime.train:main",
    "g.registry": "code:function:saebase.registry:register_model",
    "g.dataset_gate": "code:function:saebase.registry:get_dataset_plugin",
    "g.model_gate": "code:function:models.gemma2_2b_base.model_task_dataset_compatibility:assert_compatible",
    "g.pipeline": "code:function:saebase.runner:run_task",
    "g.config": "code:function:saebase.runtime.train:main",
    "g.artifact_schema": "code:function:saebase.runtime.experiment_paths:prepare_run",
}
TERM_TARGETS = {}


# Core infrastructure code sits in Harness and is imported directly from real
# source. It is a separate module from the production Code Tree's default scope.
DIRECT_TARGETS = {
    "q.transparency": ("harness/core/validate.py", "main", "校验项目架构节点与父子结构。"),
    "s.explainability": (
        "harness/visualization/export.py",
        "main",
        "生成用于浏览的结构化项目视图。",
    ),
    "g.context": (
        "harness/knowledge/validate.py",
        "validate_scientific",
        "校验科研知识、术语字典与关联结构。",
    ),
    "g.code_tree": ("harness/code/build.py", "build", "从实际 Python AST 生成代码树。"),
    "g.env_tree": ("harness/environment/probe.py", "profile", "采集当前机器的硬件与执行信息。"),
    "g.transfer": ("harness/transfer/b2.py", "command", "构建现有云端复制与校验命令。"),
    "g.b2": ("harness/transfer/b2.py", "command", "使用服务器已配置的 Backblaze B2 连接创建可审核的非删除式传输。"),
}


def direct_source_attribute(key: str) -> dict:
    path, target, explanation = DIRECT_TARGETS[key]
    module, text = parse_file(path)
    fn = next(
        (
            n
            for n in module.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == target
        ),
        None,
    )
    if fn is None:
        raise ValueError(f"Source function for {key} is absent: {path}::{target}")
    info = function_interface(fn, text, summary=explanation)
    return {
        "type": "code_reference",
        "label": "对应的真实代码",
        "title": explanation,
        "symbol_id": None,
        "path": path,
        "line": info["line"],
        "interface": info,
    }


def real_code_interface(symbol: dict) -> dict:
    """Present one original signature, its IO and a truthful source excerpt."""
    kind = symbol.get("kind")
    blocks = symbol.get("attributes", [])
    info = next((b["interface"] for b in blocks if b.get("type") == "interface"), None)
    if kind == "class":
        container = next((b for b in blocks if b.get("type") == "class_methods"), None)
        methods = container.get("methods", []) if container else []
        preferred = next(
            (m for m in methods if m["name"] in ("forward", "run", "__getitem__", "setup")), None
        )
        if preferred is None and methods:
            preferred = methods[0]
        info = preferred
    if kind == "module":
        excerpt = next((b for b in blocks if b.get("type") == "code"), None)
        if excerpt:
            return {
                "type": "code_reference",
                "label": "对应的真实代码",
                "title": symbol["title"],
                "symbol_id": symbol["id"],
                "path": symbol["path"],
                "code": excerpt["value"],
                "source_type": "module",
            }
    if info is None:
        return {
            "type": "reference",
            "label": "对应的代码实体",
            "path": symbol.get("path"),
            "line": symbol.get("line"),
        }
    return {
        "type": "code_reference",
        "label": "对应的真实代码",
        "title": symbol.get("title"),
        "symbol_id": symbol["id"],
        "path": symbol.get("path"),
        "line": info.get("line", symbol.get("line")),
        "interface": copy.deepcopy(info),
    }


def sae_method_evidence(node: dict, code_symbols: dict) -> dict | None:
    """Attach an actual Code object for a source-backed Knowledge SAE method.

    Prefer a real public function in the cited module. If a module only exposes
    private implementation helpers, the source-backed module reference remains
    explicit without inventing a public callable signature.
    """
    paths = [str(p).split(":", 1)[0] for p in node.get("evidence", [])]
    for path in paths:
        candidates = [symbol for symbol in code_symbols.values() if symbol.get("path") == path]
        if not candidates:
            continue
        callables = sorted(
            (
                symbol
                for symbol in candidates
                if symbol.get("kind") in {"function", "class", "method"}
            ),
            key=lambda item: (
                0 if item.get("source_symbol") == "execute" else 1,
                0 if item.get("kind") == "function" else 1,
                int(item.get("line") or 0),
                item["id"],
            ),
        )
        for symbol in callables:
            detail = real_code_interface(symbol)
            if detail.get("type") == "code_reference":
                return detail
        modules = [symbol for symbol in candidates if symbol.get("kind") == "module"]
        if modules:
            module = modules[0]
            return {
                "type": "code_reference",
                "label": "对应的真实代码",
                "title": module["title"],
                "symbol_id": module["id"],
                "path": path,
                "line": module.get("line", 1),
                "source_type": "module",
            }
    return None


def attach_evidence(
    tree: dict, code: dict, mapping: dict[str, str], *, skip_ids: set[str] | None = None
) -> None:
    symbols = {n["id"]: n for n in code["nodes"]}
    visible = {n["id"] for n in tree["nodes"]}
    allowed = set(mapping) - set(skip_ids or ())
    for node in tree["nodes"]:
        key = node["id"]
        if key in DIRECT_TARGETS:
            node.setdefault("attributes", []).append(direct_source_attribute(key))
            continue
        if key not in allowed:
            if key.startswith(("s.sae.", "g.sae.")):
                detail = sae_method_evidence(node, symbols)
                if detail is None:
                    raise ValueError("Missing real SAE source evidence for " + key)
                node.setdefault("attributes", []).append(detail)
            continue
        symbol_id = mapping[key]
        symbol = symbols.get(symbol_id)
        if symbol is None:
            raise ValueError(
                f"Evidence binding for {key} references an absent code entity: {symbol_id}"
            )
        detail = real_code_interface(symbol)
        attrs = node.setdefault("attributes", [])
        attrs.append(detail)
        if symbol_id in visible:
            node.setdefault("reference_ids", []).append(symbol_id)

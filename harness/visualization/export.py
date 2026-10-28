"""Export an evidence-backed, currently implemented Atlas (Environment excepted).

Source authority stays in Core/Knowledge JSON, Python AST, and machine observations.
Never publish planned/partial work as a capability or invent functions to fill a tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from hashlib import sha256
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.code.presentation import readable_code_tree  # noqa: E402 - import ordering follows runtime/bootstrap contract
from harness.knowledge.context_attributes import validate_attributes  # noqa: E402 - import ordering follows runtime/bootstrap contract
from harness.visualization.evidence import attach_evidence, CODE_TARGETS, TERM_TARGETS  # noqa: E402 - import ordering follows runtime/bootstrap contract
from harness.visualization.language import verify_public_prose  # noqa: E402 - import ordering follows runtime/bootstrap contract

DEFAULT = ROOT / "harness/visualization/data.js"
STRUCTURE = ROOT / "harness/visualization/atlas-structure.js"
ATLAS_TERMS = ROOT / "doc/knowledge/atlas_terms.json"
HIERARCHY = {"refines", "addresses", "decomposes"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_tree(graph: dict, *, view: str) -> dict:
    """Only source-backed green nodes appear; add one non-feature grouping root.

    Nodes' stable IDs are retained. If an unimplemented parent is hidden,
    an existing descendant is reattached to the closest implemented ancestor.
    Original source data is deliberately not deleted or rewritten.
    """
    if view not in {"core", "knowledge"}:
        raise ValueError("Only Core and Knowledge have this projection")
    source_nodes = graph["nodes"]
    all_ids = {node["id"] for node in source_nodes}
    green = {node["id"] for node in source_nodes if node.get("status") == "implemented"}
    parents = {}
    for edge in graph["edges"]:
        if edge["type"] in HIERARCHY and edge["target"] in all_ids:
            if edge["target"] in parents:
                raise ValueError("Not a tree: multiple parents for " + edge["target"])
            parents[edge["target"]] = edge["source"]
    root_id = f"{view}:existing"
    labels = {
        "core": (
            "SAEBase 统一科研组件与任务注册",
            "展示 SAEBase Runtime、Registry 与 Dataset、Model、Task 中已经实现的组件。",
        ),
        "knowledge": (
            "LLM SAE 科研方法与实现依据",
            "按 SAE 特征捕获、干预、生成、电路、解释、基准和训练组织有源码依据的研究方法。",
        ),
    }
    title, summary = labels[view]
    root = {
        "id": root_id,
        "kind": "root",
        "title": title,
        "summary": summary,
        "description": summary,
        "status": "implemented",
        "evidence": [],
    }
    visible = [root]
    edges = []
    for raw in source_nodes:
        if raw["id"] not in green:
            continue
        node = copy.deepcopy(raw)
        # Historical roadmaps and future tasks are deliberately not part of the
        # published existing-capability projection, including hidden metadata.
        node.pop("next", None)
        node.pop("last_change", None)
        node.pop("verification", None)
        # Avoid the retired governance semantics when describing an existing tool.
        if node["id"] == "g.context":
            node["description"] = (
                "现有知识图通过稳定节点编号、证据、三态进度和父子层级表示已经实现的研究组件。"
            )
        if node["id"] == "s.explainability":
            node["description"] = (
                "现有项目用 Core、Knowledge、Code、Environment 和术语字典分别整理架构、研究知识、源代码及执行资源。"
            )
        if node["id"] == "q.modularity":
            node["title"] = "Dataset、Model、Task Registry 解耦"
            node["description"] = "当前数据集、模型和任务已经通过注册与兼容性接口解耦。"
        elif node["id"] == "q.transparency":
            node["title"] = "Atlas 五棵 Tree 的职责与关联"
            node["description"] = "现在能够分别浏览项目架构、科研知识、源代码和环境结构。"

        specific = {
            "g.registry": "Dataset、Model、SAE Task 插件注册",
            "g.pipeline": "SAE Task 运行前的适配检查",
        }
        if node["id"] in specific:
            node["title"] = specific[node["id"]]
        node["summary"] = node.get("description", "").split("：", 1)[-1].strip()
        node["status"] = "implemented"
        visible.append(node)
        cursor = parents.get(node["id"])
        visited = set()
        while cursor and cursor not in green:
            if cursor in visited:
                raise ValueError("Cycle in historical hierarchy")
            visited.add(cursor)
            cursor = parents.get(cursor)
        parent_id = cursor or root_id
        original_type = next(
            (
                e["type"]
                for e in graph["edges"]
                if e["source"] == parent_id and e["target"] == node["id"]
            ),
            None,
        )
        # Retained actual relation where possible, otherwise a grouping-only edge.
        edges.append(
            {"source": parent_id, "target": node["id"], "type": original_type or "contains"}
        )
    output = {
        "schema_version": 1,
        "kind": view,
        "view_policy": "implemented_only",
        "title": title,
        "status_policy": "only implemented/source-backed nodes are published",
        "source": "harness/core/project_graph.json"
        if view == "core"
        else "doc/knowledge/context_graph.json",
        "nodes": visible,
        "edges": edges,
    }
    if view == "knowledge":
        organize_knowledge_topics(output)
    return output


def organize_knowledge_topics(output: dict) -> None:
    """Organize source-backed SAE research without unrelated task domains."""
    root_id = "knowledge:existing"
    group_id = "knowledge:area:sae"
    if not any(edge["source"] == root_id for edge in output["edges"]):
        return
    output["nodes"].append(
        {
            "id": group_id,
            "kind": "group",
            "title": "LLM 稀疏特征与可解释性实验",
            "summary": "按模型、SAE、Prompt 与七类研究任务浏览当前源码实现。",
            "description": "按模型、SAE、Prompt 与七类研究任务浏览当前源码实现。",
            "status": "implemented",
            "evidence": ["saebase/api/features.py"],
            "presentation_only": True,
        }
    )
    for edge in output["edges"]:
        if edge["source"] == root_id:
            edge["source"] = group_id
    output["edges"].append({"source": root_id, "target": group_id, "type": "contains"})


# Page size is enforced by the browser, never by inventing a Concept node.
# A source symbol remains attached to its actual module and class.


def readable_environment(profile: dict) -> dict:
    """Human-readable observability tree. Never pass synthetic data as actual facts."""
    synthetic = bool(profile.get("synthetic", False))
    nodes, edges = [], []
    state_default = "not_implemented" if synthetic else "implemented"
    prefix = "示例数据：" if synthetic else "本机观测："

    def present(key, kind, title, summary, parent=None, source="unknown", status=None, facts=None):
        item = {
            "id": key,
            "kind": kind,
            "title": title,
            "summary": summary,
            "description": summary,
            "status": status or state_default,
            "evidence": [str(source)] if source and source != "unknown" else [],
        }
        if facts:
            item["facts"] = facts
        nodes.append(item)
        if parent:
            edges.append({"source": parent, "target": key, "type": "contains"})

    def display_size(value):
        if not isinstance(value, (int, float)) or value < 0:
            return "待现场采集"
        gib = value / 1024**3
        if gib >= 1024:
            return f"{gib / 1024:.1f} TiB"
        return f"{gib:.1f} GiB"

    def as_text(value, default="待现场采集"):
        return (
            str(value)
            if value not in (None, "", "unknown", "not observed", "not probed")
            else default
        )

    present(
        "env:root",
        "root",
        "CPU、GPU 与科研任务执行环境",
        "当前页面使用演示环境，展示硬件资源的组织形式。"
        if synthetic
        else "当前机器的硬件与执行工具由本地探测器采集，字段记录实际观测结果。",
        source="example.json" if synthetic else "local environment probe",
    )

    platform = profile.get("platform", {})
    cpu = profile.get("cpu", {})
    memory = profile.get("memory", {})
    gpu = profile.get("gpu", {})
    storage = profile.get("storage", {})
    execution = profile.get("execution", {})

    system = as_text(platform.get("system"))
    machine = as_text(platform.get("machine"))
    present(
        "env:platform",
        "group",
        "操作系统与处理器架构",
        f"{prefix}系统 {system}；硬件架构 {machine}。",
        parent="env:root",
        source=platform.get("source"),
    )

    core_count = as_text(cpu.get("logical_count"))
    physical_count = as_text(cpu.get("physical_count"))
    present(
        "env:cpu",
        "group",
        "中央处理器",
        f"{prefix}逻辑处理单元 {core_count} 个；物理核心 {physical_count} 个。",
        parent="env:root",
        source=cpu.get("source"),
    )

    present(
        "env:memory",
        "group",
        "系统内存",
        f"{prefix}总容量 {display_size(memory.get('total_bytes'))}；"
        f"当前可用 {display_size(memory.get('available_bytes'))}。",
        parent="env:root",
        source=memory.get("source"),
    )

    devices = gpu.get("devices", [])
    gpu_state = "not_implemented" if synthetic or not devices else "implemented"
    if not devices and not synthetic and gpu.get("status") == "unknown":
        gpu_state = "partial"
    gpu_note = (
        f"演示配置列出 {len(devices)} 张图形处理器，展示设备管理结构。"
        if synthetic
        else f"观测到 {len(devices)} 张图形处理器。"
        if devices
        else "本机图形设备信息等待驱动与探测工具核验。"
    )
    present(
        "env:gpu",
        "group",
        "图形计算设备",
        gpu_note,
        parent="env:root",
        source=gpu.get("source"),
        status=gpu_state,
    )
    for i, device in enumerate(devices):
        mem = device.get("memory_mib")
        memory_text = f"{mem / 1024:.1f} GiB" if isinstance(mem, (int, float)) else "待现场采集"
        present(
            "env:gpu:" + str(device.get("index", i)),
            "device",
            f"第 {i + 1} 张图形处理器",
            (
                f"{prefix}设备显存 {memory_text}；"
                + ("型号来自演示配置。" if synthetic else "实际型号和驱动可在详细观测记录中核对。")
            ),
            parent="env:gpu",
            source=gpu.get("source"),
            facts={
                "设备名称": as_text(device.get("name")),
                "驱动版本": as_text(device.get("driver_version")),
            },
        )

    present(
        "env:storage",
        "group",
        "项目所在磁盘",
        f"{prefix}总空间 {display_size(storage.get('total_bytes'))}；"
        f"剩余空间 {display_size(storage.get('free_bytes'))}。",
        parent="env:root",
        source=storage.get("source"),
    )

    commands = execution.get("commands", {})
    command_labels = {
        "tmux": ("后台终端会话", "支持把长时间任务放在独立终端会话中执行。"),
        "sbatch": ("集群作业提交", "可从当前命令路径访问集群作业提交程序。"),
        "docker": ("容器运行环境", "可通过现有容器命令管理隔离运行环境。"),
        "podman": ("替代容器运行环境", "可使用替代容器工具创建运行环境。"),
        "python": ("Python 解释器", "可以在当前环境中启动 Python 程序。"),
        "nvidia-smi": ("图形设备检测工具", "可使用图形驱动诊断程序查询设备。"),
        "rclone": ("文件同步工具", "可以运行已有的远程文件复制与校验工具。"),
    }
    statuses = []
    for command, available in commands.items():
        if synthetic:
            state = "not_implemented"
            summary = "此项来自演示配置，运行状态由实际机器采样确定。"
        else:
            state = "implemented" if available else "not_implemented"
            summary = (
                "当前环境能够找到这个工具。" if available else "当前命令路径的检测结果为待安装。"
            )
        statuses.append(state)
        title, meaning = command_labels.get(
            command, (f"{command} 命令可用性", f"检测环境中的 {command} 命令路径与可用性。")
        )
        present(
            "env:command:" + command,
            "capability",
            title,
            meaning + " " + summary,
            parent="env:execution",
            source=execution.get("source"),
            status=state,
            facts={"命令标识": command},
        )
    exec_status = (
        "not_implemented"
        if synthetic
        else "implemented"
        if statuses and all(x == "implemented" for x in statuses)
        else "partial"
    )
    present(
        "env:execution",
        "group",
        "任务执行工具",
        (
            "示例中的工具状态展示配置组织形式。"
            if synthetic
            else "这里列出当前运行环境实际可检测的终端和执行工具。"
        ),
        parent="env:root",
        source=execution.get("source"),
        status=exec_status,
        facts={
            "Python 版本": as_text(execution.get("python_version")),
            "网络条件": "待采集"
            if execution.get("network") == "not probed"
            else as_text(execution.get("network")),
            "沙箱条件": as_text(execution.get("sandbox")),
        },
    )
    # Children were added before the execution group for simplicity; the
    # client builds its hierarchy from edges and does not depend on node order.
    return {
        "schema_version": 1,
        "kind": "environment",
        "synthetic": synthetic,
        "captured_at_utc": profile.get("captured_at_utc"),
        "nodes": nodes,
        "edges": edges,
    }


def validate_atlas_vocabulary(vocabulary: dict, glossary: dict) -> None:
    """Only curated, source-backed terms may become public Context entries."""
    if (
        vocabulary.get("schema_version") != 1
        or vocabulary.get("kind") != "atlas_context_vocabulary"
    ):
        raise ValueError("Invalid Atlas Context terminology schema")
    categories = glossary["categories"] + vocabulary.get("categories", [])
    keys = [category["id"] for category in categories]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate Context terminology category")
    ids = {term["id"] for term in glossary["terms"]}
    for term in vocabulary.get("terms", []):
        tid = term.get("id")
        if not isinstance(tid, str) or not tid.startswith("term.") or tid in ids:
            raise ValueError("Duplicate or invalid Atlas Context term: " + str(tid))
        ids.add(tid)
        if term.get("category") not in keys:
            raise ValueError("Unknown Context category for " + tid)
        for field in ("term", "definition", "why_it_matters", "foundation"):
            if not isinstance(term.get(field), str) or not term[field].strip():
                raise ValueError("Unexplained Context term: " + tid)
        if not isinstance(term.get("aliases"), list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in term["aliases"]
        ):
            raise ValueError("Invalid Context aliases: " + tid)
        sources = [term["foundation"], *term.get("source_evidence", [])]
        if len(sources) < 2:
            raise ValueError("Context term lacks evidence: " + tid)
        for source in sources:
            if not isinstance(source, str) or source.startswith("/") or "\\" in source:
                raise ValueError("Unsafe Context source: " + tid)
            resolved = (ROOT / source).resolve(strict=False)
            if not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_file():
                raise ValueError("Missing Context source: " + str(source))


def context_graph(
    glossary: dict, allowed_knowledge: set[str], curated: dict, vocabulary: dict | None = None
) -> dict:
    """Context is the sole public source for scientific and technical definitions."""
    validate_attributes(curated, {term["id"] for term in glossary["terms"]})
    if vocabulary is None:
        vocabulary = load(ATLAS_TERMS)
    validate_atlas_vocabulary(vocabulary, glossary)
    by_id = {entry["term_id"]: entry["blocks"] for entry in curated["entries"]}
    categories = glossary["categories"] + vocabulary["categories"]
    terms = glossary["terms"] + vocabulary["terms"]
    category_ids = {category["id"] for category in categories}
    root = {
        "id": "context:root",
        "kind": "root",
        "title": "科研术语定义与源码依据",
        "summary": "这里集中解释 Atlas 中使用的科研与工程术语，并给出可核对的资料来源。",
        "description": "这里集中解释 Atlas 中使用的科研与工程术语，并给出可核对的资料来源。",
        "status": "implemented",
        "evidence": ["doc/knowledge/glossary.json", "doc/knowledge/atlas_terms.json"],
    }
    nodes, edges = [root], []
    for category in categories:
        if not any(term["category"] == category["id"] for term in terms):
            continue
        catid = "context:category:" + category["id"]
        nodes.append(
            {
                "id": catid,
                "kind": "group",
                "title": category["title"],
                "summary": category["description"],
                "description": category["description"],
                "status": "implemented",
                "evidence": [
                    "doc/knowledge/glossary.json"
                    if category in glossary["categories"]
                    else "doc/knowledge/atlas_terms.json",
                ],
            }
        )
        edges.append({"source": "context:root", "target": catid, "type": "contains"})
    for term in terms:
        if term["category"] not in category_ids:
            raise ValueError("Unknown Context category " + term["category"])
        allowed_links = [
            key for key in term.get("related_node_ids", []) if key in allowed_knowledge
        ]
        attributes = [
            {
                "type": "text",
                "label": "为什么重要",
                "value": term["why_it_matters"],
                "source": term["foundation"],
            },
        ]
        if term.get("aliases"):
            attributes.append(
                {
                    "type": "text",
                    "label": "其他名称",
                    "value": "、".join(term["aliases"]),
                    "source": term["foundation"],
                }
            )
        attributes.extend(by_id.get(term["id"], []))
        # Even terms not yet associated with implemented experiments carry a
        # genuine source-backed reference, not a synthetic method.
        attributes.append(
            {
                "type": "code_reference",
                "label": "真实依据",
                "path": term["source_evidence"][0],
            }
        )
        attributes.append(
            {
                "type": "reference",
                "label": "基础解释",
                "path": term["foundation"],
            }
        )
        nodes.append(
            {
                "id": term["id"],
                "kind": "term",
                "title": term["term"],
                "summary": term["definition"],
                "description": term["definition"],
                "why_it_matters": term["why_it_matters"],
                "aliases": term["aliases"],
                "foundation": term["foundation"],
                "status": "implemented",
                "evidence": term["source_evidence"],
                "context_ids": allowed_links,
                "attributes": attributes,
            }
        )
        edges.append(
            {
                "source": "context:category:" + term["category"],
                "target": term["id"],
                "type": "contains",
            }
        )
    return {
        "schema_version": 1,
        "kind": "context",
        "view_policy": "defined_terms",
        "project": glossary.get("project"),
        "nodes": nodes,
        "edges": edges,
    }


def term_match(text: str, alias: str):
    """Match a displayed term only as a complete phrase or identifier."""
    if not alias:
        return []
    latin = re.compile(r"[A-Za-z0-9_]")
    prefix = r"(?<![A-Za-z0-9_])" if latin.match(alias[0]) else ""
    suffix = r"(?![A-Za-z0-9_])" if latin.match(alias[-1]) else ""
    return list(re.finditer(prefix + re.escape(alias) + suffix, text, re.IGNORECASE))


def explanatory_text(node: dict, view: str) -> str:
    """State the evidence-backed responsibility, not the node's name and count."""
    summary = re.sub(
        r"^(?:问题|方案|目标|定义|现状)[:：]\\s*",
        "",
        str(node.get("description") or node.get("summary") or ""),
    ).strip()
    if not summary:
        raise ValueError("Missing actual node purpose: " + node["id"])
    return summary


def semantic_terms(node: dict) -> str:
    """Search user-facing prose, never literal source code or signatures."""
    fields = [
        node.get("title", ""),
        node.get("summary", ""),
        node.get("description", ""),
        node.get("reader_explanation", ""),
    ]
    for attribute in node.get("attributes") or []:
        if attribute.get("type") == "text":
            fields.append(attribute.get("value", ""))
        elif attribute.get("type") in {"interface", "code_reference"}:
            info = attribute.get("interface") or {}
            fields.append(info.get("purpose", ""))
            fields.extend(p.get("meaning", "") for p in info.get("parameters", []))
            fields.append((info.get("returns") or {}).get("meaning", ""))
        elif attribute.get("type") == "fields":
            fields.extend(f.get("meaning", "") for f in attribute.get("fields", []))
    return " ".join(str(value) for value in fields if value)


# Source-domain terminology complements textual alias matching. A module's
# actual path is a reliable indication of the research operation it implements.
# Term identities come exclusively from the authoritative Context graph.
CODE_CONTEXT_SCOPES = (
    ("saebase/api/features.py", ("term.sae", "term.sae.capture", "term.feature_capture")),
    ("saebase/api/", ("term.sae", "term.saebase")),
    ("saebase/sae/", ("term.sae", "term.sparse_autoencoder")),
    ("saebase/store/", ("term.sae", "term.artifact")),
    ("tasks/capture/", ("term.sae", "term.sae.capture")),
    ("tasks/intervene/", ("term.sae", "term.sae.intervene")),
    ("tasks/generate/", ("term.sae", "term.sae.generate")),
    ("tasks/circuits/", ("term.sae", "term.sae.circuits")),
    ("tasks/explain/", ("term.sae", "term.sae.explain")),
    ("tasks/benchmark/", ("term.sae", "term.sae.benchmark")),
    ("tasks/train/", ("term.sae", "term.sae.train")),
    ("models/gemma2_2b_base/", ("term.gemma2_2b", "term.model")),
    ("models/", ("term.llm", "term.model")),
    ("datasets/", ("term.nlp", "term.sae.assets")),
    ("saebase/runtime/cfg/", ("term.hydra", "term.yaml")),
    ("saebase/runtime/training/", ("term.training", "term.lightning")),
    ("saebase/registry.py", ("term.registry", "term.plugin")),
    # A source node without a specialized term still has its real engineering
    # domain, rather than an invented navigation category.
    ("saebase/", ("term.saebase",)),
    ("datasets/", ("term.dataset",)),
    ("models/", ("term.model",)),
    ("tasks/", ("term.task",)),
    ("conf/", ("term.yaml",)),
)


def exact_code_context(node: dict, available: set[str]) -> list[str]:
    source = str(node.get("path") or node.get("source_path") or "")
    for prefix, concept_ids in CODE_CONTEXT_SCOPES:
        if source.startswith(prefix):
            return [term for term in concept_ids if term in available]
    return []


def attach_context_explanations(graphs: dict) -> None:
    """Populate self-contained node prose and canonical Context ID references.

    The UI always reads full terminology definitions from DATA.context.nodes;
    individual trees store only stable IDs, so definitions cannot drift.
    """
    terms = [node for node in graphs["context"]["nodes"] if node["id"].startswith("term.")]
    tokens = sorted(
        (
            (alias, term["id"])
            for term in terms
            for alias in [term["title"], *term.get("aliases", [])]
            if len(alias.strip()) >= 2
        ),
        key=lambda item: (-len(item[0]), item[1]),
    )
    context_ids = {term["id"] for term in terms}
    for view, graph in graphs.items():
        for node in graph["nodes"]:
            node["reader_explanation"] = explanatory_text(node, view)
            phrases = semantic_terms(node)
            matches = []
            for alias, term_id in tokens:
                if term_id == node["id"]:
                    continue
                for match in term_match(phrases, alias):
                    matches.append((match.start(), match.end(), term_id))
            # Longer names take precedence over embedded abbreviations.
            matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
            occupied = []
            selected = []
            for start, end, term_id in matches:
                if any(start < finish and end > begin for begin, finish in occupied):
                    continue
                occupied.append((start, end))
                if term_id not in selected:
                    selected.append(term_id)
            selected.sort(
                key=lambda tid: next((m[0] for m in matches if m[2] == tid), len(phrases))
            )
            if any(tid not in context_ids for tid in selected):
                raise ValueError("Undefined Context term in " + node["id"])
            # Generator-declared Context links are part of the node contract.
            # Validate the IDs rather than replacing them with lexical guesses.
            explicit = node.get("context_term_ids") or []
            if any(tid not in context_ids for tid in explicit):
                raise ValueError("Invalid Context reference in " + node["id"])
            if view == "code":
                code_roots = {
                    "code:root": ["term.code_tree"],
                    "code:group:saebase": ["term.saebase"],
                    "code:group:datasets": ["term.dataset"],
                    "code:group:models": ["term.model"],
                    "code:group:tasks": ["term.task"],
                    "code:group:harness": ["term.code_tree"],
                }
                selected = list(dict.fromkeys(
                    explicit + code_roots.get(node["id"], [])
                    + exact_code_context(node, context_ids) + selected
                ))
            else:
                selected = list(dict.fromkeys(explicit + selected))
            if node.get("presentation_only") and view != "context":
                if not selected:
                    raise ValueError("Generated node lacks authoritative Context: "
                                     + view + "/" + node["id"])
                title = str(node.get("title") or "")
                explanation = str(node.get("reader_explanation") or "")
                banned = (
                    r"第\\d+组|按源码职责浏览|分组浏览相关|"
                    r"按实际源码目录、定义职责|"
                    r"这是一个功能分组，名称是|"
                    r"按同一来源与主题整理已有节点"
                )
                if re.search(banned, title + " " + explanation):
                    raise ValueError("Generic synthetic node rejected: "
                                     + view + "/" + node["id"])
            node["context_term_ids"] = selected
    for view, graph in graphs.items():
        node_ids = {node["id"] for node in graph["nodes"]}
        for node in graph["nodes"]:
            if (":nav:" in node["id"] or ":public-nav:" in node["id"]
                    or re.search(r"第\\d+组", str(node.get("title") or ""))):
                raise ValueError("Synthetic navigation node rejected: " + node["id"])
        sibling_titles = set()
        for edge in graph["edges"]:
            if edge["type"] not in HIERARCHY | {"contains"}:
                continue
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                continue
            child = next((n for n in graph["nodes"] if n["id"] == edge["target"]), None)
            if not child or not child.get("presentation_only"):
                continue
            signature = (edge["source"], child["title"])
            if signature in sibling_titles:
                raise ValueError("Duplicate generated node responsibility: "
                                 + view + "/" + child["id"])
            sibling_titles.add(signature)


def glossary_graph(glossary: dict, allowed_knowledge: set[str]) -> dict:
    """Compatibility for older callers; the public view is now Context."""
    return context_graph(glossary, allowed_knowledge, {"schema_version": 1, "entries": []})


def structure_patch(graphs: dict) -> dict:
    """Compact source-backed structure for Pages' independently stored data.js.

    Static Pages snapshots are large. Publish an additive Knowledge projection
    plus moved Code/Context edges so the website and build-time tree agree.
    Stable Python source IDs remain the identity for all Code details.
    """
    code = graphs["code"]
    context = graphs["context"]
    # No fake Code navigation nodes or source-range parents are published.
    context_extra = [
        node
        for node in context["nodes"]
        if node["id"].startswith(("term.sae", "context:category:sae_"))
    ]
    context_extra_ids = {node["id"] for node in context_extra}
    context_parents = {
        edge["target"]: edge["source"]
        for edge in context["edges"]
        if edge["type"] == "contains"
        and (edge["source"] in context_extra_ids or edge["target"] in context_extra_ids)
    }
    return {
        "schema": "openresearch.atlas.structure.v1",
        "knowledge": graphs["knowledge"],
        "code": {
            "nodes": [],
            "parents": {},
            "science_links": {
                node["id"]: node.get("context_ids", [])
                for node in code["nodes"]
                if any(key.startswith(("g.sae.", "s.sae.")) for key in node.get("context_ids", []))
            },
        },
        "context": {"nodes": context_extra, "parents": context_parents},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", default=str(ROOT / "harness/core/project_graph.json"))
    parser.add_argument("--knowledge", default=str(ROOT / "doc/knowledge/context_graph.json"))
    parser.add_argument("--glossary", default=str(ROOT / "doc/knowledge/glossary.json"))
    parser.add_argument(
        "--context-attributes", default=str(ROOT / "doc/knowledge/context_attributes.json")
    )
    parser.add_argument("--code", default=str(ROOT / "doc/code/code_tree.json"))
    parser.add_argument("--environment", default=str(ROOT / "harness/environment/example.json"))
    parser.add_argument("--output", default=str(DEFAULT))
    args = parser.parse_args()
    try:
        core = existing_tree(load(Path(args.core)), view="core")
        knowledge = existing_tree(load(Path(args.knowledge)), view="knowledge")
        core_ids = {n["id"] for n in core["nodes"]}
        knowledge_ids = {n["id"] for n in knowledge["nodes"]}
        code = readable_code_tree(load(Path(args.code)))
        for node in code["nodes"]:
            node["core_ids"] = [key for key in node.get("core_ids", []) if key in core_ids]
            node["context_ids"] = [
                key for key in node.get("context_ids", []) if key in knowledge_ids
            ]
        graphs = {
            "core": core,
            "knowledge": knowledge,
            "code": code,
            "environment": readable_environment(load(Path(args.environment))),
            "context": context_graph(
                load(Path(args.glossary)), knowledge_ids, load(Path(args.context_attributes))
            ),
        }
        attach_evidence(graphs["core"], code, CODE_TARGETS)
        attach_evidence(graphs["knowledge"], code, CODE_TARGETS)
        attach_evidence(graphs["context"], code, TERM_TARGETS)
        attach_context_explanations(graphs)
        verify_public_prose(graphs)
        for name, graph in graphs.items():
            ids = {node["id"] for node in graph["nodes"]}
            if len(ids) != len(graph["nodes"]):
                raise ValueError("Duplicated displayed node ID: " + name)
            if name != "environment" and any(n["status"] != "implemented" for n in graph["nodes"]):
                raise ValueError("Non-implemented capability leaked to " + name)
            for edge in graph["edges"]:
                if edge["source"] not in ids or edge["target"] not in ids:
                    raise ValueError("Broken " + name + " edge " + str(edge))
        blob = "/* Generated from source-backed existing capabilities; no planned work. */\n"
        blob += (
            "window.OPENRESEARCH_ATLAS = "
            + json.dumps(graphs, ensure_ascii=False, separators=(",", ":"))
            + ";\n"
        )
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(blob, encoding="utf-8")
        patch = structure_patch(graphs)
        sibling = dest.parent / STRUCTURE.name
        sibling.write_text(
            "/* Source-backed Atlas navigation overlay. */\n"
            + "window.OPENRESEARCH_ATLAS_PATCH = "
            + json.dumps(patch, ensure_ascii=False, separators=(",", ":"))
            + ";\n",
            encoding="utf-8",
        )
        print(
            "Exported Atlas (existing capabilities only):",
            {name: len(g["nodes"]) for name, g in graphs.items()},
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Atlas export failed:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

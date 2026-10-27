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
from harness.code.presentation import readable_code_tree
from harness.knowledge.context_attributes import validate_attributes
from harness.visualization.evidence import attach_evidence, CODE_TARGETS, TERM_TARGETS
from harness.visualization.language import verify_public_prose, NarrativeError

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
        "core": ("OpenResearch 双核心与任务注册", "展示 DLBase、SAEBase 和 Dataset、Model、Task 注册架构中已经实现的组件。"),
        "knowledge": ("Vision 分类与 LLM SAE 研究方法", "按 CUB-200 分类及 LLM SAE 的八类源码支持方法组织研究知识，科学结论另行审计。"),
    }
    title, summary = labels[view]
    root = {"id": root_id, "kind": "root", "title": title, "summary": summary,
            "description": summary, "status": "implemented", "evidence": []}
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
            node["description"] = "现有知识图通过稳定节点编号、证据、三态进度和父子层级表示已经实现的研究组件。"
        if node["id"] == "s.explainability":
            node["description"] = "现有项目用 Core、Knowledge、Code、Environment 和术语字典分别整理架构、研究知识、源代码及执行资源。"
        if node["id"] == "q.modularity":
            node["title"] = "Dataset、Model、Task Registry 解耦"
            node["description"] = "当前数据集、模型和任务已经通过注册与兼容性接口解耦。"
        elif node["id"] == "q.transparency":
            node["title"] = "Atlas 五棵 Tree 的职责与关联"
            node["description"] = "现在能够分别浏览项目架构、科研知识、源代码和环境结构。"
        elif node["id"] == "q.cub.reproducibility":
            node["title"] = "Hydra 配置、随机种子与实验结果追溯"
            node["description"] = "已有训练配置、随机种子和分层结果目录记录。真实复现实验可依据这些记录执行。"
        specific = {
            "g.cub.dataset": "CUB-200 图像与标签读取",
            "g.cub.transforms": "CUB-200 输入尺寸与图像增强",
            "g.cub.loss": "CUB-200 分类训练损失",
            "g.cub.config": "CUB-200 Hydra 训练配置记录",
            "g.cub.seed": "训练随机种子与运行环境记录",
            "g.cub.artifacts": "CUB-200 训练结果文件追溯",
            "g.registry": "Dataset、Model、Task 插件注册",
            "g.pipeline": "Task 执行前的注册与兼容检查",
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
        original_type = next((e["type"] for e in graph["edges"]
                              if e["source"] == parent_id and e["target"] == node["id"]), None)
        # Retained actual relation where possible, otherwise a grouping-only edge.
        edges.append({"source": parent_id, "target": node["id"],
                      "type": original_type or "contains"})
    output = {
        "schema_version": 1, "kind": view, "view_policy": "implemented_only",
        "title": title, "status_policy": "only implemented/source-backed nodes are published",
        "source": "harness/core/project_graph.json" if view == "core" else
                  "doc/knowledge/context_graph.json",
        "nodes": visible, "edges": edges,
    }
    if view == "knowledge":
        organize_knowledge_topics(output)
    balance_visible_navigation(output)
    return output


def organize_knowledge_topics(output: dict) -> None:
    """Group existing research methods by scientific domain, not Work Plan IDs."""
    nodes = {node["id"]: node for node in output["nodes"]}
    group_specs = (
        ("knowledge:area:cub", "CUB-200 图像分类与 ResNet-18",
         "以 CUB-200 的图像、分类模型、训练和实验复现组织现有方法。",
         "datasets/vision/CUB_200/CUB_200.py"),
        ("knowledge:area:sae", "LLM 稀疏特征与可解释性实验",
         "以真实 SAEBase 和 LLM Task 接口组织特征解释、干预与评测方法。",
         "saebase/api/features.py"),
    )
    for key, title, summary, proof in group_specs:
        node = {"id": key, "kind": "group", "title": title,
                "summary": summary, "description": summary,
                "status": "implemented", "evidence": [proof],
                "presentation_only": True}
        output["nodes"].append(node)
        nodes[key] = node
        output["edges"].append({"source": "knowledge:existing", "target": key,
                                "type": "contains"})
    for edge in output["edges"]:
        if edge["source"] != "knowledge:existing":
            continue
        target = edge["target"]
        if target in {x[0] for x in group_specs}:
            continue
        edge["source"] = ("knowledge:area:sae" if target.startswith(("q.sae.", "s.sae.", "g.sae."))
                          else "knowledge:area:cub")
    cub_topics = {
        "data": ("CUB-200 图像、标签与输入转换", "datasets/vision/CUB_200/datamodule_setup.py"),
        "model": ("ResNet-18 主干与 200 类分类输出", "models/vision/resnet18_mlp/resnet18_mlp.py"),
        "train": ("分类损失、优化器与学习率", "tasks/vision/classification/classification_module.py"),
        "provenance": ("Hydra 配置、随机种子与结果追溯", "dlbase/experiment_paths.py"),
    }
    def topic(node_id: str) -> str:
        if node_id in {"g.cub.dataset", "g.cub.transforms", "s.cub.data"}:
            return "data"
        if node_id in {"s.cub.model", "g.cub.resnet", "g.cub.logits"}:
            return "model"
        if node_id in {"s.cub.training", "g.cub.loss", "g.cub.optimizer", "g.cub.schedule"}:
            return "train"
        return "provenance"
    cub_links = [edge for edge in output["edges"]
                 if edge["source"] == "knowledge:area:cub"]
    for key, (title, proof) in cub_topics.items():
        members = [edge for edge in cub_links if topic(edge["target"]) == key]
        if not members:
            continue
        gid = "knowledge:topic:cub:" + key
        label = {"data":"图像与数据", "model":"分类模型",
                 "train":"优化与训练", "provenance":"运行和复现"}[key]
        output["nodes"].append({
            "id": gid, "kind": "group", "title": title,
            "summary": "CUB-200 " + label + " 的源码支持研究组件。",
            "description": "展示已有输入、模型或运行接口及其可核验依据。",
            "status": "implemented", "evidence": [proof],
            "presentation_only": True,
        })
        output["edges"].append({
            "source": "knowledge:area:cub", "target": gid, "type": "contains"})
        for edge in members:
            edge["source"] = gid


def balance_visible_navigation(graph: dict, *, limit: int = 10) -> None:
    """Keep public non-Code Atlas views within ten children per visible node.

    The last-resort partitions expose actual node names as readable headings.
    Only presentation nodes are inserted; original research IDs and statuses
    remain unchanged.
    """
    by_id = {node["id"]: node for node in graph["nodes"]}
    edges = graph["edges"]
    view = graph["kind"]
    for _ in range(120):
        children = defaultdict(list)
        for edge in edges:
            if edge["type"] in HIERARCHY | {"contains"}:
                children[edge["source"]].append(edge["target"])
        over = [(key, values) for key, values in children.items()
                if len(values) > limit]
        if not over:
            return
        parent, kids = sorted(over, key=lambda x: x[0])[0]
        chosen = []
        if view == "context" and parent == "context:category:training":
            subareas = {
                "loss": {"term.cross_entropy", "term.label_smoothing"},
                "optimizer": {"term.sgd", "term.momentum", "term.weight_decay", "term.learning_rate", "term.cosine"},
                "process": {"term.batch_size", "term.epoch", "term.lightning", "term.checkpoint"},
            }
            labels = {"loss": "监督损失与标签策略",
                      "optimizer": "优化器、动量与学习率",
                      "process": "批次、训练轮次与权重检查点"}
            for key, members in subareas.items():
                subset = [id for id in kids if id in members]
                if subset:
                    chosen.append((key, labels[key], subset))
            used = {x for _k, _label, members in chosen for x in members}
            extra = [id for id in kids if id not in used]
            if extra:
                chosen.append(("other", "训练流程相关术语", extra))
        if len(chosen) < 2:
            count = max(limit, (len(kids) + limit - 1) // limit)
            buckets = [kids[i:i+count] for i in range(0, len(kids), count)]
            chosen = []
            for i, members in enumerate(buckets):
                first, last = by_id[members[0]]["title"], by_id[members[-1]]["title"]
                title = ("术语：" if view == "context" else "节点：") + first + " 至 " + last
                chosen.append((str(i), title, members))
        if len(chosen) > limit or len(chosen) < 2:
            raise ValueError("Cannot balance Atlas navigation at " + parent)
        old = {(parent, id) for id in kids}
        edges[:] = [edge for edge in edges
                    if edge["type"] not in HIERARCHY | {"contains"} or
                    (edge["source"], edge["target"]) not in old]
        for tag, title, members in chosen:
            gid = view + ":nav:" + sha256((parent+"|"+tag).encode()).hexdigest()[:14]
            if gid in by_id:
                raise ValueError("Duplicate Atlas navigation group " + gid)
            node = {"id": gid, "kind": "group", "title": title,
                    "summary": "按同一来源与主题整理已有节点。",
                    "description": "包含以下已有科研或术语节点的直接链接。",
                    "status": "implemented", "evidence": graph.get("source", []) if isinstance(graph.get("source"), list) else [graph.get("source") or "doc/knowledge/glossary.json"],
                    "presentation_only": True}
            graph["nodes"].append(node)
            by_id[gid] = node
            edges.append({"source": parent, "target": gid, "type": "contains"})
            for child in members:
                edges.append({"source": gid, "target": child, "type": "contains"})
    raise ValueError("Atlas navigation hierarchy exceeded safe rebalancing depth")


def readable_environment(profile: dict) -> dict:
    """Human-readable observability tree. Never pass synthetic data as actual facts."""
    synthetic = bool(profile.get("synthetic", False))
    nodes, edges = [], []
    state_default = "not_implemented" if synthetic else "implemented"
    prefix = "示例数据：" if synthetic else "本机观测："

    def present(key, kind, title, summary, parent=None, source="unknown",
                status=None, facts=None):
        item = {
            "id": key, "kind": kind, "title": title, "summary": summary,
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
        gib = value / 1024 ** 3
        if gib >= 1024:
            return f"{gib/1024:.1f} TiB"
        return f"{gib:.1f} GiB"

    def as_text(value, default="待现场采集"):
        return str(value) if value not in (None, "", "unknown", "not observed", "not probed") else default

    present("env:root", "root", "CPU、GPU 与科研任务执行环境",
            "当前页面使用演示环境，展示硬件资源的组织形式。" if synthetic
            else "当前机器的硬件与执行工具由本地探测器采集，字段记录实际观测结果。",
            source="example.json" if synthetic else "local environment probe")

    platform = profile.get("platform", {})
    cpu = profile.get("cpu", {})
    memory = profile.get("memory", {})
    gpu = profile.get("gpu", {})
    storage = profile.get("storage", {})
    execution = profile.get("execution", {})

    system = as_text(platform.get("system"))
    machine = as_text(platform.get("machine"))
    present("env:platform", "group", "操作系统与处理器架构",
            f"{prefix}系统 {system}；硬件架构 {machine}。",
            parent="env:root", source=platform.get("source"))

    core_count = as_text(cpu.get("logical_count"))
    physical_count = as_text(cpu.get("physical_count"))
    present("env:cpu", "group", "中央处理器",
            f"{prefix}逻辑处理单元 {core_count} 个；物理核心 {physical_count} 个。",
            parent="env:root", source=cpu.get("source"))

    present("env:memory", "group", "系统内存",
            f"{prefix}总容量 {display_size(memory.get('total_bytes'))}；"
            f"当前可用 {display_size(memory.get('available_bytes'))}。",
            parent="env:root", source=memory.get("source"))

    devices = gpu.get("devices", [])
    gpu_state = "not_implemented" if synthetic or not devices else "implemented"
    if not devices and not synthetic and gpu.get("status") == "unknown":
        gpu_state = "partial"
    gpu_note = ( f"演示配置列出 {len(devices)} 张图形处理器，展示设备管理结构。" if synthetic
                else f"观测到 {len(devices)} 张图形处理器。" if devices
                else "本机图形设备信息等待驱动与探测工具核验。")
    present("env:gpu", "group", "图形计算设备", gpu_note,
            parent="env:root", source=gpu.get("source"), status=gpu_state)
    for i, device in enumerate(devices):
        mem = device.get("memory_mib")
        memory_text = f"{mem/1024:.1f} GiB" if isinstance(mem,(int,float)) else "待现场采集"
        present("env:gpu:" + str(device.get("index", i)), "device",
                f"第 {i+1} 张图形处理器",
                (f"{prefix}设备显存 {memory_text}；"
                 + ("型号来自演示配置。" if synthetic
                    else "实际型号和驱动可在详细观测记录中核对。")),
                parent="env:gpu", source=gpu.get("source"),
                facts={"设备名称": as_text(device.get("name")),
                       "驱动版本": as_text(device.get("driver_version"))})

    present("env:storage", "group", "项目所在磁盘",
            f"{prefix}总空间 {display_size(storage.get('total_bytes'))}；"
            f"剩余空间 {display_size(storage.get('free_bytes'))}。",
            parent="env:root", source=storage.get("source"))

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
            state="not_implemented"
            summary="此项来自演示配置，运行状态由实际机器采样确定。"
        else:
            state="implemented" if available else "not_implemented"
            summary="当前环境能够找到这个工具。" if available else "当前命令路径的检测结果为待安装。"
        statuses.append(state)
        title, meaning = command_labels.get(command, (f"{command} 命令可用性",
                                                      f"检测环境中的 {command} 命令路径与可用性。"))
        present("env:command:" + command, "capability", title,
                meaning + " " + summary, parent="env:execution",
                source=execution.get("source"), status=state,
                facts={"命令标识":command})
    exec_status=("not_implemented" if synthetic else
                 "implemented" if statuses and all(x=="implemented" for x in statuses)
                 else "partial")
    present("env:execution", "group", "任务执行工具",
            ("示例中的工具状态展示配置组织形式。" if synthetic
             else "这里列出当前运行环境实际可检测的终端和执行工具。"),
            parent="env:root", source=execution.get("source"), status=exec_status,
            facts={"Python 版本":as_text(execution.get("python_version")),
                   "网络条件":"待采集" if execution.get("network") == "not probed" else as_text(execution.get("network")),
                   "沙箱条件":as_text(execution.get("sandbox"))})
    # Children were added before the execution group for simplicity; the
    # client builds its hierarchy from edges and does not depend on node order.
    return {
        "schema_version": 1, "kind": "environment",
        "synthetic": synthetic, "captured_at_utc": profile.get("captured_at_utc"),
        "nodes": nodes, "edges": edges,
    }


def validate_atlas_vocabulary(vocabulary: dict, glossary: dict) -> None:
    """Only curated, source-backed terms may become public Context entries."""
    if vocabulary.get("schema_version") != 1 or vocabulary.get("kind") != "atlas_context_vocabulary":
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
                not isinstance(alias, str) or not alias.strip() for alias in term["aliases"]):
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


def context_graph(glossary: dict, allowed_knowledge: set[str], curated: dict,
                  vocabulary: dict | None = None) -> dict:
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
        "id": "context:root", "kind": "root", "title": "科研术语定义与源码依据",
        "summary": "这里集中解释 Atlas 中使用的科研与工程术语，并给出可核对的资料来源。",
        "description": "这里集中解释 Atlas 中使用的科研与工程术语，并给出可核对的资料来源。",
        "status": "implemented", "evidence": ["doc/knowledge/glossary.json", "doc/knowledge/atlas_terms.json"],
    }
    nodes, edges = [root], []
    for category in categories:
        if not any(term["category"] == category["id"] for term in terms):
            continue
        catid = "context:category:" + category["id"]
        nodes.append({
            "id": catid, "kind": "group", "title": category["title"],
            "summary": category["description"], "description": category["description"],
            "status": "implemented", "evidence": [
                "doc/knowledge/glossary.json" if category in glossary["categories"]
                else "doc/knowledge/atlas_terms.json",
            ],
        })
        edges.append({"source": "context:root", "target": catid, "type": "contains"})
    for term in terms:
        if term["category"] not in category_ids:
            raise ValueError("Unknown Context category " + term["category"])
        allowed_links = [key for key in term.get("related_node_ids", [])
                         if key in allowed_knowledge]
        attributes = [
            {"type": "text", "label": "为什么重要", "value": term["why_it_matters"],
             "source": term["foundation"]},
        ]
        if term.get("aliases"):
            attributes.append({
                "type": "text", "label": "其他名称",
                "value": "、".join(term["aliases"]), "source": term["foundation"],
            })
        attributes.extend(by_id.get(term["id"], []))
        # Even terms not yet associated with implemented experiments carry a
        # genuine source-backed reference, not a synthetic method.
        attributes.append({
            "type": "code_reference", "label": "真实依据",
            "path": term["source_evidence"][0],
        })
        attributes.append({
            "type": "reference", "label": "基础解释",
            "path": term["foundation"],
        })
        nodes.append({
            "id": term["id"], "kind": "term", "title": term["term"],
            "summary": term["definition"], "description": term["definition"],
            "why_it_matters": term["why_it_matters"], "aliases": term["aliases"],
            "foundation": term["foundation"], "status": "implemented",
            "evidence": term["source_evidence"], "context_ids": allowed_links,
            "attributes": attributes,
        })
        edges.append({
            "source": "context:category:" + term["category"],
            "target": term["id"], "type": "contains",
        })
    return {
        "schema_version": 1, "kind": "context", "view_policy": "defined_terms",
        "project": glossary.get("project"), "nodes": nodes, "edges": edges,
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
    summary = re.sub(r"^(?:问题|方案|目标|定义|现状)[:：]\s*", "",
                     str(node.get("summary") or node.get("description") or "")).strip()
    summary = summary.rstrip("。； ")
    title = node.get("title") or node["id"]
    role = {
        "root": "这是该视图的总览节点",
        "group": "这是一个功能分组",
        "module": "这是已有源码中的一个模块",
        "class": "这是已有源码中的一个可复用组件",
        "function": "这是已有源码中的一个可调用函数",
        "method": "这是一个类内部实现的操作",
        "term": "这是由 Context 统一定义的概念",
        "device": "这是执行环境中的一项计算资源",
        "capability": "这是执行环境中的一项工具或能力",
        "question": "这是现有科研或工程问题的说明节点",
        "solve": "这是现有问题的一种实现方案",
        "goal": "这是现有实现中的具体目标",
    }.get(node.get("kind") or node.get("type"), "这是一个已记录的研究节点")
    introduction = f"{role}，名称是“{title}”。"
    statement = f"它的作用是：{summary}。" if summary else ""
    if view == "code":
        continuation = "下方的源码接口和出处可用于核对它实际执行的操作。"
    elif view == "environment":
        continuation = ("这一视图标明资源与运行工具的含义；演示配置中的数值"
                        "来自合成示例。")
    elif view == "context":
        continuation = "术语含义与出处由 Context 集中维护。"
    else:
        continuation = "下方给出可核对的实现依据和相关概念。"
    return introduction + statement + continuation


def semantic_terms(node: dict) -> str:
    """Search user-facing prose, never literal source code or signatures."""
    fields = [node.get("title", ""), node.get("summary", ""),
              node.get("description", ""), node.get("reader_explanation", "")]
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
    ("tasks/LLM/capture/", ("term.sae", "term.sae.capture")),
    ("tasks/LLM/intervene/", ("term.sae", "term.sae.intervene")),
    ("tasks/LLM/generate/", ("term.sae", "term.sae.generate")),
    ("tasks/LLM/circuits/", ("term.sae", "term.sae.circuits")),
    ("tasks/LLM/explain/", ("term.sae", "term.sae.explain")),
    ("tasks/LLM/benchmark/", ("term.sae", "term.sae.benchmark")),
    ("tasks/LLM/train/", ("term.sae", "term.sae.train")),
    ("models/LLM/gemma2_2b_base/", ("term.gemma2_2b", "term.model")),
    ("models/LLM/", ("term.llm", "term.model")),
    ("models/vision/resnet18_mlp/", ("term.resnet18", "term.model")),
    ("datasets/NLP/", ("term.nlp", "term.sae.assets")),
    ("datasets/vision/", ("term.dataset", "term.cub200")),
    ("dlbase/cfg/", ("term.hydra", "term.yaml")),
    ("dlbase/training/", ("term.training", "term.lightning")),
    ("dlbase/registry.py", ("term.registry", "term.plugin")),
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
        ((alias, term["id"]) for term in terms
         for alias in [term["title"], *term.get("aliases", [])]
         if len(alias.strip()) >= 2),
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
            selected.sort(key=lambda tid: next(
                (m[0] for m in matches if m[2] == tid), len(phrases)))
            if any(tid not in context_ids for tid in selected):
                raise ValueError("Undefined Context term in " + node["id"])
            if view == "code" and node.get("kind") in {"module", "class", "method", "function"}:
                # Put the verified research-domain terms first, then retain
                # matching technical terms from the readable description.
                selected = list(dict.fromkeys(
                    exact_code_context(node, context_ids) + selected))
            node["context_term_ids"] = selected



def glossary_graph(glossary: dict, allowed_knowledge: set[str]) -> dict:
    """Compatibility for older callers; the public view is now Context."""
    return context_graph(glossary,allowed_knowledge,{"schema_version":1,"entries":[]})



def structure_patch(graphs: dict) -> dict:
    """Compact source-backed structure for Pages' independently stored data.js.

    Static Pages snapshots are large. Publish an additive Knowledge projection
    plus moved Code/Context edges so the website and build-time tree agree.
    Stable Python source IDs remain the identity for all Code details.
    """
    code = graphs["code"]
    context = graphs["context"]
    grouped_code = [node for node in code["nodes"]
                    if node["id"].startswith("code:nav:")]
    code_parents = {
        edge["target"]: edge["source"] for edge in code["edges"]
        if edge["type"] == "contains" and
        (edge["source"].startswith("code:nav:")
         or edge["target"].startswith("code:nav:"))
    }
    context_extra = [
        node for node in context["nodes"] if
        node["id"].startswith(("term.sae", "context:category:sae_", "context:nav:"))
    ]
    context_extra_ids = {node["id"] for node in context_extra}
    context_parents = {
        edge["target"]: edge["source"] for edge in context["edges"]
        if edge["type"] == "contains" and (
            edge["source"] in context_extra_ids or
            edge["target"] in context_extra_ids)
    }
    return {
        "schema": "openresearch.atlas.structure.v1",
        "knowledge": graphs["knowledge"],
        "code": {
            "nodes": grouped_code, "parents": code_parents,
            "science_links": {
                node["id"]: node.get("context_ids", [])
                for node in code["nodes"]
                if any(key.startswith(("g.sae.", "s.sae."))
                       for key in node.get("context_ids", []))
            },
        },
        "context": {"nodes": context_extra, "parents": context_parents},
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core",default=str(ROOT/"harness/core/project_graph.json"))
    parser.add_argument("--knowledge",default=str(ROOT/"doc/knowledge/context_graph.json"))
    parser.add_argument("--glossary",default=str(ROOT/"doc/knowledge/glossary.json"))
    parser.add_argument("--context-attributes",default=str(ROOT/"doc/knowledge/context_attributes.json"))
    parser.add_argument("--code",default=str(ROOT/"doc/code/code_tree.json"))
    parser.add_argument("--environment",default=str(ROOT/"harness/environment/example.json"))
    parser.add_argument("--output",default=str(DEFAULT))
    args=parser.parse_args()
    try:
        core=existing_tree(load(Path(args.core)),view="core")
        knowledge=existing_tree(load(Path(args.knowledge)),view="knowledge")
        core_ids={n["id"] for n in core["nodes"]}
        knowledge_ids={n["id"] for n in knowledge["nodes"]}
        code=readable_code_tree(load(Path(args.code)))
        for node in code["nodes"]:
            node["core_ids"]=[key for key in node.get("core_ids",[]) if key in core_ids]
            node["context_ids"]=[key for key in node.get("context_ids",[]) if key in knowledge_ids]
        graphs={
            "core":core,
            "knowledge":knowledge,
            "code":code,
            "environment":readable_environment(load(Path(args.environment))),
            "context":context_graph(load(Path(args.glossary)), knowledge_ids,
                                    load(Path(args.context_attributes)))
        }
        balance_visible_navigation(graphs["environment"])
        balance_visible_navigation(graphs["context"])
        attach_evidence(graphs["core"],code,CODE_TARGETS)
        attach_evidence(graphs["knowledge"],code,CODE_TARGETS)
        attach_evidence(graphs["context"],code,TERM_TARGETS)
        attach_context_explanations(graphs)
        verify_public_prose(graphs)
        for name, graph in graphs.items():
            ids={node["id"] for node in graph["nodes"]}
            if len(ids)!=len(graph["nodes"]):
                raise ValueError("Duplicated displayed node ID: "+name)
            if name != "environment" and any(n["status"]!="implemented" for n in graph["nodes"]):
                raise ValueError("Non-implemented capability leaked to "+name)
            for edge in graph["edges"]:
                if edge["source"] not in ids or edge["target"] not in ids:
                    raise ValueError("Broken "+name+" edge "+str(edge))
        blob="/* Generated from source-backed existing capabilities; no planned work. */\n"
        blob+="window.OPENRESEARCH_ATLAS = "+json.dumps(graphs,ensure_ascii=False,separators=(",",":"))+";\n"
        dest=Path(args.output)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(blob,encoding="utf-8")
        patch = structure_patch(graphs)
        sibling = dest.parent / STRUCTURE.name
        sibling.write_text("/* Source-backed Atlas navigation overlay. */\n"+
                           "window.OPENRESEARCH_ATLAS_PATCH = "+
                           json.dumps(patch,ensure_ascii=False,separators=(",",":"))+";\n",
                           encoding="utf-8")
        print("Exported Atlas (existing capabilities only):",
              {name:len(g["nodes"]) for name,g in graphs.items()})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Atlas export failed:",exc,file=sys.stderr)
        return 1
    return 0

if __name__=="__main__":
    raise SystemExit(main())

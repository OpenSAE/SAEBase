"""Source-backed presentation for existing SAEBase Python interfaces."""

from __future__ import annotations
import copy
from collections import defaultdict
from pathlib import Path, PurePosixPath
from harness.code.semantic_titles import semantic_title, semantic_summary, context_for_path

GROUPS = {
    "saebase": "SAE 特征、模型适配与实验基础",
    "datasets": "NLP Prompt 数据集与来源",
    "models": "LLM 模型适配与兼容性",
    "tasks": "SAE 捕获、干预、生成、解释和评测任务",
    "harness": "Harness 代码结构与研究证据",
}
GROUP_SUMMARIES = {
    "saebase": "提供 SAE 模型表示、特征干预、评测和存储的通用实现。",
    "datasets": "组织 NLP Prompt、数据来源及其可追溯的版本化输入。",
    "models": "实现语言模型的加载、激活位置与任务兼容性检查。",
    "tasks": "执行七类 LLM SAE 科研任务并保留原有 SAEBase 接口。",
    "harness": "维护项目结构、代码树、科研术语、实验审阅与环境记录。",
}
MODULES = {
    "saebase/api/features.py": ("SAE 特征捕获与编辑", "捕获激活、编码稀疏特征并执行受控编辑。"),
    "saebase/api/model.py": ("语言模型适配", "统一输入 Token、激活读取与编辑后输出。"),
    "saebase/api/components.py": (
        "Transformer 组件分析",
        "分析 Attention Head、MLP Neuron 与残差贡献。",
    ),
    "datasets/artifacts.py": (
        "NLP Prompt 产物",
        "管理 Prompt JSONL、manifest 和数据来源。",
    ),
    "datasets/prompts.py": ("NLP 输入读取", "加载已选数据集的真实 Prompt 记录。"),
    "saebase/registry.py": ("科研组件注册", "注册和查询模型、数据集与 SAE 任务。"),
    "saebase/runtime/train.py": ("SAE 任务执行入口", "校验配置并分派已登记的研究任务。"),
    "tasks/sae_runtime.py": ("七类 SAE 任务公共入口", "核验模型与数据并调用对应任务实现。"),
}
for _task, _title in {
    "capture": "SAE 特征捕获",
    "intervene": "SAE 教师强制干预",
    "generate": "SAE 生成干预",
    "circuits": "SAE 电路分析",
    "explain": "SAE 自动解释",
    "benchmark": "SAE 基准评测",
    "train": "SAE 训练",
}.items():
    MODULES[f"tasks/{_task}/task.py"] = (_title, "调用对应的 SAE 实验实现与验证接口。")
CLASSES = {
    "SAEBaseDataModule": ("NLP 数据管理", "装配来自独立 Prompt 来源的训练与评测记录。"),
    "DatasetPlugin": ("数据集注册接口", "定义 Prompt 数据集的加载及任务兼容条件。"),
    "ModelPlugin": ("模型注册接口", "记录 LLM 构建与任务适配方式。"),
    "TaskPlugin": ("SAE Task 注册接口", "连接已实现的研究任务与调度入口。"),
}
FUNCTIONS = {
    "register_dataset": ("注册 Prompt 数据集", "将 NLP 数据源接入统一科研接口。"),
    "register_model": ("注册语言模型", "将 LLM 构建适配接入注册表。"),
    "register_task": ("注册 SAE 研究任务", "按已实现的七类科研操作绑定任务。"),
    "run_task": ("执行 SAE 任务", "将当前 Hydra 配置交给指定研究任务。"),
    "load_raw_counterfact": ("读取反事实问答数据", "加载可追溯的 CounterFact 输入样本。"),
    "build_model": ("构建 LLM 模型", "检查配置兼容条件后构建模型实例。"),
}
METHODS = {
    "run": ("执行科研任务", "根据配置启动对应的 SAE 操作。"),
    "setup": ("准备 Prompt 数据", "依据指定来源建立任务输入集合。"),
    "__getitem__": ("读取记录", "根据索引提取指定的数据条目。"),
    "__init__": ("初始化组件", "建立对象的配置和运行状态。"),
    "forward": ("运行模型前向计算", "计算输入对应的模型表示与输出。"),
}
CONCRETE = {
    "datasets": ("NLP 数据集实现", "查看语言模型 Prompt 来源、读取和产物协议。"),
    "models": ("LLM 模型实现", "查看实际语言模型和任务兼容性。"),
    "tasks": ("七类 LLM SAE 任务", "查看特征捕获、干预、生成、电路、解释、评测与训练。"),
}
INSTANCE_LABELS = {}
KEY_METHODS = {
    "code:function:saebase.api.features:capture_features": "捕获模型激活并编码为 SAE 特征。",
}
PUBLIC_EXCERPTS = frozenset(KEY_METHODS)
ROOT = Path(__file__).resolve().parents[2]


def source_excerpt(node: dict) -> str | None:
    if node["id"] not in PUBLIC_EXCERPTS:
        return None
    source = (ROOT / node["path"]).resolve(strict=False)
    if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
        return None
    lines = source.read_text(encoding="utf-8").splitlines()
    start = node.get("line", 0) - 1
    if not (0 <= start < len(lines)):
        return None
    # The snippet is a small actual contiguous extract, not the full module.
    extract = lines[start : start + 12]
    if not extract:
        return None
    minimum_indent = min(
        (len(line) - len(line.lstrip()) for line in extract if line.strip()), default=0
    )
    return "\n".join(line[minimum_indent:].rstrip() for line in extract).rstrip()


def task_configuration_nodes(name: str, parent: str, nodes: list, edges: list) -> None:
    """Attach real Hydra task YAML under the corresponding generic task."""
    directory = ROOT / "conf/task"
    configurations = []
    for path in sorted(directory.glob("*.yaml")):
        value = path.read_text(encoding="utf-8")
        names = [
            line.split(":", 1)[1].strip().split("#", 1)[0].strip()
            for line in value.splitlines()
            if line.startswith("  name:")
        ]
        if names and names[0].strip("\"'") == name:
            configurations.append((path, value))
    if not configurations:
        return
    group_id = "code:configurations:tasks:" + name
    sources = [str(path.relative_to(ROOT)) for path, _ in configurations]
    nodes.append(
        {
            "id": group_id,
            "kind": "group",
            "title": semantic_title(
                "group", name, f"tasks/{name}/task.py", curated="管理科研任务的 Hydra 配置"
            ),
            "summary": "汇集 " + name + " 任务实际使用的 Hydra YAML 选项与可执行配置。",
            "description": "以任务 " + name + " 的配置文件为准，定义调用和资源选项。",
            "status": "implemented",
            "evidence": sources,
            "presentation_only": True,
            "source_symbol": "",
            "context_term_ids": ["term.task", "term.yaml"],
        }
    )
    edges.append({"source": parent, "target": group_id, "type": "contains"})
    for path, content in configurations:
        source = str(path.relative_to(ROOT))
        label = "配置 " + name + " 任务的 " + path.stem + " 运行选项"
        summary = "读取 " + source + " 中声明的实际任务类型和执行参数。"
        config_id = "code:config:tasks:" + name + ":" + path.stem
        nodes.append(
            {
                "id": config_id,
                "kind": "group",
                "title": label,
                "summary": summary,
                "description": summary,
                "status": "implemented",
                "evidence": [source],
                "source_path": source,
                "presentation_only": True,
                "source_symbol": "",
                "context_term_ids": ["term.task", "term.yaml"],
                "attributes": [
                    {"type": "code", "label": "当前任务配置", "value": content, "source": source},
                    {"type": "reference", "label": "配置文件来源", "path": source},
                ],
            }
        )
        edges.append({"source": group_id, "target": config_id, "type": "contains"})


def concrete_groups(output: dict) -> None:
    """Group physical datasets, model adapters, and SAE tasks directly by component."""
    nodes = output["nodes"]
    edges = output["edges"]
    for area, (title, summary) in CONCRETE.items():
        parent = "code:group:" + area
        instances = defaultdict(list)
        common = []
        for node in nodes:
            if node["kind"] != "module":
                continue
            parts = PurePosixPath(node.get("path") or "").parts
            if not parts or parts[0] != area:
                continue
            if len(parts) >= 3:
                instances[parts[1]].append(node)
            else:
                common.append(node)
        selected = common + [node for group in instances.values() for node in group]
        if not selected:
            continue
        ids = {node["id"] for node in selected}
        edges = [edge for edge in edges
                 if not (edge["type"] == "contains" and edge["source"] == parent
                         and edge["target"] in ids)]
        group_id = "code:concrete:" + area
        nodes.append({
            "id": group_id, "kind": "group", "title": title,
            "summary": summary, "description": summary, "status": "implemented",
            "evidence": [node["path"] for node in selected],
            "presentation_only": True, "source_symbol": "",
            "context_term_ids": [{"datasets": "term.dataset", "models": "term.model",
                                  "tasks": "term.task"}[area]],
        })
        edges.append({"source": parent, "target": group_id, "type": "contains"})
        for module in common:
            edges.append({"source": group_id, "target": module["id"], "type": "contains"})
        for name, modules in sorted(instances.items()):
            instance_id = "code:instance:" + area + ":" + name
            sample = modules[0]["path"]
            instance_title, instance_summary = INSTANCE_LABELS.get(
                (area, name),
                (
                    ("读取" + name + " 数据集输入")
                    if area == "datasets"
                    else ("加载" + name + " 语言模型")
                    if area == "models"
                    else ("执行" + name + " SAE 科研任务"),
                    ("加载并准备 " + name + " 对应的真实 Prompt 与样本记录。")
                    if area == "datasets"
                    else ("构建 " + name + " 模型并核查任务兼容条件。")
                    if area == "models"
                    else ("执行 " + name + " 对应的 SAE 研究操作及其配置。"),
                ),
            )
            nodes.append({
                "id": instance_id, "kind": "group", "title": instance_title,
                "summary": instance_summary, "description": instance_summary,
                "status": "implemented", "evidence": [m["path"] for m in modules],
                "presentation_only": True, "source_symbol": "",
                "context_term_ids": [{"datasets": "term.dataset", "models": "term.model",
                                      "tasks": "term.task"}[area]],
            })
            edges.append({"source": group_id, "target": instance_id, "type": "contains"})
            if area == "tasks":
                task_configuration_nodes(name, instance_id, nodes, edges)
            for module in modules:
                edges.append({"source": instance_id, "target": module["id"], "type": "contains"})
    output["edges"] = edges


# Source-derived Chinese title rules live in semantic_titles.py. The original
# symbol name and source location remain in each AST node's metadata and
# expanded interface; neither is part of its public headline.

# Navigation is a view concern. A Python source node is not a research method
# merely because its parent has many children. The Atlas UI paginates children
# without adding invented scope nodes or changing authoritative source IDs.


def readable_code_tree(graph: dict) -> dict:
    """Presentation-only, already-existing component subset of an AST snapshot."""
    output = copy.deepcopy(graph)
    output["scope_note"] = (
        "仅显示当前源代码已存在的主要实现；内部辅助函数和空包初始化不作为单独功能节点。"
    )
    output["projection"] = "existing_public_api"
    selected = []
    for item in output["nodes"]:
        kind = item["kind"]
        path = item.get("path", "")
        raw = item["title"]
        if kind == "module" and path.endswith("/__init__.py"):
            continue
        if kind in ("function", "method") and raw.startswith("_"):
            if item["id"] not in KEY_METHODS:
                continue
        # Keep all other real symbols: no synthetic classes or functions.
        selected.append(item)
    visible = {node["id"] for node in selected}
    output["edges"] = [
        edge for edge in output["edges"] if edge["source"] in visible and edge["target"] in visible
    ]
    for node in selected:
        kind = node["kind"]
        original = node["title"]
        node["source_symbol"] = original
        if kind == "root":
            label, summary = (
                "浏览科研系统的实际功能",
                ("按数据集、模型、实验运行与 SAE 分析方法浏览已实现的代码。"),
            )
        elif kind == "group":
            group = node["id"].rsplit(":", 1)[-1]
            label = GROUPS.get(group, "管理科研组件与实际接口")
            summary = GROUP_SUMMARIES.get(group, "按相近的科研职责组织已有接口。")
        else:
            labels = {"module": MODULES, "class": CLASSES, "method": METHODS, "function": FUNCTIONS}
            lookup = labels.get(kind, FUNCTIONS)
            key = node.get("path", "") if kind == "module" else original
            known = lookup.get(key)
            curated_title = known[0] if known else ""
            label = semantic_title(kind, original, node.get("path", ""), curated=curated_title)
            summary = semantic_summary(label, kind, node.get("path", ""), known[1] if known else "")
        node["title"] = label
        node["summary"] = summary
        node["technical_description"] = node.get("description", "")
        node["description"] = summary
        node["status"] = "implemented"
        node["source_path"] = node.get("path", "")
        if kind in ("method", "function"):
            # The key function and exact signature are the attributes of the
            # real leaf node, not another invented capability.
            owner = node["id"].split(":")[-1]
            symbol = owner if kind == "method" else original
            node["key_function"] = symbol
            node["attributes"] = [
                {
                    "type": "function",
                    "label": "关键函数",
                    "value": symbol,
                    "signature": node.get("signature", ""),
                },
                {
                    "type": "text",
                    "label": "功能说明",
                    "value": KEY_METHODS.get(node["id"], summary),
                },
            ]
            if node["id"] in KEY_METHODS:
                node["summary"] = KEY_METHODS[node["id"]]
                node["description"] = KEY_METHODS[node["id"]]
            snippet = source_excerpt(node)
            if snippet:
                node["attributes"].append(
                    {
                        "type": "code",
                        "label": "实际源码节选",
                        "value": snippet,
                        "source": node.get("path"),
                        "line": node.get("line"),
                    }
                )
            if node.get("path"):
                node["attributes"].append(
                    {
                        "type": "reference",
                        "label": "完整代码位置",
                        "path": node["path"],
                        "line": node.get("line"),
                    }
                )
        # Only display source-backed research links that are also in the
        # existing scientific/project projection, handled by the exporter.
    output["nodes"] = selected
    concrete_groups(output)
    from harness.code.interfaces import enrich_code_tree

    enrich_code_tree(output, METHODS)
    from harness.code.functional_taxonomy import organize_saebase_concepts

    organize_saebase_concepts(output)
    return output

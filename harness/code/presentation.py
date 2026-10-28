"""Source-backed presentation for existing SAEBase Python interfaces."""

from __future__ import annotations
import copy
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from harness.code.semantic_titles import semantic_title, semantic_summary, context_for_path
from harness.code.functional_taxonomy import is_reexport_only

GROUPS = {
    "saebase": "SAEBase 科研功能与实现概览",
    "datasets": "NLP Prompt 数据集与来源",
    "models": "LLM 模型适配与兼容性",
    "tasks": "SAE 捕获、干预、生成、解释和评测任务",
    "harness": "Harness 代码结构与研究证据",
}
GROUP_SUMMARIES = {
    "saebase": "按科研概念浏览特征表示、模型接口、因果干预、基准评测、训练与验证的真实实现。",
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
# Reviewed task responsibilities. A task family, its plugin entry, and its
# implementation are distinct concepts, even when all share the same operation.
TASK_INSTANCE_SEMANTICS = {
    "capture": (
        "SAE 激活捕获与稀疏特征读取",
        "面向选定模型层采集真实激活，形成带有特征位置和重建信息的研究记录。",
    ),
    "intervene": (
        "教师强制条件下的 SAE 因果编辑",
        "在固定输入上设置特征替换条件，对比编辑前后的模型输出分数。",
    ),
    "generate": (
        "生成过程中的 SAE 特征控制",
        "在模型逐步生成文本时应用选定 SAE 特征编辑，记录基线与干预后的生成行为。",
    ),
    "circuits": (
        "基于特征与连接的因果电路分析",
        "对指定模型与输入构建候选特征电路，并保存可复查的追踪结果。",
    ),
    "explain": (
        "SAE 特征假设生成与验证",
        "通过真实解释生成器提出特征作用假设，并在留出样本上检查解释预测。",
    ),
    "benchmark": (
        "SAE 性能与研究假设的基准评测",
        "按照评测配置执行重建、干预和可靠性等指标，并汇总可追溯结果。",
    ),
    "train": (
        "SAE 字典学习与训练验收",
        "基于指定的模型激活训练稀疏自编码器，并保存检查点及训练指标。",
    ),
}
TASK_WRAPPER_SUMMARIES = {
    "capture": ("绑定激活捕获任务入口", "将 CaptureTask 注册为 capture 操作，交由统一运行器加载模型、SAE 和输入配置。"),
    "intervene": ("绑定教师强制干预任务入口", "将 InterveneTask 指向 intervene 操作；特征替换和 Logit 对比由具体执行模块负责。"),
    "generate": ("绑定生成干预任务入口", "通过 GenerateTask 声明 generate 操作，并将任务配置交给统一 SAE 运行接口。"),
    "circuits": ("绑定电路追踪任务入口", "用 CircuitsTask 声明 circuits 操作，连接通用任务注册表与电路计算流程。"),
    "explain": ("提供解释器依赖与任务分派", "ExplainTask 接受解释生成器与预测器，并在 run 时转交给解释任务执行逻辑。"),
    "benchmark": ("绑定基准评测任务入口", "BenchmarkTask 选择 benchmark 操作，由运行器继续加载配置和已注册的评测后端。"),
    "train": ("绑定 SAE 训练任务入口", "TrainTask 声明 train 操作，统一接收 Hydra 配置并交给训练实现。"),
}
TASK_IMPLEMENTATION_SUMMARIES = {
    "capture": ("提取激活特征及重建统计", "捕获指定 Hook 的 SAE 编码，返回活跃特征数、最强特征以及重建范数。"),
    "intervene": ("替换稀疏特征并测量输出变化", "对固定 Prompt 替换指定 SAE 特征，比较前后 Logit 并返回变化范数及候选 Token。"),
    "generate": ("比较逐步生成前后的特征干预", "把 Prompt 转成可定位的 Token 样本，构造特征编辑条件并调用逐步生成对照评估。"),
    "circuits": ("运行原生电路追踪并保存图结构", "核验执行授权与输入配置后调用电路追踪器，在独立目录输出分析产物。"),
    "explain": ("生成特征解释并测试泛化预测", "读取发现与留出输入，调用真实解释生成器，并检查解释对新样本的预测能力。"),
    "benchmark": ("执行配置化基准并汇总指标", "读取已存在的评测配置，核验 Gemma 模型目标，再调用真实评测运行器保存指标。"),
    "train": ("执行训练计划并写入 SAE 检查点", "校验训练授权与模型快照，构造 TrainPlan 并把结果保存到独立检查点目录。"),
}
# Reviewed modules explain a specific source responsibility. Avoid treating
# every backend file as another implementation of the same research goal.
INTERVENTION_MODULES = {
    "clean_prefix": ("重算干净前缀的编辑对照", "每步重算原始前缀，在该前缀上运行 SAE 编辑以提供明确的干净历史参考结果。"),
    "engine": ("批量执行干预并管理重试", "按计算预算调度批量 Hook 前向传播，处理微批重试和可恢复的测量任务。"),
    "hf_cached_steering": ("在持久 KV 缓存中逐 Token 编辑特征", "每个新 Token 都编码真实残差并执行特征修改，同时保存编辑过的 KV 历史。"),
    "kv_cache_exact": ("核对独立干净 KV 与编辑解码", "使用独立的干净与编辑分支检验增量解码一致性，保留其试验协议和误差证据。"),
    "operators": ("应用 SAE 特征空间编辑算子", "根据消融、缩放、钳制、添加或特征集合条件生成 FeaturePatch。"),
    "results": ("原子写入干预测量记录", "通过 JSONL 分片、校验和与 Manifest 提供崩溃恢复及实验测量追溯。"),
    "specs": ("定义可复现的干预条件与样本身份", "声明特征集合、编辑幅度、样本读出位置及实验身份的校验约束。"),
    "steering": ("逐步重算前缀进行生成对照", "使用完整前缀重算的参考解码，比较每一步 SAE 编辑与基线生成结果。"),
}
for _task, _pair in TASK_WRAPPER_SUMMARIES.items():
    MODULES[f"tasks/{_task}/task.py"] = _pair
for _task, _pair in TASK_IMPLEMENTATION_SUMMARIES.items():
    MODULES[f"tasks/{_task}/operation.py"] = _pair
for _stem, _pair in INTERVENTION_MODULES.items():
    MODULES[f"tasks/intervene/backend/{_stem}.py"] = _pair
INSTANCE_LABELS = {("tasks", key): value for key, value in TASK_INSTANCE_SEMANTICS.items()}
KEY_METHODS = {
    "code:function:saebase.api.features:capture_features": "捕获模型激活并编码为 SAE 特征。",
}
PUBLIC_EXCERPTS = frozenset(KEY_METHODS)
ROOT = Path(__file__).resolve().parents[2]
SOURCE_REVIEWS = json.loads(
    (ROOT / "harness/code/reviewed_semantics.json").read_text(encoding="utf-8")
)["nodes"]




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
        if is_reexport_only(item):
            # Do not elevate legacy import forwards into duplicate functional
            # nodes. Canonical source implementations remain in their trees.
            continue
        if kind in ("function", "method") and raw.startswith("_"):
            if item["id"] not in KEY_METHODS:
                continue
        if (kind in {"function", "method", "class"}
                and path.startswith("saebase/research_lab/tests/")):
            # An acceptance test is evidence of the Research Lab contract, not
            # dozens of separate public research capabilities. The test module
            # and original Python source remain available as evidence.
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
        review = SOURCE_REVIEWS.get(node["id"])
        if review and (
            review["source_path"] != node.get("path")
            or review["source_symbol"] != original
        ):
            raise ValueError("Stale source review for " + node["id"])
        if review:
            label, summary = review["zh_title"], review["zh_summary"]
        elif kind == "root":
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
    from harness.code.functional_taxonomy import organize_oversized_concepts

    organize_oversized_concepts(output)
    return output

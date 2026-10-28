"""Source-backed interface documentation for the existing Python code tree.

AST supplies precise signatures, parameters, return expressions, class methods,
attributes, and source excerpts. The readable descriptions serve as conservative
labels for source-backed data; repository code remains the authority.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Parameter meanings are curated across the current codebase. A type annotation,
# default and exact usage/source line accompany every parameter.
PARAMETER_MEANINGS = {
    "self": "当前类实例",
    "cls": "当前类对象",
    "x": "模型或函数接收的输入张量／数值",
    "y": "样本对应的目标标签",
    "cfg": "当前实验组合配置",
    "dm": "当前数据加载模块实例",
    "stage": "需要准备的数据阶段",
    "batch": "当前批次的输入及标签",
    "batch_idx": "当前批次在本轮中的序号",
    "dataloader_idx": "当前数据加载器的序号",
    "epoch_id": "当前训练轮次标识",
    "kwargs": "调用方传入的额外命名参数",
    "args": "调用方传入的额外位置参数",
    "in_channels": "图像输入通道数",
    "dropout": "特征层的随机失活比例",
    "small_input": "用于选择小分辨率输入的卷积配置",
    "image_size": "模型图像输入边长",
    "train": "训练阶段的数据增强选择",
    "name": "注册或检索时使用的组件名称",
    "alias": "组件别名",
    "canonical": "组件的正式注册名称",
    "context": "当前校验或日志的上下文标识",
    "value": "待校验或转换的输入值",
    "what": "正在检查的字段名称",
    "allowed": "允许使用的名称集合",
    "task_name": "科研任务的注册名称",
    "adapter_name": "汇总器的名称",
    "adapter": "用于转换任务指标的适配函数",
    "adapter_args": "传入指标适配器的附加参数",
    "split_metrics": "各数据划分及其对应指标的映射",
    "split_order": "结果表中数据划分的显示顺序",
    "metric_order": "结果表中指标的显示顺序",
    "formatters": "指标数值格式化函数映射",
    "cols": "表格单行中的列文本",
    "widths": "每一列的目标显示宽度",
    "v": "本次待处理或格式化的值",
    "path": "输入／输出文件路径",
    "cwd": "当前实验运行目录",
    "root": "文件系统或数据根目录",
    "relative_path": "相对于当前实验的子目录层级",
    "session": "本次运行的会话名称",
    "category": "结果产物的类型",
    "description": "当前分析目标的人类可读说明",
    "context_ids": "本次分析关联的知识节点编号",
    "note": "当前记录的附加说明",
    "seed": "随机数初始化种子",
    "workers": "随机性设置覆盖的数据加载工作进程",
    "verbose": "控制执行状态的输出详细度",
    "batch_size": "每次参数更新的样本数量",
    "bs": "当前训练批次的样本数量",
    "optimizer": "当前训练使用的参数优化器",
    "max_epochs": "训练的总轮次数",
    "warmup_epochs": "训练初始阶段的学习率预热轮数",
    "min_lr_ratio": "最小学习率与初始学习率的比例",
    "min_delta": "判断指标变化所需的最小幅度",
    "window": "判断趋势使用的连续轮数",
    "vals": "连续验证轮次的指标列表",
    "dirpath": "模型检查点保存目录",
    "callbacks": "训练生命周期回调列表",
    "loggers": "训练指标记录器列表",
    "enable_progress_bar": "训练进度条的启用标记",
    "run_details": "本次训练的结构化运行信息",
    "train_start": "本次训练开始时间",
    "train_end": "本次训练结束时间",
    "best_ckpt_path": "最佳模型权重文件路径",
    "eval_note": "本次评测的数据划分与口径说明",
    "eval_blocks": "已经格式化的评测结果内容",
    "cfg_text": "训练配置的文本形式",
    "logits": "模型输出的原始类别得分",
    "loss": "训练或评测损失值",
    "label_smoothing": "交叉熵标签平滑强度",
    "weights": "模型加载的参数权重",
    "model": "当前使用的分类模型对象",
    "output": "模型预测结果",
    "output_dir": "运行结果保存目录",
    "checkpoint": "模型训练状态文件",
    "url": "本地工具或服务的访问地址",
    "host": "工具绑定的网络地址",
    "start": "可用端口搜索的起点",
    "span": "端口扫描范围",
    "timeout_s": "网络响应检查的时间限制（秒）",
    "manual_cmd": "手动启动可视化服务的命令",
    "H_orig": "原图像的高度",
    "W_orig": "原图像的宽度",
    "x_orig": "原图像中关键点的横坐标",
    "y_orig": "原图像中关键点的纵坐标",
    "bbox_orig": "原图像中边界框的位置与宽高",
    "image_id": "数据文件中图像的编号",
    "local_index": "当前数据集中的样本索引",
    "index": "当前被访问的样本索引",
    "idx": "当前被访问的样本索引",
    "split": "需要评测的数据划分名称",
    "pred": "模型生成的类别预测",
    "target": "样本的真实目标",
    "targets": "样本真实标签集合",
    "threshold": "多标签预测概率的判定阈值",
    "color": "图像中部位的显示颜色",
    "num_classes": "分类任务目标类别数量",
    "setup": "数据加载模块的构建入口函数",
    "assert_task_supported": "数据集支持任务的检查函数",
    "eval_split_note": "用于描述数据划分的函数",
    "construct": "模型创建函数",
    "assert_compatible": "模型与任务组合的兼容性检查函数",
    "task_class": "具体科研任务类",
    "weights_only": "权重文件的加载方式",
    "device": "模型与张量的执行设备",
}
FIELD_MEANINGS = {
    "backbone_drop": "主干特征上的 Dropout 操作",
    "fc1": "多层头的第一层全连接映射",
    "fc2": "多层头的类别输出层",
    "relu1": "多层头第一层后的激活",
    "model": "当前训练和预测使用的模型",
    "cfg": "当前任务的组合配置",
    "criterion": "当前分类任务的损失计算对象",
    "trainer": "训练过程的统一管理器",
    "train_dataset": "训练数据集实例",
    "val_dataset": "验证数据集实例",
    "test_dataset": "测试数据集实例",
    "batch_size": "数据加载时的批量样本数",
    "num_workers": "数据加载并行进程数",
    "num_classes": "任务的类别数量",
    "image_size": "模型输入图像尺寸",
    "_label_smoothing": "交叉熵使用的标签平滑系数",
}


# Specific meanings for every current class field in the tracked repository.
# Each field is supported by its initializer/dataclass declaration above.
FIELD_MEANINGS.update(
    {
        "train": "当前数据集所读取的官方训练划分标志",
        "transforms": "加载图像后执行的预处理与数据增强函数",
        "data_root": "输入数据的根目录路径",
        "images_file": "图像编号与相对文件路径的索引文件",
        "train_test_split_file": "官方训练与测试划分索引文件",
        "bounding_boxes_file": "各图像目标边界框的标注文件",
        "image_attributes_file": "每张图像对应的属性标注文件",
        "_train_ids": "官方训练划分中的图像编号列表",
        "_test_ids": "官方测试划分中的图像编号列表",
        "_image_id_label": "从图像编号查询类别标签的映射",
        "_image_id_bbox": "从图像编号查询边界框的映射",
        "_image_id_parts": "从图像编号查询关键点集合的映射",
        "_attr_id_to_name": "从属性编号查询属性名称的映射",
        "_train_path_label": "训练样本的图像路径与类别配对列表",
        "_test_path_label": "测试样本的图像路径与类别配对列表",
        "_train_bboxes": "训练图像对应的边界框集合",
        "_test_bboxes": "测试图像对应的边界框集合",
        "_train_parts": "训练图像对应的部位关键点集合",
        "_test_parts": "测试图像对应的部位关键点集合",
        "_train_attr_sets": "训练图像对应的属性集合",
        "_test_attr_sets": "测试图像对应的属性集合",
        "train_set": "由当前配置准备的训练数据集",
        "val_set": "用于当前训练配置的验证数据集",
        "test_set": "用于训练后评测的测试数据集",
        "_cub_legacy_val_is_test": "验证数据来源采用官方测试划分的状态标记",
        "setup": "按配置装配数据加载模块的回调函数",
        "assert_task_supported": "检查数据集支持指定任务的入口函数",
        "eval_split_note": "生成数据集评测划分说明的入口函数",
        "assert_compatible": "检查模型与任务及数据集适配条件的入口函数",
        "construct": "根据配置创建模型实例的回调函数",
        "task_class": "当前注册的具体科研任务类",
        "BAR_FORMAT": "训练进度条的文字布局模板",
        "target_epochs": "预计执行的训练总轮数",
        "_t0": "训练开始时记录的时间戳",
        "_epochs_done": "当前累计完成的训练轮数",
        "val_losses": "历次验证阶段计算的损失值列表",
        "phase1_end_lr": "第一训练阶段结束时记录的学习率",
        "is_transformer": "当前配置所对应模型的 Transformer 类型标记",
        "_input_format": "当前数据集输入张量的结构形式",
        "_multi_label": "当前任务采用多标签分类的状态标记",
        "_run_loss_sum": "当前评测范围内累计的分类损失",
        "_run_acc_sum": "当前评测范围内累计的准确率统计值",
        "_run_count": "当前评测范围内累计的样本或批次计数",
        "n_samples": "当前数据划分参与评测的样本数量",
        "loss": "当前数据划分上的平均分类损失",
        "f1_micro": "当前多标签评测的微平均 F1 指标",
        "f1_macro": "当前多标签评测的宏平均 F1 指标",
        "mean_ap": "当前多标签评测的平均精确率指标",
        "acc_top1": "当前数据划分的 Top-1 分类准确率",
        "acc_top5": "当前数据划分的 Top-5 分类准确率",
    }
)
PARAMETER_MEANINGS.update(
    {
        "transforms": "按数据加载阶段执行的图像预处理和增强函数",
    }
)


def safe_expr(node) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError, TypeError):
        return ""


@lru_cache(maxsize=96)
def parse_file(path: str) -> tuple[ast.Module, str]:
    dest = (ROOT / path).resolve(strict=False)
    if not dest.is_relative_to(ROOT.resolve()) or dest.suffix != ".py" or not dest.is_file():
        raise ValueError("Expected tracked project Python source: " + path)
    text = dest.read_text(encoding="utf-8")
    return ast.parse(text, filename=path), text


def source_for(node: ast.AST, contents: str, *, limit: int = 60) -> str:
    lines = contents.splitlines()
    start = max(0, getattr(node, "lineno", 1) - 1)
    end = min(len(lines), getattr(node, "end_lineno", start + 1))
    raw = lines[start:end]
    if len(raw) > limit:
        # Start and end are real source lines, with an explicit gap marker.
        head, tail = max(12, limit - 10), 10
        raw = raw[:head] + ["    ⋮"] + raw[-tail:]
    if not raw:
        return ""
    smallest = min((len(s) - len(s.lstrip()) for s in raw if s.strip()), default=0)
    return "\n".join(s[smallest:].rstrip() for s in raw).rstrip()


def arguments(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict]:
    a = fn.args
    positional = list(a.posonlyargs) + list(a.args)
    required = len(positional) - len(a.defaults)
    entries = []

    def put(arg: ast.arg, position: str, default=None):
        annotation = safe_expr(arg.annotation)
        meaning = PARAMETER_MEANINGS.get(arg.arg)
        if not meaning:
            meaning = (
                f"{annotation} 类型的输入，参与当前函数的计算。"
                if annotation
                else "传入当前函数的具名数据，具体使用位置见源码。"
            )
        entries.append(
            {
                "name": arg.arg,
                "meaning": meaning,
                "annotation": annotation,
                "kind": position,
                "default": safe_expr(default) if default is not None else None,
            }
        )

    for idx, arg in enumerate(positional):
        put(
            arg,
            "positional_only" if idx < len(a.posonlyargs) else "positional",
            a.defaults[idx - required] if idx >= required else None,
        )
    if a.vararg:
        put(a.vararg, "var_positional")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        put(arg, "keyword_only", default)
    if a.kwarg:
        put(a.kwarg, "var_keyword")
    return entries


def return_expressions(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    result = []

    class Returns(ast.NodeVisitor):
        def visit_Return(self, node):
            expression = safe_expr(node.value) if node.value is not None else "None"
            if expression not in result and len(result) < 5:
                result.append(expression[:120])

        def visit_FunctionDef(self, node):
            if node is fn:
                for item in node.body:
                    self.visit(item)

        def visit_AsyncFunctionDef(self, node):
            if node is fn:
                for item in node.body:
                    self.visit(item)

        def visit_Lambda(self, node):
            return

    Returns().visit(fn)
    return result or ["None"]


def output_meaning(name: str, expressions: list[str], annotation: str) -> str:
    if name == "forward":
        return "模型前向计算产生的输出张量或类别得分。"
    if name == "__getitem__":
        return "当前索引对应的图像及标注样本。"
    if name == "__len__":
        return "当前数据集的样本总数。"
    if name in {"configure_optimizers"}:
        return "训练优化器及学习率调度配置。"
    if name in {"training_step", "validation_step"}:
        return "当前批次的训练或验证结果。"
    if name == "run":
        return "调用科研任务运行流程后的返回结果。"
    if annotation == "None" or expressions == ["None"]:
        return "本函数通过状态更新、训练执行或写入操作完成工作。"
    if "Path" in annotation:
        return "当前操作使用的文件或目录路径。"
    if "bool" in annotation:
        return "当前条件的判断结果。"
    if "DictConfig" in annotation:
        return "经过组合或处理的任务配置。"
    if any("loss" in e for e in expressions):
        return "当前任务计算得到的损失值。"
    if any("logit" in e for e in expressions):
        return "当前模型的类别得分张量。"
    if "str" in annotation:
        return "当前过程生成的文字说明或标识。"
    if annotation:
        return f"函数返回的 {annotation} 类型结果。"
    return "由代码中的 return 表达式产生的结果。"


def function_interface(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, contents: str, summary: str = "", *, key: str = ""
) -> dict:
    expressions = return_expressions(fn)
    annotation = safe_expr(fn.returns)
    purpose = summary or (ast.get_docstring(fn, clean=True) or "").split("\n")[0][:260]
    if not purpose:
        purpose = "按源码中的输入与逻辑执行当前处理。"
    import re

    if re.search(
        r"\b(?:not|no|never|without|cannot|rather(?:\s+than)?)\b|[不未无非]|避免|尚|缺少|不得",
        purpose,
        re.IGNORECASE,
    ):
        purpose = f"执行 {fn.name} 的输入处理和条件判断，具体行为可在下方原始源码中核对。"
    return {
        "name": fn.name,
        "signature": ("async def " if isinstance(fn, ast.AsyncFunctionDef) else "def ")
        + fn.name
        + "("
        + safe_expr(fn.args)
        + ")"
        + (" -> " + annotation if annotation else ""),
        "purpose": purpose,
        "parameters": arguments(fn),
        "returns": {
            "name": "return",
            "annotation": annotation or "由源码推导",
            "expressions": expressions,
            "meaning": output_meaning(fn.name, expressions, annotation),
        },
        "code": source_for(fn, contents),
        "line": getattr(fn, "lineno", 0),
    }


def class_fields(cls: ast.ClassDef) -> list[dict]:
    found = {}

    def assign(key: str, expression: str, annotation=""):
        if key not in found:
            found[key] = {
                "name": key,
                "meaning": FIELD_MEANINGS.get(
                    key, f"{key} 是这个组件保存的内部属性，用于其方法读取初始化状态或计算参数。"
                ),
                "annotation": annotation,
                "assignment": expression[:180],
            }

    for item in cls.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assign(
                        target.id,
                        safe_expr(item.value),
                        safe_expr(getattr(item, "annotation", None)),
                    )
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            for nested in ast.walk(item):
                if isinstance(nested, (ast.Assign, ast.AnnAssign)):
                    targets = nested.targets if isinstance(nested, ast.Assign) else [nested.target]
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            assign(
                                target.attr,
                                safe_expr(nested.value),
                                safe_expr(getattr(nested, "annotation", None)),
                            )
    return list(found.values())


def source_declaration(node: dict):
    module, source = parse_file(node["path"])
    kind = node["kind"]
    symbol = node.get("source_symbol", "")
    if kind == "module":
        return module, source
    if kind == "class":
        return next(
            (v for v in module.body if isinstance(v, ast.ClassDef) and v.name == symbol), None
        ), source
    if kind in ("function", "method"):
        identity = node["id"].split(":")[-1]
        if kind == "function":
            return next(
                (
                    v
                    for v in module.body
                    if isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef)) and v.name == symbol
                ),
                None,
            ), source
        owner, method = identity.split(".", 1)
        cls = next(
            (v for v in module.body if isinstance(v, ast.ClassDef) and v.name == owner), None
        )
        if cls:
            return next(
                (
                    v
                    for v in cls.body
                    if isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef)) and v.name == method
                ),
                None,
            ), source
    return None, source


def enrich_code_node(node: dict, methods_description: dict[str, tuple[str, str]]) -> None:
    kind = node["kind"]
    if kind not in {"module", "class", "function", "method"} or not node.get("path"):
        return
    declaration, contents = source_declaration(node)
    if declaration is None:
        raise ValueError("Real AST declaration missing for " + node["id"])
    if kind == "class":
        funcs = []
        for method in declaration.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _, summary = methods_description.get(method.name, ("", ""))
                funcs.append(function_interface(method, contents, summary, key=method.name))
        fields = class_fields(declaration)
        node["attributes"] = [
            {"type": "fields", "label": "类属性", "fields": fields},
            {"type": "class_methods", "label": f"类中全部方法 · {len(funcs)} 个", "methods": funcs},
            {
                "type": "reference",
                "label": "类定义源码",
                "path": node["path"],
                "line": node.get("line"),
            },
        ]
        node["interface_methods"] = len(funcs)
        node["interface_fields"] = len(fields)
    elif kind in ("function", "method"):
        info = function_interface(declaration, contents, node.get("summary", ""))
        node["attributes"] = [
            {"type": "interface", "label": "函数接口", "interface": info},
            {
                "type": "reference",
                "label": "完整源码",
                "path": node["path"],
                "line": node.get("line"),
            },
        ]
        node["key_function"] = node["id"].split(":")[-1]
    elif kind == "module":
        functions = [
            item
            for item in declaration.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        classes = [item for item in declaration.body if isinstance(item, ast.ClassDef)]
        first = functions[0] if functions else classes[0] if classes else None
        node["attributes"] = [
            {
                "type": "module_contents",
                "label": "源码组成",
                "functions": [v.name for v in functions],
                "classes": [v.name for v in classes],
            },
            {
                "type": "code",
                "label": "实际代码主要部分",
                "value": source_for(first, contents, limit=34)
                if first
                else source_for(declaration, contents, limit=24),
                "source": node["path"],
                "line": getattr(first, "lineno", 1),
            },
            {"type": "reference", "label": "完整模块", "path": node["path"], "line": 1},
        ]
        if functions:
            node["attributes"].append({
                "type": "class_methods",
                "label": "模块中实际定义的函数接口 · " + str(len(functions)) + " 个",
                "methods": [function_interface(fn, contents) for fn in functions],
            })


def enrich_code_tree(graph: dict, methods_description: dict[str, tuple[str, str]]) -> dict:
    for node in graph["nodes"]:
        enrich_code_node(node, methods_description)
    return graph

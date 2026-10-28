"""Build the source-ID English overlay for SAEBase's five Atlas trees.

This generator MUST follow ATLAS_AUTHORING_STANDARD.md, especially its
Context-first ID mapping, project isolation, and bilingual release gates.
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
    "make": "Build",
    "assert": "Validate",
    "is": "Check",
    "has": "Check",
    "supports": "Check",
    "can": "Check",
    "attach": "Attach",
    "append": "Append",
    "apply": "Apply",
    "publish": "Publish",
    "release": "Release",
    "close": "Close",
    "rebuild": "Rebuild",
    "resolve": "Resolve",
    "canonicalize": "Normalize",
    "serialize": "Serialize",
    "deserialize": "Deserialize",
    "subsample": "Subsample",
    "tokenize": "Tokenize",
    "reindex": "Reindex",
    "estimate": "Estimate",
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
    "readout_position": "Token position from which model logits are read; -1 selects the final token.",
    "verify_replay": "Whether to check that replaying the unedited activation preserves the baseline logits.",
    "require_affine_decode": "Whether to verify that selected SAE feature contributions decode additively.",
    "identity": "Optional SAE checkpoint and activation-site identity metadata used for consistency checks.",
}

# Verified human-facing Chinese descriptions mapped to faithful English.
# Generic source-code templates are intentionally absent from this table.
MEANING_TRANSLATIONS = {
    "当前类实例": "The current class instance, passed as self.",
    "当前类对象": "The class object being operated on.",
    "当前实验组合配置": "Composed configuration for the current experiment.",
    "模型与张量的执行设备": "Device on which the model and tensors execute.",
    "文件系统或数据根目录": "Root directory for files or datasets.",
    "当前操作使用的文件或目录路径。": "Path to the file or directory used by this operation.",
    "输入／输出文件路径": "Path for input or output files.",
    "当前条件的判断结果。": "Result of evaluating the current condition.",
    "模型预测结果": "Outputs produced by the model's prediction step.",
    "随机数初始化种子": "Seed used to initialize a reproducible random generator.",
    "注册或检索时使用的组件名称": "Component identifier used for registration or lookup.",
    "当前模型的类别得分张量。": "Tensor of class or token scores produced by the current model.",
    "调用方传入的额外位置参数": "Additional positional arguments provided by the caller.",
    "调用方传入的额外命名参数": "Additional keyword arguments provided by the caller.",
    "模型或函数接收的输入张量／数值": "Input tensor or scalar passed into the model or function.",
    "当前校验或日志的上下文标识": "Context identifier used to label a validation or log record.",
    "当前训练和预测使用的模型": "Model used for the current training and prediction step.",
    "当前实验运行目录": "Directory containing the current experiment run.",
    "待校验或转换的输入值": "Input value to validate or convert.",
    "需要评测的数据划分名称": "Name of the dataset split to evaluate.",
    "当前任务的组合配置": "Configuration composed for the current research task.",
    "允许使用的名称集合": "Set of names accepted by the current interface.",
    "模型前向计算产生的输出张量或类别得分。": "Model output tensor or scores from the forward pass.",
    "当前训练使用的参数优化器": "Optimizer used to update trainable parameters.",
    "模型检查点保存目录": "Directory where model checkpoints are saved.",
    "训练生命周期回调列表": "Callbacks invoked during the training lifecycle.",
    "训练指标记录器列表": "Loggers that collect training metrics.",
    "训练进度条的启用标记": "Whether the training progress bar is enabled.",
    "本地工具或服务的访问地址": "Address used to access a local tool or service.",
    "手动启动可视化服务的命令": "Command for starting the visualization service manually.",
    "连续验证轮次的指标列表": "Metric values collected across successive validation rounds.",
    "判断指标变化所需的最小幅度": "Minimum metric change used to detect a meaningful difference.",
    "判断趋势使用的连续轮数": "Number of consecutive epochs or rounds considered for a trend.",
    "本次训练开始时间": "Recorded start time of the current training run.",
    "本次训练结束时间": "Recorded end time of the current training run.",
    "本次训练的结构化运行信息": "Structured metadata describing the training run.",
    "最佳模型权重文件路径": "Path to the best model checkpoint weights.",
    "本次评测的数据划分与口径说明": "Description of the evaluated data split and metric definition.",
    "已经格式化的评测结果内容": "Evaluation results after formatting for output.",
    "本次运行的会话名称": "Session name associated with this execution.",
    "科研任务的注册名称": "Registry name of the research task.",
    "本次分析关联的知识节点编号": "Knowledge node identifier associated with this analysis.",
    "输入 x": "Input x.",
    "当前被访问的样本索引": "Index of the sample currently being accessed.",
    "各数据划分及其对应指标的映射": "Mapping from dataset splits to their associated metrics.",
    "相对于当前实验的子目录层级": "Subdirectory relative to the current experiment run.",
    "运行任务的注册名称": "Name identifying the registered research task.",
    "模型或 SAE 组件标识": "Identifier of the model or SAE component.",
    "实际激活位置": "Activation site used to capture or patch a model.",
    "完整模型权重与配置": "Model weights and configuration for the selected checkpoint.",
    "结果产物的类型": "Type of artifact produced by the operation.",
    "模型与任务组合的兼容性检查函数": "Callable that checks model and task compatibility.",
    "样本的真实目标": "Ground-truth target associated with the sample.",
    "网络响应检查的时间限制（秒）": "Timeout in seconds for checking a network response.",
    "训练的总轮次数": "Total number of training epochs.",
    "数据加载模块的构建入口函数": "Callable that constructs the data-loading module.",
    "数据集支持任务的检查函数": "Callable that checks whether a dataset supports a task.",
    "用于描述数据划分的函数": "Callable that describes the dataset split.",
    "当前记录的附加说明": "Additional annotation associated with the current record.",
    "检查模型与任务及数据集适配条件的入口函数": "Callable that checks compatibility across model, task, and dataset.",
    "当前批次的输入及标签": "Input records and target labels for the current batch.",
    "用于转换任务指标的适配函数": "Adapter callable that converts task metric values.",
    "模型创建函数": "Factory callable that constructs the model.",
    "需要准备的数据阶段": "Data preparation stage to run.",
    "模型训练状态文件": "File containing the model training state.",
    "当前分析目标的人类可读说明": "Human-readable description of the current analysis target.",
    "当前批次在本轮中的序号": "Index of the current batch within its epoch.",
    "当前数据加载器的序号": "Index of the current data loader.",
    "数据加载时的批量样本数": "Number of samples per loaded batch.",
    "当前任务计算得到的损失值。": "Loss value computed by the current research task.",
    "组件别名": "Alternative name used to identify the component.",
    "组件的正式注册名称": "Canonical registry name of the component.",
    "结果表中数据划分的显示顺序": "Display order of dataset splits in the results table.",
    "结果表中指标的显示顺序": "Display order of metrics in the results table.",
    "指标数值格式化函数映射": "Mapping from metrics to numerical formatting functions.",
    "训练初始阶段的学习率预热轮数": "Number of learning-rate warmup epochs at the start of training.",
    "最小学习率与初始学习率的比例": "Ratio of the minimum learning rate to the initial learning rate.",
    "模型输出的原始类别得分": "Raw class scores emitted by the model.",
    "每次参数更新的样本数量": "Number of samples contributing to each parameter update.",
    "由当前配置准备的训练数据集": "Training dataset prepared from the active configuration.",
    "用于当前训练配置的验证数据集": "Validation dataset used by the active training configuration.",
    "用于训练后评测的测试数据集": "Test dataset used for post-training evaluation.",
    "按配置装配数据加载模块的回调函数": "Callback that builds the data-loading module from configuration.",
    "检查数据集支持指定任务的入口函数": "Callable that verifies support for a specified task in the dataset.",
    "生成数据集评测划分说明的入口函数": "Callable that generates a description of evaluation splits.",
    "根据配置创建模型实例的回调函数": "Callback that constructs a model instance from configuration.",
    "当前注册的具体科研任务类": "Concrete research task class currently registered.",
    "训练进度条的文字布局模板": "Text formatting template for the training progress bar.",
    "预计执行的训练总轮数": "Planned total number of training epochs.",
    "训练开始时记录的时间戳": "Timestamp recorded when training begins.",
    "当前累计完成的训练轮数": "Number of training epochs completed so far.",
    "历次验证阶段计算的损失值列表": "List of loss values recorded across validation rounds.",
    "第一训练阶段结束时记录的学习率": "Learning rate recorded at the end of the first training phase.",
    "具体科研任务类": "Concrete research task class used for execution.",
    "多标签预测概率的判定阈值": "Probability threshold used to classify multilabel predictions.",
    "汇总器的名称": "Name identifying the result aggregator.",
    "传入指标适配器的附加参数": "Additional arguments supplied to the metric adapter.",
    "随机性设置覆盖的数据加载工作进程": "Data-loader workers covered by random seed initialization.",
    "控制执行状态的输出详细度": "Verbosity setting controlling execution-status output.",
}
# These explanations are reviewed against the actual evaluate_sar implementation.
# The key is a stable source symbol, not an English/Chinese sentence fragment.
PARAMETER_CONTEXT = {
    "evaluate_sar": {
        "p": "Probability passed to the original SAR coalition sampler to control feature inclusion.",
        "readout_position": "Token index for baseline and intervention logit readouts; -1 selects the last token.",
        "verify_replay": "Checks that patching the captured, unedited activation reproduces baseline logits before evaluating coalitions.",
        "identity": "Optional SAEIdentity metadata validated against the hook, layer and captured SAE dimensions; returned with the SAR result.",
        "K": "Number of highest-absolute-activation SAE features selected for coalition analysis.",
        "M": "Number of feature coalitions generated and evaluated with model patches.",
        "n_fit": "Number of coalition samples assigned to fitting; the remaining samples are held out.",
        "readout": "Function mapping the logits at the readout position to a single scalar response.",
        "paper": "Original SAR paper module providing coalition sampling and the SAR estimator.",
        "require_affine_decode": "Checks that ablating selected SAE features has additive effects on decoded activations.",
        "site": "Activation hook and layer at which the SAE features are captured and edited.",
        "model": "Model adapter used for activation capture and controlled logit evaluation.",
        "sae": "Sparse autoencoder used to encode activations and decode feature edits.",
        "prompt": "Optional text input to tokenize when explicit token IDs are not supplied.",
        "tokens": "Optional pretokenized model input passed to the model adapter.",
        "seed": "Random seed passed to the original SAR coalition sampler.",
    }
}


# Stable reviewed descriptions for the most prominent SAEBase interfaces.
# Add research-critical entries here only after inspecting the actual source.
REVIEWED_CODE = {
    "code:class:datasets.controlled_inputs:BinaryFactor": (
        "Define a named binary experimental factor",
        "Represent one controlled-input factor with exactly two distinct, nonempty string levels.",
    ),
    "code:class:datasets.controlled_inputs:ControlledInputState": (
        "Represent one controlled prompt state",
        "Store the binary factor mask, exact prompt, sample identity, and token readout position.",
    ),
    "code:class:datasets.controlled_inputs:ControlledInputDesign": (
        "Construct a complete binary controlled-input design",
        "Represent one prompt for each combination of independent binary factors, with ordered masks and traceable sample identities.",
    ),
    "code:class:datasets.controlled_inputs:ControlledInputCapture": (
        "Record activations and provenance for a controlled-input design",
        "Bundle collected sample records, collection statistics, and the saved controlled-input manifest.",
    ),
    "code:function:saebase.registry:register_dataset": (
        "Register datasets with task eligibility checks",
        "Register a named dataset, its loader, the supported-task validation callback, "
        "and the evaluation split description in the shared registry.",
    ),
    "code:function:saebase.registry:register_model": (
        "Register language models and compatibility rules",
        "Store the model's constructor and compatibility checker under a registered name.",
    ),
    "code:function:saebase.registry:register_task": (
        "Register a research task and its implementation",
        "Bind a named research operation to its task implementation through the common registry.",
    ),
    "code:function:saebase.runner:run_task": (
        "Dispatch a configured SAE research task",
        "Select and execute the registered research task under the composed Hydra configuration.",
    ),
    "code:function:saebase.api.features:capture_features": (
        "Capture model activations and encode SAE features",
        "Read activations from a selected model site and encode the resulting sparse features.",
    ),
    "code:class:saebase.api.features:FeaturePatch": (
        "Represent an SAE feature patch and its activation",
        "Store the edited feature codes and corresponding model activation as one typed result.",
    ),
    "code:class:saebase.api.model:ActivationSite": (
        "Identify the layer and hook used for activation capture",
        "Represent a precise model activation site used by the adapter and SAE interfaces.",
    ),
    "code:class:saebase.api.model:ModelAdapter": (
        "Adapt language models for activation reads and edits",
        "Define the interface used to capture activations and evaluate edits for a selected model.",
    ),
    "code:function:models.gemma2_2b_base.model_task_dataset_compatibility:assert_compatible": (
        "Check Gemma-2 2B compatibility with selected datasets and SAE tasks",
        "Validate the selected research task, prompt source, and Gemma-2 2B model identity.",
    ),
}
DATASET_NAMES = {
    "BBH": "BIG-Bench Hard",
    "CounterFact": "CounterFact",
    "GreaterThan": "GreaterThan",
    "IOI": "IOI",
    "MMLUPro": "MMLU-Pro",
    "SimpleCopying": "SimpleCopying",
    "SubjectVerbAgreement": "Subject-Verb Agreement",
}
TASK_NAMES = {
    "capture": "SAE activation capture",
    "intervene": "teacher-forced SAE interventions",
    "generate": "generation-time SAE steering",
    "circuits": "circuit tracing",
    "explain": "feature explanation",
    "benchmark": "SAE benchmarking",
    "train": "sparse autoencoder training",
}
MODULE_ACTIONS = {
    "prepare": "Prepare inputs",
    "tokenize": "Tokenize and filter prompts",
    "registry": "Register and retrieve components",
    "runner": "Dispatch registered operations",
    "artifacts": "Manage versioned input artifacts",
    "prompts": "Load and validate input prompts",
    "contracts": "Validate interface contracts",
    "schema": "Define and validate data records",
    "results": "Store and inspect research results",
    "loading": "Load validated model and SAE weights",
    "experiment_paths": "Manage traceable experiment directories",
}
SOURCE_VERBS = (
    "Load",
    "Read",
    "Return",
    "Build",
    "Construct",
    "Prepare",
    "Extract",
    "Filter",
    "Calculate",
    "Compute",
    "Capture",
    "Decode",
    "Encode",
    "Generate",
    "Validate",
    "Verify",
    "Compare",
    "Apply",
    "Record",
    "Save",
    "Write",
    "Run",
    "Evaluate",
    "Configure",
    "Register",
    "Select",
    "Resolve",
    "Estimate",
)
SOURCE_REJECT = ("暂无", "请结合", "待审阅", "source-linked detail", "no verified")


def human(identifier: str) -> str:
    """Make source words readable while preserving technical abbreviations."""
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", identifier)
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    parts = [word for word in re.split(r"[_\-\s]+", value.strip("_")) if word]
    acronyms = {
        "sae",
        "llm",
        "api",
        "gpu",
        "cpu",
        "hf",
        "ioi",
        "rss",
        "sar",
        "gpt",
        "kv",
        "nll",
        "id",
        "ids",
        "json",
        "yaml",
        "mmlu",
        "bbh",
    }
    return (
        " ".join(word.upper() if word.lower() in acronyms else word for word in parts)
        or "component"
    )


def role(path: str) -> str:
    for prefix, desc in ROLES:
        if path.startswith(prefix):
            return desc
    return "research software interfaces"


def domain(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "datasets":
        return DATASET_NAMES.get(parts[1], human(parts[1])) + " examples"
    if len(parts) >= 3 and parts[0] == "models":
        name = parts[1]
        return "Gemma-2 2B model" if name == "gemma2_2b_base" else human(name) + " model"
    if len(parts) >= 3 and parts[0] == "tasks":
        return TASK_NAMES.get(parts[1], human(parts[1]) + " research")
    if len(parts) >= 4 and parts[:2] == ["saebase", "data"]:
        return DATASET_NAMES.get(parts[2], human(parts[2])) + " prompt compatibility"
    if len(parts) >= 3 and parts[:2] == ["saebase", "interventions"]:
        return "feature editing and steering"
    if len(parts) >= 4 and parts[:3] == ["saebase", "examples", "interventions"]:
        return "intervention research examples"
    if len(parts) >= 3 and parts[:2] == ["saebase", "release"]:
        return "software release verification"
    if len(parts) == 2 and parts == ["saebase", "runner.py"]:
        return "registered SAE research tasks"
    return role(path)


def source_description(node: dict) -> str:
    """Use source docstrings only where they are legible English evidence."""
    value = str(node.get("technical_description") or "").strip()
    if not value or CJK.search(value) or any(bad in value.lower() for bad in SOURCE_REJECT):
        return ""
    first = value.splitlines()[0].strip().strip(" *")
    if len(first) < 18 or len(first) > 220 or "\ufffd" in first:
        return ""
    return first


def action_label(node: dict, name: str, path: str) -> str:
    scope = domain(path)
    owner = ""
    if node.get("kind") == "method":
        owner = human(node["id"].rsplit(":", 1)[-1].rsplit(".", 1)[0])
    if name == "to_dict":
        return "Serialize " + (owner or scope) + " as a dictionary"
    if name == "from_dict":
        return "Reconstruct " + (owner or scope) + " from a dictionary"
    if name == "__init__":
        return "Initialize " + (owner or scope) + " state"
    if name in {"default_data_dir", "data_dir"}:
        return "Locate the default data directory for " + scope
    if name == "assert_compatible":
        return "Check task and dataset compatibility for " + scope
    if name in {"n_states", "n_factors"}:
        unit = "states" if name == "n_states" else "experimental factors"
        return "Count " + unit + " in " + (owner or scope)
    if name in {"reconstruction_mse", "reconstruction_stats"}:
        return "Compute SAE reconstruction metrics for " + scope
    if name == "canonicalize_behavior":
        return "Normalize experiment behavior labels for " + scope
    if name.startswith("from_") and owner:
        return "Build " + owner + " from " + human(name[5:])
    words = name.lstrip("_").split("_", 1)
    verb = VERBS.get(words[0].lower())
    if verb and len(words) > 1:
        return verb + " " + human(words[1]) + " for " + scope
    if verb:
        return verb + " " + scope
    # The Chinese source annotation supplies a semantic operation verb.
    return evidence_action(node, name, path)


# Fallback verbs must preserve a Chinese source action when there is no reviewed
# English docstring. They describe the role, not an unverified scientific effect.
CHINESE_ACTION_VERBS = (
    ("定位", "Locate"),
    ("查找", "Find"),
    ("读取", "Read"),
    ("加载", "Load"),
    ("校验", "Validate"),
    ("验证", "Verify"),
    ("检查", "Check"),
    ("筛选", "Filter"),
    ("解析", "Resolve"),
    ("计算", "Compute"),
    ("评估", "Evaluate"),
    ("评测", "Evaluate"),
    ("比较", "Compare"),
    ("重建", "Reconstruct"),
    ("构建", "Build"),
    ("生成", "Generate"),
    ("记录", "Record"),
    ("保存", "Save"),
    ("导出", "Export"),
    ("注册", "Register"),
    ("采集", "Collect"),
    ("捕获", "Capture"),
    ("干预", "Intervene on"),
    ("应用", "Apply"),
    ("装配", "Assemble"),
    ("转换", "Convert"),
    ("规范化", "Normalize"),
    ("格式化", "Format"),
    ("去除", "Remove"),
    ("删除", "Delete"),
    ("更新", "Update"),
    ("排序", "Sort"),
    ("切分", "Split"),
    ("合并", "Merge"),
    ("判断", "Check"),
    ("汇总", "Summarize"),
    ("整理", "Organize"),
    ("关闭", "Close"),
    ("启动", "Start"),
    ("训练", "Train"),
    ("为", "Attach"),
    ("标注", "Annotate"),
    ("计算", "Compute"),
    ("提供", "Expose"),
    ("定义", "Define"),
    ("执行", "Execute"),
    ("处理", "Process"),
    ("管理", "Manage"),
)


def evidence_action(node: dict, name: str, path: str) -> str:
    """Translate an existing Chinese action and attach the real AST symbol."""
    chinese = str(node.get("title", ""))
    prefix = next((english for zh, english in CHINESE_ACTION_VERBS if chinese.startswith(zh)), "")
    if not prefix:
        prefix = "Handle" if node.get("kind") in {"function", "method"} else "Implement"
    if chinese.startswith("为") and not name.startswith("attach_"):
        prefix = "Prepare"
    # Source symbol and module scope remain visible in the expanded Code panel.
    subject = human(name)
    if name in {"main", "__main__"}:
        subject = "the entry point"
    if name.endswith(("_path", "_dir", "_root")) and prefix in {"Process", "Handle"}:
        prefix = "Locate"
    if name.startswith(("is_", "has_", "supports_", "can_")) and prefix == "Handle":
        prefix = "Check"
    return prefix + " " + subject + " for " + domain(path)


def class_action(node: dict, name: str, scope: str) -> str:
    """Use the class's actual responsibility rather than a generic Provide."""
    chinese = str(node.get("title", ""))
    normalized = human(name)
    if name.endswith(("Error", "Exception")):
        verb = "Report"
    elif name.endswith(("Model", "Adapter", "Backend")):
        verb = "Adapt"
    elif name.endswith(("Writer", "Logger")):
        verb = "Write"
    elif name.endswith(("Registry", "Catalog")):
        verb = "Register"
    elif name.endswith(("Loader", "DataModule")):
        verb = "Load"
    elif name.endswith(("Plugin", "Contract", "Interface")):
        verb = "Define"
    elif chinese.startswith(("管理", "执行", "处理")):
        verb = "Manage"
    else:
        verb = "Represent"
    return verb + " " + normalized + " for " + scope


# Reviewed navigation-language rules derived from actual source responsibilities.
# These are display categories, not additional research or implementation nodes.
NAVIGATION_EN = {
    "激活捕获与特征读取": "Capture activations and inspect SAE features",
    "特征干预与生成": "Apply feature interventions during model execution",
    "电路追踪与归因": "Trace candidate circuits and feature attribution",
    "解释与假设检验": "Generate and test feature explanations",
    "指标、对照与评测": "Evaluate metrics, baselines, and controls",
    "训练与参数更新": "Train and update model parameters",
    "数据、产物与存储": "Manage datasets, artifacts, and storage",
    "约束、校验与配置": "Validate interfaces and experiment configuration",
    "运行入口与调度": "Dispatch research runs and execution backends",
    "公共辅助函数与接口": "Inspect shared source interfaces",
    "API 与调用接口": "Inspect public APIs and call boundaries",
    "SAE 字典、加载与覆盖率": "Load sparse dictionaries and verify coverage",
    "SAE 特征与实验产物存储": "Store sparse features and experiment artifacts",
    "LLM 模型与 SAE 任务": "Connect language models to SAE research tasks",
    "NLP 样本与 Prompt": "Prepare language inputs and prompt datasets",
    "SAE 评测与可靠性": "Benchmark SAE methods and assess reliability",
    "评测稳定性与统计检验": "Evaluate statistical reliability and controls",
    "SAR、RSS 与外部基准集成": "Integrate SAR, RSS, and external benchmarks",
    "任务执行后端": "Execute registered research tasks",
    "任务入口与公共接口": "Dispatch tasks through shared interfaces",
    "可扩展基准接口": "Register and invoke benchmark extensions",
    "外部基准适配": "Adapt external benchmarking methods",
    "评测指标计算": "Compute benchmarking and evaluation metrics",
    "评测结果发布与验证": "Validate and publish evaluation reports",
    "原生实现与基线": "Run native algorithms and baseline comparisons",
    "生成式 SAE 特征干预": "Steer sparse features during generation",
    "教师强制与特征干预": "Compare teacher-forced feature interventions",
    "SAE 特征捕获": "Capture sparse features from model activations",
    "电路追踪与干预": "Trace circuits and intervene on candidate paths",
    "SAE 自动解释": "Generate and assess feature explanations",
    "SAE 训练": "Train sparse autoencoder dictionaries",
    "管理科研任务的 Hydra 配置": "Compose research task configuration with Hydra",
    "配置科研任务的运行参数": "Set experiment and task execution parameters",
    "组织模型构建与兼容检查": "Build model adapters and check compatibility",
    "组织数据集加载与注册": "Load and register prompt datasets",
    "管理稀疏特征研究软件与运行接口": "Inspect sparse-feature APIs and runtime calls",
    "管理SAE 研究验证工具": "Validate SAE research methods and source evidence",
    "管理Research Lab 数据服务与证据核查": "Serve and verify Research Lab evidence",
    "管理SAE 基准任务规划与指标计算": "Plan SAE benchmarks and calculate evaluation metrics",
    "管理SAE 研究能力验证与执行": "Run SAE capability and integration checks",
    "管理实验产物的目录管理": "Manage reproducible experiment output directories",
    "管理训练配置与执行约束": "Validate training settings and execution constraints",
    "管理生成过程中的 SAE 干预": "Control SAE interventions during generation",
    "管理vLLM 文本生成计算后端": "Execute generation through the vLLM backend",
    "研究概念与依据": "Explore scientific concepts and evidence",
}

# Group summaries express the actual responsibility captured by the Chinese
# source, instead of reporting a count of nested implementation interfaces.
GROUP_SUMMARIES = {
    "code:root": "Browse dataset loaders, language-model adapters, seven SAE research task families, and shared research APIs.",
    "code:group:datasets": "Organize prompt datasets, source acquisition, and reproducible input artifacts.",
    "code:group:models": "Organize model loading, activation sites, and task compatibility checks.",
    "code:group:saebase": "Collect SAE representations, feature interventions, benchmarks, and storage interfaces.",
    "code:group:tasks": "Group the seven supported SAE research operations and their execution interfaces.",
    "code:concrete:datasets": "Locate prompt sources, dataset readers, and reproducible input artifact contracts.",
    "code:concrete:models": "Locate concrete model adapters and research-task compatibility implementations.",
    "code:concrete:tasks": "Locate capture, intervene, generate, circuits, explain, benchmark, and train implementations.",
}


def code_group_summary(node: dict, title: str, source_path: str) -> str:
    """Explain an existing source hierarchy using its scope and actual role."""
    node_id = node["id"]
    if node_id in GROUP_SUMMARIES:
        return GROUP_SUMMARIES[node_id]
    tokens = node_id.split(":")
    if len(tokens) >= 4 and tokens[1] == "instance":
        family, item = tokens[2], tokens[3]
        if family == "datasets":
            return (
                "Group "
                + DATASET_NAMES.get(item, human(item))
                + " prompt loading and preparation interfaces."
            )
        if family == "models":
            return "Group " + human(item) + " model construction and compatibility interfaces."
        if family == "tasks":
            return (
                "Group "
                + TASK_NAMES.get(item, human(item))
                + " configuration and execution interfaces."
            )
    if node.get("navigation_scope") == "source_range":
        return "Locate a contiguous source section of " + domain(source_path) + " interfaces."
    if node.get("title") in NAVIGATION_EN:
        return NAVIGATION_EN[node["title"]].rstrip(".") + " in " + domain(source_path) + "."
    if source_path:
        return (
            "Navigate " + domain(source_path) + " implementations represented by this source group."
        )
    return title.rstrip(".") + " across existing SAEBase components."


def first_source_path(node_id: str, children: dict, by_id: dict) -> str:
    """Ground a presentation-only group in the first real descendant file."""
    pending = [node_id]
    visited = set()
    while pending:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        node = by_id.get(current, {})
        path = node.get("path") or node.get("source_path") or ""
        if path.endswith(".py"):
            return path
        for evidence in node.get("evidence", []):
            source = evidence.split(":", 1)[0]
            if source.endswith(".py"):
                return source
        pending.extend(children.get(current, []))
    return ""


def group_label(node: dict, descendants: list[dict], children: dict, by_id: dict) -> str:
    """Write navigable action labels from source-backed groups, not opaque IDs."""
    node_id = node["id"]
    if node_id == "code:root":
        return "Explore SAEBase research implementations"
    named = {
        "code:group:datasets": "Browse prompt datasets and source preparation",
        "code:group:models": "Browse language-model adapters and compatibility",
        "code:group:saebase": "Browse SAEBase feature APIs, caches, and evaluation",
        "code:group:tasks": "Browse SAE research tasks and execution",
        "code:group:harness": "Browse research Harness and source verification",
        "code:concrete:datasets": "Explore individual research datasets",
        "code:concrete:models": "Explore individual language-model adapters",
        "code:concrete:tasks": "Explore seven SAE research operations",
    }
    if node_id in named:
        return named[node_id]
    tokens = node_id.split(":")
    if len(tokens) >= 4 and tokens[1] == "instance":
        family = tokens[2]
        label = human(tokens[3])
        if family == "datasets":
            return "Prepare and load " + DATASET_NAMES.get(tokens[3], label) + " prompts"
        if family == "models":
            return "Build and validate " + label + " language models"
        if family == "tasks":
            return "Execute " + TASK_NAMES.get(tokens[3], label) + " research operations"
    if len(tokens) >= 4 and tokens[1] == "configurations":
        task = tokens[3]
        return "Compose Hydra configuration for " + TASK_NAMES.get(task, human(task))
    if len(tokens) >= 5 and tokens[1] == "config":
        task = tokens[3]
        return "Set execution options for " + TASK_NAMES.get(task, human(task))
    source = first_source_path(node_id, children, by_id)
    scope = domain(source) if source else "SAEBase research operations"
    if node.get("navigation_scope") == "source_range":
        section = re.search(r"第(\d+)组", node.get("title", ""))
        number = section.group(1) if section else "1"
        return "Inspect " + scope + " interfaces, source section " + number
    label = NAVIGATION_EN.get(node.get("title", ""))
    if label:
        return label + " for " + scope
    if node.get("title", "").startswith("按源码职责浏览"):
        return "Inspect source-defined operations for " + scope
    if node.get("title", "").startswith("分组浏览"):
        return "Inspect related source interfaces for " + scope
    if descendants:
        first = next((item for item in descendants if item.get("path")), {})
        return "Browse " + domain(first.get("path") or source) + " source interfaces"
    return "Browse source-backed SAEBase component interfaces"


def make_code(node: dict, children: dict, by_id: dict) -> dict:
    kind = node.get("kind", "component")
    path = node.get("path") or node.get("source_path") or ""
    symbol = node.get("source_symbol") or ""
    stem = path.rsplit("/", 1)[-1].removesuffix(".py")
    name = symbol.rsplit(".", 1)[-1] if symbol else stem
    scope = domain(path)
    reviewed = REVIEWED_CODE.get(node["id"])
    doc = source_description(node)
    reviewed_status = "reviewed" if reviewed else ("source_docstring" if doc else "source_identity")
    if reviewed:
        title, summary = reviewed
    elif kind in {"function", "method"}:
        title = action_label(node, name, path)
        if doc and len(doc) <= 120 and doc.split(" ", 1)[0] in SOURCE_VERBS:
            title = doc.rstrip(".")
        summary = doc or title + "."
    elif kind == "class":
        if name.endswith(("Config", "Cfg", "Options", "Spec")):
            title = "Configure " + human(name) + " for " + scope
        elif name.endswith(("Result", "Record", "Manifest", "Snapshot", "State")):
            title = "Represent " + human(name) + " for " + scope
        else:
            title = class_action(node, name, scope)
        summary = doc or title + "."
    elif kind == "module":
        title = MODULE_ACTIONS.get(stem)
        if title is None:
            title = evidence_action(node, stem, path)
        else:
            title += " for " + scope
        if doc and len(doc) <= 120 and doc.split(" ", 1)[0] in SOURCE_VERBS:
            title = doc.rstrip(".")
        summary = doc or title + "."
    else:
        descendants = [by_id[c] for c in children.get(node["id"], []) if c in by_id]
        title = group_label(node, descendants, children, by_id)
        source_path = first_source_path(node["id"], children, by_id)
        summary = code_group_summary(node, title, source_path)
        reviewed_status = "source_hierarchy"

    interface = next(
        (
            item.get("interface")
            for item in node.get("attributes", [])
            if item.get("type") == "interface" and item.get("interface")
        ),
        {},
    )
    args = [
        p.get("name", "")
        for p in interface.get("parameters", [])
        if p.get("name") and not CJK.search(str(p["name"]))
    ]
    returns = interface.get("returns") or {}
    annotation = str(returns.get("annotation") or "")
    details = []
    if args:
        details.append("Declared inputs: " + ", ".join(args) + ".")
    if annotation and annotation != "None" and not CJK.search(annotation):
        details.append("Declared return type: " + annotation + ".")
    if path:
        details.append("Source: " + path + ".")
    description = summary.rstrip(".") + "."
    if details:
        description += " " + " ".join(details)
    return {
        "title": title,
        "summary": summary,
        "description": description,
        "reader_explanation": description,
        "translation_origin": reviewed_status,
    }


def disambiguate_code_titles(nodes: dict, sources: dict) -> None:
    """Prefer real subdomains for duplicate titles; never append raw symbols."""
    duplicates = defaultdict(list)
    for node_id, en in nodes.items():
        duplicates[en["title"]].append(node_id)
    for ids in duplicates.values():
        if len(ids) == 1:
            continue
        for node_id in ids:
            node = sources[node_id]
            if node.get("kind") == "group":
                # The navigation hierarchy already carries its source scope.
                # Repeated roles under separate parents can share a title.
                continue
            path = node.get("path") or node.get("source_path") or ""
            section = human(path.rsplit("/", 1)[-1].removesuffix(".py"))
            if node.get("kind") == "method":
                section = human(node_id.rsplit(":", 1)[-1].rsplit(".", 1)[0])
            if section and section != "component":
                nodes[node_id]["title"] += " in " + section
    # A source location is unique even where duplicate methods share a name.
    # Any remaining collisions are caught by stable IDs and the source panel.


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
            disambiguate_code_titles(nodes, by_id)
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
        "parameter_context": PARAMETER_CONTEXT,
        "meaning_translations": MEANING_TRANSLATIONS,
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

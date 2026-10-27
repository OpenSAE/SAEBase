"""Semantic Chinese Atlas headlines derived from actual Python source identities.

Headlines describe an operation and its research object. They deliberately do
not render source file paths, CamelCase names, or function signatures. Source
symbols and paths remain available in each node's structured attributes.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

# Curated descriptions apply only to observable source responsibilities.
# Even the generic fallback chooses an actual research object, not a filename.
BENCHMARK_FILES = {
    "activation_health.py": "检查 SAE 激活的数值稳定性",
    "analyze.py": "分析 SAE 基准评测的结果",
    "controlled_inputs.py": "构造受控输入与成对对照",
    "health_results.py": "汇总模型激活的健康检查",
    "stage_d.py": "核验模型与 SAE 的覆盖范围",
    "backend/adapters/localization.py": "接入特征局部性评测协议",
    "backend/adapters/papers.py": "复用论文基准的评测定义",
    "backend/adapters/saebench.py": "接入 SAEBench 评测器",
    "backend/metrics/core.py": "计算 SAE 基础指标",
    "backend/metrics/intervention.py": "测量 SAE 特征干预效果",
    "backend/metrics/reconstruction.py": "计算模型激活的重建损失",
    "backend/planner.py": "规划 SAE 评测任务",
    "backend/registry.py": "注册基准评测器与指标",
    "backend/report.py": "生成 SAE 评测报告",
    "backend/results.py": "保存 SAE 基准执行结果",
    "backend/runner.py": "运行 SAE 基准评测",
    "backend/schema.py": "定义基准任务与结果的数据协议",
    "extensions/contracts.py": "定义可扩展评测的输入协议",
    "extensions/registry.py": "注册外部评测扩展",
    "extensions/runtime.py": "加载外部 SAE 评测资源",
    "integrations/localization.py": "连接特征局部性实验",
    "integrations/rss.py": "评估 SAE 特征响应变化",
    "integrations/sar.py": "评估 SAE 特征组合效应",
    "integrations/sources.py": "追溯基准评测的数据来源",
    "native/benchmarks.py": "运行原生 SAE 对照基准",
    "native/custom_saebench.py": "执行自定义 SAE Bench 基准",
    "reliability/bootstrap.py": "估计评测分数的置信区间",
    "reliability/controls.py": "构建可靠性评测的对照组",
    "reliability/discriminability.py": "测量评测指标的区分能力",
    "reliability/ground_truth.py": "组织可靠性评测的参考答案",
    "reliability/ground_truth_report.py": "报告参考答案与实际评测的差异",
    "reliability/interaction_truth.py": "构建特征交互的参考结果",
    "reliability/localization.py": "评估特征局部性结果的稳定性",
    "reliability/observations.py": "汇总评测实验的观测记录",
    "reliability/ranking.py": "评估 SAE 排序的一致性",
    "reliability/report.py": "汇总可靠性检验结果",
    "reliability/resampling.py": "执行重采样稳定性检验",
    "reliability/rss_truth.py": "建立特征响应的参考结果",
    "reliability/runner.py": "运行可靠性评测流程",
    "reliability/schema.py": "定义可靠性评测数据结构",
}

DATASET_SUBJECTS = {
    "BBH": "BBH 推理题目",
    "CounterFact": "CounterFact 反事实知识样本",
    "GreaterThan": "数字大小比较样本",
    "IOI": "IOI 间接宾语任务样本",
    "MMLUPro": "MMLU-Pro 多学科问答样本",
    "SimpleCopying": "序列复制任务样本",
    "SubjectVerbAgreement": "主谓一致任务样本",
}

SPECIAL_SYMBOLS = {
    "native_worker": "启动原生 SAE 基准计算进程",
    "audit_distributions": "审查基准实验的数据分布",
    "audit_inventory": "审查评测样本与模型清单",
    "failure_record": "生成模型配对失败记录",
    "rebuild_canonical_status": "重建模型配对验收状态",
    "redact_secrets": "清理评测日志中的敏感信息",
    "relative_under": "解析实验产物的相对路径",
    "n_factors": "统计受控实验的因素数量",
    "from_renderer": "从输入渲染结果构造受控实验",
    "strategy_for_model_key": "选择模型对应的分词策略",
    "behaviors_for_model": "读取模型支持的数据任务",
    "model_forward_smoke": "检查语言模型的前向计算",
    "final_layer": "读取 SAE 适配的最后一层位置",
    "reference_layer": "读取 SAE 特征对照层位置",
    "cmd_probe_layer": "检测模型的可用激活层",
    "n_layers": "读取模型的 Transformer 层数",
    "forward_logits": "计算语言模型的输出分数",
    "allowed_model_keys": "列出 SAE 可适配的模型名称",
    "model_candidates": "列出模型的候选加载配置",
    "prompts_path": "定位 Prompt 数据产物",
    "prompt_file": "定位模型输入的 Prompt 文件",
    "analyze_prompt_activation": "分析 Prompt 引起的特征激活",
    "to_tokens": "将输入文本编码为 Token 序列",
    "tokens": "读取模型输入的 Token 序列",
    "from_prompts": "从 Prompt 集合构建受控实验",
    "prompt_group_id": "生成 Prompt 实验分组编号",
    "prompts_by_state": "按照实验条件组织 Prompt",
    "tokens_per_second": "统计模型生成的 Token 吞吐量",
    "estimator_source_sha256": "计算评估器源码的校验值",
    "source_identity": "核对基准评测的数据来源身份",
    "import_native": "载入原生基准实现",
    "loss_recovered": "计算 SAE 重建损失的恢复比例",
    "target_nll": "计算目标序列的负对数似然",
    "atomic_json": "原子化保存评测结果",
    "discover_extensions": "发现可用的评测扩展",
    "external_sae": "载入外部 SAE 评测资源",
    "external_pair_sources": "整理外部模型与 SAE 配对来源",
    "native_pilot": "执行原生 SAE 试点评测",
    "fingerprint": "生成实验产物的内容指纹",
    "sha256_state_dict": "计算模型权重的 SHA256 校验值",
    "stable_json": "生成规范化实验记录",
    "key": "生成实验记录的身份索引",
    "to_dict": "序列化实验记录字段",
    "from_dict": "还原实验记录字段",
    "packed_row": "打包生成请求与结果字段",
    "vllm_available": "检测 vLLM 生成服务的可用性",
    "release": "释放模型运行资源",
    "stop_vllm": "停止 vLLM 生成服务",
    "worker_cmd": "构造生成后端的工作命令",
    "cmd_dump_resid": "导出生成模型的残差激活",
    "cmd_generate": "执行模型生成请求",
    "cmd_smoke": "运行生成后端的基础检查",
    "autostart_enabled": "判断生成服务的自动启动条件",
    "post_completions": "提交并处理文本生成请求",
    "dry_run": "执行干预前的配置检查",
    "apply_condition": "应用 SAE 特征编辑条件",
    "remote_ndif_capture": "通过 NDIF 远程采集模型激活",
    "interventions_per_second": "估算每秒可执行的特征干预次数",
    "close": "关闭当前任务资源",
    "flush": "写入缓存中的运行结果",
    "committed": "确认运行结果的持久化状态",
    "step": "推进一次任务执行步骤",
    "default_data_dir": "定位数据集的默认存储目录",
    "default_source_dir": "定位原始数据的来源目录",
    "filtered_json_path": "定位筛选后的数据文件",
    "greater_than_json_path": "定位数字比较任务的数据文件",
    "ioi_json_path": "定位 IOI 样本文件",
    "copying_json_path": "定位序列复制任务的数据文件",
    "attach_token_ids": "为输入样本附加 Token 编码",
    "attach_ioi_token_ids": "为 IOI 样本附加 Token 编码",
    "attach_greater_than_token_ids": "为数字比较样本附加 Token 编码",
    "attach_copying_token_ids": "为序列复制样本附加 Token 编码",
    "attach_sva_token_ids": "为主谓一致样本附加 Token 编码",
    "attach_token_ids_first_token_yy": "标注年份首个 Token 的位置",
    "attach_token_ids_single_token_yy": "标注年份单 Token 编码",
    "attach_and_filter_argmax": "附加预测编码并筛选最大概率结果",
    "attach_and_filter_logit_diff": "筛选预测分数差异样本",
    "attach_distractors": "附加多选问答的干扰选项",
    "drop_na_and_remap": "清理缺失回答并重映射答案",
    "filter_argmax": "筛选最大概率预测样本",
    "filter_logit_diff": "按预测分数差异筛选样本",
    "filter_counterfact_with_gpt2": "用 GPT-2 筛选反事实知识样本",
    "filter_greater_than_with_gpt2": "用 GPT-2 筛选数字比较样本",
    "filter_copying_with_gpt2": "用 GPT-2 筛选序列复制样本",
}

GENERIC_CURATED = {
    "执行科研任务",
    "初始化组件",
    "执行任务",
    "读取数据",
    "管理源码对象",
    "运行任务入口",
    "处理任务接口",
}


def module_semantics(path: str) -> str:
    parts = PurePosixPath(path).parts
    if len(parts) >= 4 and parts[:2] == ("research_datasets", "NLP"):
        subject = DATASET_SUBJECTS.get(parts[2], "语言模型评测样本")
        filename = parts[-1].lower()
        if filename == "tokenize.py":
            return "为" + subject + "构建 Token 编码"
        if filename == "prepare.py":
            return "准备" + subject + "的原始输入"
        if filename.lower() == (parts[2].lower() + ".py"):
            return "读取" + subject + "与目标标签"
    if path.startswith("tasks/LLM/benchmark/"):
        relative = path[len("tasks/LLM/benchmark/") :]
        if relative in BENCHMARK_FILES:
            return BENCHMARK_FILES[relative]
    filename = PurePosixPath(path).name
    simple = {
        "registry.py": "管理" + context_for_path(path) + "的注册信息",
        "report.py": "汇总" + context_for_path(path) + "的结果报告",
        "results.py": "保存" + context_for_path(path) + "的实验结果",
        "runner.py": "执行" + context_for_path(path) + "的任务流程",
        "operation.py": "执行" + context_for_path(path) + "任务",
        "task.py": "组织" + context_for_path(path) + "的任务入口",
        "schema.py": "定义" + context_for_path(path) + "的数据协议",
        "contracts.py": "校验" + context_for_path(path) + "的接口约束",
        "settings.py": "配置" + context_for_path(path) + "的执行参数",
        "engine.py": "执行" + context_for_path(path) + "的核心计算",
        "loading.py": "加载" + context_for_path(path) + "的模型与参数",
        "identity.py": "记录" + context_for_path(path) + "的身份信息",
        "shards.py": "保存" + context_for_path(path) + "的分片数据",
        "collect.py": "采集" + context_for_path(path) + "的输入记录",
        "api.py": "提供" + context_for_path(path) + "的调用接口",
        "utils.py": "整理" + context_for_path(path) + "的公共工具",
        "pipeline_checks.py": "检查" + context_for_path(path) + "的执行条件",
        "launch.py": "启动" + context_for_path(path) + "的执行服务",
        "server.py": "提供" + context_for_path(path) + "的服务端接口",
    }
    return simple.get(filename, "实现" + context_for_path(path))


# Long compound objects take precedence over individual identifier tokens.
OBJECTS = (
    (r"token.*ids?|attach.*token|first.*token", "输入文本的 Token 编码"),
    (r"source.*sha|sha256|checksum|fingerprint", "实验产物的完整性校验值"),
    (r"source.*identity|pair.*source", "模型与数据的来源身份"),
    (r"controlled.*input|binary.*factor|factor.*condition", "受控输入的实验条件"),
    (r"localiz|feature.*location", "SAE 特征局部性"),
    (r"extension|adapter|external", "外部基准与扩展"),
    (r"nll|negative.*log.*likelihood", "目标序列的负对数似然"),
    (r"loss.*recover|reconstruction.*loss", "SAE 重建损失"),
    (r"rank|ordering|spearman", "SAE 评测排名"),
    (r"stable.*json|to.*dict|from.*dict|serialize", "实验记录的字段结构"),
    (r"vllm|worker.*cmd|completions", "模型生成服务"),
    (r"argmax|logit.*diff|filter|distractor", "模型预测与数据筛选"),
    (r"input.*capture|state.*capture", "受控实验的激活记录"),
    (r"plan.*node|benchmark.*target", "基准执行目标"),
    (r"hydra.*(cfg|config)|config.*hydra", "Hydra 训练配置"),
    (r"checkpoint|check.point", "模型检查点"),
    (r"train.*(val|test).*split|split.*(train|test)|data.*split", "训练与评测数据划分"),
    (r"task.*dataset|dataset.*task", "数据集与任务兼容关系"),
    (r"feature.*activ|activ.*feature|capture.*feature", "SAE 特征激活"),
    (r"sae.*(identity|profile)|model.*sae.*pair", "模型与 SAE 身份"),
    (r"module.*(hook|site)|activation.*site|trace.*location", "模型激活位置"),
    (r"generation.*edit|generate.*edit|steering", "生成过程中的特征干预"),
    (r"teacher.*forc|fixed.*input", "教师强制干预"),
    (r"residual.*(stream|contribution)", "残差流贡献"),
    (r"model.*(registry|plugin)|register.*model", "模型注册信息"),
    (r"dataset.*(registry|plugin)|register.*dataset", "数据集注册信息"),
    (r"task.*(registry|plugin)|register.*task", "任务注册信息"),
    (r"(auroc|auprc|f1|accuracy|top.?[15])", "分类与排序评测指标"),
    (r"(metric|measure|score|result|report)", "评测指标与结果"),
    (r"(reliability|bootstrap|resampl|confidence|variance)", "评测可靠性"),
    (r"(circuit|pathway|attribution)", "候选因果电路"),
    (r"(explan|interpret|hypothes)", "SAE 特征解释"),
    (r"(sparsify|sparse.autoencoder|sae.*train)", "稀疏自编码器训练"),
    (r"(feature|code|latent)", "稀疏特征表示"),
    (r"(prompt|tokens?|vocab|text)", "模型输入文本"),
    (r"(tensor|activation|gradient)", "模型张量与梯度"),
    (r"(device|cuda|gpu|memory)", "设备与显存配置"),
    (r"(coverage|accepted|acceptance|verification)", "模型与 SAE 覆盖验收"),
    (r"(dataset|datamodule|sample|loader)", "数据样本与加载流程"),
    (r"(training|train|optim|scheduler|learning.rate|epoch)", "模型训练参数"),
    (r"(checkpoint|artifact|shard|storage|manifest)", "实验产物与存储"),
    (r"(schema|contract|policy|permission|guard)", "接口约束与数据协议"),
    (r"(config|setting|param)", "任务运行配置"),
    (r"(model|forward|backward|layer)", "模型执行过程"),
    (r"(registry|plugin)", "组件注册表"),
    (r"(runner|pipeline|runtime|operation)", "任务执行流程"),
    (r"(file|path|dir|cache)", "实验文件与缓存"),
)

OPERATIONS = (
    (r"^(attach|annotate|label|tag)_?", "标注"),
    (r"^(filter|select|drop|exclude|keep)_?", "筛选"),
    (r"^(import|discover|locate|search)_?", "查找"),
    (r"^(fingerprint|hash|sha256)_?", "计算"),
    (r"^(release|close|stop|shutdown)_?", "关闭"),
    (r"^(flush|commit|publish|sync)_?", "保存"),
    (r"^(copy|clone|replicate)_?", "复制"),
    (r"^(apply|condition)_?", "应用"),
    (r"^(merge|join|combine|aggregate|reduce)_?", "汇总"),
    (r"^(construct|compose|pack)_?", "构建"),
    (r"^(post|submit|send)_?", "提交"),
    (r"^(default|path|source)_?", "定位"),
    (r"^(validate|verify|assert|check|guard|ensure|require|test)_?", "校验"),
    (r"^(evaluate|score|benchmark|assess|compare|measure)_?", "评估"),
    (r"^(collect|capture|observe|trace|record|sample)_?", "采集"),
    (r"^(decode|reconstruct)_?", "重建"),
    (r"^(encode|tokenize)_?", "编码"),
    (r"^(edit|intervene|patch|steer|ablate|replace|scale|zero)_?", "干预"),
    (r"^(train|fit|optimize|update)_?", "训练"),
    (r"^(calculate|compute|count|estimate)_?", "计算"),
    (r"^(infer|predict|classify)_?", "推断"),
    (r"^(generate|sample|propose)_?", "生成"),
    (r"^(load|read|fetch|get|retrieve|lookup|find)_?", "读取"),
    (r"^(save|write|export|persist|dump|store)_?", "保存"),
    (r"^(build|make|create|construct|prepare|assemble|setup|initialize|init)_?", "构建"),
    (r"^(register|add|insert|append)_?", "注册"),
    (r"^(resolve|parse|normalize|convert|transform|map)_?", "解析"),
    (r"^(format|describe|summarize|render|display|print|report)_?", "整理"),
    (r"^(schedule|plan|dispatch|route)_?", "调度"),
    (r"^(run|execute|launch|start|main|call|invoke)_?", "执行"),
    (r"^(clear|delete|remove|reset|cleanup|prune)_?", "清理"),
    (r"^(list|enumerate|iter|walk)_?", "列出"),
    (r"^(is|has|can|should|supports|enabled)_?", "判断"),
)

# Actual module locations determine research context, never the displayed
# dotted path or code symbol. Specific mappings precede general families.
SCOPES = (
    ("saebase/api/features.py", "SAE 特征捕获与受控编辑"),
    ("saebase/api/model.py", "模型激活读取与前向预测"),
    ("saebase/api/components.py", "Transformer 组件贡献分析"),
    ("saebase/api/nnsight_local.py", "模型内部激活与梯度追踪"),
    ("saebase/sae/coverage.py", "SAE 模型与位置覆盖验收"),
    ("saebase/sae/loading.py", "SAE 权重加载与实例化"),
    ("saebase/sae/registry.py", "SAE 模型注册与身份"),
    ("saebase/store/", "SAE 特征数据存储"),
    ("saebase/sae/", "稀疏自编码器配置与加载"),
    ("saebase/api/", "SAE 特征捕获与模型干预"),
    ("tasks/LLM/benchmark/reliability/", "SAE 评测可靠性"),
    ("tasks/LLM/benchmark/integrations/", "SAR 与 RSS 评测适配"),
    ("tasks/LLM/benchmark/", "SAE 基准评测"),
    ("tasks/LLM/intervene/", "教师强制特征干预"),
    ("tasks/LLM/generate/", "生成过程中的 SAE 干预"),
    ("tasks/LLM/circuits/", "候选因果电路分析"),
    ("tasks/LLM/explain/", "SAE 特征解释"),
    ("tasks/LLM/capture/", "SAE 特征捕获"),
    ("tasks/LLM/train/", "稀疏自编码器训练"),
    ("tasks/LLM/", "语言模型任务执行"),
    ("models/LLM/", "语言模型加载与运行"),
    ("datasets/NLP/", "模型文本输入数据"),
    ("dlbase/cfg/", "训练配置与执行约束"),
    ("dlbase/training/", "训练回调与优化调度"),
    ("dlbase/registry.py", "数据、模型与任务注册"),
    ("dlbase/experiment_paths.py", "实验产物的目录管理"),
    ("dlbase/", "模型训练与实验记录"),
    ("research_datasets/", "数据集加载与注册"),
    ("models/", "模型构建与兼容检查"),
    ("tasks/", "科研任务执行与评测"),
    ("harness/", "科研代码与项目结构校验"),
)


def context_for_path(path: str) -> str:
    value = str(path or "")
    for prefix, label in SCOPES:
        if value.startswith(prefix):
            return label
    return "科研组件与执行接口"


TECHNICAL = ("SAE", "LLM", "Hydra", "GPU", "CPU", "SAR", "RSS", "NLP", "API")


def _words(identifier: str) -> str:
    value = str(identifier or "").split(".")[-1]
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"_+", "_", value.strip("_")).lower()


def object_for(symbol: str, path: str) -> str:
    words = _words(symbol)
    for pattern, label in OBJECTS:
        if re.search(pattern, words.replace("_", " ")) or re.search(pattern, words):
            return label
    return context_for_path(path)


def operation_for(symbol: str) -> str:
    words = _words(symbol)
    for pattern, verb in OPERATIONS:
        if re.search(pattern, words):
            return verb
    return "处理"


def _clean_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" 。·-")
    value = re.sub(r"\b(?:WP\d+|stage[_ -]?[A-D])\b", "", value, flags=re.I)
    return value


def semantic_title(kind: str, symbol: str, path: str, curated: str = "", summary: str = "") -> str:
    """One Chinese operation/concept, not a function-name/path inventory."""
    if symbol in {"to_dict", "from_dict", "key"} and kind != "module":
        action = {
            "to_dict": "序列化",
            "from_dict": "还原",
            "key": "生成",
        }[symbol]
        subject = (
            "SAE 模型身份记录"
            if "saebase/store/identity.py" in path
            else "SAE 元数据"
            if "saebase/api/metadata.py" in path
            else "受控输入的实验参数"
            if "controlled_inputs.py" in path
            else "基准评测计划"
            if "/planner.py" in path
            else "基准结果的数据字段"
            if "/schema.py" in path
            else "实验记录"
        )
        return _clean_title(action + subject)
    if symbol in SPECIAL_SYMBOLS and kind != "module":
        return _clean_title(SPECIAL_SYMBOLS[symbol])
    if (
        curated
        and curated not in GENERIC_CURATED
        and re.search(r"[\u3400-\u9fff]", curated)
        and " · " not in curated
    ):
        return _clean_title(curated)
    context = context_for_path(path)
    if kind == "root":
        return "浏览已实现的科研功能与代码结构"
    if kind == "group":
        return _clean_title(curated) if curated else "管理科研任务与组件"
    if kind == "module":
        return _clean_title(module_semantics(path))
    if kind == "class":
        noun = object_for(symbol, path)
        if noun == context:
            noun = {
                "BatchEngine": "批量干预执行流程",
                "ExecutionStats": "干预任务的性能统计",
                "RunBudget": "模型干预的计算预算",
                "CleanKVStep": "KV 缓存逐步校验",
                "KVParityError": "KV 缓存一致性异常",
                "Condition": "特征编辑实验条件",
                "ExperimentIdentity": "干预实验的身份记录",
                "GenerationSpec": "生成干预的参数协议",
                "Edit": "一次特征编辑操作",
                "EditPlan": "多步骤特征干预计划",
                "PlanNode": "基准任务执行步骤",
                "BenchmarkTarget": "SAE 基准评测目标",
                "BinaryFactor": "受控输入的二元实验条件",
                "ControlledInputCapture": "受控输入的模型激活",
                "ControlledInputDesign": "受控输入的实验设计",
                "ControlledInputState": "受控输入的执行状态",
                "ExtensionBundle": "可扩展评测组件",
                "ExternalObservation": "外部评测观测记录",
                "ExternalSAE": "外部 SAE 身份与适配",
            }.get(symbol, noun)
        return _clean_title("管理" + noun)
    action = operation_for(symbol)
    subject = object_for(symbol, path)
    if subject == context and action == "处理":
        return _clean_title("处理" + context)
    if action == "读取" and subject == "科研组件与执行接口":
        subject = context
    separator = " " if subject and subject[0].isascii() and subject[0].isalpha() else ""
    return _clean_title(action + separator + subject)


def semantic_summary(title: str, kind: str, path: str, existing: str = "") -> str:
    if (
        existing
        and re.search(r"[\u3400-\u9fff]", existing)
        and not re.search(r"(?:源文件 |实际定义了 |当前源码中已定义)", existing)
    ):
        return existing
    return f"负责{title}；源码依据和调用参数可在节点的展开说明中查看。"


def title_is_semantic(title: str) -> bool:
    return (
        bool(re.search(r"[\u3400-\u9fff]", title))
        and " · " not in title
        and "/" not in title
        and "\\" not in title
        and not re.search(r"(?:^|\s)(?:def|class)\s|\.py\b|WP\d+", title, re.I)
    )

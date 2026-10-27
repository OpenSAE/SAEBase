"""Readable presentation of existing Python AST symbols.

Only the canonical AST snapshot defines what actually exists. This layer
filters routine boilerplate and supplies conservative, human-readable labels;
it never creates synthetic code functions or alters stable symbol IDs.
"""
from __future__ import annotations
import copy
import re
from hashlib import sha256
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath
from harness.code.semantic_titles import (
    semantic_title, semantic_summary, context_for_path,
)

GROUPS = {
    "dlbase":"DLBase 训练配置、运行与实验产物",
    "saebase":"SAEBase 特征表示与模型干预",
    "research_datasets":"Dataset 读取、划分与注册",
    "models":"Model 构建、插件与兼容性",
    "tasks":"Vision 分类与 LLM SAE 任务",
    "harness":"Harness 代码树、环境与知识验证",
}

GROUP_SUMMARIES = {
    "dlbase": "为多种科研任务提供公共训练入口、实验记录、优化调度和组件注册工具。",
    "saebase": "为 LLM 稀疏特征捕获、干预、生成、电路、解释、基准与训练提供公共算法和可复用接口。",
    "research_datasets": "把数据样本和标签组织为训练、验证、测试批次，并提供统一的数据集注册入口。",
    "models": "创建神经网络模型，检查模型与输入、任务的兼容关系，并提供统一构建入口。",
    "tasks": "按通用任务类型组织训练和评测流程，具体模型与数据集由实验配置选择。",
    "harness": "负责检查代码结构、记录科研知识、管理运行环境和生成 Atlas 浏览页面。",
}

MODULES = {
    "datasets/vision/CUB_200/CUB_200.py":("鸟类图像数据集","读取现有 CUB 鸟类图像、标签和训练测试划分。"),
    "datasets/vision/CUB_200/datamodule_setup.py":("鸟类数据加载流程","组装训练、验证和测试数据，设置当前图像预处理。"),
    "datasets/vision/CUB_200/dataset_task_policy.py":("数据集任务适配检查","检查分类任务是否属于该数据集已支持的用法。"),
    "datasets/datamodule.py":("统一数据加载接口","按数据集配置构造训练、验证与测试批次。"),
    "datasets/plugins.py":("数据组件注册","把已实现的数据集接入公共组件注册表。"),
    "dlbase/cfg/config_validate.py":("训练配置校验","训练前检查必要配置和已注册的组件名称。"),
    "dlbase/cfg/contracts.py":("配置约束工具","集中实现数据集与任务名称的配置断言。"),
    "dlbase/experiment_paths.py":("实验结果目录管理","构建分层结果路径并保护历史运行文件。"),
    "dlbase/plugin_bootstrap.py":("组件加载","为数据、模型与任务加载已有注册入口。"),
    "dlbase/registry.py":("组件注册中心","管理数据集、模型与任务的注册、查询和接口约束。"),
    "dlbase/runtime_check.py":("运行依赖检查","在训练启动前检查 Python 与必要的训练依赖。"),
    "dlbase/summary_log.py":("实验结果汇总","生成和校验任务独立的评测表与实验摘要。"),
    "dlbase/train.py":("统一训练入口","校验 Hydra 组合配置后运行已注册的科研任务。"),
    "dlbase/training/facets.py":("训练功能开关","根据训练配置启用或关闭已经存在的训练组件。"),
    "dlbase/training/schedulers.py":("学习率调度工具","生成训练使用的学习率预热和余弦变化方案。"),
    "dlbase/training/seed.py":("随机性与复现配置","统一设置训练种子和确定性执行条件。"),
    "dlbase/training/standard_lightning.py":("标准训练流程","创建训练器、回调、日志与模型检查点。"),
    "dlbase/training/training_callbacks.py":("训练进度反馈","显示训练进度、估算剩余时间并记录验证趋势。"),
    "models/build.py":("模型实例构建","结合当前配置检查兼容性并创建模型实例。"),
    "models/plugins.py":("模型注册","把已实现的模型接入统一构建入口。"),
    "models/vision/resnet18_mlp/model_task_dataset_compatibility.py":("模型任务兼容性","校验模型是否适用于指定的任务与数据集。"),
    "models/vision/resnet18_mlp/resnet18_mlp.py":("残差网络分类模型","实现 ResNet 主干及可配置线性或多层分类头。"),
    "tasks/vision/classification/classification_module.py":("分类训练模块","封装分类预测、损失计算、优化器和训练步骤。"),
    "tasks/vision/classification/classification_task.py":("分类任务接口","将通用科研任务接口接到已有分类训练流程。"),
    "tasks/vision/classification/pipeline_checks.py":("分类运行前检查","在开始训练前核验组件兼容性与分类输出形状。"),
    "tasks/vision/classification/post_train_eval.py":("训练后分类评测","计算分类结果并准备可汇总的指标记录。"),
    "tasks/vision/classification/run.py":("分类训练执行流程","组织训练、保存检查点和训练后评测。"),
    "tasks/plugins.py":("任务注册","将已实现的任务接入统一任务分发。"),
    "tasks/runner.py":("任务分发入口","按照任务名称启动当前已注册的训练任务。"),
}
# Source-backed summaries for the newly integrated SAEBase and LLM tasks.
MODULES.update({
    "saebase/api/features.py": ("SAE 特征捕获和编辑接口", "捕获模型激活、编码 SAE 特征并进行可验证的干预。"),
    "saebase/api/model.py": ("语言模型接口", "统一模型的 token、激活读取和干预后输出接口。"),
    "saebase/api/components.py": ("Transformer 组件贡献", "区分真实残差差值、Attention Head 与 MLP Neuron 投影贡献。"),
    "datasets/NLP/artifacts.py": ("NLP Prompt 产物协议", "统一真实数据源、模型可用性、JSONL、manifest 和路径。"),
    "tasks/LLM/benchmark/controlled_inputs.py": ("受控输入评测", "构造二元输入实验并采集有来源的模型激活。"),
    "tasks/LLM/intervene/backend/steering.py": ("SAE 生成干预", "逐 token 比较基线生成和实际 SAE 特征改写后的生成。"),
    "tasks/LLM/circuits/backend/circuits.py": ("电路分析", "导入有来源的因果电路并调用实际原生电路追踪工具。"),
    "tasks/LLM/explain/interpretation.py": ("自动特征解释", "使用真实生成器产生特征解释假设并在独立样本上检验。"),
    "tasks/LLM/benchmark/backend/runner.py": ("SAE 基准评测", "对指定 SAE、指标和数据样本运行真实的基准评测。"),
    "tasks/LLM/benchmark/stage_d.py": ("SAE 覆盖率验证与模型配对验收", "通过真实 SAE 覆盖记录、模型配对与运行状态组织计算验收。"),
    "tasks/LLM/train/backend/training.py": ("SAE 训练", "用原生 SAE Lens 或 Sparsify 训练器执行受约束的训练。"),
    "tasks/LLM/sae_runtime.py": ("七个 LLM 任务的公共入口", "统一校验模型和数据，并将执行分发到各任务自己的 operation.py。"),
    "datasets/NLP/prompts.py": ("NLP 数据集输入", "从选定数据集的实际提示样本文件读取文本。"),
})
for _task_name, _task_label in {
    "capture": "特征捕获", "intervene": "教师强制干预",
    "generate": "生成干预", "circuits": "电路分析",
    "explain": "自动解释", "benchmark": "基准评测", "train": "SAE 训练",
}.items():
    MODULES[f"tasks/LLM/{_task_name}/task.py"] = (
        _task_label, "该通用任务经任务注册表调用 SAEBase 的实际执行接口。"
    )

CLASSES = {
    "CUB200":("鸟类图像数据集","读取 CUB 图像、标签及对应标注。"),
    "DLBaseDataModule":("训练数据管理器","按当前数据配置构建训练、验证与测试数据加载器。"),
    "DatasetPlugin":("数据集插件接口","记录数据集创建方式与任务支持检查。"),
    "ModelPlugin":("模型插件接口","描述模型构建与组件兼容性检查。"),
    "TaskModule":("科研任务抽象接口","定义任务需要实现的名称和执行入口。"),
    "TaskPlugin":("任务插件接口","把已实现的任务接到统一注册机制。"),
    "CompactTQDMProgressBar":("简洁训练进度条","显示训练批次和当前指标摘要。"),
    "EpochEtaHoursCallback":("剩余训练时间估算","根据轮次完成情况估算预计耗时。"),
    "ValLossHistoryAndPhaseLrCallback":("验证损失与学习率记录","记录验证损失及训练阶段结束时的学习率。"),
    "Identity":("恒等特征映射","将输入特征原样传递至输出。"),
    "MLPModuleResnet":("多层分类头","通过多个全连接层把视觉特征映射为类别分数。"),
    "Resnet18MLP":("残差网络分类器","连接 ResNet 主干与当前选定的分类头。"),
    "ClassificationModule":("分类训练与优化","封装模型前向预测、损失、优化和评测步骤。"),
    "ClassificationTask":("分类任务执行接口","通过统一任务协议调用分类训练流程。"),
    "MultiLabelSplitMetrics":("多标签评测记录","保存多标签评测的分组指标。"),
    "SplitMetrics":("分类评测记录","保存指定数据划分的分类指标。"),
}
FUNCTIONS = {
    "__getattr__":("按需获取组件","在请求模块属性时按现有规则返回相应对象。"),
    "_component":("验证目录名称","防止不合法的路径组件进入实验结果目录。"),
    "_cub_transforms":("构建鸟类图像变换","创建当前分类任务使用的图像裁剪与预处理。"),
    "_default_metric_text":("格式化缺省指标","为缺少值的评测指标提供规范的显示文本。"),
    "_dlbase_repo_root":("定位项目根目录","取得训练模块所属仓库的根目录位置。"),
    "_ensure_plugins":("确认组件已加载","在获取注册组件之前触发已有插件加载流程。"),
    "_evaluate_multi_label":("多标签评测","计算当前多标签分类任务的预测结果指标。"),
    "_evaluate_single_label":("单标签评测","计算单标签分类任务的预测结果指标。"),
    "_find_free_port":("查找可用网络端口","为本地训练可视化服务寻找可用端口。"),
    "_fmt_pb_float":("整理进度条数值","缩短训练进度条中的浮点数字显示。"),
    "_format_row":("格式化结果表行","把指标值按既定列结构写成一行。"),
    "_get_output":("读取模型预测结果","从模型返回值中提取当前分类所需输出。"),
    "_map_bbox_to_input_space":("映射图像边界框","把原图标注转换到模型输入图像的坐标系。"),
    "_map_point_to_input_space":("映射图像关键点","把图像关键点映射到预处理后的输入坐标系。"),
    "_multi_label_f1_micro":("计算多标签综合分数","汇总多标签预测的微平均分类结果。"),
    "_normalize_metric_keys_per_split":("统一评测字段名称","整理不同数据划分下的指标字段。"),
    "_register_builtin_datasets":("登记内置数据集","把已经存在的数据集加入注册表。"),
    "_register_builtin_models":("登记内置模型","把已经存在的模型加入注册表。"),
    "_register_builtin_tasks":("登记内置任务","把已有任务接入统一任务注册表。"),
    "_single_label_acc":("计算单标签准确率","统计单标签分类预测正确的样本比例。"),
    "_tensorboard_argv":("构建训练曲线启动参数","准备启动本地训练曲线服务的命令参数。"),
    "_validate_split_metrics_payload":("校验评测数据结构","检查各数据划分的指标结构是否符合约定。"),
    "_wait_tensorboard_http":("检查训练曲线服务就绪","等待本地训练曲线页面响应。"),
    "apply_lightning_reproducibility":("应用复现设置","设置训练随机种子和执行确定性选项。"),
    "artifact_path":("定位实验结果文件","依据产物类型返回该实验的具体保存位置。"),
    "assert_classification_task":("检查分类任务约束","确认任务配置与分类运行接口相符。"),
    "assert_compatible":("检查模型组合兼容性","确认指定模型能用于该数据集与任务组合。"),
    "assert_dataset_name_in":("核验数据集名称","按配置约束检查数据集名称。"),
    "assert_task_in":("核验任务类型","按配置约束检查当前任务类型。"),
    "assert_task_supported":("核验任务是否受支持","确保数据集已声明支持当前任务。"),
    "build_eval_blocks_with_adapter":("组装评测报告","将任务侧指标转换为公共报告中的评测表。"),
    "build_lightning_loggers":("创建训练日志记录器","连接训练指标与可视化日志存储。"),
    "build_model":("构建兼容模型","先检查组合可用性，再创建实际模型。"),
    "build_model_checkpoint_callbacks":("创建模型保存回调","按验证指标设置最佳模型权重保存。"),
    "build_warmup_cosine":("创建学习率变化方案","构建预热和余弦衰减的学习率调度。"),
    "check_run_location":("检查实验输出位置","核验当前工作路径与 Hydra 约定位置一致。"),
    "classification_eval_splits":("组织分类评测数据划分","返回分类报告使用的训练、验证或测试指标记录。"),
    "classification_eval_train_test_only":("整理训练和测试评测","将同源的验证与测试指标汇总成统一表项。"),
    "classification_logits_shape_dry_run":("预检查分类输出形状","通过最小前向运行核验类别维度。"),
    "clear_experiment_leaf_dir":("清理单次运行目录","清理已明确允许覆盖的运行叶子中的临时产物。"),
    "construct_model":("创建模型权重实例","按模型配置初始化具体网络。"),
    "count_module_parameters":("统计模型参数量","统计评测报告需要的模型规模。"),
    "create_analysis_group":("创建分析结果分组","为已有实验创建带解释和产物分类的子目录。"),
    "create_layout":("初始化结果目录结构","建立模型权重、日志、表格和报告的标准目录。"),
    "cub_eval_split_note":("说明鸟类评测数据划分","输出当前鸟类数据集验证和测试来源的说明。"),
    "describe_classification_input_for_summary":("描述分类模型输入","为实验报告整理输入张量的形状。"),
    "early_stopping_patience_epochs":("计算提前停止等待轮次","根据训练总轮次确定等待验证改善的时间。"),
    "ensure_plugins_loaded":("加载已有组件注册","确认需要的内置插件已经注册。"),
    "env_suppresses_tensorboard_auto_launch":("判断是否自动打开曲线页面","根据当前运行环境决定是否启动浏览器。"),
    "evaluate_classification_loader":("运行分类数据评测","遍历数据加载器并计算分类指标。"),
    "format_split_metrics_table":("生成分类指标表","按照数据划分整理各项分类指标。"),
    "get_dataset_plugin":("获取数据集插件","根据名称读取已注册的数据组件。"),
    "get_model_plugin":("获取模型插件","根据名称读取已注册的模型组件。"),
    "get_task_plugin":("获取任务插件","根据名称读取已注册的任务组件。"),
    "guard_existing_run":("防止覆盖既有实验","在运行目录已有产物时检查是否允许写入。"),
    "lightning_trainer_kwargs":("整理训练器配置","生成公共训练器所需参数。"),
    "list_registered_dataset_keys":("查看可用数据集","列出现有注册的数据集名称。"),
    "list_registered_model_keys":("查看可用模型","列出现有注册的模型名称。"),
    "list_registered_task_keys":("查看可用任务","列出现有注册的任务名称。"),
    "main":("运行训练主入口","按已经组合的配置启动训练任务。"),
    "make_warmup_cosine_lambda":("创建学习率调度函数","生成预热和余弦衰减对应的步进系数。"),
    "maybe_launch_tensorboard_browser":("按需打开训练曲线","在允许的情况下启动训练指标网页。"),
    "merge_classification_task_facets":("组合分类训练开关","合并分类任务专属的运行功能配置。"),
    "merge_dynamic_segmentation_task_facets":("组合分割训练开关","对已有分割任务选项进行配置合并。"),
    "merge_train_facets":("组合通用训练开关","从配置中整理可选的训练功能。"),
    "missing_packages":("核验依赖安装情况","汇总训练环境需要补充安装的依赖项。"),
    "prepare_run":("准备实验运行目录","创建可追溯的实验目录与元信息。"),
    "print_python_identity":("显示 Python 环境信息","汇报解释器及运行环境的基本信息。"),
    "register_dataset":("注册数据集","把数据集及其创建入口写入注册表。"),
    "register_dataset_alias":("添加数据集别名","为已注册的数据集添加备用名称。"),
    "register_model":("注册模型","把模型及其创建入口写入注册表。"),
    "register_task":("注册任务","把任务及其执行入口写入注册表。"),
    "resolve_best_checkpoint_path":("定位最佳模型权重","找到当前实验记录中的最佳模型检查点。"),
    "resolve_dataset_name":("确定数据集正式名称","把输入的数据集别名解析为已注册名称。"),
    "run_classification_training":("执行分类训练","组织模型训练、指标记录和权重保存。"),
    "run_final_pre_training_checks":("运行训练前组合校验","检查数据、模型与分类输出的完整性。"),
    "run_leaf":("定位单次实验目录","由数据、模型、任务和种子得到运行叶子目录。"),
    "run_task":("分发已注册任务","根据配置名称调用已有科研任务入口。"),
    "setup_cub_datamodule":("配置鸟类数据加载器","为鸟类分类实例装配数据划分与变换。"),
    "task_components":("解析任务目录层级","从任务配置读取合法的子任务路径。"),
    "task_root":("定位任务实验根目录","由数据集、模型与任务得到实验目录位置。"),
    "task_tmp_dir":("创建任务临时目录","把临时产物放在对应任务之下。"),
    "tqdm_epoch_progress_bar":("创建训练进度反馈","显示轮次、当前损失和训练指标。"),
    "val_still_strictly_decreasing":("判断验证损失趋势","检查最近几个验证轮次的损失是否持续下降。"),
    "validate_hydra_cfg":("校验 Hydra 训练配置与注册组件","检查训练配置是否可交给注册表执行。"),
    "verify_training_runtime":("核验训练环境","检查训练启动所需的运行时依赖。"),
    "write_summary_log":("写入实验报告","把评测摘要保存为可阅读的运行报告。"),
    "write_tensorboard_jump_page":("生成训练曲线入口页面","为当前运行写入本地训练曲线链接。"),
}
METHODS = {
    "__init__":("初始化组件","创建组件并设置当前配置与内部状态。"),
    "__getitem__":("读取一条数据样本","按索引取得图像及相应的标注。"),
    "__len__":("获取数据集样本数","返回当前数据集中可读取的样本数量。"),
    "_get_attributes":("读取鸟类属性定义","加载鸟类的属性名称与对应编码。"),
    "_get_id_to_bbox":("读取图像边界框","获取图像到边界框标注的映射。"),
    "_get_id_to_label":("读取图像类别标签","获取图像到鸟类类别的映射。"),
    "_get_id_to_parts":("读取图像局部部位","加载鸟类各部位关键点的标注映射。"),
    "_get_image_attributes":("读取图像属性","加载样本对应的鸟类属性标注。"),
    "_get_path_label":("读取图像路径和标签","对应图像文件位置与类别。"),
    "_sgd_param_groups":("配置不同层的优化参数","按骨干网络和分类头组织优化器参数。"),
    "_step":("执行一次分类计算","计算当前批次预测与损失。"),
    "_train_test_split":("读取训练和测试划分","根据官方标识区分训练与测试样本。"),
    "as_dict":("转换为结构化指标","把评测记录转成便于保存和汇总的数据结构。"),
    "configure_optimizers":("配置优化器与学习率","创建当前任务所使用的参数更新与调度方案。"),
    "eval_split_note":("解释评测数据来源","输出数据划分与评测限制的简要说明。"),
    "forward":("计算网络前向输出","把输入特征交给当前模型并产生输出。"),
    "get_bbox_by_local_index":("读取输入空间的边界框","按样本索引获得预处理图像上的边界框。"),
    "get_colored_parts_by_local_index":("读取彩色部位关键点","提取可见部位的位置及对应的颜色标记。"),
    "get_metrics":("读取训练指标摘要","返回训练进度条需要显示的指标。"),
    "name":("返回任务名称","提供当前任务注册时使用的名称。"),
    "on_fit_end":("记录训练结束状态","在训练结束时保留必要的指标记录。"),
    "on_train_epoch_start":("处理训练轮次开始","在新轮次开始时更新训练进度与必要状态。"),
    "on_train_start":("初始化训练时间估算","在训练开始时记录后续估算所需的状态。"),
    "on_validation_batch_start":("处理验证批次开始","在验证批次开始时更新当前展示状态。"),
    "on_validation_epoch_end":("处理验证轮次结束","在一个验证轮次结束后更新历史指标。"),
    "predict_description":("显示预测任务标题","生成预测进度显示的简短文字。"),
    "run":("执行科研任务","调用已经实现的任务运行流程。"),
    "sanity_check_description":("显示预检查阶段信息","提供训练前检查阶段的显示文字。"),
    "setup":("装配数据集","准备训练、验证和测试所需的数据对象。"),
    "test_dataloader":("加载测试批次","返回测试阶段使用的数据加载器。"),
    "test_description":("显示测试阶段信息","提供测试进度条上的文字。"),
    "train_dataloader":("加载训练批次","返回训练阶段使用的数据加载器。"),
    "training_step":("执行训练批次","计算一个训练批次的预测和损失。"),
    "val_dataloader":("加载验证批次","返回验证阶段使用的数据加载器。"),
    "validation_description":("显示验证阶段信息","生成验证进度条上的文字。"),
    "validation_step":("执行验证批次","计算验证批次的预测和指标。"),
}

# Presentation-only groups organize existing Python modules by their source
# directory. Each concrete component owns its implementation details.
CONCRETE = {
    "research_datasets": ("CUB-200 与 NLP 数据集", "逐一查看 CUB-200、NLP 样本的读取、划分和适配实现。"),
    "models": ("Vision 与 LLM 模型实现", "查看 ResNet-18、Gemma-2 2B 的结构、构建与兼容性实现。"),
    "tasks": ("分类训练与 LLM SAE 任务", "查看分类训练和七项 LLM SAE 任务的执行流程与配置。"),
}

# Human-readable names describe verified existing directories. Newly added
# component directories still receive a scoped branch automatically.
INSTANCE_LABELS = {
    ("research_datasets", "CUB_200"): (
        "CUB-200 鸟类数据集", "该数据集的图像、标注、划分与数据适配实现。"
    ),
    ("models", "resnet18_mlp"): (
        "ResNet-18 模型", "残差网络、分类头与模型兼容性实现。"
    ),
    ("tasks", "classification"): (
        "分类任务", "通用的分类训练、预测与评测流程；运行组合由配置决定。"
    ),
}

# Details refer to implemented source only. Never use these to invent new APIs.
KEY_METHODS = {
    "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward":
        "按当前 head_style 处理输入：MLP 模式使用 ResNet 特征并传给 MLP 头；线性模式经过 ResNet 各层与池化，再由全连接分类层输出 logits。",
    "code:method:research_datasets.vision.CUB_200.CUB_200:CUB200.__getitem__":
        "根据数据集的局部索引读取图像与已有标注，并执行配置中的预处理。",
    "code:method:tasks.vision.classification.classification_task:ClassificationTask.run":
        "把当前任务配置交给已存在的分类训练执行函数。",
    "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers":
        "构建现有分类任务使用的优化器和学习率调度。",
    "code:method:tasks.vision.classification.classification_module:ClassificationModule.training_step":
        "计算训练批次的损失和可记录的训练指标。",
    "code:method:research_datasets.datamodule:DLBaseDataModule.setup":
        "根据当前数据集配置准备训练、验证和测试数据。",
}

# Only a small approved set of source methods are excerpted for the public site.
# The snippet always comes from the actual tracked file, not LLM-authored code.
PUBLIC_EXCERPTS = frozenset({
    "code:method:models.vision.resnet18_mlp.resnet18_mlp:Resnet18MLP.forward",
    "code:method:research_datasets.vision.CUB_200.CUB_200:CUB200.__getitem__",
    "code:method:tasks.vision.classification.classification_task:ClassificationTask.run",
    "code:method:tasks.vision.classification.classification_module:ClassificationModule.configure_optimizers",
})
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
    extract = lines[start:start + 12]
    if not extract:
        return None
    minimum_indent = min((len(line) - len(line.lstrip()) for line in extract if line.strip()), default=0)
    return "\n".join(line[minimum_indent:].rstrip() for line in extract).rstrip()


def task_configuration_nodes(name: str, parent: str, nodes: list, edges: list) -> None:
    """Attach real Hydra task YAML under the corresponding generic task."""
    directory = ROOT / "conf/task"
    configurations = []
    for path in sorted(directory.glob("*.yaml")):
        value = path.read_text(encoding="utf-8")
        names = [line.split(":", 1)[1].strip().split("#", 1)[0].strip()
                 for line in value.splitlines() if line.startswith("  name:")]
        if names and names[0].strip("\"'") == name:
            configurations.append((path, value))
    if not configurations:
        return
    group_id = "code:configurations:tasks:" + name
    sources = [str(path.relative_to(ROOT)) for path, _ in configurations]
    nodes.append({
        "id": group_id, "kind": "group", "title": semantic_title("group", name, f"tasks/LLM/{name}/task.py", curated="管理科研任务的 Hydra 配置"),
        "summary": "任务配置按独立选项组织展示。",
        "description": "任务配置按独立选项组织展示。",
        "status": "implemented", "evidence": sources,
        "presentation_only": True, "source_symbol": "",
    })
    edges.append({"source": parent, "target": group_id, "type": "contains"})
    for path, content in configurations:
        source = str(path.relative_to(ROOT))
        label = ("配置科研任务的运行参数" if path.stem == name
                 else "配置任务输入与执行参数")
        summary = "定义任务类型、运行层级与训练和评测选项。"
        config_id = "code:config:tasks:" + name + ":" + path.stem
        nodes.append({
            "id": config_id, "kind": "group", "title": label,
            "summary": summary, "description": summary, "status": "implemented",
            "evidence": [source], "source_path": source,
            "presentation_only": True, "source_symbol": "",
            "attributes": [
                {"type": "code", "label": "当前任务配置", "value": content, "source": source},
                {"type": "reference", "label": "配置文件来源", "path": source},
            ],
        })
        edges.append({"source": group_id, "target": config_id, "type": "contains"})


def concrete_groups(output: dict) -> None:
    """Present generic code apart from Vision/NLP/LLM implementations."""
    nodes = output["nodes"]
    edges = output["edges"]
    for area, (title, summary) in CONCRETE.items():
        parent_id = "code:group:" + area
        families = ("vision", "NLP") if area == "research_datasets" else ("vision", "LLM")
        family_modules = {family: {} for family in families}
        family_shared = {family: [] for family in families}
        for node in nodes:
            if node["kind"] != "module":
                continue
            parts = PurePosixPath(node.get("path", "")).parts
            if len(parts) < 3 or parts[0] != area or parts[1] not in family_modules:
                continue
            family = parts[1]
            if len(parts) >= 4:
                family_modules[family].setdefault(parts[2], []).append(node)
            else:
                family_shared[family].append(node)
        scoped = {item["id"] for instances in family_modules.values()
                  for group in instances.values() for item in group}
        scoped.update(item["id"] for group in family_shared.values() for item in group)
        if not scoped:
            continue
        edges = [edge for edge in edges
                 if not (edge["type"] == "contains"
                         and edge["source"] == parent_id and edge["target"] in scoped)]
        group_id = "code:concrete:" + area
        nodes.append({
            "id": group_id, "kind": "group", "title": title,
            "summary": summary, "description": summary,
            "status": "implemented",
            "evidence": [n["path"] for n in nodes if n["id"] in scoped],
            "presentation_only": True, "source_symbol": "",
        })
        edges.append({"source": parent_id, "target": group_id, "type": "contains"})
        for family in families:
            instances = family_modules[family]
            if not instances and not family_shared[family]:
                continue
            family_id = f"code:family:{area}:{family}"
            family_title = {
                ("research_datasets", "vision"): "Vision 图像数据集",
                ("research_datasets", "NLP"): "NLP 文本输入数据集",
                ("models", "vision"): "Vision 图像模型",
                ("models", "LLM"): "LLM 语言模型",
                ("tasks", "vision"): "Vision 分类训练任务",
                ("tasks", "LLM"): "LLM 稀疏特征任务",
            }.get((area, family), f"{family} · {area} 实现")
            family_evidence = [node["path"] for node in family_shared[family]]
            family_evidence.extend(n["path"] for group in instances.values() for n in group)
            nodes.append({
                "id": family_id, "kind": "group", "title": family_title,
                "summary": f"{family_title} 领域的具体组件及其公共接口。",
                "description": f"{family_title} 领域的具体组件及其公共接口。",
                "status": "implemented", "evidence": family_evidence,
                "presentation_only": True, "source_symbol": "",
            })
            edges.append({"source": group_id, "target": family_id, "type": "contains"})
            for shared in family_shared[family]:
                edges.append({"source": family_id, "target": shared["id"],
                              "type": "contains"})
            for name, modules in sorted(instances.items()):
                instance_id = "code:instance:" + area + ":" + name
                family_path = ("tasks/LLM/" if area == "tasks" else
                               "models/LLM/" if area == "models" else
                               "datasets/NLP/") + name + "/task.py"
                instance_title, instance_summary = INSTANCE_LABELS.get(
                    (area, name), (
                        "组织" + context_for_path(family_path),
                        "汇总该科研功能的真实模型、数据与任务实现及接口。",
                    ))
                nodes.append({
                    "id": instance_id, "kind": "group", "title": instance_title,
                    "summary": instance_summary, "description": instance_summary,
                    "status": "implemented", "evidence": [m["path"] for m in modules],
                    "presentation_only": True, "source_symbol": "",
                })
                edges.append({"source": family_id, "target": instance_id,
                              "type": "contains"})
                if area == "tasks":
                    task_configuration_nodes(name, instance_id, nodes, edges)
                for module in modules:
                    edges.append({"source": instance_id, "target": module["id"],
                                  "type": "contains"})
    output["edges"] = edges



# Source-derived Chinese title rules live in semantic_titles.py. The original
# symbol name and source location remain in each AST node's metadata and
# expanded interface; neither is part of its public headline.

# An AST contains real source entities but is not itself a readable navigation
# hierarchy. Insert source-backed presentation groups without renaming actual
# functions, changing their IDs, or modifying the authoritative AST snapshot.
MAX_ATLAS_CHILDREN = 10
CODE_ROLES = (
    ("capture", "激活捕获与特征读取", r"capture|activat|feature|encode|decode|token"),
    ("intervene", "特征干预与生成", r"interven|edit|patch|steer|generat|forward|infer"),
    ("circuit", "电路追踪与归因", r"circuit|attribut|trac|gradient|residual"),
    ("explain", "解释与假设检验", r"explain|interpret|hypothesi"),
    ("evaluation", "指标、对照与评测", r"bench|metric|evaluat|score|assess|loss|auroc|report|reliab|rank|truth"),
    ("training", "训练与参数更新", r"train|optim|schedul|backward|fit|epoch|checkpoint|learn"),
    ("data", "数据、产物与存储", r"data|dataset|prompt|artifact|manifest|shard|store|file|path|read|write|save|load|collect"),
    ("contract", "约束、校验与配置", r"valid|check|verify|guard|config|schema|contract|require|assert|policy"),
    ("runtime", "运行入口与调度", r"runtime|runner|run|execute|setup|init|launch|start|stop|build|create|make|construct"),
)
CODE_DIR_LABELS = {
    "api": "API 与调用接口", "sae": "SAE 字典、加载与覆盖率",
    "store": "SAE 特征与实验产物存储",
    "llm": "LLM 模型与 SAE 任务", "nlp": "NLP 样本与 Prompt",
    "vision": "视觉数据集、模型与分类",
    "benchmark": "SAE 评测与可靠性",
    "reliability": "评测稳定性与统计检验",
    "integrations": "SAR、RSS 与外部基准集成",
    "backend": "任务执行后端", "adapters": "外部基准适配",
    "metrics": "评测指标计算",
    "release": "评测结果发布与验证",
    "native": "原生实现与基线",
    "extensions": "可扩展基准接口",
    "generate": "生成式 SAE 特征干预",
    "intervene": "教师强制与特征干预",
    "capture": "SAE 特征捕获",
    "circuits": "电路追踪与干预",
    "explain": "SAE 自动解释",
    "train": "SAE 训练",
    "entry": "任务入口与公共接口",
}


def _source_role(node: dict) -> str:
    symbol = str(node.get("source_symbol") or node.get("title") or "")
    path = PurePosixPath(node.get("path") or "")
    # Exact source symbols have priority over the surrounding file's domain.
    text = symbol.lower().replace("_", " ")
    for key, _title, pattern in CODE_ROLES:
        if re.search(pattern, text):
            return key
    text = path.stem.lower().replace("_", " ")
    for key, _title, pattern in CODE_ROLES:
        if re.search(pattern, text):
            return key
    return "helpers"


def _code_buckets(parent_id: str, child_ids: list[str], by_id: dict) -> tuple[str, list] | None:
    children = [by_id[key] for key in child_ids]
    # Source directory boundaries outrank speculative method categories.
    directories = [
        PurePosixPath(node.get("path") or "").parts[:-1] for node in children
    ]
    if all(directories):
        prefix = []
        for parts in zip(*directories):
            if len(set(parts)) != 1:
                break
            prefix.append(parts[0])
        depth = len(prefix)
        scopes = defaultdict(list)
        for node, parts in zip(children, directories):
            piece = parts[depth] if depth < len(parts) else "entry"
            scopes[piece.lower()].append(node["id"])
        if 1 < len(scopes) <= MAX_ATLAS_CHILDREN:
            labels = {
                key: CODE_DIR_LABELS.get(key, "管理" + context_for_path(
                    by_id[scopes[key][0]].get("path", "")))
                for key in scopes
            }
            return "directory", [(key, labels[key], members)
                                 for key, members in sorted(scopes.items())]
    roles = defaultdict(list)
    for node in children:
        roles[_source_role(node)].append(node["id"])
    if 1 < len(roles) <= MAX_ATLAS_CHILDREN:
        labels = {key: title for key, title, _ in CODE_ROLES}
        labels["helpers"] = "公共辅助函数与接口"
        return "role", [(key, labels[key], members)
                        for key, members in sorted(roles.items())]
    # Explicit, deterministic source ranges are a last resort for a single
    # very large class or densely populated package. This preserves its
    # source order instead of inventing research-method categories.
    ordered = sorted(children, key=lambda n: (
        n.get("path") or "", int(n.get("line") or 0), n["id"]))
    chunk_size = max(MAX_ATLAS_CHILDREN,
                     (len(ordered) + MAX_ATLAS_CHILDREN - 1) // MAX_ATLAS_CHILDREN)
    chunks = [ordered[i:i + chunk_size]
              for i in range(0, len(ordered), chunk_size)]
    if len(chunks) < 2:
        return None
    entries = []
    for i, chunk in enumerate(chunks):
        paths = {n.get("path") for n in chunk}
        first, last = chunk[0], chunk[-1]
        if len(paths) == 1 and first.get("line") and last.get("line"):
            title = "分组浏览相关接口与方法（第" + str(i + 1) + "组）"
        else:
            title = "按源码职责浏览相关操作（第" + str(i + 1) + "组）"
        entries.append((str(i), title, [n["id"] for n in chunk]))
    return "source_range", entries


def balance_code_hierarchy(output: dict, *, max_children: int = MAX_ATLAS_CHILDREN) -> None:
    """Keep every direct Code child list within a readable navigational bound."""
    if max_children != MAX_ATLAS_CHILDREN:
        raise ValueError("Atlas Code navigation uses ten-child presentation scopes")
    by_id = {n["id"]: n for n in output["nodes"]}
    edges = output["edges"]
    for iteration in range(48):
        child_map = defaultdict(list)
        for edge in edges:
            if edge["type"] == "contains":
                child_map[edge["source"]].append(edge["target"])
        offenders = [(parent, kids) for parent, kids in child_map.items()
                     if len(kids) > max_children]
        if not offenders:
            return
        parent, kids = sorted(offenders, key=lambda pair: pair[0])[0]
        partition = _code_buckets(parent, kids, by_id)
        if partition is None:
            raise ValueError("Cannot form bounded source-backed Code scopes at " + parent)
        method, buckets = partition
        if len(buckets) > max_children or len(buckets) < 2:
            raise ValueError("Invalid Code navigation partition at " + parent)
        original_edges = {(parent, key) for key in kids}
        edges[:] = [edge for edge in edges
                    if edge["type"] != "contains" or
                    (edge["source"], edge["target"]) not in original_edges]
        for key, label, members in buckets:
            digest = sha256((parent + "|" + method + "|" + key).encode()).hexdigest()[:14]
            gid = "code:nav:" + digest
            if gid in by_id:
                raise ValueError("Duplicate Code navigation node " + gid)
            provenance = sorted({by_id[child].get("path") or ""
                                 for child in members if by_id[child].get("path")})
            node = {
                "id": gid, "kind": "group",
                "title": label,
                "summary": "按实际源码目录、定义职责或声明位置组织 "
                           + str(len(members)) + " 个已存在的源码实体。",
                "description": "本导航分组涵盖对应源码实体，其接口和来源仍在各自原始节点中。",
                "status": "implemented", "presentation_only": True,
                "navigation_scope": method, "source_symbol": "",
                "path": "", "evidence": provenance,
                "context_ids": [], "core_ids": [],
            }
            output["nodes"].append(node)
            by_id[gid] = node
            edges.append({"source": parent, "target": gid, "type": "contains"})
            for child in members:
                edges.append({"source": gid, "target": child, "type": "contains"})
    raise ValueError("Atlas hierarchy exceeded safe rebalancing depth")


def readable_code_tree(graph: dict) -> dict:
    """Presentation-only, already-existing component subset of an AST snapshot."""
    output = copy.deepcopy(graph)
    output["scope_note"] = "仅显示当前源代码已存在的主要实现；内部辅助函数和空包初始化不作为单独功能节点。"
    output["projection"] = "existing_public_api"
    selected = []
    for item in output["nodes"]:
        kind=item["kind"]
        path=item.get("path","")
        raw=item["title"]
        if kind=="module" and path.endswith("/__init__.py"):
            continue
        if kind in ("function","method") and raw.startswith("_"):
            if item["id"] not in KEY_METHODS:
                continue
        # Keep all other real symbols: no synthetic classes or functions.
        selected.append(item)
    visible = {node["id"] for node in selected}
    output["edges"] = [edge for edge in output["edges"]
                       if edge["source"] in visible and edge["target"] in visible]
    for node in selected:
        kind=node["kind"]
        original=node["title"]
        node["source_symbol"]=original
        if kind == "root":
            label, summary = "浏览科研系统的实际功能", (
                "按数据集、模型、实验运行与 SAE 分析方法浏览已实现的代码。"
            )
        elif kind == "group":
            group = node["id"].rsplit(":", 1)[-1]
            label = GROUPS.get(group, "管理科研组件与实际接口")
            summary = GROUP_SUMMARIES.get(group, "按相近的科研职责组织已有接口。")
        else:
            labels = {"module": MODULES, "class": CLASSES,
                      "method": METHODS, "function": FUNCTIONS}
            lookup = labels.get(kind, FUNCTIONS)
            key = node.get("path", "") if kind == "module" else original
            known = lookup.get(key)
            curated_title = known[0] if known else ""
            label = semantic_title(kind, original, node.get("path", ""),
                                   curated=curated_title)
            summary = semantic_summary(label, kind, node.get("path", ""),
                                       known[1] if known else "")
        node["title"]=label
        node["summary"]=summary
        node["technical_description"]=node.get("description","")
        node["description"]=summary
        node["status"]="implemented"
        node["source_path"]=node.get("path","")
        if kind in ("method", "function"):
            # The key function and exact signature are the attributes of the
            # real leaf node, not another invented capability.
            owner = node["id"].split(":")[-1]
            symbol = owner if kind == "method" else original
            node["key_function"] = symbol
            node["attributes"] = [
                {"type": "function", "label": "关键函数",
                 "value": symbol, "signature": node.get("signature", "")},
                {"type": "text", "label": "功能说明",
                 "value": KEY_METHODS.get(node["id"], summary)},
            ]
            if node["id"] in KEY_METHODS:
                node["summary"] = KEY_METHODS[node["id"]]
                node["description"] = KEY_METHODS[node["id"]]
            snippet = source_excerpt(node)
            if snippet:
                node["attributes"].append({
                    "type": "code", "label": "实际源码节选", "value": snippet,
                    "source": node.get("path"), "line": node.get("line"),
                })
            if node.get("path"):
                node["attributes"].append({
                    "type": "reference", "label": "完整代码位置",
                    "path": node["path"], "line": node.get("line"),
                })
        # Only display source-backed research links that are also in the
        # existing scientific/project projection, handled by the exporter.
    output["nodes"] = selected
    concrete_groups(output)
    from harness.code.interfaces import enrich_code_tree
    enrich_code_tree(output, METHODS)
    balance_code_hierarchy(output)
    return output

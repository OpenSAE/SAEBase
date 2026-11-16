# 当前实现与接口迁移

## 目的和主要功能

说明科研实现的唯一归属、当前接口，以及旧调用迁入当前实现的方法。

## 适用范围

供使用当前 SAEBase 源码、配置和插件的开发者迁移调用；科研记录继续保留原始数值、来源和计算身份。

## 当前入口

八类任务位于根级 tasks。激活健康评测由 benchmark 提供，`saebase` 命令直接接入该用途。保留准备提示词、环境检查、干预、评测、可靠性分析和合成对照命令。多模型比较通过 `saebase-benchmark --config <比较配置> --dry-run/--real/--report-only` 使用。正式执行与模型下载继续需要相应授权。

| 旧实现入口 | 当前实现入口 |
| --- | --- |
| saebase.interventions | tasks.intervene.backend 或 tasks.intervene.native |
| saebase.benchmarks | tasks.benchmark.execution |
| saebase.reliability | tasks.benchmark.reliability |
| saebase.integrations | tasks.benchmark.methods |
| saebase.generate | tasks.generate.backend |
| saebase.circuits.graph | tasks.circuits.backend.circuits |
| saebase.explore | tasks.capture.explore |
| saebase.release 中的比较计算 | tasks.benchmark.comparison |
| tasks.benchmark.extensions | saebase.extensions |
| saebase.data | saebase.datasets，物理实现留在根级 datasets |

插件只使用 `opensae.saebase.extensions`。模型与 SAE、指标共享现有扩展对象和登记状态。旧插件组名退出当前接入，当前发行物中的示例与文档使用同一名称。

任务入口返回 `EvaluationResult`；具体操作函数保持自己的测量结果。旧 Task 的字典结果现在保存在外层结果的 summary 中。Precheck 的 PASS、FAIL、BLOCKED、NOT_RUN 继续表达其检查状态。

共同配置不再要求 train.max_epochs。运行目录清理选项使用 runtime.clear_experiment_leaf，训练特有计划继续放在训练任务。Task 的 SAE 和层默认值由所选模型与资源决定。

## 保留的能力与移除的机制

### Capture 的特征数量

`task.top_feature_num` 控制 capture 返回的特征数量，缺省为 10。返回数量不超过实际特征总数，排序继续使用激活绝对值，记录保留原始带符号的激活值。零、负数、布尔值和非整数在加载模型前被拒绝。该参数由 capture 任务持有，直接操作入口复用同一检查，不改变 SAE 编码或模型激活。

固定 0.2.0 的发布链路、旧发布命令和专属资产检查退出活动工程。当前软件发布使用现有手工发布流程，其校验目标为精确源码与当前分发包。多模型比较保留权重身份、缓存、统计分析和结果摘要，已解除软件版本与至少两个模型族的固定要求。

原生训练合并为任务内部的一份实现，保留精度设置、数据输入检查、原生恢复和检查点身份。模型的平铺重复目录退出活动代码，模型键仍指向对应族与实例。

NNsight 工作程序保留独立可用的 baseline 生成、运行检查、层探测和激活导出。依赖已不存在研究模块的编辑与 teacher-force 分支移除；特征编辑使用当前 generate/intervene 接口。程序对不支持的编辑请求给出明确错误，不替换为基线计算。

## 科研身份与版本文字

工具文件按用途命名。数值身份中已经参与指纹计算的固定格式字符串保留原值，包括激活、编码、样本及干预测量的标识。它们表达当前单一计算身份，不引入旧版本选择分支。改动这些字符串会改变缓存键，因此与普通文件改名分开处理。

已有真实测量与来源资料保持原始字节。当前工作区未发现 experiment 或 runs 运行目录，本轮没有执行结果转换。外部挂载的在用结果需要用户明确指定后，按其实际 Manifest 和内容身份迁移；不推断未知数据位置。

## 验证状态

重构候选仅完成源码静态检视。新增和更新的测试源码覆盖任务契约、配置独立性、结果查找和指标完整性，但没有执行 pytest、构建、安装、模型下载或 GPU 验收。数值等价性和安装行为须依据实际执行的证据报告。

## 功能一致性改进的迁移说明

以上验证状态描述前次重构交付。本次功能改进执行离线产品检查与安装态检查，具体结果按本次交付记录报告；合成验证不表示真实权重或科学验收通过。

1. capture 的 `task.top_feature_num` 开始实际生效；默认值仍为 10，超过特征总数时返回全部特征。
2. 多层 SAE 不再隐式取层列表最后一项。通过 `task.layer`、预检查列表中的 `layer` 或 `model.default_layers.<sae_key>` 明确选择。默认科研主配置已明确选择原有 Gemma 层位 25。
3. 预检查与执行使用同一资源解析。外部资源需先显式登记；隔离预检查还需要可安装的 `task.extensions` entry point 名称。提供者可以声明能力，加载后仍检查实际能力。
4. SAEBase 统一从 `conf/config.yaml` 启动，所有任务显式组合全局 model、dataset 和 task。Benchmark 的 `task.protocol` 只描述评测方法；`task.benchmark_config` 文件中的模型、数据和显式样本与总控冲突时拒绝，不能覆盖或删除全局选择。摘要记录实际对象与方法来源，相对输入路径按原启动目录解析。
5. 任务结果新增完成摘要与格式版本；旧记录保持可读。脚本应同时读取执行状态和完成摘要，处理退出码 0、1、2，不再以存在报告或进程正常返回推断全部工作完成。
6. `runtime.mode` 支持 `new`、`resume`、`overwrite`，空配置等效于 `new`。旧 `experiment.overwrite` 转换为对应意图，冲突配置被拒绝。恢复保留原始配置、账本和错误历史；缺少恢复身份的旧运行不会被自动补成可恢复。
7. 结果查找返回候选而不是复用许可。旧账本缺少样本或协议身份时，比较保留不可比较原因，不伪造差值。`compare_runs(..., pairs=[((model_a, sae_a), (model_b, sae_b))])` 显式选择跨资源配对。
8. Research Lab 仅转发经原有发布流程批准的完成摘要。旧记录未携带摘要时保持缺失，不推断为完成；本次不会公开本地运行或私有输入。

activation-health 的准备样本由既有 PromptSource 及其 manifest 契约检查，与实际读取路径一致；SAE 层位也进入公共选择解析。原生模型比较的 `verify_matrix` 仍要求至少两个模型族、每模型至少两个 SAE，以及完整指标和审计。本轮不将软件执行完成替换为科研验收通过；前次说明中的“解除模型族要求”不代表该原生验收条件已解除。

本次不改动历史数值、既有指标公式和缓存身份格式，不替换正式 Atlas，也不自动推送或发布。

已删除并行的研究启动配置与 Benchmark 专用启动配置，不保留旧启动别名。当前调用者改为默认总控或 `compose(config_name="config")`；activation-health 通过同一总控组合所需组件。Gemma 的默认层位 25 回到所属模型配置，总控不包含特定模型的层位设置。

`saebase` 主命令现在直接进入公共 Runtime，使用 `model=... dataset=... task=...`。激活健康评测的专用参数转换仍由其科学工具持有，转换后组合相同总控；不再把该评测工具作为 SAEBase 主入口。

目录统一来自全局选择；Task 自定义公共目录和删除全局 model/dataset 的逻辑已移除。模型集合使用 `model.members`，任务必须声明支持该输入能力；集合不会触发 Runtime 分派子任务。旧目录与历史科研结果不自动移动或重写，源码及配置身份变化不能静默绕过恢复校验。

## Precheck 自动准备

执行 Precheck 后默认尝试安装缺失分析依赖、下载选定权重和准备选定数据。网络、安装和下载默认开启，仍可显式设置离线模式；仅配置与展示配置不会触发准备。准备失败保留具体阶段、实际尝试和完整日志，再返回 BLOCKED。

移除 `allow_compute`、`allow_gpu`、`allow_native` 开关；已选择的检查计划在资源就绪后继续执行。设备和原生计划仍需有效，训练检查等实际计算预算保持有效。准备时间、数据准备时间和实际检查时间分别配置。

IOI 等数据集通过已有 Dataset 准备方法生成和过滤。最小准备预算明确记录为 32 个候选、至少 1 个保留样本；它不放宽原有过滤条件，不将最小检查数据称为完整科研数据集。已有数据保持原始字节；新数据经过暂存、校验和目录提交。


## Capture 的 non_whiten / whiten 模式

`task.mode` 是 capture 方法参数，默认 `non_whiten`，历史未声明此字段的运行按 non_whiten 读取。该默认值保持原始激活绝对值 Top-K；`whiten` 使用同一选定 Dataset、Model、SAE、Hook 的所有 prompt token 出现位置拟合逐 feature 均值与总体标准差，再按末 token 的绝对 z-score 排名，并列按 feature ID 升序。两种模式使用不同的方法身份，不能互相作为已完成结果跳过。

示例：`python -m saebase.runtime.train dataset=humaneval model=gpt2small task=capture task.execute=true task.mode=whiten experiment.sub_name=whiten_run`。切换原有方法使用 `task.mode=non_whiten`。`task.whiten_chunk_size` 默认256，仅控制原生 tokenwise SAE 编码的块大小并写入统计身份，FP64 累积中央矩并以 token 总数 N 为方差分母。使用当前选定的完整数据集合拟合描述性统计，包括原生零激活；包含真实输入的 BOS 和提示后缀，不追加答案、不包含 padding。显式 task.prompt 时仍用 Dataset 全部选定输入拟合，再输出该显式输入的结果。

whiten 不改变 feature ID、SAE 权重、原生激活值或解码方式。每条结果保留 `activation`，新增 `zscore`、`mean`、`std`、`zero_native_activation`、`whiten_eligible`。标准差为0时分数置0并排除 Top-K；有效项不足 top_feature_num 时返回所有有效项。所有维度均为常数时排名记为数学未定义。原生零激活项可以因负 z-score 入选。`active_features` 继续表示原生非零项；`top_features` 按模式对全部合格维度排名。此模式没有 SAR 专用的语义 blocklist。

统计保存为 `metadata/capture_normalization.npz` 与 `capture_normalization.json`，记录均值、方差、标准差、有效维度、token总数、模型/SAE字节身份、输入 token 摘要、方法版本和产物校验摘要。统计文件是该结果的必要产物。逐 prompt 暂存 CPU 末 token snapshot，排名阶段复用它，避免第二次模型前向。统计阶段任何输入失败都会使本轮失败，不发布部分数据拟合的标准化参数。BatchTopK 等跨 token 竞争编码器需另行定义统计协议，当前明确报告不支持。


Capture 两种模式沿用相同 Dataset/Model/capture 目录层级，使用独立 experiment.sub_name 保存各次运行。manifest 新增通用 task_mode 元数据，已有 list_runs 支持 task_mode 过滤，例如 list_runs(root, task="capture", task_mode="whiten")。旧 capture 记录缺少 mode 时，通过 CaptureTask 的默认配置规则在读取时识别为 non_whiten，历史文件不回写。whiten 与 non_whiten 的模式、统计身份、配置比较分别保留，查询或续跑不会将二者互相替代。


## Neuronpedia 对应仓库更名

Llama Scope LXR-8x 的 `fnlp/Llama3_1-8B-Base-LXR-8x` 与 `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` 已核实指向同一个仓库和提交，解释源匹配识别这一明确更名。仍要求 SAE 物理文件夹、维度和 Hook 对应；其他同名仓库不自动认作同一资源。此修复同时用于 precheck 和 capture 的既有解释获取路径。新运行自动获取对应解释，历史结果不会被读取操作改写。

本次 HotpotQA 的两种 capture 另存 `reports/capture_explanations.json`，以原 capture 账本的 SHA-256 绑定解释补充记录，按 feature ID 查询解释及其来源，不重算或改写原始激活。公开源没有解释的 feature 仍明确保留缺失状态，不生成替代标注。


## 任务归属与关键检查调整（2026-11-14）

公共入口在准备运行目录前调用 Task 自己的参数校验，并将已校验的 Task 实例交给执行入口，避免正常链路重复调用同一校验。`tasks/sae_runtime.py` 只保留共享资源操作及少量输入类型辅助函数；具体参数规则位于各 Task 的 `operation.py`。直接使用 Task 或 `execute_task` 仍在入口校验配置。通用执行层不再替任务补写“完成”；具体任务声明自己的完成摘要。

自动 precheck 对 capture／intervene／generate 保留当前真实 SAE 抽样探测，对 benchmark／circuits／explain／train 调用任务自己的 `prepare_execution`，准备协议、输入和后端，不重复执行正式科研计算。训练准备不再要求正式计划最多两步；解释准备不再要求生成假说必须已经有预测评分。原生任务自己的计算与资源预算继续生效。原生训练、解释和图任务不为外层模型标签下载无关权重。

intervene／generate／circuits 的现有单输入方法在结果中明确写入 `input_scope`：显式 prompt 或 Dataset 第一条、请求和处理数量。此变更没有新增这些任务的全 Dataset 评测协议。公共 prompt 读取不再统一限制4000字符；上下文和原生后端的实际限制由相应执行实现负责。

显式 Dataset 输入文件的内容摘要进入已有样本选择键；同一路径内容改变会产生新选择，历史选择不重写。相对输入路径沿原启动目录解析。自然数据跨 split 的抽样、原始标签、科研公式和数值精度不变。

SAE 加载记录携带同一次解析得到的公开解释来源。Llama Scope 使用 SAE Lens 发布条目的稳定键 `llama_scope_lxr_8x` 关联物理 SAE 与解释库，删除解释模块中的特定新旧仓库名转换。没有登记来源时明确记录解释未取得，不再将其隐式当成全部解释完成。

CaptureTask 提供 `refresh_annotations(folder)` 与 `read_samples(folder)`。前者复用已保存的 feature ID 和真实加载身份补取解释，不加载权重、不重算激活；后者在读取时合并解释补充记录。Runtime 的普通 `read_run`／`inspect_run` 使用 Task 提供的小型完成更新，历史 manifest 和激活账本不改写，也不在常规运行发现时遍历全部样本。

同一个已加载资源的嵌套观察会话复用已有权重摘要和解释快照，最外层结束时进行原有最终核对。已准备的 IOI 样本不再触发无必要的 SAE CPU/GPU 往返。

本次不修改 tools、tests、scripts、Research Lab 和 Atlas；不新增平行注册、运行或测试框架。测试中绑定旧仓库名特殊转换、旧解释完成含义和旧内部调用签名的断言需另行更新，不能将其失败解释为实际科研计算已失败。


## Capture 的 SAR 模式与特征数量改名

Capture 的配置参数 `task.top_k` 更名为 `task.top_feature_num`，默认值仍为 10。旧参数会明确报错；调用者应修改配置或命令。生成采样、SAE 架构及可靠性分析中各自的 `top_k` 保持原义。历史运行记录保持原始内容，新配置不自动复用旧运行。

`task.mode` 支持 `non_whiten`、`whiten`、`sar`。SAR 按原生激活绝对值选择 `top_feature_num` 个特征，同值按 feature ID 排序；它测量整组特征相对于指定行为的组合性，不产生单特征 SAR 排名。特征原始激活、Neuronpedia 注释和正常 capture 结果继续保留。

```bash
saebase model=pythia70m dataset=ioi task=capture \
  task.mode=sar task.layer=3 task.top_feature_num=16 \
  task.sar.source=/absolute/path/to/SAR/sar.py
```

以上是仅配置入口；实际计算需要 `task.execute=true`、已有资源及相应执行授权。`task.sar.source` 指向用户提供的可信原版 `sar.py` 或其仓库目录；配置校验只检查输入，不执行该文件。执行时导入其 `coalitions` 和 `sar`，保留并检查源文件 SHA256，不复制或重写估计器。

默认每样本抽取 384 个子集，256 个用于拟合、128 个用于留出评估，每特征保留概率为 0.5。拟合样本数至少覆盖 `1 + K + K*(K-1)/2` 个二阶系数，并要求拟合、留出至少各两个样本。K 必须至少为 2 且不超过 SAE 维数，不能在 SAR 模式静默截断。随机种子来自全局 `seed`。

行为暂支持指定输出位置的目标与对照 token logit 差。Dataset 样本可提供 `target_token_id`、`reference_token_id`，或 IOI 已有的单 token `tokenization.correct_ids`、`tokenization.distractor_ids`。显式 `task.prompt` 使用 `task.sar.target_token_id` 和 `task.sar.reference_token_id`。标签冲突、多 token、缺失或不合法时明确失败，不从预测结果猜标签，不拼接答案。Dataset 仍持有样本范围和来源。

干预复用同一次 capture 的 snapshot，只清除子集中未保留的选中特征，其余特征与原始重构残差保持原义。默认执行解码可加性检查和原样重放检查；`require_affine_decode=false` 明确使用另一个 decoder-mediated 协议身份。检查不是对任意解码器线性的数学证明。

每条样本结果新增 `sar`，包含指标、未定义指标名、特征 ID、行为、采样参数和源文件身份。`reports/sar/*.npz` 保存 masks、拟合/留出索引、响应、原始基线、tokens、捕获激活及选中特征值，并登记到 required_artifacts。常数行为标为数学未定义，沿用现有 EvaluationResult 的已评估语义；注释刷新保留该数学状态。SAR 的未定义指标序列化为 null，不伪造数值。

SAR 实现归 `tasks/capture/sar.py` 与 `sar_mode.py`，原版文件加载复用 `saebase/api/paper_sources.py`。`tasks.benchmark.methods` 继续导出同一个 `evaluate_sar` 供已有调用者复用；旧的 `tasks.benchmark.methods.sar` 文件路径迁入 capture。公共 Runtime 与 Research Lab 本轮没有修改。现阶段子集干预按顺序执行，不包含上游专门的批量或后缀重放加速。

SAR 的解码可加性诊断对 FP16/BF16 模型激活使用 FP32 加减，避免把多个独立补丁的低精度舍入差误判为解码器非线性。该提升仅用于诊断，正式子集干预仍使用原始 snapshot 和模型原生精度；非线性解码器的默认拒绝保持有效。


## 工具能力归入正式任务（2026-11-14）

本节接续上面的任务归属调整：本轮已处理 tools、tests 和 scripts。已有公共命令名不变，`saebase-benchmark`、`saebase-intervene`、`saebase-audit`、`saebase-ground-truth` 分别调用对应 Task 的实现。Benchmark 与批量干预的真实计算经过统一 Runtime，记录运行、执行自动 precheck 并传递完成状态；`--output` 在真实计算时表示运行工作区根目录，产物使用现有 Dataset/Model/Task 层级。原工具模块路径停止使用。

- 批量干预的方法配置位于 `task.protocol`，模型和 SAE 从全局选择取得，token 输入使用 `dataset.samples`。JSON 命令入口仅将原有实验文件转换到此配置，不另行执行一套科研流程。
- 所有 token 的稀疏特征捕获通过 `task=capture task.output_scope=all_tokens task.mode=non_whiten` 使用。`task.max_tokens` 和 `task.max_active_per_token` 分别声明输入长度和每 token 完整非零行的上限，超出时明确失败而不截断稀疏行；结果仍保留末 token 的摘要。该输出范围目前不支持 whiten/SAR，不能将它们解释成已实现。
- 多条件生成通过 `task=generate task.feature_selection=last_active_content_token` 自动选择末端内容 token 上的特征，或在 `task.conditions` 中显式指定已有原生干预条件。结果记录每个条件的基线、修改后 token 与干预步骤。
- SAE 单独准备和原生 encode/decode 探测归 precheck：`task=precheck task.mode=sae task.resource_only=true`；独立入口为 `python -m tasks.precheck.checkpoint --sae-key <key> --layer <layer> --device auto --output <directory>`，不会加载基础模型。它使用明确标注的合成输入，仅证明软件可用。历史覆盖工作流可显式提供 `--coverage-catalog` 保留原覆盖协议的来源字段。
- 数值参考的记录与比较由 `saebase.api.reference` 提供；站点覆盖与上游 SAEBench 工作进程归 Benchmark。保留的固定参考验收入口调用这些实现，不再维护独立算法。
- 多模型、多 Dataset 的 capture 调度使用 `python -m tasks.capture.matrix --capture-mode whiten --tag <run_tag>`。选择清单在 `conf/experiment/capture_matrix.yaml`，每个组合仍构造 SAEBase 主配置，由正式入口自动 precheck 后 capture。旧固定批次驱动的重复实现已迁入正式矩阵入口。

旧的 Qwen/vLLM 临时性能排查、重复 GPU/资源探测、已完成的实验交接脚本和仅包装其他命令的 shell 文件已移除。测试保留原生算子数值、身份/样本、失败传播、恢复和安装态行为；不再以删除包是否存在、文档数量或迁移文件数量判定功能正确。已完成的 Atlas 源码目录迁移断言一并退役；Atlas 原文、中央校验与正式发布流程不作修改。历史科研记录与冻结文档保持原文；其中的旧命令只描述当时的执行。

当前开发文档导航检查仅验证活动文档链接；中央历史归档校验继续负责冻结原文的摘要，不在项目导航检查里重复进行。Research Lab 的专属维护继续暂停。


## 用功能名称表达实现与结果（2026-11-14）

活动代码的说明、报错、报告和工作流选项改用实际功能名称：检查点数值参考、激活与特征编码缓存、批量干预、Benchmark、可靠性分析、SAE 训练、回路追踪和远程追踪。不再要求使用者先了解开发阶段编号。

可靠性报告标题改为 `Reliability Audit`。检查点参考比较函数为 `compare_reference_activation_probe`；可靠性分析内部读取函数为 `_read_benchmark`，内部缓存和测量查询同样按用途命名。现有公共 CLI 命令保持不变，参考验收使用 `--reference`，可靠性分析使用 `--results-dir`。旧参数别名仅用于解析已有调用。手动真实参考验收工作流的选项为 `reference`、`activation-cache`、`interventions` 和 `all`。

已保存的结果、格式标识、协议身份、缓存目录和来源字段不重写；这些机器标识继续按现有格式读取。模型名称、层号、稀疏度、范数、数值精度等科研记号保留原义。冻结文档保留历史原文，Research Lab 本轮不维护。以上调整不改变数值计算和样本选择。


## SAR 的完整正确答案行为

`task.sar.readout=answer_mean_logprob` 使用 `mean_t log P(y_t* | x, y_<t*, patched_h)`。Dataset 的 `answers.py` 负责从真实标注提取正确答案和来源字段；显式单条输入可用 `task.sar.answer` 提供标准答案。选择题解析到正确选项正文，CounterFact 使用 target_true，GSM8K 使用 #### 后的最终答案，HumanEval 保留标准代码的缩进与换行。HarmBench 的行为描述与 XSTest 的分类标签都不当作标准回答，缺少参考答案时记录 unsupported。

特征从原始问题末 token 捕获一次；答案不参与 SAE 编码或特征选择。答案续写单独以 `add_special_tokens=false` 编码，并追加到原始 prompt token 后；前向输入只含答案前缀，使用严格移位的标签计算每个答案 token 的条件概率。无额外 BOS/EOS。文本答案的空格、代码补全所需换行作为明确的续写格式记录。干预始终定位原问题末 token，不随答案长度移动到答案 token。FP32 log_softmax 后以 FP64 求均值，保留未干预与原样重放的行为一致性检查。

完整答案 token ID、长度、原始答案及来源、格式前缀和聚合定义写入 SAR 行为身份，原始 npz 增加 answer_token_ids。原版 SAR 采样、拟合、留出及指标定义保持不变。此实现提供正确答案支持度上的 SAR 测量，用户给出的 Selection Wins、Preservation、Search 表格是方法参考，本轮未将其声明为已复现结果。


## 模型执行窗口统一（2026-11-14）

新加载模型使用 `min(模型原生上下文上限, 4096)`。窗口规则由 Model 的 `models/loading_policy.py` 负责，读取实际模型文本配置；不使用 tokenizer 的无穷大占位长度。TransformerLens 构建前传入 `n_ctx`，使 Llama 的原有 2048 默认窗口提高到 4096，也避免先创建过大的注意力缓冲。Hugging Face 包装层提供相同的 `cfg.n_ctx`，公共输入和完整答案评分共用它。

本批 GPT-2 Small 的执行窗口仍为 1024，Gemma、Llama 和 Qwen 的所选九个模型为 4096。超过执行窗口的显式 token 输入会明确报告不支持；现有 TransformerLens 字符串分词的截断行为保持原义，SAR 仍要求完整题目和答案。旧结果和已停止的批次不改写、不自动重跑。窗口变化会影响实际输入，新实验结果按其实际 token 身份记录，不能把旧的截断结果称为完整输入结果。


## Whiten 特征的 SAR 配对结果（2026-11-14）

`task.mode=sar task.sar.feature_selection=whiten` 使用选定数据集拟合已有标准化统计，按绝对标准化分数选出 `task.top_feature_num` 个特征；SAR 的组合干预直接使用这些 ID。默认仍为原生激活排序，不改变旧实验。

设置 `task.sar.top_pairs=5` 后，结果新增 `sar.pairwise`。二阶拟合复用原版 SAR 的 `design` 和已有拟合组合，其系数描述两个保留指示变量的交互项，按绝对值排序；这是二阶代理模型系数，不是每对特征各有一个 SAR 分数。原有整组指标与留出评估保持原义，拟合秩随结果记录。每对保存两端的原生激活、标准化分数和 Neuronpedia 公开解释；交互方向说明单独标注为拟合解释。全部配对 ID 与系数保存于原有 SAR 数组产物，解释补取沿用 capture 的正式接口。


## 运行记录的读取与继续处理

`EvaluationResult.diagnostics` 保存问题的具体原因、对任务的影响和下一步建议。预检查的 `reason`、`repair` 与实际证据直接用于形成说明。旧记录在读取时补充这一视图，原文件保持原样；缺少原因的旧状态如实标记原因尚未记载，并提示读取任务日志和预检查报告。

结果的正式执行状态为 `EvaluationResult.status`，`completion` 继续表达请求完成情况。新结果的 `meta` 保留种子等补充信息；调用者原来读取 `meta.execution_status` 或 `meta.executed` 时，分别改为读取 `status` 或判断 `status == "executed"`。

运行内部产物在保存时采用相对路径，`read_run(folder)` 返回基于当前目录解析后的文件位置。旧的任务结果绝对路径通过正式任务结果文件的位置识别其原运行目录，再映射到当前目录；外部来源继续保留其实际位置。读取过程保持历史文件原样。

`manifest.configuration` 是当前配置的正式来源，`original_configuration` 与首次 `config.yaml` 保留初始设置；恢复尝试继续保存。当前配置摘要统一使用 `configuration_sha256`。新写入过程清理被替代的 `latest_configuration.json` 与旧的 `config_sha256` 字段。

`list_runs(root, errors=issues, **filters)` 返回可读运行，并向 `issues` 追加无法读取的路径、具体原因、影响和处理建议；默认调用通过警告指出这些异常。单独读取指定运行时仍返回实际错误，供调用者定位。

## 工程精简后的调用迁移

- 模型调用使用 `models.plugins.get_model_adapter(key)` 或已登记的模型接口，系列中的 `specs.py` 保存声明；实例 `model.py` 和 `compatibility.py` 的转发位置退出。
- 提示身份、路径、可用性及读写统一从 `saebase.datasets.artifacts` 导入，原 `artifact_layout`、`prompt_schema` 和 `prompt_store` 调用迁入该位置，数据内容与格式标识保持原值。
- 组件捕获从 `saebase.api.components` 导入，解释生成从 `tasks.explain.interpretation` 导入；生成后端直接使用 `tasks.generate.backend`。
- 直接读取单个组件 YAML 的代码使用 `conf.load_group(group, name)`；Hydra 选择语法保留，目录表中的预设合成原公共配置。新增同系列模型无需新增实例转发包。
- 批次通过 `tasks.capture.matrix` 或 `scripts/capture_queue.py` 启动。旧实验与测量文件保持原位置；源码身份变化仍按原恢复条件处理。

固定解码方向的 KV 缓存生成使用 `tasks.generate.backend.decoder_steering`；interp-engine 的 SAE 位置编辑使用 `tasks.generate.backend.sae_interventions`。这两项独立科研能力保留原算法、参数与协议，实际后端依赖按原方式延迟加载。

提示产物默认位置统一采用原正常读取入口的规则：源码运行使用项目根下的 `data/` 和 `experiment/`，安装后使用调用目录；显式指定的路径保持其原义。构建会重新生成所属包的输出，移除旧构建目录残留的已删除模块；分发检查按实际组件目录表和正常安装调用确认完整性。


## 工具入口归入所属功能

原工具目录的能力已经迁入所属模块。数据准备和环境展示保留原命令名称。配对检查使用 `python -m tasks.precheck.sae_checks`，数值参考使用 `python -m saebase.api.reference`。短评测通过 `saebase-benchmark --pair-protocol cpu_pair|reference_pair --real` 执行；CPU 协议选择 `--sae-key`，参考协议选择 `--model`。

真实权重的存储和干预回归分别使用 `python -m tests.product.test_activation_store` 与 `python -m tests.product.test_feature_interventions`，原参数与输出内容继续适用。工程和 Lab 命令详见[贡献指南](../../../.github/CONTRIBUTING.md)。旧调用者改用这些当前入口。历史结果中的格式标识和身份保留原意。

## 示例与评测模块归属

原示例包的调用改为两个源码演示脚本或对应教程。测试用合成组件通过 `tests.fixtures.synthetic` 读取。原评测公共类型与规划函数通过 `tasks.benchmark.execution` 导入，指标登记归 `tasks.benchmark.metrics.registry`，方法接入归 `tasks.benchmark.methods`，已有测量通过 `tasks.benchmark.results.metrics` 读取。

统计函数继续由 `tasks.benchmark.reliability` 提供：样本组重采样与Bootstrap同属 `bootstrap.py`，已知因子与退化测量同属 `ground_truth.py`，SAR与RSS对照同属 `controls.py`，统计报告读写归 `tasks.benchmark.results.audit`。

位点批次使用 `python -m tasks.benchmark.sites.matrix`，单个位点检查使用 `python -m tasks.benchmark.sites.checks`，实际评测使用 `python -m tasks.benchmark.sites.runner`。目录声明和来源查询归 `sites.catalog`，恢复记录归 `sites.records`。证据核对使用 `python -m tasks.benchmark.results.evidence --output <运行目录>`，资料包使用 `python -m tasks.benchmark.results.archive --help`。这些操作继续读取原科研身份与协议；位点、精度、种子、划分和结果格式保持原意。

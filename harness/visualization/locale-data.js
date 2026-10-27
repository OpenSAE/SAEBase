/* English locale data is keyed by immutable Atlas node IDs.
 * Chinese text comes from the source-backed graphs; IDs and evidence are not translated.
 * The public site uses this same file, so its snapshots and source repository agree.
 */
(function () {
"use strict";
const TERM_EN = {
 "term.sae":["Sparse Autoencoder (SAE)","An autoencoder that encodes model activations into sparse features and reconstructs them, enabling feature-level analysis."],
 "term.sae.reproducible":["Reproducible SAE experiments","A provenance protocol recording the model, SAE, input, activation site, and intervention settings for each experiment."],
 "term.sae.assets":["Model and prompt artifact identity","The model adapter, SAE profile, prompt data files, and their verified origins used in a research run."],
 "term.sae.capture":["SAE feature capture","Read tensors from a specified model activation site and encode them into SAE feature activations."],
 "term.sae.intervene":["Teacher-forced feature intervention","Edit features or model activations under a fixed input sequence and activation site, then compare the resulting outputs."],
 "term.sae.generate":["SAE intervention during generation","Apply selected feature edits as the model generates successive tokens and compare the continuation with the baseline."],
 "term.sae.circuits":["SAE causal circuit analysis","Trace candidate paths among model components and feature interventions; causal interpretations require independent tests."],
 "term.sae.explain":["Automatic SAE feature explanation","Propose semantic hypotheses for features and evaluate their predictions on separate examples."],
 "term.sae.benchmark":["SAE benchmarking and reliability","Evaluate reconstruction, interventions, controlled comparisons, and reliability through a common benchmark runner."],
 "term.sae.sar_rss":["SAR and RSS benchmark adapters","Integrate behavioral compositionality and response-sensitivity evaluations into the SAE benchmark interface."],
 "term.sae.train":["SAE training","Optimize sparse autoencoder parameters with a registered training task and its native training backend."],
 "term.dataset":["Dataset","A configurable component that supplies input examples, labels, and data splits to the research pipeline."],
 "term.model":["Model","A computational component that receives inputs and produces predictions or representations using a defined architecture."],
 "term.task":["Task","A research operation specifying what to train or evaluate; the runtime selects its model and dataset."],
 "term.registry":["Registry","A shared mapping from component names to registered datasets, models, SAE profiles, and tasks."],
 "term.plugin":["Plugin","A component implementing an agreed interface that can be registered and called through the common runtime."],
 "term.ast":["Abstract Syntax Tree (AST)","A structured Python source-code representation used to identify real classes, functions, methods, and relationships."],
 "term.api":["Application Programming Interface (API)","An agreed callable interface specifying arguments, return values, and communication between software components."],
 "term.yaml":["YAML","A human-readable configuration format used to compose experiment settings."],
 "term.tensorboard":["TensorBoard","A visualization tool for training logs, including loss curves and other metrics over time."],
 "term.tqdm":["tqdm","A progress-display utility for monitoring iterations such as training batches."],
 "term.callback":["Training callback","A function invoked at defined points in training to log metrics, save checkpoints, or run auxiliary operations."],
 "term.code_tree":["Code Tree","An interactive view of actual Python modules, classes, functions, and methods extracted from source code."],
 "term.core_tree":["Core Tree","An interactive view of project architecture, shared component responsibilities, and management goals."],
 "term.knowledge_tree":["Knowledge Tree","An interactive view of scientific questions, proposed methods, and independently evaluated research goals."],
 "term.environment_tree":["Environment Tree","An interactive view of hardware, software, and runtime resources relevant to executing research."],
 "term.context_tree":["Context Tree","The shared terminology and evidence view that explains terms used in Atlas nodes."],
 "term.cpu":["Central Processing Unit (CPU)","A general-purpose processor executing operating-system and program instructions."],
 "term.gpu":["Graphics Processing Unit (GPU)","A parallel processor commonly used to accelerate neural-network workloads."],
 "term.gib":["GiB","A binary data capacity unit; one GiB equals 1,024 MiB."],
 "term.tib":["TiB","A binary data capacity unit; one TiB equals 1,024 GiB."],
 "term.tmux":["tmux","A terminal session manager for keeping long-running commands accessible after disconnecting."],
 "term.sbatch":["sbatch","A cluster job-submission command that hands work to a scheduler."],
 "term.docker":["Docker","A platform for running applications and their dependencies inside containers."],
 "term.podman":["Podman","A tool for creating and running application containers."],
 "term.rclone":["rclone","A file synchronization tool for copying and checking data across storage locations."],
 "term.nvidia_smi":["nvidia-smi","A command-line utility for inspecting NVIDIA GPU devices, memory, and utilization."],
 "term.python":["Python","The programming language used for this project's experiments, data processing, and tools."],
 "term.function_signature":["Function signature","An interface declaration describing a function name, parameters, and their types."],
 "term.artifact":["Experiment artifact","A persisted experiment output such as a metric, log, checkpoint, or report."],
 "term.datamodule":["DataModule","A data management component that builds training, validation, and test data loaders."],
 "term.saebase_runtime":["SAEBase Runtime","The shared implementation of experiment configuration, component registration, task dispatch, and artifact management."],
 "term.saebase":["SAEBase","Research infrastructure for capturing, editing, explaining, evaluating, and training sparse features in language models."],
 "term.llm":["Large Language Model (LLM)","A model for processing and generating language whose internal computations can also be analyzed."],
 "term.nlp":["Natural Language Processing (NLP)","Methods for converting language inputs and behaviors into analyzable research data."],
 "term.sparse_autoencoder":["Sparse Autoencoder (SAE)","An autoencoder that uses a small number of active features to encode and reconstruct neural-network activations."],
 "term.gemma2_2b":["Gemma-2 2B","A roughly two-billion-parameter language model in Google's Gemma 2 family; the experiments use the base pretrained variant."],
 "term.feature_capture":["Feature capture","Read activations from a specified language-model layer and extract feature values using the corresponding SAE."],
 "term.intervene":["Teacher-forced intervention","Edit features while holding the complete input token sequence fixed, then compare the model's outputs."],
 "term.generate":["Generation intervention","Change selected features during autoregressive generation and compare generated behavior against a baseline."],
 "term.circuit_analysis":["Circuit analysis","Analyze relationships among internal model components and record the observed graph with its provenance."],
 "term.explain":["Automatic explanation","Generate hypotheses from feature-activating examples and assess their predictive ability on held-out data."],
 "term.benchmark":["Benchmark evaluation","Evaluate representation or intervention methods using fixed metrics, samples, and computational protocols."],
 "term.sae_training":["SAE model training","Learn sparse feature encodings and reconstructions with a training backend, producing inspectable weights and artifacts."],
 "term.dlbase":["SAEBase Runtime","Shared configuration, registration, task execution, and experiment artifact management."]
};
const GROUP_EN = {
 "context:root":["Research terminology and source evidence","Definitions and verifiable sources for scientific and engineering terms used throughout Atlas."],
 "context:category:sae_methods":["SAE feature analysis and interventions","Terms for SAE capture, interventions, generation, circuits, explanations, and training."],
 "context:category:sae_resources":["SAE models, inputs, and provenance","Definitions of model identities, SAE profiles, prompts, and reproducible experiment conditions."],
 "context:category:sae_evaluation":["SAE benchmarking and scientific validation","Concepts for reconstruction, intervention, reliability, SAR, and RSS evaluations."],
 "context:category:platform":["Framework, code, and tasks","Shared concepts for datasets, models, tasks, and source interfaces."],
 "context:category:runtime":["Hardware and execution tools","Definitions of computing devices, capacity units, and execution utilities."]
};
const CORE_EN={
 "q.research":"How can research projects become reproducible systems?",
 "q.modularity":"How can research components maintain clear boundaries?",
 "q.reproducibility":"How can every experimental run be traced?",
 "q.transparency":"How can people and agents jointly understand the project?",
 "q.mobility":"How can experiments be reproduced and moved across devices?",
 "s.registry":"Decoupled plugin registry",
 "s.compatibility":"Three-layer compatibility checks",
 "s.hydra":"Hydra configuration and a single execution entry",
 "s.artifacts":"Hierarchical experiment artifact protocol",
 "s.explainability":"Context management across research trees",
 "s.mobility":"Environment probes and verified data transfer",
 "g.registry":"Pluggable component registration",
 "g.dataset_gate":"Dataset and task compatibility",
 "g.model_gate":"Model compatibility checks",
 "g.pipeline":"Pre-execution task validation",
 "g.config":"Reproducible experiment configuration",
 "g.artifact_schema":"Typed experiment output directories",
 "g.context":"Structured scientific knowledge and evidence",
 "g.code_tree":"Source-verified classes and functions",
 "g.env_tree":"Machine environment snapshots",
 "g.transfer":"Data backup and recovery",
 "g.recovery":"Offline experiment reproduction",
 "g.b2":"Remote B2 backups",
 "q.multi_domain":"How can SAE research tasks share reusable components?",
 "s.parallel_cores":"Unified SAEBase and runtime task interfaces",
 "g.domain_families":"Organize datasets, models, and SAE tasks",
 "g.llm_capture":"Capture LLM features",
 "g.llm_teacher_forcing":"Teacher-forced LLM interventions",
 "g.llm_generation":"LLM feature interventions during generation",
 "g.llm_circuits":"Analyze LLM circuits",
 "g.llm_explain":"Automatically explain LLM features",
 "g.llm_benchmark":"Benchmark LLM sparse features",
 "g.llm_sae_training":"Train sparse autoencoders",
 "q.agent_flow":"How can agent research work be audited?",
 "s.agent_flow":"Plan-driven, human-reviewed execution",
 "g.agent_execution_flow":"Auditable agent research workflow"
};
const KNOWLEDGE_EN={
 "q.sae.methods":"How can LLM SAE research be made reproducible?",
 "q.sae.assets":"How can Gemma-2 2B, SAE profiles, and prompts form verifiable inputs?",
 "s.sae.assets":"Model, SAE, and prompt provenance protocol",
 "g.sae.prompt_artifacts":"Store traceable prompts by dataset and model",
 "g.sae.gemma_model":"Load the Gemma-2 2B model adapter",
 "g.sae.sae_registry":"Select sparse dictionaries by SAE identity",
 "q.sae.capture":"How can SAE feature responses be captured at fixed positions?",
 "s.sae.capture":"Capture SAE features and read activations",
 "g.sae.activation_capture":"Capture actual activations and SAE features",
 "g.sae.capture_task":"Capture features through the task interface",
 "q.sae.intervene":"How do feature interventions affect fixed-input outputs?",
 "s.sae.intervene":"Teacher-forced interventions and controlled comparisons",
 "g.sae.feature_edit":"Apply SAE feature-editing operations",
 "g.sae.teacher_forced":"Execute the teacher-forced intervention task",
 "q.sae.generate":"How do SAE feature edits affect generation?",
 "s.sae.generate":"Generation interventions and baseline continuations",
 "g.sae.generation_task":"Execute the SAE generation comparison",
 "g.sae.generation_engine":"Dispatch the generation intervention backend",
 "q.sae.circuits":"How can candidate causal circuits be traced and tested?",
 "s.sae.circuits":"Circuit tracing and local path interventions",
 "g.sae.circuit_trace":"Invoke the source-backed circuit tracer",
 "g.sae.circuit_edit":"Intervene on candidate circuit paths",
 "q.sae.explain":"How can SAE feature explanations be generated and independently tested?",
 "s.sae.explain":"Feature explanation hypotheses and held-out tests",
 "g.sae.interpretation":"Generate feature explanation hypotheses",
 "g.sae.explanation_task":"Run explanations through the LLM task interface",
 "q.sae.benchmark":"How can SAEs be compared with controlled reliability metrics?",
 "s.sae.benchmark":"SAE benchmarks, reliability, and compositionality adapters",
 "g.sae.benchmark_runner":"Run benchmarks on selected SAEs and examples",
 "g.sae.reliability":"Test evaluation reliability and controls",
 "g.sae.sar_adapter":"Run SAR behavioral compositionality evaluations",
 "g.sae.rss_adapter":"Run RSS response-sensitivity evaluations",
 "q.sae.train":"How can traceable SAEs be trained and archived?",
 "s.sae.train":"SAE training and checkpoint interfaces",
 "g.sae.train_task":"Start SAE training through the task interface",
 "g.sae.train_backend":"Execute the native SAE training backend"
};
const TEXT_EN={
 "为什么重要":"Why this matters", "其他名称":"Alternative names",
 "真实依据":"Source evidence","基础解释":"Foundational explanation",
 "专业词汇 · Context":"Technical terms · Context",
 "在 Code Tree 中查看对应实现":"View the corresponding implementation in Code Tree",
 "在 Context 中查看":"View in Context",
 "展开节点属性与依据":"Expand attributes and evidence",
 "这个节点在做什么":"What this node does",
 "科研术语定义与源码依据":"Research terminology and source evidence",
 "用于理解 Atlas 中的相关节点和源码职责。":"This term helps explain responsibilities of Atlas nodes and source-backed interfaces.",
 "帮助准确理解科研组件的职责、输入和输出。":"This term clarifies the responsibilities, inputs, and outputs of research components.",
 "已有实现":"Implemented","部分实现":"Partially implemented","待审阅":"Pending review"
};
// Interface and presentation labels are shared across all five trees.
Object.assign(TEXT_EN,{
  "SAEBase 研究架构与任务注册": "SAEBase architecture and task registration",
  "LLM SAE 科研方法与实现依据": "LLM SAE methods and implementation evidence",
  "SAE 技术术语及源码依据": "SAE terminology and source evidence",
  "CPU、GPU 与任务执行环境": "CPU, GPU, and execution environments",
  "SAEBase Python 实现树": "SAEBase Python implementation tree",
  "Atlas 五棵 Tree 的职责与关联": "Atlas tree responsibilities and relationships",
  "模型、数据与 SAE Task 插件注册": "Model, dataset, and SAE task registration",
  "SAE Task 运行前的兼容检查": "SAE task pre-execution compatibility checks",
  "Prompt 数据集": "Prompt datasets",
  "LLM 模型实现": "LLM model implementations",
  "LLM SAE 任务执行": "LLM SAE task execution",
  "SAE 研究任务": "SAE research tasks",
  "已有的项目功能": "Existing project functionality",
  "研究概念与依据": "Scientific concepts and evidence",
  "机器与执行条件": "Machine and execution conditions",
  "现有代码能力": "Implemented code capabilities",
  "已有的模块化组件架构": "Existing modular component architecture",
  "已有的项目上下文组织": "Existing project context organization",
  "已有的训练配置与产物管理": "Training configuration and artifact management",
  "定义训练损失": "Define training loss",
  "记录 Hydra 组合配置": "Record composed Hydra configurations",
  "确定随机性与执行环境": "Specify randomness and execution environment",
  "保存可追溯的结果文件": "Save traceable result files",
  "可插拔组件注册": "Pluggable component registration",
  "Task 运行前检查": "Pre-execution task checks",
  "这个节点在做什么": "What this node does",
  "展开或收起子节点": "Expand or collapse child nodes",
  "收起节点属性": "Collapse node attributes",
  "查看功能说明与源码依据": "View functional explanation and source evidence",
  "展开节点说明与依据": "Expand node explanation and evidence",
  "尝试其他检索词": "Try another search term",
  "可核对的科研概念示意": "Diagram of a verifiable research concept",
  "输入 x": "Input x",
  "残差变换 F(x)": "Residual transformation F(x)",
  "输出": "Output",
  "捷径连接": "Skip connection",
  "查看实体源码": "View entity source",
  "参数与作用": "Parameters and roles",
  "输入参数": "Input parameters",
  "主要作用": "Primary purpose",
  "现有源代码实现的处理流程。": "Processing workflow implemented in the current source.",
  "函数签名": "Function signature",
  "返回值": "Return value",
  "源代码表达式": "Source-code expression",
  "实际代码主要部分": "Relevant implementation excerpt",
  "查看完整函数源码": "View complete function source",
  "成员数据": "Member attributes",
  "类": "Class",
  "函数": "Function",
  "查看源文件": "View source file",
  "在 Code Tree 中查看对应实现": "View implementation in Code Tree",
  "查看完整来源": "View complete source",
  "查看依据": "View evidence",
  "选择节点查看证据与来源": "Select a node to inspect evidence and provenance",
  "演示数据（未实测）": "Synthetic demo data (not measured)",
  "源代码标识": "Source identifier",
  "方法签名": "Method signature",
  "设备与执行信息": "Device and execution information",
  "演示环境 · 合成设备配置": "Demonstration environment · synthetic devices",
  "SAEBase API 与特征接口": "SAEBase API and feature interfaces",
  "SAE 字典与加载": "SAE dictionaries and loading",
  "特征数据与分片存储": "Feature data and sharded storage",
  "语言模型与 SAE 操作": "Language models and SAE operations",
  "NLP 样本与 Prompt": "NLP examples and prompts",
  "SAE 指标与可靠性评测": "SAE metrics and reliability evaluation",
  "统计可靠性": "Statistical reliability",
  "SAR、RSS 与基准集成": "SAR, RSS, and benchmark integrations",
  "计算执行后端": "Computation backends",
  "外部基准适配": "External benchmark adapters",
  "评测指标计算": "Evaluation metric computation",
  "评测结果发布": "Evaluation result publication",
  "原生执行与对照": "Native execution and controls",
  "扩展评测接口": "Extended evaluation interfaces",
  "特征捕获": "Feature capture",
  "受控特征干预": "Controlled feature intervention",
  "生成时特征编辑": "Feature editing during generation",
  "电路追踪与干预": "Circuit tracing and interventions",
  "特征自动解释": "Automatic feature explanation",
  "SAE 训练": "SAE training",
  "模型实现": "Model implementation",
  "数据集与输入": "Datasets and inputs",
  "科研任务": "Research tasks",
  "训练与实验产物管理": "Training and experiment artifacts",
  "SAEBase 可复用算法": "Reusable SAEBase algorithms",
  "特征捕获与读取": "Feature capture and reading",
  "特征编辑与生成": "Feature editing and generation",
  "电路与归因": "Circuits and attribution",
  "特征解释": "Feature explanations",
  "评测与统计": "Evaluation and statistics",
  "训练与优化": "Training and optimization",
  "数据与保存": "Data and storage",
  "参数与校验": "Parameters and validation",
  "运行与调度": "Execution and dispatch",
  "源码模块": "Source module",
  "其他公共源码接口": "Other shared source interfaces",
  "依据当前已发布的源文件与方法标识组织导航。": "Navigation is organized from published source files and method identities."
});
const graphs=window.OPENRESEARCH_ATLAS||{};
const CJK=/[\u3400-\u9fff]/;
function add(chinese,english){
 if(typeof chinese==="string"&&CJK.test(chinese)&&typeof english==="string"&&english.trim()&&!CJK.test(english)) TEXT_EN[chinese.trim()]=english;
}
function cleanName(node){
 const raw=(node.source_symbol||node.symbol||node.path||node.id||"").split(/[/:.]/).filter(Boolean).pop()||"component";
 return raw.replace(/([a-z])([A-Z])/g,"$1 $2").replace(/[_-]+/g," ").replace(/\b\w/g,c=>c.toUpperCase());
}
const familyHint={core:"project structure",knowledge:"research evidence",code:"source code",environment:"execution environment",context:"technical terminology"};
for(const [view,graph] of Object.entries(graphs)){
 for(const node of graph?.nodes||[]){
  const item=TERM_EN[node.id]||GROUP_EN[node.id];
  const special=view==="core"?CORE_EN[node.id]:view==="knowledge"?KNOWLEDGE_EN[node.id]:null;
  const englishTitle=item?.[0]||special||(CJK.test(node.title||"")?cleanName(node):node.title);
  add(node.title,englishTitle);
  const englishSummary=item?.[1]||(view==="core"||view==="knowledge"
     ? ("Source-backed "+(view==="core"?"project design":"research method")+": "+englishTitle+". Scientific outcomes require their own evidence.")
     : ("Source-backed "+(familyHint[view]||"research")+" for "+englishTitle+". See the linked implementation or experiment record for details."));
  add(node.summary,englishSummary);
  add(node.description,englishSummary);
  add(node.why_it_matters, view==="context"?
    ("Understanding "+englishTitle+" supports interpreting source evidence and research conditions."):englishSummary);
  for(const attr of node.attributes||[]){
   if(!attr||typeof attr!=="object")continue;
   const label=attr.label;
   if(label&&CJK.test(label))add(label,TEXT_EN[label]||("Source-backed "+(attr.type||"attribute")+" detail"));
   if(typeof attr.value==="string"&&CJK.test(attr.value)){
    if(label==="为什么重要")add(attr.value,"Understanding "+englishTitle+" clarifies source identity, research assumptions, and interpretation.");
    else if(label==="其他名称")add(attr.value,(node.aliases||[]).filter(x=>!CJK.test(x)).join(", ")||englishTitle);
    else add(attr.value,"See the cited source for the precise "+(attr.type||"research")+" content associated with "+englishTitle+".");
   }
  }
 }
}
const DOC_EN={
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
  "harness/transfer/README.md": "harness/transfer/README.en.md"
};
window.OPENRESEARCH_ATLAS_I18N={en:TEXT_EN,terms:TERM_EN,core:CORE_EN,knowledge:KNOWLEDGE_EN,docs:DOC_EN};
})();
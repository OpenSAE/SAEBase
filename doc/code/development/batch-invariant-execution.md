# 批量评分的一致性与加速

## 目的和主要功能

使同一题目、同一特征干预在不同batch下保持相同评分，并在此前提下选择最快批量执行。说明共同执行策略、任务与Precheck的职责、来源和实测入口。

## 适用范围

当前单GPU的Capture、Intervene、Generate以及对应性能预检。正式大批次保持停止，本轮只运行有界真实性能与一致性检查。该文档记录验证范围，不以有限探针承诺所有硬件、所有输入的全局数学证明。

## 流程与职责

1. Runtime在任何模型加载或CUDA计算之前选定`runtime.numerics`，将当前计算版本写入配置后判断续跑。相同策略但版本不同的历史运行需要重新计算。同一工作进程保持一种数值策略，切换使用新进程。
2. `models/numerics.py`统一设置GPU计算策略，调用已复用的上游算子；HF的Attention使用Torch现成math后端以固定计算路径；模型适配器在加载前沿用该策略。各任务不分别设置环境或创建新的推理服务。
3. 任务继续使用已有模型、SAE、特征筛选和标准答案评分接口。公共缓存与批处理位于模型API，算法和指标位于对应任务。
4. Runtime加载一次模型和SAE，Precheck与正式任务共享。Precheck复用正常任务执行路径，对同一输入逐条计算与不同batch进行比较，测量吞吐和显存，再把推荐交给正式执行。`vllm`数值策略下要求误差为0；不通过时保留失败原因，不能把静默退回逐条评分算作通过。
5. 正式任务沿用相同策略并保留每题控制。显存不足时可减小batch，但减小后仍须通过一致性检查。数值策略及上游来源随运行结果、性能记录、归一化身份和行为产物保存；不能与旧数值策略的缓存混用。

## 复用范围

数值设置、矩阵乘法与log-softmax实现来自vLLM0.27.1，固定commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，源码路径`vllm/model_executor/layers/batch_invariant.py`。softmax使用较新的固定行归约实现，来自commit `68e478651a20e3d23bf939912c38883d511ac3ee`的`vllm/model_executor/determinism/batch_invariant.py`。接入层保留FP16/BF16的FP32累积及输出类型语义。

复制的函数定义、许可证与来源清单保存在`saebase/_vendor/`。原矩阵核显式指定IEEE FP32，使用适合本卡共享内存的上游tile配置；这些适配逐项写入来源清单。GPU计算代码由上游提供，本项目只维护接入、生命周期和实际任务验证。

本机原环境使用Torch2.11，项目所列vLLM0.27.1完整依赖要求Torch2.13。当前采取复用必要能力的方式，保留已验证的模型/SAE加载与干预能力，避免为取得几个计算算子升级整个实验环境。普通文本生成的现有vLLM入口不等于已支持本项目的中间层干预评分。

`runtime.numerics=vllm`为标准Runtime配置；直接使用低层模型API且未指定策略时保持`native`，历史配置缺少该字段时同样按`native`执行。该设置保持权重和指标定义，可能改变历史低精度舍入结果，因此身份和来源必须记录。CPU执行不安装CUDA算子，配置检查不执行GPU设置。

## 工程默认值

标准配置使用以下已验证设置，普通Runtime和矩阵worker在加载模型前执行同一份配置：

| 项目 | 默认设置 | 用途 |
| --- | --- | --- |
| CPU计算线程 | `runtime.cpu_threads=4`，Torch及BLAS统一限制 | 采用前轮四模型实测设置，避免普通启动恢复到本机24个Torch线程 |
| 文本切词并行 | 关闭独立线程池 | 单模型worker已有批处理，减少额外线程竞争 |
| GPU计算 | `runtime.numerics=vllm`，Attention使用math后端 | 保留已验证的批次一致性 |
| 计算精度 | 沿用各模型和SAE原加载精度，稳定策略使用IEEE FP32 | 保持模型身份与评分定义 |
| 批量大小 | 自动实测，默认候选最高64，显存预留10% | 按吞吐选取；显存不足时减小并检查结果 |
| 观测及缓存 | Intervene/Generate默认复用；模型和SAE加载一次 | 沿用已通过实测的公共计算复用 |
| 编译 | 保持现有直接执行路径，采用已复用的Triton算子 | 控制启动成本与动态干预兼容性 |
| 模型进程 | 现有矩阵调度按模型顺序执行 | 单个矩阵入口一次运行一个模型worker |

CPU线程控制由`saebase/runtime/environment.py`负责：子进程启动时设置线程环境；普通Runtime加载前用Torch和`threadpoolctl`现成接口限制已加载的线程池。`runtime.cpu_threads`可显式覆盖，配置检查保持无副作用。旧配置缺少该字段时，直接Runtime保留原CPU设置；模型子进程使用共同启动默认值。显存分配器沿用已验证默认行为，按批量搜索和显存余量控制负载。

性能记录同时保存实际Torch线程数、已加载BLAS线程池和相关环境设置。固定工程设置与自动batch搜索各自记录，避免把已有组合优化的收益重复算到本轮CPU设置上。四种Capture及Intervene/Generate的前轮实测已使用4线程和观测复用，本次将其落实为正常入口默认值。

默认入口复核：普通Runtime的一题GPT2/GSM8K生成采用默认8token，Top10比较全部完成；自动预检PASS，正式任务直接复用batch10，评分及离散输出控制误差0。矩阵的一题I也完成，原输入、答案、候选及全部1024个评分与前轮v4产物逐项完全相同。实际Torch/BLAS线程均为4。原始运行路径和核对结果见`experiment/maintenance/20261116_batch_invariance/defaults_verification.json`，详细过程续写同目录`execution.md`。

## 验证与产物

### Qwen累加与评分加速修正（2026-11-17）

真实100题暴露了旧小样本未覆盖的Qwen GDN累加差异：CUDA `cumsum`的归约布局随行数改变。逐算子追踪在累加处首次出现约1.9e-6差异，后续低精度运算将其放大。本次复用vLLM所含FLA的逐行scan内核，固定warp配置；当前Transformers 5.23.0 GDN函数保持原算法，仅替换该累加。源版本、SHA和许可证见`saebase/_vendor/gdn_source.json`、`vllm_scan.py`、`hf_qwen_gdn.py`、`FLA_LICENSE`。

Qwen GPU数值身份单独升级为`saebase.vllm_batch_invariant.qwen_gdn_scan.v5`。数学指标、原精度、模型权重和题目保持，固定归约顺序改变了旧Qwen舍入结果，因此旧Qwen Capture不能与新结果混用，需要重新计算。其余模型仍保持v4，既有完成结果可复用。配置在续跑比较前绑定对应版本，实际加载时安装对应模型实现。

原生答案评分现在为HF模型复用已有`SiteReplay`/`AnswerReplay`，与TL模型一样缓存前层；逐条和合批共用该缓存。每题保留完整原生前向对照，避免仅比较两个相同优化实现。batch策略按完整评分序列长度的2次幂桶及控制状态数复用，消除按精确答案长度反复搜索；同一已加载模型/SAE的预检及Capture模式共享策略。遇到OOM降低batch后保存新选择，后续同桶不会重复从过大batch开始。

旧54道失败题的原生与batch1/2/8/16控制比较全部通过；其中一题完整1024状态在batch2/8/16/32/64下与逐条结果完全相同。实际测速、软件检查和正式Precheck的来源在`experiment/maintenance/20261117_capture_acceleration/`，恢复运行记录续写原执行文档。

本轮工作记录与结果位于`experiment/maintenance/20261116_batch_invariance/`。最终v4每组真实探针使用原有模型/SAE/层、GSM8K3题、四种Capture和Intervene/Generate；预热1次、测量2次，Generate生成2个token。I完整计算10个候选的1024状态，并从每个模型的已保存产物独立复算1024状态；评分控制比较最高batch64。不同长度/顺序及尾批次、概率累积精度和续跑隔离的回归另行保存。

读取性能报告时区分：同策略下批量评分相对逐条评分的吞吐、完整每题耗时、首次搜索成本与模型加载等公共阶段。所有收益使用实际测量，未测部分不作倍数推断。

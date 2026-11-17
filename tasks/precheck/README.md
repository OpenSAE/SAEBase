# Precheck 资源准备与实际检查

## 目的和主要功能

准备并检查选定数据、模型、SAE 与任务条件，提供执行判断和具体原因。

## 适用范围

用于明确选择的 Precheck 或正式任务的自动检查；资源动作按当前请求与配置处理。

Precheck 使用 SAEBase 唯一总控 `conf/config.yaml`，复用现有 Dataset、Model、SAE 和科研操作接口。没有新增下载任务、管理任务或 Registry。

## 执行流程

`task.execute=false` 或 `--cfg job` 保持配置用途，不进行安装、下载或计算。设置 `task.execute=true` 后：

1. 解析准确的模型、SAE、层位和检查目标。
2. 尝试安装缺少的分析依赖，固定已安装的 Torch、NumPy 等模型数值栈版本，并在新进程验证导入。
3. 优先检查本地完整缓存；缺少文件时尝试下载选定模型及所需 SAE 文件，记录仓库版本和缓存位置。一个 SAE 检查不会下载同仓库全部层位。
4. 未给出显式提示时，通过选定 Dataset 的准备接口补齐数据，复用已有生成、编码、过滤和 manifest 规则。
5. 检查数据产物，再运行选定的 SAE 或任务检查。资源准备完成不等于实际检查通过。
6. 保存准备记录、原始日志、检查结果和产物审计。

```bash
saebase task=precheck task.execute=true \
  task.mode=task 'task.check_tasks=[capture]' \
  task.device=cpu experiment.sub_name=basic
```

Dataset 和 Model 仍从总控的 `dataset=...`、`model=...` 选择。`task.mode` 支持 `sae`、`task`、`all`。任务检查列表为空时沿用全部七个科研任务；需要原生计划的 benchmark、circuits、explain、train 必须提供实际参数，Precheck 不生成虚构计划。

## 默认准备策略

- `auto_setup`、`allow_network`、`allow_install`、`allow_download` 默认开启。显式关闭网络或准备功能仍可用于离线诊断。
- 模型与 SAE 的缺失文件下载预算默认 16 GiB；准备超时 1800 秒，数据准备另有 1800 秒。实际检查仍使用 `max_seconds`、`max_samples`、`max_tokens`。
- 默认数据准备生成 32 个候选、要求至少保留 1 个。它是明确记录的最小检查数据准备预算，不改变 Dataset 原有过滤条件，不代表完整评测数据或科研验收。
- 已有完整数据直接验证并复用；数据准备采用锁与临时目录提交，已有不完整产物不会被覆盖。
- 设备由 `task.device` 指定，实际检查计划由 `task.task_configs` 指定。删除 `allow_compute`、`allow_gpu`、`allow_native` 二次确认开关；执行请求不会在下载后再等待人工确认。
- 原生训练检查保留最多两步等实际预算要求。缺少输入计划或硬件不可用会报告具体原因。

## 记录与失败

结果保存在当前 Runtime 运行目录：

- `metadata/precheck_result.json`：准备阶段及实际检查结果。
- `metadata/precheck_audit.json`：报告文件完整性检查。
- `tmp/precheck_jobs/<阶段>/metadata/`：请求和子进程回执。
- `tmp/precheck_jobs/<阶段>/logs/`：完整标准输出、错误输出，以及安装、导入和下载事件。

读取时先查看请求范围、阶段及实际检查对象，再读取对应结果与日志。结果中的具体原因和处理建议用于补足依赖、资源或输入，已经完成的内容按所属任务协议复用。开发检查与记录功能时，同时安排这些内容在运行结束后的读取和实际处理。

安装、来源访问、下载、数据准备、预算不足等准备失败分别记录 BLOCKED 原因。软件执行错误仍保留 FAIL，不伪装成环境问题。超时停止整个工作进程组，避免安装或原生计算子进程留在后台。

这些记录可以证明软件执行到哪一步；不能自动证明模型性能或独立科研验收。离线测试使用明确标记的合成响应和模型，不替代真实链路运行。

## 性能预检与自动调优

`task.mode=performance` 对 `task.check_tasks=[capture,intervene,generate]` 指定的真实任务做小规模性能试验。Capture 默认检查 `whiten`、`non_whiten`、`g_s`、`I`，可用 `+task.capture_modes=[g_s,I]` 缩小范围。一次性能预检选择一个模型、一个 SAE 和层位；多个资源分别运行。

```bash
python -m saebase.runtime.train model=gpt2small dataset=gsm8k \
  task=precheck task.mode=performance task.execute=true \
  'task.check_tasks=[capture,intervene,generate]' \
  '+task.capture_modes=[g_s,I]' experiment.sub_name=performance_probe
```

实际 `capture`、`intervene`、`generate` 默认开启 `task.performance.enabled=true`，正式 Runtime 的自动预检会先运行对应性能探针，将推荐 batch 交给任务。设置为 false 保留标量执行，不进行性能调优；GPU数值策略仍由runtime.numerics统一决定。配置检查（`--cfg job` / `execute=false`）不执行探针。Qwen3.5 直接命令需使用 `conf/runtime/environments.yaml` 中的专用解释器；矩阵入口通过 Runtime 自动选择。

公共预算由 `saebase/runtime/performance.py` 负责，默认：

| 参数（位于 task.performance） | 默认值 | 用途 |
| --- | --- | --- |
| enabled | true（标准任务配置） | 开启自动性能预检与报告 |
| batch_size | auto | 自动选取；也可指定正整数 |
| batch_sizes | [1,2,4,8,16,32,64] | 实测候选，并补上实际单位数上限 |
| probe_samples | 3 | 性能探针题数；不修改生产批次的 Dataset 数量 |
| probe_states | 64 | Capture 批量搜索的状态控制数；I 的实际探针仍完整计算1024状态 |
| warmup / repeats | 1 / 3 | 基准和候选都预热，并多次测量 |
| max_seconds | 120 | 每次候选搜索的软时间预算；正在执行的前向会完成 |
| memory_fraction | 0.9 | 当前进程峰值 reserved 显存占设备总显存的上限 |
| atol / rtol | native: 1e-6 / 0；vllm: 0 / 0 | 答案平均 log 概率的数值检查阈值 |
| logit_atol | native: 1e-4；vllm: 0 | 干预/生成浮点指标阈值；生成 token、排名、字符串等离散输出必须完全一致 |

未列在 YAML 中的预算使用 Hydra 的 `+task.performance.<字段>=...` 覆盖。`task.task_configs` 可传实际任务配置，如清零生成的 `generate: {intervention: ablate, value: 0, max_new_tokens: 8, reuse_prompt_observations: true}`。应使用待运行任务的相同参数比较。

调优比较**同一输入、同一工作量**，选择数值检查通过、保留显存余量、实测吞吐最高的候选；自动模式下若候选比原配置慢则保留原配置。OOM 会记录并停止向更大的 batch 搜索，实际运行遇到 OOM 会降低 batch 并做数值检查。结果是给定预算和候选中的最好实测值，不承诺全局最优。按模型、任务和长度区间复用参数；没有净收益的决定也会缓存。正常 Runtime 自动预检把这些推荐交给紧接着执行的任务，不把所有长度强制设成同一个 batch。Capture 每题及特征任务的代表性控制继续检查；这是一组实测控制，不是对所有可能输入的数值证明。

主要工程优化：

- TL 残差前/后位置及 HF decoder 的前半模型缓存；HF 准备缓存时在选定层停止，避免无用尾层与全部词表输出。
- Capture 状态真实合批、相同原生 code 状态去重、不变解码基准复用，以及向量化四格交互排名。解码仍逐行调用原生 SAE，保留运算顺序及 LN 统计。
- Intervene 的公共观测与 identity 检查复用、尾层重放及特征条件批处理。
- Generate 的公共基线生成、原生特征条件批处理；batch1与较大batch使用相同实现。每条生成轨迹保留自己的完整前缀和原随机种子，复用不包含陈旧的编辑后 KV。
- inference_mode、仅保留需要的 logits 切片，避免缓存向量的视图保留整个 sequence×vocab 张量；模型和 SAE 复用及检查点校验租约在同一预检范围内共享。

`reports/performance_precheck.md` 是性能预检总报告，JSON 保存各探针的阶段分布、逐样本记录及全部候选的吞吐/显存/误差/拒绝原因。每个任务在 `reports/performance.json`、`performance.md` 保存实际运行耗时。报告提供平均、标准差、最小/最大、P50/P90/P95/P99，区分 question、feature comparison 与 answer state 的摊销耗时；阶段可嵌套，不能直接相加。

收益按相同探针的原配置基准与候选的预热后中位数计算，候选缓存准备计入吞吐比较，搜索、I 全部状态的真实执行及共享预处理分别记录。报告同时列出含调优及扣除首次调优后的每题耗时、搜索开销和预计回本单位数。少量探针的分位数只是描述性统计；它们不是500题全量结果或总体置信区间。没有逐项做消融的优化只汇报组合收益，不把同一收益重复归因给多个优化。

四个主要模型的本次真实性能探针、原始来源及比较口径见[性能实测报告](/workspace/SAEBase/experiment/maintenance/20261116_task_performance/性能实测报告.md)。

标准运行采用`runtime.numerics=vllm`，在模型加载前统一复用vLLM的稳定计算策略，严格比较评分和输出，差异必须为0；不把静默回退标量算作通过。历史数值路径使用`runtime.numerics=native`，同一工作进程不可混用。设计与来源见[批量评分的一致性与加速](../../doc/code/development/batch-invariant-execution.md)。

正常Runtime入口只加载一次模型和SAE，自动预检与正式任务共享，并直接消费本次推荐。计算版本在续跑判断前写入配置；版本变化时重新计算，同版本继续按任务完成策略恢复。I的性能预检另从已保存的一个题目产物读取全部1024状态，换batch独立复算并要求逐项相同。

标准工程设置已固定为`runtime.cpu_threads=4`，统一控制Torch和BLAS线程，关闭切词器的额外并行；Intervene/Generate默认开启观测复用。普通Runtime、预检和矩阵worker使用相同设置，实际值随结果及性能报告保存。计算精度、稳定Attention、现有编译路径、显存余量和单模型顺序调度沿用已有验证，具体选择见[工程默认值](../../doc/code/development/batch-invariant-execution.md#工程默认值)。

显存候选测量前释放未使用的分配器缓存，再重置并测量峰值；90%显存上限继续包含本次工作真实需要的预留空间。此处理避免前题或较大batch残留缓存导致所有候选被误拒绝。候选记录包含`reserved_before_cleanup_bytes`及`memory_measurement_policy`，用于识别测量口径；数值一致性条件保持原值。

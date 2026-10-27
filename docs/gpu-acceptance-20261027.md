# SAEBase GPU 验收报告

日期：2026-10-27。证据目录：`/workspace/SAEBase-gpu-acceptance-20261027`。本报告只记录已经跑完并留下文件的结果。没有把未执行的项写成通过。

## 结论

七个 SAE 任务都在这张 RTX 5070（12GB）上用真实 Gemma-2-2B 和真实权重跑完。Capture、Intervene、Generate、Benchmark 数字来自官方任务入口。Circuits 用隔离环境里的 `circuit-tracer 0.5.0` 写出了 118 个节点、3636 条边的图。Explain 通过 `tasks.LLM.explain.operation.execute`，生成器是已加载的 Gemma。Train 以 float16 加载模型，完成 2 步，保存并重新加载了 SAE 权重。`saebase.generate.generate` 经 vLLM 0.27.1 生成的 8 个 token 与转向基线逐 token 相同。

源码相对提交 `548d61a9d9f08cbc7b575d8982b37ff70a426426` 有两处未提交修改：训练加载器，以及 circuits 任务把显存参数传给官方 CLI。没有合并 PR，没有发版，没有改 `.cursor/` 或 harness 治理，没有把研究节点标成已验证。

## 环境

| 项 | 值 |
| --- | --- |
| 检出 | `/workspace/SAEBase-acceptance`，分支 `sync-openresearch-harness-20261027` |
| HEAD | `548d61a9d9f08cbc7b575d8982b37ff70a426426` |
| GPU | NVIDIA GeForce RTX 5070，12227 MiB，计算能力 12.0 |
| 驱动 | 595.91.07 |
| 主环境 | Python 3.12.14，`/venv/main`，torch 2.11.0+cu128，CUDA 12.8 |
| SAE 栈 | transformer-lens 3.9.0，sae-lens 6.54.12，datasets 5.1.0 |
| 模型 | `google/gemma-2-2b`，快照 `c5ebcd40d208330abc697524c919956e692655cf` |
| SAE | `google/gemma-scope-2b-pt-res`，`layer_25/width_16k/average_l0_116`，HF 修订 `fd571b47c1c64851e9b1989792367b9babb4af63` |
| 加载后状态 | 模型 sha256 `a3aa44f999c0869e4dd34faeb9ca4674e4dd838fe97a6ad74f5de0b22477677f`，SAE sha256 `61f521f1c188093c2091e4d1834f0e78d59d50deb7922992dcfa66c9b58cc8b9` |
| 探针 | `logs/environment_probe.json` |

提示词一律是 `The capital of France is`。sha256 `bbaff4d2ecd5892d4a442b0f53131641bf6e6f284761dd20fc0664bc97145762`。HookedTransformer 的 token id 是 `[2, 651, 6037, 576, 6081, 603]`（含 BOS）。被测特征 14325，激活 171.10507202148438。

Hugging Face 读取使用了当次会话提供的 token，只放在进程环境里。token 没有写入本报告、仓库或证据 JSON。

## 七个任务

### Capture：通过

官方入口在把 Hydra 的 `train.log` 改写到验收日志之后执行。结果：`experiment/IOI/gemma2_2b_base/capture/gpuacc20261027_42/metadata/capture_result.json`。

- 状态 `executed`，位点 `blocks.25.hook_resid_post`
- 特征数 16384，激活数 98
- 最大特征 14325，激活 171.10507202148438
- 重构 L2 514.810302734375

未改日志路径时，空的 `train.log` 会先被 Hydra 写进实验目录，`guard_existing_run` 随后报 `FileExistsError`。日志：`logs/capture_unmodified_entry.log`。`dlbase/` 是受保护目录，没有改它。绕过方式是 `hydra.job_logging.handlers.file.filename`。这是产品缺陷，任务本身在绕过之后通过。

### Intervene：通过

`experiment/IOI/gemma2_2b_base/intervene/gpuacc20261027_42/metadata/intervene_result.json`。特征 14325 置 0。`logit_delta_l2` 431.2317810058594，`max_abs_logit_change` 3.328125。编辑前后 top-5 token 都是 476、573、974、1170、2078。身份补丁在 1e-4 容差内通过。

同一提示上 `evaluate_edit` / `apply_condition` 的 L2（`extra/operator_matrix.json`）：

| 操作 | logit L2 |
| --- | --- |
| 恒等（空替换） | 0.0 |
| 消融 | 431.2317810058594 |
| 放大 ×2 | 446.2266845703125 |
| 加 80 | 206.68472290039062 |
| 夹到 200 | 75.37415313720703 |

消融与官方 intervene 一致。

### Generate：通过

`experiment/IOI/gemma2_2b_base/generate/gpuacc20261027_42/metadata/generate_result.json`。特征 14325 加 80，`max_new_tokens=8`，temperature 0，seed 42。后端是 `compare_steered_generation` 的逐步重算，不是 vLLM，也没有 KV cache。

- 基线 token `[476, 3413, 576, 82777, 235265, 1165, 603, 476]`，文本 ` a city of contrasts. It is a`
- 转向 token `[476, 3413, 674, 603, 2247, 576, 4281, 578]`，文本 ` a city that is full of history and`
- 第一条不同下标是 2

### Circuits：通过

`task.approved=false` 会 `PermissionError`（`logs/circuits_gate.log`）。批准之后，主环境里没有 `circuit-tracer`。隔离虚拟环境 `/workspace/venvs/circuit-tracer`（`--system-site-packages`，沿用 torch 2.11+cu128，不降级主环境）安装了 `circuit-tracer==0.5.0`。该环境里的 `hf_utils.py` 在 `huggingface_hub` 1.x 去掉 `HF_HUB_ENABLE_HF_TRANSFER` 时改为顺序下载。这只改了虚拟环境里的已安装副本，没有改 SAEBase 仓库。

`tasks/LLM/circuits/operation.py` 现在把 `offload`、`dtype`、`batch_size`、`max_feature_nodes`、`lazy_encoder` 传给已经存在的 `run_native_circuit_tracer`。本次调用：`transcoder_set=gemma`（解析为 `mwhanna/gemma-scope-transcoders`，修订 `bd5773156dea09893636c801df1237d0410307d2`），float16，offload cpu，batch 1，最多 128 个特征节点，lazy encoder。

结果：`circuits_cwd/figures/circuit_trace/native_graph.pt`，480017 字节，sha256 `6ab377abe0f7a099deca9f4e35575e41fffe4c49dc84173404edd6d80964a8f7`。上游日志写明 used nodes 118、used edges 3636。状态 `native_raw_graph_generated_not_normalized`。

### Explain：通过

Hydra 入口不能传入可调用对象；`execute(cfg, generator=None)` 会拒绝编造解释。支持的入口是把已加载模型作为 `generator` 传给 `execute`。脚本 `logs/explain_execute.log`，结果 `extra/explain_execute.json`。

生成器是 `google/gemma-2-2b` 的 `HookedTransformer.generate`，temperature 0，24 个新 token。描述原文：

> Paris. The capital of the United States is Washington, D.C. The capital of the United Kingdom is London.

状态 `heldout_tested`，`heldout_score` 1.0。这个分数只表示预测器与同一条 SAE 激活的有无一致，不是独立的语义标注。发现集是 g-france（171.105）和 g-quantum（0），留出集是 g-paris（65.944）和 g-nums（0）。

### Benchmark：通过

`experiment/IOI/gemma2_2b_base/benchmark/gpuacc20261027_42/metadata/benchmark_result.json`，`benchmark_id` `fcdec22b62be9218605c8e1c`。

| 指标 | 值 |
| --- | --- |
| core.l0 | 97.0 |
| core.reconstruction_mse | 25.435324005031127 |
| core.explained_variance | 0.826606639166795 |
| reconstruction.loss_recovered | 0.9797510631328847 |
| 消融 logit L2 | 430.14892578125 |
| 加 80 logit L2 | 205.2344512939453 |

Capture 的激活数是 98，benchmark 的 L0 是 97。消融 L2 与 `evaluate_edit` 的 431.2317810058594 接近但不是逐位相同。两套数都来自真实前向，报告并列，不把它们说成同一个量。

### Train：通过（2 步冒烟）

预算：层 0，`blocks.0.hook_resid_post`，`d_sae=64`，`k=8`，topk，2 步，batch 16，上下文 16，32 条本地文本。`task.approved=false` 会拒绝，日志里有 `Native SAE training requires task.approved=true`。

float32 的 Gemma-2-2B 在 12GB 上 OOM（约 11.46 GiB）。`LanguageModelSAERunnerConfig.dtype` 不会传给 HookedTransformer。修复是 `model_from_pretrained_kwargs={"dtype": "float16", "center_writing_weights": false}`，并把 `save_to_disk` 语料用 `load_from_disk` 交给 runner。快照目录名映射成 `google/gemma-2-2b`，否则 TransformerLens 拒绝该路径。

结果：`experiment/IOI/gemma2_2b_base/train/gpuacc_smoke_42/native_sae_training/checkpoint/inference/sae_weights.safetensors`（1189416 字节）。`SAE.load_from_disk` 得到 `TopKSAE`，`d_in=2304`，`d_sae=64`，`W_enc` 形状 `[2304, 64]`。随机输入的 encode 绝对值和 0.5275761485099792，重构 MSE 1.0148569345474243。记录：`extra/train_reload.json`。耗时 15.78 秒。这是冒烟，不是层 25 的完整训练。

## vLLM 生成

主环境的 `vllm` 只有约 30 个 Python 文件，缺少 `logging_utils`，不能导入。没有往主环境重装，以免换掉 cu128 的 torch。

使用 `/workspace/SAEReasoning/.venv` 里已经装好的 vLLM 0.27.1 和 torch 2.13.0+cu130，`PYTHONPATH` 指向本次 SAEBase 检出，调用 `saebase.generate.generate`。Gemma-2 拒绝 float16（数值不稳定），改用 bfloat16。`max_model_len=256`，`gpu_memory_utilization=0.80`，seed 42。脚本必须是真实文件：从 stdin 启动时 vLLM 子进程会找不到 `/workspace/<stdin>`。

结果 `extra/vllm_generate.json`：8 个新 token `[476, 3413, 576, 82777, 235265, 1165, 603, 476]`，文本 ` a city of contrasts. It is a`。与 Generate 任务的未转向基线逐 token 相同。vLLM 的 prompt 长度是 5，没有 HookedTransformer 的 BOS id 2。

## 数据

`saebase.data` 与 `research_datasets.NLP` 在 seed 42、n=32 下生成的 IOI、GreaterThan、SimpleCopying、SubjectVerbAgreement 提示完全相同。`extra/dataset_prompt_identity.json`。源文件 diff 主要是 IOI 名字列表的换行，生成结果一致。

CounterFact 归一化在 `azhx/counterfact` 修订 `c01c413f856ee38f5c080c9fc5e87aff478e2ff9` 的前 20 行上一致。该划分 19728 行。

MMLU-Pro `TIGER-Lab/MMLU-Pro` 修订 `b189ec765aa7ed75c8acfea42df31fdae71f97be`，test 12032 行。

BBH `lukaemon/bbh` 必须指定 config。本次用 `boolean_expressions`，修订 `982bb89fd79532a8ac676a61fc42eb1aeec63f99`，test 250 行。`extra/dataset_bbh.json`。

## B2

rclone 远程 `research-5090`，类型 b2。用 `harness/transfer/b2.py` 做 `copy`（不是 sync），目标 `research-5090:research-5090/saebase-gpu-acceptance-20261027`。载荷是两份不含凭证的小文件。随后 `verify`（`rclone check --one-way`）：0 differences，2 matching files。没有删除桶内其他对象。

## CLI 与 Atlas

源码 `saebase --help` 有完整参数，`--list` 列出 task `sae` 以及 Gemma-2-2B 模型。`python -m dlbase.train --cfg job --help` 列出 dataset、model、task（含 capture 到 train）。

本地 wheel：`wheel/opensae_saebase-0.5.0-py3-none-any.whl`。装进 `/workspace/venvs/saebase-wheel` 后，`saebase --help` 和 `saebase --list` 可用。这个 wheel 不包含 `dlbase`，所以 `python -m dlbase.train` 必须从源码树运行。没有上传 PyPI。

Atlas：`harness.code.build` 与 `harness.visualization.export` 写到验收目录 `atlas/`，没有改仓库里的 harness 页面。`build_pages --check` 对仓库现有页面返回 0。导出计数：core 26，knowledge 29，code 1857，environment 15，context 64。这些是现有能力的导出，没有把节点标成已验证。

## 与 v0.5.0

`saebase/api/features.py`、`saebase/api/model.py`、`saebase/api/__init__.py`、`saebase/sae/loading.py`、`models/hooked_lm_common.py` 与标签 v0.5.0 字节相同。Capture 和编辑数字就是这套实现的数字。没有为了对照再占一次 GPU 重跑同一份代码。

## 代码改动（未提交）

1. `tasks/LLM/train/backend/training.py`：快照目录映射为官方模型 id；float16 传入 `from_pretrained`；本地 `save_to_disk` / JSON 语料；缩小 activation buffer。
2. `tasks/LLM/circuits/operation.py`：把任务配置里的 offload、dtype、batch、特征上限和 lazy encoder 传给已有的 circuit-tracer 包装。默认不填时行为与原来相同，`task.approved` 仍然必须为真。

## 仍然存在的产品缺陷

- 未改路径时，`python -m dlbase.train` 会因为 Hydra 先写入空 `train.log` 而拒绝新的实验目录。
- 发布的 wheel 不包含 `dlbase`。
- 主环境的 vLLM 安装不完整。可用的引擎在 SAEReasoning 的虚拟环境里，版本是 0.27.1。
- Explain 的 Hydra 配置不能携带 Python 可调用对象。带生成器的 `execute` 是实际跑通的入口。
- `heldout_score` 不是语义正确性。

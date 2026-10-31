# QA500 · 实验定义与运行手册

**目录归属** 此文件属于独立 Experiment，仅组合现有 Dataset / Model / SAE / Task。


## 这是一项确定性执行工作

**工作计划** `harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md`。

**固定机器配置** `conf/experiment/qa500/protocol.json`，包含十个数据集、每源500题、十个模型/20个 SAE 层位点、 seed、token预算、所有任务采样规模和参数。研究决策已经完成。Agent 只按照本文件执行、记录失败、恢复原样配置的失败计算、交付收据。禁止修改实验协议以获得“通过”。

**唯一入口** `scripts/experiments/qa500.sh`，它检查源分支、Python依赖、盘空间、22个通用Harness contracts的既有绑定、QA单元测试、全模型矩阵。脚本自动创建本地 `.venv-qa500/` 和 `experiment/qa500/`，所有运行输出留在本地，不会自动推送 HF。

## 0. Agent 收到本工作后做什么

先运行

```bash
git fetch origin
git switch feat/qa-evaluation-500-v1
bash scripts/experiments/qa500.sh preflight
```

然后在真实 GPU 已授权的机器上，仅运行下面 **一条** 标准命令

```bash
bash scripts/experiments/qa500.sh full --allow-local-gpu
```

这条命令依次自动运行软件测试、10数据集冻结、源验证、20个SAE位点的10题真实 Pilot、50,000 QA Baseline、100,000 Capture、4,000个特征消融、1,000个短生成、每位点/数据集两个 source explanation、独立完整性审计、六个HF新配置的本地 staging。它**不会执行 Train、Circuits、自动 HF 线上发布**。任一独立执行单元资源不满足时，记录对应 blocked 与原始报错，继续其他可运行单元，最后以非零退出码与部分审计结果交付；不允许 agent 自行替换样本和模型。

额外的原生实验由单独授权驱动

```bash
bash scripts/experiments/qa500.sh native --allow-native-compute
```

此任务调用上游 `circuit-tracer` 真实 Gemma tracer 和 SAE-Lens bounded TopK training；原生工具未安装、transcoder/gated模型不存在、独立语料不可达时按实际 blocked，不得标记完成。建议先让 Agent 运行 `full`，主数据先完成后再使用 native。

如果 GPU 任务中断，使用同样的命令继续。即使 GPQA 权限阻塞，脚本仍会尝试其他九个冻结数据集并生成明确标识为 partial 的本地 staging；partial staging 无法通过线上发布门槛。主程序对完成的 sample 做哈希身份匹配、拒绝输入/SAE身份变更。失败仍然应该留在不可覆盖的独立 failure file，完整结果应由独立审计评定。

## 1. 单独阶段重跑

| 执行目标 | 完整命令 |
| --- | --- |
| 离线环境/单元测试/contract 检查 | `bash scripts/experiments/qa500.sh preflight` |
| 冻结十个 Hugging Face 数据集 | `bash scripts/experiments/qa500.sh prepare` |
| 全模型十题试跑 | `bash scripts/experiments/qa500.sh pilot --allow-local-gpu` |
| 五千问题 × 二十 SAE 位点及QA baseline | `bash scripts/experiments/qa500.sh core --allow-local-gpu` |
| 特征消融、生成、解释 | `bash scripts/experiments/qa500.sh effects --allow-local-gpu` |
| Circuits 与独立SAE train | `bash scripts/experiments/qa500.sh native --allow-native-compute` |
| 独立计数和来源审计 | `bash scripts/experiments/qa500.sh audit` |
| 独立审计与HF staging | `bash scripts/experiments/qa500.sh stage` |

复杂故障无需决定新方法。 Agent 在原命令重试一次后仍失败，将原始错误及定位信息交付维护者；对没有手段修复的 HF 权限与权重门槛标记 `ENV_BLOCKED`，继续其他独立模型/数据集时只保留真实完成的结果。

运行前检查实际剩余盘空间不少于 20GiB（这只是最低启动下限，真实运行需要更充足的模型缓存空间）。模型设备为 CUDA 时要求 `torch.cuda.is_available()`，不把 CPU/synthetic 输出冒充GPU结果。某些模型需要HF gating，缺失时保留阻塞，不用更小模型替换。

## 2. 模型输入的具体检查

每条数据必须有真实 `sample_id`、`source_revision`、`question`、`context`、`choices`、`answer`、`answer_type`、`prompt_sha256`。输入 `render_prompt` 只包含题干/上下文/答案候选，不包含 gold；`make_input` 使用对应模型的 tokenizer，Gemma3 IT 使用实际 `apply_chat_template(..., add_generation_prompt=True)`。序列超过2048 tokens 需要显式 failure，不能自动截断或替换题目。

QA多选（MMLU、ARC、CommonsenseQA、MedQA、GPQA）以真实 log probabilities 评估所有候选**完整 token 序列**；BoolQ在 Yes/No 上评分；GSM8K 为数值 final answer，Hotpot/BBH 为固定 normalized exact match；MATH-500 使用 `math-verify==0.9.0` 等价判断，若grader不可用则标记 `needs_math_equivalence`。准确率的分母是实际 `correct != null` 的样本数，不把 unsupported 算作答错。

**真实 Model/SAE 校验**：`load_registered`、真实 `torch.no_grad()`、model_forward logits finite、`capture_features` 对应 hook 、最后一个 prompt token、SAE `d_in/d_sae` 对齐、decoder reconstruction MSE有限、真正非零 Top10 SAE ID 位于 [0,d_sae)。该 Top10 为**最后一个 prompt token** 激活，不是整段 prompt 的全部激活 feature。每个 sample 存储 token IDs SHA、真实 SAE 权重 SHA256；每个模型执行收据另外计算真实模型 state_dict SHA256，并核对两个 SAE 层加载了同一套模型权重。禁止将历史 Stage D 的单层存在性证明当成本次实际实验成功。

## 3. 标准结果和失败收据

```text
experiment/qa500/
  inputs/{dataset_id}/
    questions.jsonl
    manifest.json
  results/qa_standard_500_v1/
    _model_baselines/{model_key}/{dataset_id}/
      pilot.jsonl, all.jsonl, all_summary.json
    {model_key}/{sae_key}/L{layer}/{dataset_id}/
      pilot.jsonl, pilot_summary.json
      capture.jsonl, capture_summary.json
      benchmark_summary.json
      intervene.jsonl, intervene_summary.json
      generate.jsonl, generate_summary.json
      explain.json
      *_failures.jsonl
    _receipts/{model_key}/{pilot,core,effects}.json
    _receipts/{pilot,core,effects}_matrix.json
    _native/circuits_gemma2_2b/*.json
  native_train/
    data_and_model.json
    execution/
    summary.json
  neuronpedia/
    mapping.json
    annotations.jsonl
    neuronpedia_sync.json
  hf_stage/
    qa_v1_inputs/data.csv
    qa_v1_sites/data.csv
    qa_v1_runs/data.csv
    qa_v1_metrics/data.csv
    qa_v1_per_question/data.csv
    qa_v1_task_records/data.csv
    publication_manifest.json
  logs/
    driver_{phase}_{utc}.log
    frozen_plan.json, frozen_sites.json
```

所有完成行含真正实测的数值、源ID/sha、model/SAE/site/layer hook、feature activation/feature ID、`status=executed`。Task的结果是独立原始记录，不能只上报一张成功表。恢复时必须校验相同输入和同一 SAE 权重。如果只有 499 个实际可处理样本，那就是 499/500 partial，而不是 500/500 的替代样本。

每个 Agent 最终交付 `_receipts`、`qa_report audit`、`hf_stage/publication_manifest.json`（仅在 staging 成功后存在）、全部错误摘要、`git rev-parse HEAD` 和实际硬件信息；必要时附 5 条去敏示例供独立审阅。不得删除失败日志或原始证据。

## 4. Neuronpedia feature explanation

Source-backed 解释只做**核对后关联**，而非自动猜测。Step 1 用完整 Capture 的实际 SHA 确认模型/SAE/layer 精确坐标；Step 2 获得 Neuronpedia 公开 source 的**原始对应关系证据**；Step 3 写入 `experiment/qa500/neuronpedia/mapping.json`，该文件至少列出 site_id 对应 `{"model":"...","source":"...","sae_sha256":"<64hex>"}`；Step 4 执行以下命令（不调用大量单feature API）

```bash
python -m experiments.qa500.neuronpedia_sync \
  --mapping experiment/qa500/neuronpedia/mapping.json \
  --result-root experiment/qa500/results \
  --output experiment/qa500/neuronpedia/annotations.jsonl
```

之后带 `--neuronpedia-map` 和 `--neuronpedia-records` 的 `qa_batch effects` 或 `qa_pipeline capture` 可读取精确匹配描述。缺少经过独立验证的 source 映射时，主动留空为 `explanation=null,status=unmapped`。Neuronpedia 的自然语言解释是第三方提供的 hypothesis，不能通过 `heldout_top10_count` 判断解释语义正确。

原生 Circuits 的 transcoder nodes 也不能当作某个独立SAE字典的 feature解释。所有解释需要来源链接及版本，不能填人工想象文本。

## 5. 审计、HF staging 与线上发布

审核命令

```bash
python -m experiments.qa500.report audit
python -m experiments.qa500.report stage
```

审计要求 10 个源各500题、100个 model×dataset baseline 完成、200个 site×dataset capture 完成、200个干预任务单元各20条、200个生成任务单元各5条，并验证所有 sample ID、SAE hash、feature维度、token/源身份与数值。某些子任务如 Explain、Circuits、Train 在主矩阵以外，使用原生 source-only/not-verified 状态单独展示。

只有研究人员明确批准对 HF dataset 的修改、完成外部权利审核和独立验收后，才可在本地加 `--approved` 执行

```bash
python -m experiments.qa500.publish \
  --stage experiment/qa500/hf_stage --approved
```

此命令将 QA500 六个新 config 作为新 cohort 加入 `OpenSAE/SAEBase-Evaluation`，对 HF parent commit 作并发保护，不能覆盖以前的 25 个配置。Agent 不得因为用户批准运行 GPU 就自行执行 publish。

当前代码包含实现与离线测试；**实际500题、100,000个 SAE 测量、20模型位点、native GPU 执行及 HF 写入均需本地真实机器提供证据后才算完成**。

## 6. Agent 可直接复制的 Prompt

> 使用 SAEBase 分支 `feat/qa-evaluation-500-v1`。阅读 `harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md`、`experiments/qa500/README.md` 和 `conf/experiment/qa500/protocol.json`，并使用 Registry 解析各阶段 CT contract。所有研究选择已经冻结，禁止自行修改题目、样本数量、模型、层、seed、feature selection、metrics、prompt模板、训练语料、HF schema 或发布门槛。先执行 `bash scripts/experiments/qa500.sh preflight`，然后在已经获得本地GPU授权的机器上执行 `bash scripts/experiments/qa500.sh full --allow-local-gpu`。失败时保留真实报错和状态，对同一命令恢复重试，不替换不兼容资产，不将 synthetic/failure 记录为成功。独立 native 工作只在收到 `--allow-native-compute` 许可后执行。最终提交可审查的 run receipts、source/model/SAE SHA、日志、审计结果和拟发布 staging；等待人工批准后才可进行实际 HF 线上发布。

# 数据集与提示来源

## 目的和主要功能

说明当前 Dataset 登记与既有 PromptSource 输入的对应关系，帮助选择实际数据及配置。

## 适用范围

用于数据接入、样本准备与任务输入选择。真实可选项由 `datasets/plugins.py` 和 `conf/dataset/` 维护。

## Dataset 入口

当前 Dataset 登记包括 HotpotQA、BBH、CounterFact、GreaterThan、IOI、MMLUPro、SimpleCopying、RAVEL、ARCChallenge、GSM8K、HumanEval、TruthfulQA、HarmBench 和 XSTest。任务读取各 Dataset 声明的样本、划分与来源。准备和读取方式见 `datasets/AGENTS.md`，配置采用实际数据路径。

## 既有 PromptSource 接口

下面保存既有提示来源的名称与输入用途；其中的协议编号对应原数据选择记录。Dataset 与 PromptSource 各自按实际登记接口选择。

| Behavior | Hydra group | Notes |
|----------|-------------|-------|
| SimpleCopying | `simple_copying` | Sequence / induction-style copying |
| GreaterThan | `greater_than` | Numeric / comparative relation |
| IOI | `ioi` | Indirect Object Identification |
| SubjectVerbAgreement | `subject_verb_agreement` | Morphosyntactic agreement |
| CounterFact | `counterfact` | Factual association probes |
| MMLUPro | `mmlu_pro` | TIGER-Lab MMLU-Pro MCQ (0-shot direct by default) |
| BBH | `bbh` | Big-Bench Hard；协议 05 / 10 题单 |
| ARCChallenge | `arc_challenge` | ARC-Challenge；协议 05 测试 IDs |
| MedQA | `medqa` | MedQA-USMLE；协议 05 测试 IDs |
| MATH500 | `math500` | MATH-500；`max_tokens=2048` |
| GPQAExtended | `gpqa_extended` | GPQA Extended；gated 时协议 10 记 blocked |
| LiveCodeBench | `livecodebench` | 协议 10：05 的 350 + 150（seed `20260913`） |

Artifacts live under `{paths.data_dir}/{Behavior}/{model_key}/v1/`.

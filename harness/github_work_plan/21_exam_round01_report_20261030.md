# Work Plan 21.1 · round01 正式综合考试结果

记录日期：2026-10-30  
性质：**真实 SAEBase 工作流考试的独立 Review 归档**。不是 Atlas 审批、不是产品发布、不是评分器修订。

考试执行当时禁止改 GitHub。本文件由随后的用户指令授权归档到仓库。

## 1. 身份

| 项目 | 记录 |
| --- | --- |
| 仓库 | https://github.com/OpenSAE/SAEBase |
| 试卷分支 | `exam/wp21-1-ready` |
| 冻结提交 | `8a78be1209bb30c652d500f934c226a5d54684c9` |
| 产品基线 SHA | `41a54cbfdee5cce24d3d8f7517aadef0c752c5ac` |
| Run ID | `round01` |
| 授权来源 | 2026-10-30 12:00 UTC-5 Cursor 用户指令，明确启动 R01–R20，独立工作树，真实生产接口，环境阻塞记 `ENV_BLOCKED` |

机器成绩表：[`harness/tests/wp21/runs/round01/summary.md`](../tests/wp21/runs/round01/summary.md)。逐题独立判定：[`harness/tests/wp21/runs/round01/reviews/`](../tests/wp21/runs/round01/reviews/)。

## 2. 成绩

PASS 只能来自独立 `review_final.json`。SELF_TESTED / EVIDENCE_READY_FOR_REPLAY 都不是 PASS。

| 题 | 最终状态 | 分数 |
| --- | --- | ---: |
| R01–R07 | PASS | 100 |
| R08 | ENV_BLOCKED | 0 |
| R09–R16 | PASS | 100 |
| R17 | FAIL | 57 |
| R18–R19 | PASS | 100 |
| R20 | INCOMPLETE | 65 |

计数：**PASS=17，ENV_BLOCKED=1，FAIL=1，INCOMPLETE=1**。未把阻塞或审计缺口写成 PASS。

## 3. 环境

- Python 3.12.14，torch 2.11.0+cu128，RTX 5070 12GB，一卡一模型作业
- Gemma `google/gemma-2-2b` revision `c5ebcd40d208330abc697524c919956e692655cf`
- SAE `gemma-scope-2b-pt-res` revision `fd571b47c1c64851e9b1989792367b9babb4af63`，layer 25，`blocks.25.hook_resid_post`
- 正式 probe：`REAL_MODEL_PROBE_OK_NOT_EXAM_PASS`（带 PYTHONPATH 的第二次探测；无 PYTHONPATH 的首次探测按不可变失败保留）

原始实验叶子、命令日志和大张量留在考试工作树，不随本报告提交。

## 4. 非 PASS 归因

### R08 `ENV_BLOCKED`

需要冻结 Pythia-14m revision `d31f659f077f9ea500c4d7faa999893d7f28cbdf`。本地缓存不存在。按手册不下载、不用 toy。

### R17 `FAIL`（57）

三次 native SAELens 训练（各 1024 token，seed 7/7/11）和 validation heldout encode/decode 实际发生。seed 已进入 `plan.json` 与 runner。同 seed 权重 `W_dec` max_abs≈0.083，达不到冻结 rtol=1e-4 / atol=1e-5，按题卡保留失败。另：`native_training_logs/` 因审计器按尾斜杠精确文件名匹配而无法 `EVIDENCE_READY_FOR_REPLAY`。

### R20 `INCOMPLETE`（65）

独立正反例全部复跑通过（真实 `capture_features`、`evaluate_edit`、`BenchmarkConfig` + `runner.execute(real=True)`）。`benchmark_output/` 同样无法通过目录产物审计，硬门槛 `candidate_evidence_ready=false`，故不是 PASS。

## 5. Harness 改进建议（本归档不改评分器）

1. 目录类 `required_artifacts` 的尾斜杠应匹配目录前缀，而不是 `Path.as_posix()` 精确文件名。
2. `benchmarks.runner` 未向 `BatchEngine` 提供 `pad_token_id`，变长样本会失败；需文档或转发 pad。
3. runner 默认 `load_registered` 为 CPU；GPU 题应记录 CUDA loader。
4. 16-step CUDA TopK 很难满足 1e-4/1e-5 同 seed 容差；或收紧确定性，或题卡改为记录比较。
5. probe 必须设置 `PYTHONPATH` 为包装 ROOT，否则首次失败记录不可改。
6. `record.py` CLI 把子进程非零码当作自身退出码，bash 循环会被负例打断。

## 6. 本归档不表示

- 不修改冻结 cases、Contracts、评分器、Hooks、Runtime 语义
- 不合并考生产品 diff 进 `main`
- 不构成 Atlas / 人工 / 发布批准

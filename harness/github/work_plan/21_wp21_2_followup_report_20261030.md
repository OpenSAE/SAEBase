# Work Plan 21.2 · 本地 GPU Examiner 归档（2026-10-30）

性质：**WP21.2 F00–F05 本地复核归档**。不是 Atlas 审批、不是产品发布、不是评分器修订，也不改写 WP21.1 round01 历史 `review_final`。

- 工具分支：`harness/wp21-2-hardening-20261030`（`473784823419f57099b1a6e0f62aafca54f89c23`），PR #106
- 冻结试卷：`8a78be1209bb30c652d500f934c226a5d54684c9`
- 产品基线：`41a54cbfdee5cce24d3d8f7517aadef0c752c5ac`
- 新考试 run：`round02`（仅 R08）
- 机器摘要：[`harness/tests/wp21/runs/round02/`](../../tests/wp21/runs/round02)

未纳入本归档：考生工作树、GPU 张量、checkpoint、Reviewer ZIP、含绝对路径的原始 stdout/stderr。

## 1. 成绩（新旧并列）

| 题 | round01（不变） | round02 / 复核 |
| --- | --- | --- |
| R01–R07, R09–R16, R18, R19 | PASS 100 | 未重考 |
| R08 | ENV_BLOCKED 0 | **PASS 100**（5/5 + 4/4） |
| R17 | FAIL 57 | 新审计 `EVIDENCE_READY_FOR_REPLAY`；同 seed `W_dec` 仍超容差。旧 FAIL 保留 |
| R20 | INCOMPLETE 65 | 新审计 `EVIDENCE_READY_FOR_REPLAY`；未自动改 PASS |

## 2. F00–F05

| ID | 状态 | 要点 |
| --- | --- | --- |
| F00 | DONE | `check-spec` SPEC_VALID；pytest 108 passed；补齐冻结 Pythia-14m 后 preflight 20/20。Gemma+SAE CUDA probe 与 Pythia CPU probe 均为 `REAL_MODEL_PROBE_OK_NOT_EXAM_PASS` |
| F01 | DONE | 环境准备者锁定 `EleutherAI/pythia-14m` revision `d31f659f077f9ea500c4d7faa999893d7f28cbdf`。新独立评审 PASS。历史 ENV_BLOCKED 不动 |
| F02 | DONE | 目录审计缺口已由 WP21.2 工具修复。`W_dec` max_abs=0.0827，failing_elements=147398/147456。seed 已进入 SAELens；第一个差异来源是 CUDA TopK 非确定性 |
| F03 | DONE | `benchmark_output/` 7 个可哈希文件。足以发起新独立 finalize，本归档不改 round01 INCOMPLETE |
| F04 | DONE | 7 vs 32 tokens：无 pad → `mixed token lengths require explicit pad_token_id`；pad=0 → RAN。`load_registered` 默认 cpu；显式 cuda 时参数在 cuda:0 |
| F05 | DONE | 15 个 canonical Reviewer 目录可归档；R03/R06/R13/R15/R18 脚本哈希漂移，完整证据在对应 `*b`。ZIP 留本地 |

## 3. R08 round02

- 架构 GPTNeoXForCausalLM，6 层，宽 128，CPU float32
- `model.safetensors` sha256 `116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1`
- identity patch max_abs = 0.0
- [`reviews/R08/review_final.json`](../../tests/wp21/runs/round02/reviews/R08/review_final.json)

## 4. 归因

| 代码 | 责任方 | 说明 |
| --- | --- | --- |
| ENV_BLOCKED | environment | round01 R08 缺 Pythia；本轮已钉扎后重考 |
| EVALUATOR_SCOPE_GAP | evaluator | round01 目录产物审计；工具已修，历史成绩保留 |
| IMPLEMENTATION_ERROR | baseline_product | R17 CUDA TopK 同 seed 不可复现；Benchmark runner 不转发 device/`pad_token_id` |

产品 runner 修复应另开产品 PR，见 [`product_fix_candidate.md`](../../tests/wp21/runs/round02/benchmark/product_fix_candidate.md)。Harness 评分与冻结容差未改。
